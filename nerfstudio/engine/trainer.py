# zinyou note:
# trainer in principle should not be modified. 
# modification should be only within pipeline or lower level: model and data manager. 

# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from typing import Dict, List, Tuple
from numpy import isin
import json
from pathlib import Path

import torch
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

import plotly.graph_objects as go

from nerfstudio.configs import base_config as cfg
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers, setup_optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.utils.load_utils import check_load_step

CONSOLE = Console(width=120)


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

        self.base_dir = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        # set up viewer if enabled
        viewer_log_path = self.base_dir / config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and local_rank == 0:
            self.viewer_state, banner_messages = viewer_utils.setup_viewer(config.viewer, log_filename=viewer_log_path)
        self._check_viewer_warnings()
        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / config.logging.relative_log_dir
        writer.setup_event_writer(config, log_dir=writer_log_path)
        writer.setup_local_writer(
            config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
        profiler.setup_profiler(config.logging)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank, load_step = check_load_step(self.config),
        )
        self.optimizers = setup_optimizers(self.config, self.pipeline.get_param_groups())

        self._load_checkpoint()

        self.training_attributes = TrainingCallbackAttributes(
                optimizers=self.optimizers,  # type: ignore
                grad_scaler=self.grad_scaler,  # type: ignore
                pipeline=self.pipeline,  # type: ignore
                config=self.config.trainer,  # type: ignore
            )

        self.callbacks = self.pipeline.get_training_callbacks(self.training_attributes)

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self._init_viewer_state()

        # plotly scene 
        if self.config.trainer.visualize_scene:
            scene_plotly_data = self.pipeline.get_scene_plotly_figure()
            if scene_plotly_data:
                fig = go.Figure(data=scene_plotly_data)
                writer.put_plotly(name="scene", figure=fig)
                CONSOLE.log("Scene plotly is uploaded.")

        self.training_time = 0.0

        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            step = self._start_step
            self._update_viewer_state(step)
            for step in range(self._start_step, num_iterations):
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:

                    self.pipeline.train()

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )
                    
                    start_time = time.time()

                    # time the forward pass
                    loss, loss_dict, metrics_dict = self.train_iteration(step)

                    self.training_time += time.time() - start_time

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.config.pipeline.datamanager.train_num_rays_per_batch / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)

                if step_check(step, self.config.trainer.steps_per_save):
                    self.save_checkpoint(step)

                self.eval_iteration(step)

                writer.write_out_storage()
            # save checkpoint at the end of training
            self.save_checkpoint(step)
            self.save_running_performance()

            CONSOLE.rule()
            CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
            if not self.config.viewer.quit_on_train_completion:
                CONSOLE.print("Use ctrl+c to quit", justify="center")
                self._always_render(step)

    @check_main_thread
    def _always_render(self, step):
        if self.config.is_viewer_enabled():
            while True:
                self.viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(step)

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if self.config.is_viewer_enabled():
            string = (
                "[NOTE] Not running eval iterations since only viewer is enabled."
                " Use [yellow]--vis wandb[/yellow] or [yellow]--vis tensorboard[/yellow] to run with eval instead."
            )
            CONSOLE.print(f"{string}")

    @check_main_thread
    def save_running_performance(self):
        output_path = self.checkpoint_dir / f"../running_performance.json"

        performance_info = {
            "n_parameters": self.pipeline.model.n_parameters() / 1024 / 1024,
            "training_time": self.training_time,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(performance_info, indent=2), "utf8")

        CONSOLE.log(f'model parameters: {performance_info["n_parameters"]}, training time: {performance_info["training_time"]}')

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            dataset=self.pipeline.datamanager.train_dataset,
            start_train=self.config.viewer.start_train,
        )
        if not self.config.viewer.start_train:
            self._always_render(self._start_step)

    @check_viewer_enabled
    def _update_viewer_state(self, step: int):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as _:
            num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
            try:
                self.viewer_state.update_scene(self, step, self.pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert self.viewer_state.vis is not None
                self.viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int):
        """Performs update on rays/sec calclation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_dir
        # try to find checkpoint dir 
        if load_dir is None:
            load_dir_try =  self.config.get_checkpoint_dir()
            if load_dir_try.exists():
                load_dir = load_dir_try
        
        if load_dir is not None:
            load_step = self.config.trainer.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                # load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
                load_step = sorted(int(x.replace('-', '.').split('.')[-2]) for x in os.listdir(load_dir))[-1]
            load_path = Path(load_dir) / Path(f"model.{load_step:09d}.ckpt")
            if not load_path.exists():
                load_path = Path(load_dir) / Path(f'step-{load_step:09d}.ckpt')  # old format
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], load_step, load_dir)
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.trainer.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}, starting from step {self._start_step}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"model.{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        self.pipeline.call_customized_save(step=step, checkpoint_dir=self.checkpoint_dir)
        # possibly delete old checkpoints
        if self.config.trainer.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if int(str(f).split('.')[-2]) != step:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str = self.device.split(":")[0]
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        # try:
        #     torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), 10.0, error_if_nonfinite=True)
        #     # torch.nn.utils.clip_grad_value_(self.pipeline.model.parameters(), 10.0)
        # except Exception as e:
        #     CONSOLE.print(f"Error: {e}")
        #     CONSOLE.print("Error: gradient clipping detected nonfinite number, skipping updating. ")
        #     self.optimizers.scheduler_step_all(step)
        #     self.optimizers.zero_grad_all()
        #     return loss, loss_dict, metrics_dict
        
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step):
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.trainer.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.trainer.steps_per_eval_image):
            time.sleep(1)
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.trainer.steps_per_eval_all_images):
            time.sleep(5)
            metrics_dict_ave, raw_data = self.pipeline.get_average_eval_image_metrics(step=step)

            # display evaluated metrics and images per view (the storage requirement is huge)
            if self.config.trainer.visualize_seperate_eval_images:
                # evaluating images from all views
                metrics_dict_list = raw_data[0]
                metrics_groups = metrics_dict_list[0].keys()
                for group in metrics_groups:
                    if group == "image_filename":
                        continue
                    group_name = f'Verbose Eval - Metric({group})'
                    for metrics_dict in metrics_dict_list:
                        writer.put_scalar(name=f'{group_name}/{metrics_dict["image_filename"]}', scalar=metrics_dict[group], step=step)

                images_dict_list = raw_data[1]
                image_groups = images_dict_list[0].keys()
                for group in image_groups:
                    if group == "image_filename":
                        continue
                    group_name = f'Verbose Eval - Image({group})'
                    for image_dict in images_dict_list:
                        writer.put_image(name=f"{group_name}/{image_dict['image_filename']}", image=image_dict[group], step=step)

            # display average evaluated metrics and images
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict_ave, step=step)
        