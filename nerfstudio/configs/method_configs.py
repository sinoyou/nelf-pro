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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nelfpro_dataparser import NeLFProDataParserConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import MultiStepSchedulerWithWarmupConfig

from nerfstudio.models.nelfpro import NeLFProModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

method_configs: Dict[str, Config] = {}
descriptions = {
    'nelf-pro-small': 'default training config for small-scale scenes', 
    'nelf-pro-large': 'default training config for large-scale scenes',
}

method_configs["nelf-pro-small"] = Config(
    method_name="nelf-pro-small",
    trainer=TrainerConfig(
        steps_per_eval_batch=2000, 
        steps_per_eval_image=2000,
        steps_per_eval_all_images=20000,
        steps_per_save=2000, 
        max_num_iterations=20001, 
        mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NeLFProDataParserConfig(data_num_basis=64, data_num_core=3),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeLFProModelConfig(eval_num_rays_per_chunk=1 << 12, num_basis=64, num_core=3),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=20001, warm_up_steps=1),
        },
        "field_mlp": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=20001, warm_up_steps=1),
        },
        "field_basis": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=20001, warm_up_steps=1),
        }, 
        "field_core_angular": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=20001, warm_up_steps=1),
        }, 
        "field_core_radial": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=20001, warm_up_steps=1),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

method_configs["nelf-pro-large"] = Config(
    method_name="nelf-pro-large",
    trainer=TrainerConfig(
        steps_per_eval_batch=2000, 
        steps_per_eval_image=2000,
        steps_per_eval_all_images=60000,
        steps_per_save=5000, 
        max_num_iterations=60001, 
        mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NeLFProDataParserConfig(point_cloud_sample_num=10000, data_num_basis=256, data_num_core=16),
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeLFProModelConfig(eval_num_rays_per_chunk=1 << 12, 
                                 proposal_net_args_list=[{'angular_resolution': 128, 'radial_resolution': 128, 'feat_dim': 8}, 
                                                         {'angular_resolution': 256, 'radial_resolution': 256, 'feat_dim': 8}],
                                 num_basis=256,
                                 num_core=16, 
                                 proposal_weights_anneal_max_num_iters=500),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=60001, warm_up_steps=1),
        },
        "field_mlp": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=60001, warm_up_steps=1),
        },
        "field_basis": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=60001, warm_up_steps=1),
        }, 
        "field_core_angular": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=60001, warm_up_steps=1),
        }, 
        "field_core_radial": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerWithWarmupConfig(max_steps=60001, warm_up_steps=1),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
