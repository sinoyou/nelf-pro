#!/usr/bin/env python
"""Train a radiance field with nerfstudio.
For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations

import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from rich.console import Console

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.config_merger import merge_customized_config
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import Trainer
from nerfstudio.utils import comms, profiler

CONSOLE = Console(width=120)
DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: cfg.Config, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = Trainer(config, local_rank, world_size)
    trainer.setup()
    trainer.train()

def launch(
    main_func: Callable,
    config: Optional[cfg.Config] = None,
) -> None:
    """Function that spawns muliple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        config (Config, optional): config file specifying training regimen.
    """
    assert config is not None
    # world_size=0 uses one CPU in one process.
    # world_size=1 uses one GPU in one process.
    try:
        main_func(local_rank=0, world_size=1, config=config)
    except KeyboardInterrupt:
        # print the stack trace
        CONSOLE.print(traceback.format_exc())
    finally:
        profiler.flush_profiler(config.logging)


def main(config: cfg.Config) -> None:
    """Main function."""

    config.set_timestamp()
    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.dataparser.data")
        config.pipeline.datamanager.dataparser.data = config.data

    if config.trainer.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.trainer.load_config}")
        read_config = yaml.load(config.trainer.load_config.read_text(), Loader=yaml.Loader)
        if isinstance(read_config, cfg.Config):
            config = read_config
        elif isinstance(read_config, dict):
            config = merge_customized_config(config, read_config)
        else:
            raise ValueError(f"Invalid loaded config type: {type(read_config)}")

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
