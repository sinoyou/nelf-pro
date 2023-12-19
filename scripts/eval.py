#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Name of the output images dir.
    output_images_path: Path = Path("output_images/")

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        metrics_dict, pack = pipeline.get_average_eval_image_metrics()
        metrics_dict_list, images_dict_list = pack
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_images_path.mkdir(parents=True, exist_ok=True)

        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

        image_groups = images_dict_list[0].keys()
        for group in image_groups:
            if group == "image_filename":
                continue
            for image_dict in images_dict_list:
                if group == 'depth':
                    cv2.imwrite(
                        str(self.output_images_path / Path(f"{group}_{image_dict['image_filename']}.png")),
                        (image_dict[group].cpu().numpy() * 255.0).astype(np.uint8)[..., ::-1],
                    )
                else:
                    cv2.imwrite(
                        str(self.output_images_path / Path(f"{group}_{image_dict['image_filename']}.png")),
                        (image_dict[group].cpu().numpy() * 255.0).astype(np.uint8)[..., ::-1],
                    )
        CONSOLE.print(f"Saved rendering results to: {self.output_images_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
