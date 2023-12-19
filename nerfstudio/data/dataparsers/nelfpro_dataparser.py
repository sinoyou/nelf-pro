from __future__ import annotations
from curses import meta

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.utils.plotly_utils_nelfpro import plot_point3d

from nerfstudio.data.utils.probe_sample import FactorPoseGenerator

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.data.dataparsers.raw_dataset_loader.llff_dataset_raw_loader import LLFFRawLoader
from nerfstudio.data.dataparsers.raw_dataset_loader.kitti360_dataset_raw_loader import KITTI360RawLoader
from nerfstudio.data.dataparsers.raw_dataset_loader.bungee_dataset_raw_loader import BungeeRawLoader
from nerfstudio.data.dataparsers.raw_dataset_loader.nerf_dataset_raw_loader import NeRFStudioRawLoader

CONSOLE = Console(width=120)

@dataclass
class NeLFProDataParserConfig(DataParserConfig):
    """Configuration for the SpherRiFDataParser."""

    _target: Type = field(default_factory=lambda: NeLFProDataParser)
    
    # raw dataset loader config
    raw_loader: Literal["llff", "kitti360", "bungee", "nerfstudio"] = "llff"
    data: Path = Path("./data/please_fill_in_the_path_to_your_raw_dataset")
    eval_interval: int = 8
    eval_type: Literal["dev"] = "dev"

    # camera pose config
    scale_factor: float = 1.0
    downscale_factor: Optional[int] = None
    scenebox_scale: int = 1.0
    orientation_method: Literal["none", "up", "pca"] = "up"
    center_poses: bool = True
    auto_scale_poses: bool = True

    # probe generation config
    data_num_core: int = 3
    data_num_basis: int = 64
    use_kmeans_core: bool = True
    use_fps_basis: bool = True
    factor_pos_noise_scale: float = 0.02

    # point cloud config
    point_cloud_sample_num: int = -1


@dataclass
class NeLFProDataParser(DataParser):
    """Dataset Parser for Raw Mipnerf360 dataset."""

    config: NeLFProDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        data_dir = Path(self.config.data)

        if self.config.raw_loader == "llff":
            raw_loader = LLFFRawLoader(data_dir, downscale_factor=self.config.downscale_factor, partition_index=None)
        elif self.config.raw_loader == "kitti360":
            raw_loader = KITTI360RawLoader(data_dir, downscale_factor=self.config.downscale_factor, partition_index=None, eval_type=self.config.eval_type)
        elif self.config.raw_loader == 'bungee':
            raw_loader = BungeeRawLoader(data_dir, downscale_factor=self.config.downscale_factor, partition_index=None)
        elif self.config.raw_loader == "nerfstudio":
            raw_loader = NeRFStudioRawLoader(data_dir, downscale_factor=self.config.downscale_factor, partition_index=None)
        else:
            raise NotImplementedError("unknown raw dataset loader {}".format(self.config.raw_loader))
        
        loaded_raw = raw_loader.get_loaded_data()
        image_filenames = loaded_raw['image_filenames']
        poses = loaded_raw['poses']
        fx = loaded_raw['fx']
        fy = loaded_raw['fy']
        cx = loaded_raw['cx']
        cy = loaded_raw['cy']
        height = loaded_raw['height']
        width = loaded_raw['width']
        distort = loaded_raw['distort']
        point3d_xyz = loaded_raw['point3d_xyz']
        point3d_rgb = loaded_raw['point3d_rgb']

        train_indices, eval_indices = raw_loader.get_train_val_indices(self.config.eval_interval)

        if split == "train":
            indices = train_indices
        elif split in ["val", "test"]:
            indices = eval_indices
        else:
            raise ValueError(f"Unknown split {split}")

        # orinetation and translation
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_poses=self.config.center_poses,
        )
        
        # point cloud processing
        if point3d_xyz is not None:
            # sparse sampling
            if self.config.point_cloud_sample_num > 0:
                self.config.point_cloud_sample_num = min(self.config.point_cloud_sample_num, point3d_xyz.shape[0])
                sample_indices = np.random.choice(point3d_xyz.shape[0], self.config.point_cloud_sample_num, replace=False)
                point3d_xyz = point3d_xyz[sample_indices]
                point3d_rgb = point3d_rgb[sample_indices]
                CONSOLE.log(f'Sample {self.config.point_cloud_sample_num} points from point cloud.')
            # coordinate transformation
            point3d_xyz_homo = np.concatenate([point3d_xyz, np.ones((point3d_xyz.shape[0], 1))], axis=1)  # num_points, 4
            point3d_xyz = point3d_xyz_homo @ transform.T.numpy()  # num_points, 3
        
        # scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        CONSOLE.log(f"scale factor is {scale_factor}")
        poses[:, :3, 3] *= scale_factor
        point3d_xyz = point3d_xyz * scale_factor if point3d_xyz is not None else None

        # generate basis factor positions
        if self.config.use_fps_basis:
            CONSOLE.log(f'Generate {self.config.data_num_basis} basis factors with greedy furthest point sampling strategy.')
            sampler = FactorPoseGenerator(strategy='fps')
            index = sampler.sample(poses, self.config.data_num_basis)
            basis_factor_poses = poses[index][:, :3, 3]
        else:
            CONSOLE.log(f'Generate {self.config.data_num_basis} basis factors with random strategy.')
            sampler = FactorPoseGenerator(strategy='random')
            index = sampler.sample(poses, self.config.data_num_basis)
            basis_factor_poses = poses[index][:, :3, 3]
        basis_factor_poses += FactorPoseGenerator.get_random_offset(basis_factor_poses.shape, scale=self.config.factor_pos_noise_scale, seed=1737)
        basis_factor_list = []
        for i in range(basis_factor_poses.shape[0]):
            _data = {
                'x': float(basis_factor_poses[i, 0].detach().item()),
                'y': float(basis_factor_poses[i, 1].detach().item()),
                'z': float(basis_factor_poses[i, 2].detach().item()),
            }
            basis_factor_list.append(_data)
        CONSOLE.log('basis factor poses: {}'.format(basis_factor_list))

        # generate core factor positions
        if self.config.use_kmeans_core:
            CONSOLE.log(f'Generate {self.config.data_num_core} core factor positions with kmeans cluster strategy.')
            sampler = FactorPoseGenerator(strategy='kmeans', return_type='position')
            core_factor_poses = sampler.sample(poses, self.config.data_num_core)
        else:
            CONSOLE.log(f'Generate {self.config.data_num_core} core factor positions with random strategy.')
            sampler = FactorPoseGenerator(strategy='random')
            index = sampler.sample(poses, self.config.data_num_core)
            core_factor_poses = poses[index][:, :3, 3]
        core_factor_poses += FactorPoseGenerator.get_random_offset(core_factor_poses.shape, scale=self.config.factor_pos_noise_scale, seed=1737)
        core_factor_list = []
        for i in range(core_factor_poses.shape[0]):
            _data = {
                'x': float(core_factor_poses[i, 0].detach().item()), 
                'y': float(core_factor_poses[i, 1].detach().item()), 
                'z': float(core_factor_poses[i, 2].detach().item()), 
            }
            core_factor_list.append(_data)
        CONSOLE.log('core factor poses: {}'.format(core_factor_list))

        # combine core and basis config
        probe_config = {}
        probe_config['basis'] = basis_factor_list
        probe_config['core'] = core_factor_list
        probe_config = [probe_config]  # embeded in a list for compatibility with nerf dataset loader

        # filter by the indices given by the split
        image_filenames = [image_filenames[i] for i in indices]
        poses = poses[indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distort_params = torch.stack(distort, dim=0)[idx_tensor]

        # build scene aabb box
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scenebox_scale, -self.config.scenebox_scale, -self.config.scenebox_scale], 
                 [self.config.scenebox_scale, self.config.scenebox_scale, self.config.scenebox_scale]
                ], dtype=torch.float32
            )
        )

        camera_type = CameraType.PERSPECTIVE
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distort_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            probe_config=probe_config, 
            image_filenames=[e.name for e in image_filenames],
        )

        assert raw_loader.get_downscale_factor() is not None, "downscale factor should not be None"
        if raw_loader.get_downscale_factor() > 1:
            cameras.rescale_output_resolution(scaling_factor=1.0 / raw_loader.get_downscale_factor(), round_hw=True)

        other_data = dict()
        plot_data = self._get_plotly_data(split, cameras, cameras.probe, point3d_xyz, point3d_rgb)
        other_data['scene_plotly'] = plot_data 

        dataparser_outputs = DataparserOutputs(image_filenames=image_filenames, cameras=cameras, scene_box=scene_box, other_data=other_data)

        return dataparser_outputs
    
    def _get_plotly_data(self, split, cameras, probe=None, point3d_xyz=None, point3d_rgb=None):
        plot_data = []
        plot_data += cameras.get_plotly(camera_group='train' if split == 'train' else 'eval')
    
        # we only plot factors and point clouds when the split = 'train', otherwise the eval split plot would be same and abundant. 
        if split == 'train':
            if probe is not None:
                plot_data += probe.get_plotly()
        
            if point3d_xyz is not None:
                lb = np.percentile(point3d_xyz, 10, axis=0)
                ub = np.percentile(point3d_xyz, 90, axis=0)
                point3d_mask = (point3d_xyz[:, 0] > lb[0]) & (point3d_xyz[:, 0] < ub[0]) & (point3d_xyz[:, 1] > lb[1]) & (point3d_xyz[:, 1] < ub[1]) & (point3d_xyz[:, 2] > lb[2]) & (point3d_xyz[:, 2] < ub[2])
                point3d_xyz_plot = point3d_xyz[point3d_mask]
                point3d_rgb_plot = point3d_rgb[point3d_mask]
                plot_data += plot_point3d(point3d_xyz_plot, point3d_rgb_plot)
        
        return plot_data
