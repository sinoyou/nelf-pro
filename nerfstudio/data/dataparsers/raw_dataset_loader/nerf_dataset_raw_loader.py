import os
import numpy as np
from pathlib import Path, PurePath
from PIL import Image

from nerfstudio.data.utils.colmap_utils import read_points3d_binary

from rich.console import Console
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.raw_dataset_loader.raw_loader import RawLoader

from nerfstudio.utils.io import load_from_json
import json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

class NeRFStudioRawLoader(RawLoader):

    def get_loaded_data(self) -> dict:
        meta = load_from_json(self.data_dir / 'transforms.json')

        num_skipped_image_filenames = 0

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break

        fnames = []
        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_frame(filepath, self.data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]
        
        if self.partition_index is not None:
            frames = frames[self.partition_index[0]:self.partition_index[1]]
            CONSOLE.log(f"Load dataset partition of {self.partition_index[0]}-{self.partition_index[1]}")
        else:
            CONSOLE.log(f"Load entire dataset of size = {len(frames)}")

        image_filenames = []
        poses = []
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in frames:
            filepath = PurePath(frame["file_path"])
            fname = self._get_frame(filepath,self.data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            transform_matrix = np.array(frame["transform_matrix"])
            if (transform_matrix[:3, 3] > 100).any() or (transform_matrix[:3, 3] < -100).any():
                CONSOLE.log(f'detect abnormal camera center {transform_matrix[:3, 3]}, skip this image.')
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                fx.append(float(frame["fl_x"]))
            else:
                fx.append(float(meta["fl_x"]))

            if not fy_fixed:
                fy.append(float(frame["fl_y"]))
            else:
                fy.append(float(meta["fl_y"]))

            if not cx_fixed:
                cx.append(float(frame["cx"]))
            else:
                cx.append(float(meta["cx"]))

            if not cy_fixed:
                cy.append(float(frame["cy"]))
            else:
                cy.append(float(meta["cy"]))
            
            if not height_fixed:
                height.append(int(frame["h"]))
            else:
                height.append(int(meta["h"]))
            
            if not width_fixed:
                width.append(int(frame["w"]))
            else:
                width.append(int(meta["w"]))

            if not distort_fixed:
                raise NotImplementedError("No support for variable distortion parameters yet.")
            else:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(transform_matrix)

        # check
        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """

        # load point clouds (if available)
        point3d_xyz = None
        point3d_rgb = None
        if os.path.exists(os.path.join(self.data_dir, 'sparse', '0', 'points3D.bin')):
            point3d = read_points3d_binary(os.path.join(self.data_dir, 'sparse', '0', 'points3D.bin'))
            point3d_xyz = np.stack([item.xyz for item in point3d.values()], axis=0)
            point3d_rgb = np.stack([item.rgb for item in point3d.values()], axis=0)
            CONSOLE.log(f"Loaded {point3d_xyz.shape[0]} point cloud points.")
        
        if os.path.exists(os.path.join(self.data_dir, 'colmap', 'sparse', '0', 'points3D.bin')):
            point3d = read_points3d_binary(os.path.join(self.data_dir, 'colmap', 'sparse', '0', 'points3D.bin'))
            point3d_xyz = np.stack([item.xyz for item in point3d.values()], axis=0)
            point3d_rgb = np.stack([item.rgb for item in point3d.values()], axis=0)
            CONSOLE.log(f"Loaded {point3d_xyz.shape[0]} point cloud points.")

        if isinstance(point3d_xyz, np.ndarray):
            point3d_xyz = point3d_xyz[:, np.array([1, 0, 2])]
            point3d_xyz[..., 2] *= -1

        self.ret_data = {
            'poses': poses, 
            'image_filenames': image_filenames,
            'fx': fx,
            'fy': fy, 
            'cx': cx, 
            'cy': cy, 
            'height': height, 
            'width': width, 
            'distort': distort,
            'point3d_xyz': point3d_xyz,
            'point3d_rgb': point3d_rgb
        }

        # import ipdb
        # ipdb.set_trace()

        return self.ret_data

    def _get_frame(self, filepath: PurePath, data_dir: PurePath, downsample_folder_prefix='images_') -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """
        if self.downscale_factor is None:
            test_img = Image.open(data_dir / filepath)
            w, h = test_img.size
            max_res = max(w, h)
            df = 0
            while True:
                if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                    break
                if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                    break
                df += 1

            self.downscale_factor = 2**df
            CONSOLE.log(f"Auto-detected downscale factor: {self.downscale_factor}")

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        else:
            return data_dir / filepath

    def get_train_val_indices(self, eval_interval):
        all_indices = np.arange(len(self.ret_data['image_filenames'])).astype(np.int32)
        filter_func = lambda x: x % eval_interval == 0
        train_indices = all_indices[~filter_func(all_indices)]
        val_indices = all_indices[filter_func(all_indices)]
        return train_indices, val_indices