import os
import numpy as np
from pathlib import Path, PurePath
from PIL import Image

from nerfstudio.data.utils.colmap_utils import read_points3d_binary

from rich.console import Console
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.raw_dataset_loader.raw_loader import RawLoader

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

class LLFFRawLoader(RawLoader):
    def get_loaded_data(self) -> dict:
        num_skipped_image_filanems = 0

        # obtain all image files in the directory
        fnames_raw = []  # image names without considering resolution constraint.
        for frame in os.listdir(os.path.join(self.data_dir, "images")):
            if frame.endswith(".JPG") or frame.endswith(".jpg"):
                fnames_raw.append(os.path.join("images", frame))

        CONSOLE.log(f"Detected image files number: {len(fnames_raw)}")

        # adjust resolution and assign correct directory
        fnames = []
        for frame in fnames_raw:
            file_path_scale = self._get_frame(PurePath(frame), self.data_dir)
            if file_path_scale.exists():
                fnames.append(file_path_scale)
            else:
                num_skipped_image_filanems += 1

        # obtain loaded image width and height
        img_width, img_height = self._get_size(fnames[0])
        CONSOLE.log(f"Loaded image resolution: {img_height}x{img_width}")

        if num_skipped_image_filanems > 0:
            CONSOLE.log(f"Skip {num_skipped_image_filanems} images due to not correct resolution images found.")
        CONSOLE.log(f"Detected image files number after resolution check: {len(fnames)}")

        # sort images by names
        inds = np.argsort(fnames)
        frames = [fnames[i] for i in inds]
        CONSOLE.log(f"Detected total image files number: {len(frames)}")
        if self.partition_index is not None:
            frames = frames[self.partition_index[0]:self.partition_index[1]]
            CONSOLE.log(f"Load dataset partition of {self.partition_index[0]}-{self.partition_index[1]}")
        
        # load poses data
        poses_bounds = np.load(os.path.join(self.data_dir, 'poses_bounds.npy'))
        assert poses_bounds.shape[0] == len(frames), "loaded poses and image frames do not match {} and {}.".format(poses_bounds.shape[0], len(frames))
        poses_bounds = poses_bounds[:, :15].reshape(poses_bounds.shape[0], 3, 5)
        poses_matrix = np.concatenate(
            [poses_bounds[:, :3, :4], np.tile(np.array([0, 0, 0, 1]), (poses_bounds.shape[0], 1, 1))], axis=1
        )
        assert (poses_bounds[:, 0, 4] == poses_bounds[0, 0, 4]).all(), "image height is not consistent."
        assert (poses_bounds[:, 1, 4] == poses_bounds[0, 1, 4]).all(), "image width is not consistent."

        # load point clouds (if available)
        point3d_xyz = None
        point3d_rgb = None
        if os.path.exists(os.path.join(self.data_dir, 'sparse', '0', 'points3D.bin')):
            point3d = read_points3d_binary(os.path.join(self.data_dir, 'sparse', '0', 'points3D.bin'))
            point3d_xyz = np.stack([item.xyz for item in point3d.values()], axis=0)
            point3d_rgb = np.stack([item.rgb for item in point3d.values()], axis=0)
            CONSOLE.log(f"Loaded {point3d_xyz.shape[0]} point cloud points.")

        # camera axis convention: from mipnerf360 to blender/nerf.
        # original: down, right, backwards
        # blender: right, up, backwards
        mipnerf360_to_blender = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        poses_matrix[:, :3, :3] = poses_matrix[:, :3, :3] @ mipnerf360_to_blender.T
        # nothing to do with point cloud here, because coordinate transformation only affects rotation matrix. (no translation)

        # generate the returned data 
        poses, image_filenames = [], []
        fx, fy, cx, cy, height, width, distort = [], [], [], [], [], [], []
        for i, frame in enumerate(frames):
            fx.append(float(poses_bounds[i, 2, 4]))
            fy.append(float(poses_bounds[i, 2, 4]))
            # principle point is assumed to be at the center.
            # careful!: cx corresponds to width while cy corresponds to height.
            cx.append(float(poses_bounds[i, 1, 4] / 2))
            cy.append(float(poses_bounds[i, 0, 4] / 2))
            height.append(int(poses_bounds[i, 0, 4]))
            width.append(int(poses_bounds[i, 1, 4]))

            distort.append(camera_utils.get_distortion_params(k1=0, k2=0, k3=0, k4=0, p1=0, p2=0))

            image_filenames.append(frame)
            poses.append(poses_matrix[i])
        
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
        
    def _get_size(self, filepath: PurePath) -> tuple:
        img = Image.open(filepath)
        w, h = img.size
        return w, h

    def get_train_val_indices(self, eval_interval):
        all_indices = np.arange(len(self.ret_data['image_filenames'])).astype(np.int32)
        filter_func = lambda x: x % eval_interval == 0
        train_indices = all_indices[~filter_func(all_indices)]
        val_indices = all_indices[filter_func(all_indices)]
        return train_indices, val_indices