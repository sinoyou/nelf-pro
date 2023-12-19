import os
import ipdb
import numpy as np
from pathlib import Path, PurePath
from PIL import Image

from rich.console import Console
from scipy import linalg
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.raw_dataset_loader.raw_loader import RawLoader

from nerfstudio.utils.kitti360_utils import loadCalibrationCameraToPose, load_intrinsics, load_poses

CONSOLE = Console(width=120)

def path_cam0_to_cam1(path: Path):
    """Converts a path from cam0 to cam1."""
    parts = list(path.parts)
    assert parts[-3] == "image_00"
    parts[-3] = "image_01"
    return Path(*parts)

class KITTI360RawLoader(RawLoader):

    def __init__(self, data_dir: Path, downscale_factor: int = None, partition_index: tuple = None, **kwargs):
        super().__init__(data_dir, downscale_factor, partition_index, **kwargs)
        self.img_width = 1408
        self.img_height = 376

    def get_loaded_data(self) -> dict:
        self.downscale_factor = 1
        eval_type = self.other_args['eval_type']
        if eval_type == 'dev':
            self.ret_data, _ = self._load_split(self.data_dir, load_stereo_camera=True, midt=None)
        else:
            raise NotImplementedError(f"Unknown eval_type {eval_type} for KITTI360RawLoader.")

        return self.ret_data

    def _load_split(self, data_dir, load_stereo_camera: bool=True, midt=None) -> dict:
        dataset_dir = data_dir.parent.parent    # e.g. ./data/KITTI360
        task_dir = data_dir.parent              # e.g. ./data/KITTI360/data_2d_nvs_drop50
        scene = data_dir.name                   # e.g. train_01, test_01
        data_dir_dir = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        assert len(data_dir_dir) == 1, f"KITTI360RawLoader only support a single sequence, but got {data_dir_dir}"
        seq_id = data_dir_dir[0]
        img_h, img_w = self.img_height, self.img_width
        CONSOLE.log(f"Loading kitti360 nvs task scene {scene} ...")

        # obtain all training images (from camera_0 and camera_1)
        index_txt = os.path.join(task_dir, '{}.txt'.format(scene))
        with open(index_txt) as f:
            fnames_raw = f.read().splitlines()
        fnames = []
        for frame in fnames_raw:
            # for camera 0 
            fnames.append(Path(os.path.join(data_dir, frame)))
        
        # loading intrinsics 
        K0, R_rect_0 = load_intrinsics(os.path.join(dataset_dir, 'calibration', 'perspective.txt'), 0)
        # K for camera 1 has shape of 3x4 and its[:3, 3] is non-zero (due to rectification for the translation on width direction, but it will be considered by T120)
        _, R_rect_1 = load_intrinsics(os.path.join(dataset_dir, 'calibration', 'perspective.txt'), 1)

        # loading extrinsics (poses)
        pose_file = os.path.join(dataset_dir, 'data_poses', seq_id, 'cam0_to_world.txt')
        frame_idxs, poses_all = load_poses(pose_file)
        cam0_poses = []
        for fname in fnames:
            img_idx = int(fname.name.split('.')[0])
            pose_idx = np.argwhere(frame_idxs == img_idx)[0][0]
            cam0_poses.append(poses_all[pose_idx])
        cam0_poses = np.stack(cam0_poses, axis=0)
        # translation based on middle frame
        # we avoid using invmid because it will cause world coordinate back to the perspective camera coordinate. 
        # invmid = np.linalg.inv(cam0_poses[cam0_poses.shape[0] // 2])
        # cam0_poses = invmid @ cam0_poses
        if midt is None:
            midt = cam0_poses[cam0_poses.shape[0] // 2][:3, 3].copy()
            cam0_poses[:, :3, 3] -= midt
        else:
            cam0_poses[:, :3, 3] -= midt

        # loading camera -> GMU coordinates 
        Tr = loadCalibrationCameraToPose(os.path.join(dataset_dir, 'calibration/calib_cam_to_pose.txt'))
        T0, T1 = Tr['image_00'], Tr['image_01']
        T0 = T0 @ np.linalg.inv(R_rect_0)
        T1 = T1 @ np.linalg.inv(R_rect_1)
        T021 = np.linalg.inv(T1) @ T0 
        T120 = np.linalg.inv(T021)
        cam1_poses = cam0_poses @ T120[None]

        # coordinate conversion
        # kitti360: right, down, forward
        # blender: right, up, backwards
        kitti360_to_blender = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        cam0_poses[:, :3, :3] = cam0_poses[:, :3, :3] @ kitti360_to_blender.T
        cam1_poses[:, :3, :3] = cam1_poses[:, :3, :3] @ kitti360_to_blender.T

        # generate the returned data 
        poses, image_filenames = [], []
        fx, fy, cx, cy, height, width, distort = [], [], [], [], [], [], []

        for i, fname in enumerate(fnames):
            if load_stereo_camera:
                poses += [cam0_poses[i], cam1_poses[i]]
                image_filenames += [fname, path_cam0_to_cam1(fname)]
                fx += [K0[0, 0], K0[0, 0]]
                fy += [K0[1, 1], K0[1, 1]]
                cx += [K0[0, 2], K0[0, 2]]
                cy += [K0[1, 2], K0[1, 2]]
                height += [img_h, img_h]
                width += [img_w, img_w]
                distort += [camera_utils.get_distortion_params(k1=0, k2=0, k3=0, k4=0, p1=0, p2=0), camera_utils.get_distortion_params(k1=0, k2=0, k3=0, k4=0, p1=0, p2=0)]
            else:
                poses += [cam0_poses[i]]
                image_filenames += [fname]
                fx += [K0[0, 0]]
                fy += [K0[1, 1]]
                cx += [K0[0, 2]]
                cy += [K0[1, 2]]
                height += [img_h]
                width += [img_w]
                distort += [camera_utils.get_distortion_params(k1=0, k2=0, k3=0, k4=0, p1=0, p2=0)]

        ret_data = {
            'poses': poses, 
            'image_filenames': image_filenames,
            'fx': fx,
            'fy': fy, 
            'cx': cx, 
            'cy': cy, 
            'height': height, 
            'width': width, 
            'distort': distort,
            'point3d_xyz': None,
            'point3d_rgb': None, 
        }

        return ret_data, midt
    
    def get_train_val_indices(self, eval_interval):
        all_indices = np.arange(len(self.ret_data['image_filenames'])).astype(np.int32)
        if self.other_args['eval_type'] == 'dev':
            filter_func = lambda x: (x % eval_interval == 0)
            train_indices = all_indices[~filter_func(all_indices)]
            val_indices = all_indices[filter_func(all_indices)]
        else:
            raise NotImplementedError(f"Unknown eval_type {self.other_args['eval_type']} for KITTI360RawLoader.")
        return train_indices, val_indices