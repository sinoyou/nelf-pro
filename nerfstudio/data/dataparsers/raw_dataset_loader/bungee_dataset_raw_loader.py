import os
import numpy as np
import cv2
import json
from pathlib import Path, PurePath
import ipdb

from nerfstudio.data.utils.colmap_utils import read_points3d_binary

from rich.console import Console
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.raw_dataset_loader.raw_loader import RawLoader

CONSOLE = Console(width=120)

def _downsample_google_data(basedir, factor):
    img_basedir = basedir
    img_folder = 'images'
    imgdir = os.path.join(img_basedir, img_folder)

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]
    downsample_image_filenames = []
    
    output_folder = f'images_{factor}'
    output_dir = os.path.join(img_basedir, output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgfiles_downsampled = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]
    if len(imgfiles_downsampled) == len(imgfiles):
        check_list = [Path(imgfiles[i]).name == Path(imgfiles_downsampled[i]).name for i in range(len(imgfiles))]
        if all(check_list):
            skip_flag = True
            CONSOLE.log('Skip writing downsampled image files. ')
        else:
            skip_flag = False
    else:
        skip_flag = False
    

    sh = np.array(cv2.imread(imgfiles[0]).shape)
    for f in imgfiles:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        
        # format convert
        if im.shape[-1] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        
        # downsample
        im = cv2.resize(im, (sh[1]//factor, sh[0]//factor), interpolation=cv2.INTER_AREA)
        
        output_file = os.path.join(output_dir, PurePath(f).name)
        if not skip_flag:
            cv2.imwrite(output_file, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        downsample_image_filenames.append(Path(output_file))

    data = json.load(open(os.path.join(basedir, 'poses_enu.json')))
    poses = np.array(data['poses'])[:, :-2].reshape([-1, 3, 5])
    
    # downsample poses is not needed here, the general parser will handle this. 
    # poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
    # poses[:, 2, 4] = poses[:,2, 4] * 1./factor 

    scene_scaling_factor = data['scene_scale']
    scene_origin = np.array(data['scene_origin'])
    scale_split = data['scale_split']

    return downsample_image_filenames, poses, scene_scaling_factor, scene_origin, scale_split


def load_multiscale_data(basedir, factor=3):
    image_filenames, poses, scene_scaling_factor, scene_origin, scale_split = _downsample_google_data(basedir, factor=factor)
    CONSOLE.log(f'Loaded {len(image_filenames)} images from {basedir} with factor {factor}')
    return image_filenames, poses, scene_scaling_factor, scene_origin, scale_split


class BungeeRawLoader(RawLoader):
    def get_loaded_data(self) -> dict:
        image_filenames, poses, scene_scaling_factor, scene_origin, scale_split = load_multiscale_data(self.data_dir, factor=self.downscale_factor)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        # coordinate transformation
        # original: x-right, y-up, z-backward
        # same conventions as nerfstudio (original nerf paper), no need for further change. 

        # generate exported files
        poses_list, filenames = [], []
        fx, fy, cx, cy, height, width, distort = [], [], [], [], [], [], []
        for i, frame in enumerate(image_filenames):
            fx.append(focal)
            fy.append(focal)
            cx.append(W/2)
            cy.append(H/2)
            
            height.append(H)
            width.append(W)
            distort.append(camera_utils.get_distortion_params(k1=0, k2=0, k3=0, k4=0, p1=0, p2=0))

            filenames.append(frame)
            poses_list.append(np.concatenate([poses[i], np.array([[0, 0, 0, 1]])], axis=0))

        self.ret_data = {
            'poses': poses_list, 
            'image_filenames': filenames,
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
        
        return self.ret_data

    def get_train_val_indices(self, eval_interval):
        assert eval_interval == 16, 'eval_interval is expected to be 16 in bungee dataset.'
        all_indices = np.arange(len(self.ret_data['image_filenames']))
        filter_func = lambda x: x % eval_interval == 0
        train_indices = all_indices[~filter_func(all_indices)]
        val_indices = all_indices[filter_func(all_indices)]
        return train_indices, val_indices