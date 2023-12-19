from pathlib import Path
from rich.progress import Console
from abc import abstractmethod

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from nerfstudio.cameras.probes import Probes

from nerfstudio.field_components.resolution_wraping import (
    apply_theta_wraping, 
    apply_phi_wraping, 
)

CONSOLE = Console(width=120)
    
class FactorField(nn.Module):
    '''
    This field is on CUDA device and directly used in the computation graph. 
    '''
    
    @staticmethod
    def factory(total_num: int, feat_dim: int, size: tuple):
        CONSOLE.log(f'using CUDA fields for factors. ')
        return CUDAField(total_num, feat_dim, size)
    
    def register_optimizer_params(self, optimizer: torch.optim.Optimizer):
        pass

    @abstractmethod
    def upsample(self, target_size: tuple):
        pass

    def forward(self):
        raise NotImplementedError('this is not called, only for inheritance. ')

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError('load method not implemented. ')

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError('save method not implemented. ')
    
    @abstractmethod
    def grid_sample(self, x, y, camera_indices, **kwargs):
        pass

    @abstractmethod
    def get_cuda_fields(self):
        pass

    def save_cpu_fields(self, step: int, checkpoint_dir: Path, name: str):
        pass

    def load_cpu_fields(self, step: int, checkpoint_dir: Path, name: str):
        pass

    @abstractmethod
    def check_field_number_consistency(self, probe_num: int):
        pass


class CUDAField(FactorField):
    def __init__(self, total_num: int, feat_dim: int, size: tuple):
        '''
        total_num: total number of basis/core factors in the scene. 
        feat_dim: feature dimension of each field.
        size: size of the field, 2d, 3d, etc. But the external operations (e.g. grid_sample) should conform to the size.
        '''
        super(CUDAField, self).__init__()

        self.total_num = total_num
        self.feat_dim = feat_dim

        self.fields = torch.nn.Parameter(0.1 * torch.randn(feat_dim, total_num, *size), requires_grad=True)

    def register_optimizer_params(self, optimizer: torch.optim.Optimizer):
        return # do nothing here, we don't need to track the optimizer's params.

    def upsample(self, target_size: tuple):
        assert len(target_size) == 2, '2d upsampling is expected. '
        old_size = self.fields.shape
        self.fields = torch.nn.Parameter(F.interpolate(self.fields, size=target_size, mode='bilinear', align_corners=True), requires_grad=True)
        CONSOLE.log(f'CUDA fields are upsampled from {old_size} to {self.fields.shape}.')
        return self.fields

    def load(self, path: str):
        return # do nothing here, torch.nn.Module will handle this. 
    
    def save(self, path: str):
        return # do nothing here, torch.nn.Module will handle this.

    def grid_sample(self, x, y, camera_indices, **kwargs):
        '''
        Given a batch of camera indices, check if the corresponding fields are in the buffer.
        If not, load the fields from the memory to the buffer and kicked out fields back to the memory.
        '''
        z_template = torch.linspace(-1.0, 1.0, self.total_num, device=x.device)
        z = z_template[camera_indices]

        grid = torch.stack([x, y, z], dim=-1)
        grid = grid.unsqueeze(dim=0)

        sample_outputs = F.grid_sample(self.fields.unsqueeze(dim=0), grid, **kwargs) 
        sample_outputs = sample_outputs.squeeze(dim=0).permute(1, 2, 3, 0)  # (N_rays, N_samples, num_near_xxx, feat_dim)

        return sample_outputs
    
    def get_cuda_fields(self):
        return self.fields

    def check_field_number_consistency(self, probe_num: int):
        assert self.total_num == probe_num, f'probe_num {probe_num} does not match the field number {self.total_num}. '


def rescale(x, src_start, src_end, tar_start, tar_end):
    return (x - src_start) / (src_end - src_start) * (tar_end - tar_start) + tar_start

def disparity_warping(x):
    return 1.0 / (x + 1)

def sample_basis_factor_features(
        num_near_basis: int,         
        field: FactorField, 
        probes: Probes, 
        positions: torch.Tensor, 
        camera_indices: torch.Tensor,
        freq_theta: float, 
        freq_phi: float
    ):
    '''
    Get factor features from the basis factor fields via bilinear interpolation sampling. 
    '''
    
    index_near_basis, pos_near_basis = probes.get_nearby_basis_index_and_pos(camera_indices[:, 0], num_near_basis)  # (N_rays, num_near), (N_rays, num_near_basis, 3)
    index_near_basis = index_near_basis.unsqueeze(dim=1).repeat(1, positions.shape[1], 1)
    
    # compute theta and phi
    displacement = positions[:, :, None, :] - pos_near_basis[:, None, :, :]  # (N_rays, N_samples, num_near_basis, 3)
    direction = F.normalize(displacement, dim=-1, eps=1e-6)
    theta = torch.acos(direction[..., 2])  # [0, pi]
    phi = torch.atan2(direction[..., 1], direction[..., 0]) 
    
    # frequency warping and rescaling
    theta = apply_theta_wraping(theta, freq_theta)
    phi = apply_phi_wraping(phi, freq_phi)        
    theta = rescale(theta, 0, np.pi, -1, 1)
    phi = rescale(phi, -np.pi, np.pi, -1, 1)

    # sample features
    features = field.grid_sample(phi, theta, index_near_basis, align_corners=True, mode='bilinear', padding_mode='zeros')  # N_rays, N_samples, num_near_basis, feat_dim
    ret = {}
    ret['features'] = features
    return ret


def sample_core_factor_features(
        num_near_core: int,  
        field_angular: FactorField, 
        field_radial: FactorField, 
        probes: Probes, 
        positions: torch.Tensor, 
        camera_indices: torch.Tensor, 
        return_combined: bool = True,
    ):
    '''
    Get factor features from the core factor fields via bilinear interpolation sampling. 
    '''
    index_near_core, pos_near_core = probes.get_nearby_core_index_and_pos(camera_indices[:, 0], num_near_core)  # (N_rays, num_near_core), (N_rays, num_near_core, 3)
    index_near_core = index_near_core.unsqueeze(dim=1).repeat(1, positions.shape[1], 1)
    
    # compute theta, phi and t
    displacement = positions[:, :, None, :] - pos_near_core[:, None, :, :]  # (N_rays, N_samples, num_near_core, 3)  
    direction = F.normalize(displacement, dim=-1)
    theta = torch.acos(direction[..., 2])
    phi = torch.atan2(direction[..., 1], direction[..., 0])
    t = torch.norm(displacement, dim=-1)
    
    # rescaling
    theta = rescale(theta, 0, np.pi, -1, 1)
    phi = rescale(phi, -np.pi, np.pi, -1, 1)
    t = disparity_warping(t) * 2 - 1.0 # z-axis, range (-1, 1)
    t = t.clamp(-1.0, 1.0)

    angular_features = field_angular.grid_sample(x=phi, y=theta, camera_indices=index_near_core, mode='bilinear', padding_mode='zeros', align_corners=True)  # N_rays, N_samples, num_near_core, feat_dim
    center = torch.zeros_like(phi, device=phi.device)
    radial_features = field_radial.grid_sample(x=t, y=center, camera_indices=index_near_core, mode='bilinear', padding_mode='zeros', align_corners=True)  # N_rays, N_samples, num_near_core, feat_dim
    
    core_features = angular_features * radial_features

    ret = {}
    if return_combined:
        ret['features'] = core_features
    else:
        ret['angular_features'] = angular_features
        ret['radial_features'] = radial_features
    
    return ret