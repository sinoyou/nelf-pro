from dataclasses import dataclass, field
from rich.console import Console

import torch
from torchtyping import TensorType

from nerfstudio.utils.plotly_utils_nelfpro import plot_spheres

CONSOLE = Console(width=120)

@dataclass
class Probes:
    camera_to_worlds: TensorType["num_cams", 3, 4]
    probe_config: dict

    num_basis: int = field(init=False)
    num_core: int = field(init=False)
    pos_basis: TensorType["num_basis", 3] = field(init=False)
    pos_core: TensorType["num_core", 3] = field(init=False)
    sorted_basis_index: TensorType["num_cams", "num_basis"] = field(init=False)
    sorted_core_index: TensorType["num_cams", "num_core"] = field(init=False)

    def __post_init__(self):
        """
        For each scene camera, generate the index of the ascending <camera, factor> distance. 
        """
        # extract camera position
        if len(self.camera_to_worlds.shape) == 2:
            self.camera_to_worlds = self.camera_to_worlds.unsqueeze(dim=0)
        camera_pos = self.camera_to_worlds[:, :3, 3].view(self.camera_to_worlds.shape[0], 1, 3)  # num_cam, 1, 3
        
        self.__init_basis_factor__(camera_pos)
        self.__init_core_factor__(camera_pos)
    
    def __init_basis_factor__(self, camera_pos):
        # basis camera number
        self.basis_factor_list = self.probe_config[0]['basis']
        self.num_basis = len(self.basis_factor_list)

        # prepare basis factor position 
        self.pos_basis = [torch.tensor([factor['x'], factor['y'], factor['z']]) for factor in self.basis_factor_list]
        self.pos_basis = torch.stack(self.pos_basis, dim=0)
        self.pos_basis = self.pos_basis.view(self.num_basis, 3).to(self.device) # num_basis, 3

        camera_basis_dist = torch.norm(camera_pos - self.pos_basis.unsqueeze(dim=0), dim=-1)  # num_cam, num_basis
        _, self.sorted_basis_index = torch.sort(camera_basis_dist, dim=1)
    
    def __init_core_factor__(self, camera_pos):
        self.core_factor_list = self.probe_config[0]['core']
        self.num_core = len(self.core_factor_list)

        self.pos_core = [torch.tensor([factor['x'], factor['y'], factor['z']]) for factor in self.core_factor_list]
        self.pos_core = torch.stack(self.pos_core, dim=0)
        self.pos_core = self.pos_core.view(self.num_core, 3).to(self.device) # num_core, 3
        
        camera_core_dist = torch.norm(camera_pos - self.pos_core.unsqueeze(dim=0), dim=-1)  # num_cam, num_core
        _, self.sorted_core_index = torch.sort(camera_core_dist, dim=1)

    def get_plotly(self):
        plotly_data = []

        # basis
        basis_plot_data = {'center': self.pos_basis.cpu().numpy()}    
        plotly_data += plot_spheres(basis_plot_data, name='basis factor')

        # core
        core_plot_data = {'center': self.pos_core.cpu().numpy()}
        plotly_data += plot_spheres(core_plot_data, name='core factor', scatter_size=9, color='rgba(255, 192, 203, 1)')
        
        return plotly_data

    @property
    def device(self):
        """Returns the device that the data is on."""
        return self.camera_to_worlds.device
    
    def get_nearby_basis_index_and_pos(self, camera_indices: TensorType["bs"], near_num: int) -> TensorType["bs", "near_num", 3]:
        """Get the indices and the positions of each ray's nearby basis factor. """
        camera_indices = camera_indices.squeeze(-1)
        assert len(camera_indices.shape) == 1, 'squeezed camera_indices should be TensorType["bs", ], but got {}'.format(camera_indices.shape)
        assert self.num_basis >= near_num, 'near_num should be smaller than total basis factor number. '

        return self._get_nearby_factor_index_and_pos(camera_indices, near_num, self.sorted_basis_index, self.pos_basis)

    def get_nearby_core_index_and_pos(self, camera_indices: TensorType["bs"], near_num: int) -> TensorType["bs", "near_num", 3]:
        """Get the indices and the positions of each ray's nearby core factor."""
        camera_indices = camera_indices.squeeze(-1)
        assert len(camera_indices.shape) == 1, 'squeezed camera_indices should be TensorType["bs", ], but got {}'.format(camera_indices.shape)
        assert self.num_core >= near_num, 'near_num should be smaller than total core factor number. '

        return self._get_nearby_factor_index_and_pos(camera_indices, near_num, self.sorted_core_index, self.pos_core)
    
    def _get_nearby_factor_index(self, camera_indices: TensorType["bs"], near_num: int, sorted_factor_index: TensorType["num_cam, num_factor"]) -> TensorType["bs", "near_num"]:
        """Get nearby factor index for each ray. """
        sorted_core_index_prefix = sorted_factor_index[:, :near_num]
        return sorted_core_index_prefix[camera_indices, :]

    def _get_nearby_factor_index_and_pos(self, camera_indices: TensorType["bs"], near_num: int, sorted_factor_index: TensorType["num_cam, num_factor"], factor_pos: TensorType["num_factor", 3]) -> TensorType["bs", "near_num", 3]:        
        bs = camera_indices.shape[0]
        selected_index = self._get_nearby_factor_index(camera_indices, near_num, sorted_factor_index)
        selected_pos = factor_pos[selected_index.view(-1), :].view(bs, near_num, 3)
        return selected_index, selected_pos
    
    def get_num_basis(self):
        return self.num_basis

    def get_num_core(self):
        return self.num_core