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
Proposal network field.
"""


from typing import Optional

import torch
from torchtyping import TensorType

from nerfstudio.field_components.factor_field import FactorField, sample_core_factor_features
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field

class NeLFDensityField(Field):
    """
    A lightweight density field module using spherical panorama coordinates. 

    Args:
        num_core: number of core factors
        near_core: number of near core factors for each ray

        angular_resolution: resolution of the panorama
        radial_resolution: resolution of the depth direction
        feat_dim: dimension of features
        
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
    """

    def __init__(
        self, 
        num_core,
        near_core, 
        angular_resolution,
        radial_resolution,
        feat_dim, 
        num_layers: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.field_num = num_core
        self.num_near_grids = near_core

        self.angular_field_stream = FactorField.factory(
            total_num=num_core,
            feat_dim=feat_dim,
            size = (angular_resolution, angular_resolution * 2), 
        )

        self.radial_field = FactorField.factory(
            total_num=num_core,
            feat_dim=feat_dim,
            size = (1, radial_resolution),
        )

        self.mlp_density_geometry = []
        self.mlp_density_geometry.append(torch.nn.Linear(feat_dim, hidden_dim))
        self.mlp_density_geometry.append(torch.nn.ReLU())
        for _ in range(num_layers - 1):
            self.mlp_density_geometry.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.mlp_density_geometry.append(torch.nn.ReLU())
        self.mlp_density_geometry.append(torch.nn.Linear(hidden_dim, 1))
        self.mlp_density_geometry = torch.nn.Sequential(*self.mlp_density_geometry)

    def get_density(self, ray_samples: RaySamples):
        probes = ray_samples.probes

        ret = sample_core_factor_features(
            num_near_core=self.num_near_grids,
            field_angular=self.angular_field_stream,
            field_radial=self.radial_field,
            probes=probes,
            positions=ray_samples.frustums.get_positions(),
            camera_indices=ray_samples.camera_indices,
        )

        grid_field_features = ret['features']
        weighted_features = torch.mean(grid_field_features, dim=-2)

        density_before_activation = self.mlp_density_geometry(weighted_features.flatten(start_dim=0, end_dim=-2))
        density = trunc_exp(density_before_activation.view(*ray_samples.frustums.shape, -1))

        return density, None
    
    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}