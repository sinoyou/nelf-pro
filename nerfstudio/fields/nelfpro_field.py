from rich.progress import Console
from typing import Dict, Optional, Union

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.factor_field import (
    FactorField, 
    sample_basis_factor_features,
    sample_core_factor_features,
)
from nerfstudio.field_components.rsh import rsh_cart_3

from nerfstudio.cameras.rays import RaySamples

CONSOLE = Console(width=120)

class NeLFProField(Field):
    def __init__(
        self,
        num_images, 

        num_basis: int, 
        near_basis: int, 
        dim_basis: int, 
        resolution_basis: int, 

        num_core: int,
        near_core: int,
        dim_core: int, 
        resolution_core_angular: int, 
        resolution_core_radial: int, 

        freq_theta: int,
        freq_phi: int,  

        use_appearance_embedding: bool,
        apperance_embedding_dim: int = 16,
        
        num_layers_geometry: int = 2, 
        hidden_dim_geometry: int = 128, 
        geo_feat_dim: int = 31, 
        num_layers_color: int = 3, 
        hidden_dim_color: int = 64, 
    ):
        super().__init__()

        # config for basis factor
        self.num_basis = num_basis 
        self.near_basis = near_basis
        self.dim_basis = dim_basis
        self.resolution_basis = resolution_basis

        # config for core factor
        self.num_core = num_core 
        self.near_core = near_core 
        self.dim_core = dim_core 
        self.resolution_core_angular = resolution_core_angular
        self.resolution_core_radial = resolution_core_radial

        # config for geometry and color mlps
        self.num_layers_geometry = num_layers_geometry  
        self.hidden_dim_geometry = hidden_dim_geometry 
        self.geo_feat_dim = geo_feat_dim 
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        # config for frequency warping on basis factor fields
        self.freq_theta = freq_theta
        self.freq_phi = freq_phi

        # config for apperance warping
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images

        self.field_basis = FactorField.factory(
            total_num = self.num_basis,
            feat_dim  = self.dim_basis, 
            size = (self.resolution_basis, self.resolution_basis * 2), 
        )

        self.field_core_angular = FactorField.factory(
            total_num = self.num_core,
            feat_dim  = self.dim_core, 
            size = (self.resolution_core_angular, self.resolution_core_angular * 2), 
        )
            
        self.field_core_radial = FactorField.factory(
            total_num = self.num_core,
            feat_dim  = self.dim_core, 
            size = (1, self.resolution_core_radial), 
        )

        # linear mapping for basis factor
        self.mlp_projection_basis = nn.Sequential(
            nn.Linear(self.dim_basis, self.dim_core, bias=True),
        )

        # factor aggregation network
        self.attn_basis = nn.Sequential(nn.Linear(self.dim_core, 1), nn.Sigmoid())
        self.attn_core_angular = nn.Sequential(nn.Linear(self.dim_core, 1), nn.Sigmoid())
        self.attn_core_radial = nn.Sequential(nn.Linear(self.dim_core, 1), nn.Sigmoid())

        # density prediction network
        self.bn = nn.BatchNorm1d(self.dim_core)
        self.mlp_density_geometry = MLP(in_dim=self.dim_core, 
                                        num_layers=self.num_layers_geometry, 
                                        layer_width=self.hidden_dim_geometry,
                                        out_dim=self.geo_feat_dim + 1)

        # color prediction network
        self.direction_encoding = lambda x: rsh_cart_3(x * 2 - 1)
        self.mlp_rgb_head = MLP(in_dim=self.geo_feat_dim + 16 + 16 * self.use_appearance_embedding,
                                num_layers=self.num_layers_color,
                                layer_width=self.hidden_dim_color,
                                out_dim=3,
                                out_activation=nn.Sigmoid())
        
        # appearance embedding
        if self.use_appearance_embedding:
            self.appearance_embedding_dim = apperance_embedding_dim
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)

    def get_basis_fields(self):
        return {self.field_basis.get_cuda_fields()}
    
    def get_core_fields(self, name):
        if name == 'radial':
            return {self.field_core_radial.get_cuda_fields()}
        elif name == 'angular':
            return {self.field_core_angular.get_cuda_fields()}
        else:
            raise NameError(f'core field {name} not found. ')
    
    def get_stream_field(self, name):
        if name == 'coefficient_fields':
            return self.field_basis
        elif name == 'coefficient_grids_angular':
            return self.field_core_angular
        elif name == 'coefficient_grids_radial':
            return self.field_core_radial
        else:
            raise NameError(f'stream field {name} not found. ')

    def get_density(self, ray_samples: RaySamples):
        """Project ray samples points to basis and core factors, decode and get density + geometric feature."""
        positions = ray_samples.frustums.get_positions()

        # check number of probes generated by data parser
        self.field_basis.check_field_number_consistency(ray_samples.probes.get_num_basis())
        self.field_core_angular.check_field_number_consistency(ray_samples.probes.get_num_core())
        self.field_core_radial.check_field_number_consistency(ray_samples.probes.get_num_core())
        
        # base field class
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        
        # sample factor features
        probes = ray_samples.probes
        # core factor component
        core_ret = sample_core_factor_features(
            num_near_core=self.near_core,
            field_angular=self.field_core_angular,
            field_radial=self.field_core_radial,
            probes=probes,
            positions=positions,
            camera_indices=ray_samples.camera_indices,
            return_combined=False, 
        )
        # basis factor component
        basis_ret = sample_basis_factor_features(
            num_near_basis=self.near_basis, 
            field=self.field_basis, 
            probes=probes, 
            positions=positions, 
            camera_indices=ray_samples.camera_indices,
            freq_phi=self.freq_phi, 
            freq_theta=self.freq_theta
        )
        
        feature_core_radial = core_ret['radial_features']
        feature_core_angular = core_ret['angular_features']
        feature_basis = basis_ret['features']

        # basis feature projection
        feature_basis = self.mlp_projection_basis(feature_basis)  # bs, samples, near_probes, fusion_dim

        # factor aggregation
        weight_core_radial = self.attn_core_radial(feature_core_radial)
        fused_core_radial = torch.sum(feature_core_radial * weight_core_radial, dim=-2)
        weight_core_angular = self.attn_core_angular(feature_core_angular)
        fused_core_angular = torch.sum(feature_core_angular * weight_core_angular, dim=-2)
        weight_basis = self.attn_basis(feature_basis)
        fused_basis = torch.sum(feature_basis * weight_basis, dim=-2)  # bs, samples, feat_dim
        tau = fused_core_angular * fused_core_radial * fused_basis # bs, samples, feat_dim
        tau = self.bn(tau.flatten(end_dim=-2)).view(*tau.shape[:-1], -1)
        
        # density and geometry feature decoding
        h = self.mlp_density_geometry(tau)  # N_rays, N_samples, geo_feat_dim + 1    
        _density_before_activation, geometry_feat = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = _density_before_activation
        density = trunc_exp(_density_before_activation)

        return density, geometry_feat

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Union[Dict, TensorType]]):
        assert density_embedding is not None
        outputs = dict()

        directions = ray_samples.frustums.directions
        direction_enc = self.direction_encoding((directions + 1.0) / 2.0)
        density_features = density_embedding
        h = torch.cat([density_features, direction_enc], dim=-1)

        if self.use_appearance_embedding:
            camera_indices = ray_samples.camera_indices.squeeze(dim=-1)
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                embedded_appearance = torch.ones(
                    (*ray_samples.frustums.directions.shape[:-1], self.appearance_embedding_dim), device=ray_samples.frustums.directions.device, 
                ) * self.embedding_appearance.mean(dim=0)
            h = torch.concat([h, embedded_appearance], dim=-1)
        rgb = self.mlp_rgb_head(h)

        outputs.update({'rgb': rgb})
        if self.training:
            outputs.update({'basis': self.field_basis.get_cuda_fields()})
            outputs.update({'core_angular': self.field_core_angular.get_cuda_fields()})

        return outputs

    def get_field_coefficients(self):
        return self.field_basis.get_cuda_fields().detach().cpu()
    
    def upsample_basis(self, target_resolution):
        self.field_basis.upsample((target_resolution, target_resolution * 2))

    def upsample_core(self, target_resolution):
        target_res_angular, target_res_radial = target_resolution
        self.field_core_angular.upsample((target_res_angular, target_res_angular * 2))
        self.field_core_radial.upsample((1, target_res_radial))
