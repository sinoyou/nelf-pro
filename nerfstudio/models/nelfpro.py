from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.fields.nelfpro_field import NeLFProField
from nerfstudio.fields.density_fields import NeLFDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    field_tv_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, EarthCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color

@dataclass
class NeLFProModelConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: NeLFProModel)
    
    # basis factor configs
    num_basis: int = 64
    """Number of basis factors."""
    near_basis: int = 16
    """Number of near basis factors for each ray. """
    dim_basis: int = 2
    """Feature dimension of basis factor. """
    resolution_basis: int = 256
    """Tensor resolution of basis factor. """
    freq_theta: float = 1
    """Frequency of multiplicative warping for theta. """
    freq_phi: float = 1
    """Frequency of multiplicative warping for phi."""

    # core factor configs
    num_core: int = 3
    """Number of core factors."""
    near_core: int = 3
    """Number of near core factors for each ray."""
    dim_core: int = 32
    """Feature dimension of core factor."""
    resolution_core_angular: int = 64
    """Initial tensor resolution of angular core factor. (a.k.a theta and phi)"""
    resolution_max_core_angular: int = 128
    """Max tensor resolution of angular core factor. (a.k.a theta and phi)"""
    resolution_core_radial: int = 64
    """Initial tensor resolution of radial core factor. (a.k.a depth direction)"""
    resolution_max_core_radial: int = 1024
    """Max tensor resolution of radial core factor. (a.k.a depth direction)"""
    iters_core_upsampling: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """Iterations for upsampling core factor. """

    # apperance embedding settings
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding. """

    # sampler config
    near_plane: float = 0.05
    """Near plane for initial ray sampler. """
    far_plane: float = 1000.0
    """Far plane for initial ray sampler."""
    use_earth_collider: bool = False
    """Whether to use earth model collider, must pass scene-specific scale. (e.g. for bungeenerf dataset)"""
    earth_collider_scale: float = 1.0
    """Scale of the earth model collider. """
    use_single_jitter: bool = False
    """Whether to use single jitter for initial ray sampler."""
    init_sampler: Literal['uniform', 'sqrt', 'log', 'uniformdisp'] = 'uniformdisp'
    """Initial ray sampler function type. """

    # proposal network config
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    num_proposal_samples_per_ray_1: int = 256
    """Number of proposal samples per ray for the #1 proposal network iteration."""
    num_proposal_samples_per_ray_2: int = 96
    """Number of proposal samples per ray for the #2 proposal network iteration."""
    num_nerf_samples_per_ray: int = 48 
    """Number of nerf samples per ray for the main field. """
    proposal_update_every: int = 5
    """Update proposal network every # iterations."""
    proposal_warmup: int = 5000
    """Warmup proposal network by linear learning rate for the first # iterations."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"angular_resolution": 64, "radial_resolution": 128, "feat_dim": 8, }, 
            {"angular_resolution": 128, "radial_resolution": 256, "feat_dim": 8, }, 
        ]
    )
    """List of proposal network arguments for each proposal networks"""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 100
    """Max num iterations for the annealing function."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    
    # rendering and training config
    background_color: Literal["random", "last_sample", "white", "black"] = "random"  
    """Background color for rendering when accumulation doesn't reach 1."""
    interlevel_loss_mult: float = 1.0
    """Interlevel loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Initial distortion loss multiplier."""
    distortion_loss_mult_factor_max: int = 1
    """Max multiplication factor for distortion loss multiplier."""
    distortion_loss_mult_factor_iters: Tuple[int, ...] = (500, 1000, 2000, 4000)
    """Iterations for upsampling distortion loss multiplier."""""
    basis_tv_loss_mult: float = 0.0
    """Tv loss multiplier for basis factor."""
    core_tv_loss_mult: float = 0.0
    """Tv loss multiplier for core factor."""


def get_upsample_steps(res_base, res_max, num_iters):
    x = (
        np.round(
            np.exp(
                np.linspace(
                    np.log(res_base), 
                    np.log(res_max), 
                    num_iters + 1, 
                )
            )
        ).astype("int").tolist()[1:]
    )
    return x


class NeLFProModel(Model):
    config: NeLFProModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Resolution and Loss Multiplier Upsampling Config
        if self.config.resolution_core_angular < self.config.resolution_max_core_angular or self.config.resolution_core_radial < self.config.resolution_max_core_radial:
            self.core_angular_upsamling_step = get_upsample_steps(self.config.resolution_core_angular, self.config.resolution_max_core_angular, len(self.config.iters_core_upsampling))
            self.core_radial_upsamling_step = get_upsample_steps(self.config.resolution_core_radial, self.config.resolution_max_core_radial, len(self.config.iters_core_upsampling))
            
        if self.config.distortion_loss_mult_factor_max != 1.0:
            self.distortion_loss_mult_factor_step = get_upsample_steps(1.0, self.config.distortion_loss_mult_factor_max, len(self.config.distortion_loss_mult_factor_iters))
            self.current_distort_loss_mult_factor = self.config.distortion_loss_mult * 1.0
        else:
            self.current_distort_loss_mult_factor = self.config.distortion_loss_mult * 1.0

        # Main Field
        self.field = NeLFProField(
            num_images=self.num_train_data, 

            num_basis = self.config.num_basis,
            near_basis = self.config.near_basis,
            dim_basis = self.config.dim_basis,
            resolution_basis = self.config.resolution_basis,

            num_core = self.config.num_core,
            near_core = self.config.near_core,
            dim_core = self.config.dim_core,
            resolution_core_angular = self.config.resolution_core_angular,
            resolution_core_radial = self.config.resolution_core_radial,

            freq_theta = self.config.freq_theta, 
            freq_phi = self.config.freq_phi, 

            use_appearance_embedding=self.config.use_appearance_embedding,
        )

        # Proposal Networks
        self.proposal_networks = torch.nn.ModuleList()
        assert len(self.config.proposal_net_args_list) == self.config.num_proposal_iterations, 'proposal_net_args_list should have the same length as num_proposal_iterations'
        for i in range(self.config.num_proposal_iterations):
            prop_net_args = self.config.proposal_net_args_list[i]
            network = NeLFDensityField(num_core=self.config.num_core, near_core=self.config.near_core, **prop_net_args)
            self.proposal_networks.append(network)
        self.density_fns = [network.density_fn for network in self.proposal_networks]

        # Proposal Sampler
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            init_sampler=self.config.init_sampler, 
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=[self.config.num_proposal_samples_per_ray_1, self.config.num_proposal_samples_per_ray_2],
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        if self.config.use_earth_collider:
            self.collider = EarthCollider(scene_scaling_factor=self.config.earth_collider_scale)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Renders
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type='vgg')

        # adjust step-dependent components
        self.adjust_step_dependent_components()
        
    def n_parameters(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.field.parameters()) + sum(p.numel() for p in self.proposal_networks.parameters())
    
    def adjust_step_dependent_components(self) -> None:
        """Call the model's customized load function.
        Args:
            checkpoint_dir: directory of checkpoint
        """
        load_step = self.load_step
        if not load_step:
            return
        
        assert load_step >= 0, f"load_step must be non-negative, got {load_step}"
        # distortion factor 
        if self.config.distortion_loss_mult_factor_max != 1.0:
            i = 0
            while len(self.distortion_loss_mult_factor_step) > 0 and load_step >= self.config.distortion_loss_mult_factor_iters[i]:
                i += 1
                self.current_distort_loss_mult_factor = self.distortion_loss_mult_factor_step.pop(0) * self.config.distortion_loss_mult
            assert len(self.config.distortion_loss_mult_factor_iters) - i == len(self.distortion_loss_mult_factor_step), 'distortion_loss_mult_factor_step should have the same length as distortion_loss_mult_factor_iters'
        
        # core upsampling
        if self.config.resolution_core_angular < self.config.resolution_max_core_angular or self.config.resolution_core_radial < self.config.resolution_max_core_radial:
            i = 0
            while len(self.core_radial_upsamling_step) > 0 and load_step >= self.config.iters_core_upsampling[i]:
                i += 1
                angular_res = self.core_angular_upsamling_step.pop(0)
                radial_res = self.core_radial_upsamling_step.pop(0)
                self.field.upsample_core((angular_res, radial_res))
            assert len(self.config.iters_core_upsampling) - i == len(self.core_radial_upsamling_step), 'core_radial_upsamling_step should have the same length as iters_core_upsampling'
        
        # proposal network annealing: ignore as its update frequency is very high. 


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # seperate the parameters of the field into two groups: fields (e.g. MLP) and fields_coef (e.g. camera fields)
        field_basis = self.field.get_basis_fields()
        field_core_angular = self.field.get_core_fields(name='angular')
        field_core_radial = self.field.get_core_fields(name='radial')
        
        param_groups["field_mlp"] = list()
        param_groups["field_basis"] = list()
        param_groups["field_core_angular"] = list()
        param_groups["field_core_radial"] = list()
        param_groups["proposal_networks"] = list()
        
        for field_params in self.field.parameters():
            if field_params in field_basis:
                param_groups["field_basis"].append(field_params)
            elif field_params in field_core_angular:
                param_groups["field_core_angular"].append(field_params)
            elif field_params in field_core_radial:
                param_groups["field_core_radial"].append(field_params)
            else:
                param_groups["field_mlp"].append(field_params)

        for proposal_parameter in self.proposal_networks.parameters():
            param_groups["proposal_networks"].append(proposal_parameter)
        
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def reinitialize_optimizer(params_name, training_callback_attributes, step):
            '''reinitialize optimizer and scheduler after upsampling. '''
            optimizers_config = training_callback_attributes.optimizers.config
            data = training_callback_attributes.pipeline.get_param_groups()[params_name]
            lr_init = optimizers_config[params_name]["optimizer"].lr
            
            opt_state_param_groups = training_callback_attributes.optimizers.optimizers[params_name].state_dict()['param_groups']
            training_callback_attributes.optimizers.optimizers[params_name] = optimizers_config[params_name]["optimizer"].setup(params=data)
            # note: we load_state_dict() for loading param_groups's _lr_initial, which is used for scheduler's last_epoch. 
            training_callback_attributes.optimizers.optimizers[params_name].load_state_dict({
                'state': training_callback_attributes.optimizers.optimizers[params_name].state_dict()['state'], 
                'param_groups': opt_state_param_groups
            })

            if optimizers_config[params_name]["scheduler"]:
                # save current state dict
                training_callback_attributes.optimizers.schedulers[params_name] = optimizers_config[params_name]["scheduler"].setup(
                    optimizer=training_callback_attributes.optimizers.optimizers[params_name], lr_init=lr_init, last_epoch=step, 
                )
        
        # upsampling core factor
        if self.config.resolution_core_angular < self.config.resolution_max_core_angular or self.config.resolution_core_radial < self.config.resolution_max_core_radial:
            def upsample_core(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
                angular_res = self.core_angular_upsamling_step.pop(0)
                radial_res = self.core_radial_upsamling_step.pop(0)
                self.field.upsample_core((angular_res, radial_res))

                reinitialize_optimizer('field_core_angular', training_callback_attributes, step)
                reinitialize_optimizer('field_core_radial', training_callback_attributes, step)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION], 
                    iters=self.config.iters_core_upsampling, 
                    func=upsample_core, 
                    args=[self, training_callback_attributes]
                )
            )
        
        # update distortion loss multiplier
        if self.config.distortion_loss_mult_factor_max != 1.0:
            def update_distortion_loss_mult_factor(self, step: int):
                self.current_distort_loss_mult_factor = self.distortion_loss_mult_factor_step.pop(0) * self.config.distortion_loss_mult

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION], 
                    iters=self.config.distortion_loss_mult_factor_iters, 
                    func=update_distortion_loss_mult_factor, 
                    args=[self]
                )
            )

        # Proposal Network Annealing and Status Update
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs['density'])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs['rgb'], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.training:
            outputs["tensors"] = {
                'basis': field_outputs['basis'], 
                'core_angular': field_outputs['core_angular'],
            }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            loss_dict["distortion_loss"] = self.current_distort_loss_mult_factor * metrics_dict["distortion"]

            if self.config.basis_tv_loss_mult != 0.0:
                loss_dict["basis_tv_loss"] = self.config.basis_tv_loss_mult * field_tv_loss(outputs["fields"]["basis"])

            if self.config.core_tv_loss_mult != 0.0:    
                loss_dict["core_tv_loss"] = self.config.core_tv_loss_mult * field_tv_loss(outputs["fields"]["core_angular"])

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb, 
            "accumulation": combined_acc, 
            "depth": combined_depth, 
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
