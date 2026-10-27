from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.algo.common.models import (
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    ConvDownsample1d,
    Mish,
    SinusoidalPosEmb,
)
from src.algo.streaming_flow import StreamingFlowForecasterTrainer


_THICK_TUBE_SIGMA_MIN = 0.005
_THICK_TUBE_SIGMA_MAX = 0.5


def sample_thick_tube_inputs_and_targets(
    xi_t: Tensor,
    dxi_dt: Tensor,
    t: Tensor,
    k: float,
    sigma0: float,
    p_original_noise: float,
) -> Tuple[Tensor, Tensor]:
    """Sample contracting thick-tube noise and its velocity target."""
    batch_size = xi_t.shape[0]
    scheduled_sigma = sigma0 * torch.exp(-k * t)
    log_sigma = math.log(_THICK_TUBE_SIGMA_MIN) + torch.rand(
        batch_size,
        device=xi_t.device,
        dtype=torch.float32,
    ) * (math.log(_THICK_TUBE_SIGMA_MAX) - math.log(_THICK_TUBE_SIGMA_MIN))
    augmented_sigma = torch.exp(log_sigma)

    if p_original_noise == 1.0:
        noise_scale = scheduled_sigma
    elif p_original_noise == 0.0:
        noise_scale = augmented_sigma
    else:
        use_original = torch.rand(
            batch_size,
            device=xi_t.device,
            dtype=torch.float32,
        ) < p_original_noise
        noise_scale = torch.where(use_original, scheduled_sigma, augmented_sigma)

    sampled_error = noise_scale.unsqueeze(-1) * torch.randn_like(xi_t)
    noisy_state = xi_t + sampled_error
    target_velocity = -k * sampled_error + dxi_dt
    return noisy_state, target_velocity


class PosteriorUnetEncoder1D(nn.Module):
    """Down-only conditional UNet that predicts a diagonal Gaussian posterior."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        latent_dim: int,
        sin_embedding_scale: float,
        time_embed_dim: int = 256,
        down_dims: Sequence[int] = (32, 64, 128),
        kernel_size: int = 5,
        n_groups: int = 8,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        down_dims = tuple(down_dims)
        if input_dim <= 0 or global_cond_dim < 0 or latent_dim <= 0:
            raise ValueError("Posterior dimensions must be positive.")
        if time_embed_dim <= 0 or time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be a positive even number.")
        if not down_dims or any(dim <= 0 for dim in down_dims):
            raise ValueError("down_dims must contain positive channel widths.")
        if n_groups <= 0 or any(dim % n_groups != 0 for dim in down_dims):
            raise ValueError(
                "Every posterior channel width must be divisible by n_groups."
            )
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max.")

        self.input_dim = int(input_dim)
        self.global_cond_dim = int(global_cond_dim)
        self.latent_dim = int(latent_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        cond_dim = time_embed_dim + global_cond_dim

        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        down_modules = []
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        nn.Identity() if is_last else ConvDownsample1d(dim_out),
                    ]
                )
            )

        mid_dim = down_dims[-1]
        self.down_modules = nn.ModuleList(down_modules)
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.posterior_head = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            Mish(),
            nn.Linear(mid_dim, 2 * latent_dim),
        )

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        hist_cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if sample.ndim != 3 or sample.shape[-1] != self.input_dim:
            raise ValueError(
                "posterior sample must have shape [batch, horizon, input_dim], "
                f"got {tuple(sample.shape)}."
            )
        batch_size = sample.shape[0]
        if hist_cond.shape != (batch_size, self.global_cond_dim):
            raise ValueError(
                "hist_cond has the wrong shape, got "
                f"{tuple(hist_cond.shape)} instead of "
                f"{(batch_size, self.global_cond_dim)}."
            )
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=sample.device, dtype=torch.float32).expand(
            batch_size
        )

        condition = torch.cat(
            [self.time_encoder(timestep), hist_cond],
            dim=-1,
        )
        x = sample.moveaxis(-1, -2)
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, condition)
            x = resnet2(x, condition)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, condition)

        posterior_stats = self.posterior_head(self.pool(x).squeeze(-1))
        mean, logvar = posterior_stats.chunk(2, dim=-1)
        return mean, logvar.clamp(self.logvar_min, self.logvar_max)


class VariationalStreamingFlowModel(nn.Module):
    """Register the training-only posterior and inference-time velocity model."""

    def __init__(
        self,
        posterior_encoder: PosteriorUnetEncoder1D,
        velocity_unet: ConditionalUnet1D,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if posterior_encoder.latent_dim != latent_dim:
            raise ValueError("Posterior and model latent dimensions must match.")
        if velocity_unet.latent_dim != latent_dim:
            raise ValueError("Velocity UNet and model latent dimensions must match.")

        self.posterior_encoder = posterior_encoder
        self.velocity_unet = velocity_unet
        self.latent_dim = int(latent_dim)

    def encode_posterior(
        self,
        xi_t: Tensor,
        trajectory_timestep: Tensor,
        timestep: Tensor,
        hist_cond: Tensor,
        target_trajectory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, horizon, _ = target_trajectory.shape
        scaled_t = trajectory_timestep * (horizon - 1)
        lower = scaled_t.floor().long().clamp(0, max(horizon - 2, 0))
        upper = (lower + 1).clamp(max=horizon - 1)
        batch_idx = torch.arange(batch_size, device=target_trajectory.device)
        xi_lower = target_trajectory[batch_idx, lower]
        xi_upper = target_trajectory[batch_idx, upper]
        evidence = torch.stack([xi_t, xi_lower, xi_upper], dim=1)
        return self.posterior_encoder(
            evidence,
            timestep,
            hist_cond,
        )

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        hist_cond: Optional[Tensor] = None,
        latent: Optional[Tensor] = None,
    ) -> Tensor:
        return self.velocity_unet(
            sample=sample,
            timestep=timestep,
            hist_cond=hist_cond,
            latent=latent,
        )


class VariationalStreamingFlowForecasterTrainer(StreamingFlowForecasterTrainer):
    def __init__(
        self,
        model: VariationalStreamingFlowModel,
        dataloader,
        beta_kl: float = 2e-3,
        crps_num_samples: int = 8,
        crps_diversity_weight: float = 0.5,
        crps_beta: float = 1.999,
        use_thick_tube: bool = False,
        thick_tube_p_original_noise: float = 0.25,
        sigma0: float = 0.0,
        k: float = 0.0,
        **kwargs,
    ) -> None:
        crps_beta = float(crps_beta)
        if crps_num_samples < 2:
            raise ValueError("crps_num_samples must be at least 2.")
        if not 0.0 < crps_beta <= 2.0:
            raise ValueError("crps_beta must be in (0, 2].")
        if use_thick_tube:
            if not 0.0 <= thick_tube_p_original_noise <= 1.0:
                raise ValueError(
                    "thick_tube_p_original_noise must be in [0, 1], got "
                    f"{thick_tube_p_original_noise}."
                )
        super().__init__(
            model=model,
            dataloader=dataloader,
            sigma0=sigma0,
            k=k,
            **kwargs,
        )
        self.beta_kl = float(beta_kl)
        self.crps_num_samples = int(crps_num_samples)
        self.crps_diversity_weight = float(crps_diversity_weight)
        self.crps_beta = crps_beta
        self.use_thick_tube = use_thick_tube
        self.thick_tube_p_original_noise = thick_tube_p_original_noise

    def _sample_inputs_and_targets(
        self,
        xi_t: Tensor,
        dxi_dt: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not self.use_thick_tube:
            return super()._sample_inputs_and_targets(xi_t, dxi_dt, t)
        return sample_thick_tube_inputs_and_targets(
            xi_t,
            dxi_dt,
            t,
            k=self.k,
            sigma0=self.sigma0,
            p_original_noise=self.thick_tube_p_original_noise,
        )

    def _prepare_velocity_model_kwargs(
        self,
        *,
        xi_t: Tensor,
        trajectory_timestep: Tensor,
        timestep: Tensor,
        hist_cond: Tensor,
        target_trajectory: Tensor,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        mean, logvar = self.model.encode_posterior(
            xi_t,
            trajectory_timestep,
            timestep,
            hist_cond,
            target_trajectory,
        )
        sample_shape = (-1, self.crps_num_samples, -1)
        latent = self.model.reparameterize(
            mean[:, None, :].expand(*sample_shape),
            logvar[:, None, :].expand(*sample_shape),
        )
        kl_loss = (
            -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(dim=-1).mean()
        )
        return {"latent": latent}, self.beta_kl * kl_loss

    def _call_velocity(
        self,
        state: Tensor,
        t: Tensor,
        hist_cond: Tensor,
        model_kwargs: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        latent = model_kwargs.get("latent")
        if latent is None or latent.ndim == 2:
            return super()._call_velocity(state, t, hist_cond, model_kwargs)

        batch_size, num_samples, _ = latent.shape
        model_kwargs["latent"] = latent.flatten(0, 1)
        velocity = super()._call_velocity(
            state.repeat_interleave(num_samples, dim=0),
            t.repeat_interleave(num_samples, dim=0),
            hist_cond.repeat_interleave(num_samples, dim=0),
            model_kwargs,
        )
        return velocity.reshape(batch_size, num_samples, -1)

    def _velocity_loss(
        self,
        velocity_pred: Tensor,
        target_velocity: Tensor,
    ) -> Tensor:
        target_difference = velocity_pred - target_velocity[:, None, :]
        pairwise_difference = (
            velocity_pred[:, :, None, :] - velocity_pred[:, None, :, :]
        )
        accuracy = target_difference.abs().pow(self.crps_beta).mean()
        diversity = pairwise_difference.abs().pow(self.crps_beta).mean()
        return accuracy - self.crps_diversity_weight * diversity

    def extra_artifacts(self):
        artifacts = super().extra_artifacts()
        artifacts.update(
            {
                "latent_dim": self.model.latent_dim,
                "latent_feature_dim": self.model.velocity_unet.latent_feature_dim,
                "latent_weight": self.model.velocity_unet.latent_weight,
                "use_thick_tube": self.use_thick_tube,
                "thick_tube_p_original_noise": self.thick_tube_p_original_noise,
                "beta_kl": self.beta_kl,
                "crps_num_samples": self.crps_num_samples,
                "crps_diversity_weight": self.crps_diversity_weight,
                "crps_beta": self.crps_beta,
            }
        )
        return artifacts
