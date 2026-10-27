import copy
import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor
import torch.nn as nn


class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        x = x.float() * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvDownsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class LinearDownsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, seq_len = x.shape
        x = x.reshape(batch_size, channels * seq_len)
        x = self.linear(x)
        return x.reshape(batch_size, channels, seq_len)


class LinearUpsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, seq_len = x.shape
        x = x.reshape(batch_size, channels * seq_len)
        x = self.linear(x)
        return x.reshape(batch_size, channels, seq_len)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, out_channels),
            Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        sin_embedding_scale: float,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
        latent_dim: Optional[int] = None,
        latent_feature_dim: int = 256,
        latent_weight: float = 0.9,
    ) -> None:
        super().__init__()
        if latent_dim is not None:
            if latent_dim <= 0:
                raise ValueError("latent_dim must be positive.")
            if latent_feature_dim <= 0:
                raise ValueError("latent_feature_dim must be positive.")
            if not 0.0 <= latent_weight <= 1.0:
                raise ValueError("latent_weight must be in [0, 1].")

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim, scale=sin_embedding_scale),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        if latent_dim is not None and latent_feature_dim != mid_dim:
            raise ValueError(
                "bottleneck_sum requires latent_feature_dim to equal the "
                f"bottleneck width, got {latent_feature_dim} and {mid_dim}."
            )

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

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            downsample = LinearDownsample1d(dim_out) if not is_last else nn.Identity()

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
                        downsample,
                    ]
                )
            )

        up_modules = []
        for dim_in, dim_out in reversed(in_out[1:]):
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        LinearUpsample1d(dim_in),
                    ]
                )
            )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.latent_dim = latent_dim
        self.latent_feature_dim = int(latent_feature_dim)
        self.latent_weight = float(latent_weight)
        self.latent_encoder = (
            nn.Sequential(
                nn.Linear(latent_dim, latent_feature_dim),
                nn.SiLU(),
                nn.Linear(latent_feature_dim, latent_feature_dim),
            )
            if latent_dim is not None
            else None
        )
        self.down_modules = nn.ModuleList(down_modules)
        self.up_modules = nn.ModuleList(up_modules)
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        hist_cond: Optional[Tensor] = None,
        latent: Optional[Tensor] = None,
    ) -> Tensor:
        sample = sample.moveaxis(-1, -2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device, dtype=torch.float32)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.to(sample.device).float().expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        latent_feature = None
        if self.latent_encoder is not None:
            if latent is None:
                raise ValueError(
                    "latent must be provided to a latent-conditioned UNet."
                )
            if (
                latent.ndim != 2
                or latent.shape[0] != sample.shape[0]
                or latent.shape[1] != self.latent_dim
            ):
                raise ValueError(
                    "latent must have shape [batch_size, latent_dim], got "
                    f"{tuple(latent.shape)}."
                )
            latent_feature = self.latent_encoder(latent)
        elif latent is not None:
            raise ValueError(
                "latent was provided to a UNet without latent conditioning."
            )

        if hist_cond is not None:
            global_feature = torch.cat([global_feature, hist_cond], dim=-1)

        x = sample
        skip_connections = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            skip_connections.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        if self.latent_encoder is not None:
            latent_feature = latent_feature.unsqueeze(-1).expand(
                -1,
                -1,
                x.shape[-1],
            )
            x = (
                (1.0 - self.latent_weight) * x
                + self.latent_weight * latent_feature
            )

        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.75) -> None:
        self.decay = decay
        self.shadow_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, value in model_state.items():
            shadow = self.shadow_state[name]
            if torch.is_floating_point(value):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow_state, strict=True)

    def clone_model(self, model: nn.Module) -> nn.Module:
        ema_model = copy.deepcopy(model)
        self.copy_to(ema_model)
        return ema_model
