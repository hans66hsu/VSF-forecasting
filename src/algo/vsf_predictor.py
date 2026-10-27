"""Deterministic and variational JEPA streaming-flow predictors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.algo.common.xfmr_predictor import (
    Mish,
    SinusoidalPosEmb,
    Transformer,
)


class SFPredictor(nn.Module):
    """Causal velocity predictor for teacher-forced JEPA streaming flow."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        flow_step_embed_dim: int = 128,
        sin_embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.horizon = num_frames
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )
        self.flow_step_encoder = nn.Sequential(
            SinusoidalPosEmb(flow_step_embed_dim, scale=sin_embedding_scale),
            nn.Linear(flow_step_embed_dim, flow_step_embed_dim * 4),
            Mish(),
            nn.Linear(flow_step_embed_dim * 4, hidden_dim),
        )

    def _call_velocity(self, x: Tensor, c: Tensor, flow_time: Tensor) -> Tensor:
        flow_condition = self.flow_step_encoder(flow_time)
        x = self.dropout(x + self.pos_embedding[:, : x.shape[1]])
        return self.transformer(x, c, flow_condition)

    def predict(self, x: Tensor, c: Tensor) -> Tensor:
        context_length = x.shape[1]
        flow_time = torch.arange(context_length, device=x.device, dtype=torch.float32)[
            None
        ].expand(x.shape[0], -1) / float(self.horizon)
        return x + self._call_velocity(x, c, flow_time) / float(self.horizon)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        return self.predict(x, c)

    def train_loss(self, states: Tensor, actions: Tensor) -> Tensor:
        batch_size, horizon, _ = actions.shape
        local_phase = torch.rand(
            batch_size, 1, device=states.device, dtype=torch.float32
        )
        interval = torch.arange(
            horizon, device=states.device, dtype=torch.float32
        ).unsqueeze(0)
        flow_time = (interval + local_phase) / float(horizon)
        interpolated, target_velocity = batched_linear_interpolate_trajectory(
            states, flow_time
        )
        predicted_velocity = self._call_velocity(interpolated, actions, flow_time)
        return F.mse_loss(predicted_velocity, target_velocity)


class TrajectoryPosteriorEncoder(nn.Module):
    """Lightweight causal Transformer posterior over a complete trajectory."""

    def __init__(
        self,
        *,
        num_frames: int,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
        flow_step_embed_dim: int = 128,
        sin_embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("The trajectory posterior requires at least 2 layers.")
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            hidden_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )
        self.posterior_head = nn.Linear(hidden_dim, 2 * latent_dim)
        self.posterior_condition_encoder = nn.Linear(3 * input_dim, input_dim)
        self.flow_step_encoder = nn.Sequential(
            SinusoidalPosEmb(flow_step_embed_dim, scale=sin_embedding_scale),
            nn.Linear(flow_step_embed_dim, flow_step_embed_dim * 4),
            Mish(),
            nn.Linear(flow_step_embed_dim * 4, hidden_dim),
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        curr_state: Tensor | None = None,
        flow_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("The posterior requires one more state than action.")
        if states.shape[1] != self.num_frames + 1:
            raise ValueError("The posterior trajectory has the wrong horizon.")
        if curr_state is None or flow_time is None:
            raise ValueError("curr_state and flow_time are required by trajectory.")
        evidence = curr_state
        condition = self.posterior_condition_encoder(
            torch.cat(
                (actions, states[:, :-1], states[:, 1:]), dim=-1
            )
        )
        flow_condition = self.flow_step_encoder(flow_time)
        evidence = self.dropout(evidence + self.pos_embedding)
        stats = self.posterior_head(
            self.transformer(evidence, condition, flow_condition)[:, -1]
        )
        mean, logvar = stats.chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 4.0)


class VSFPredictor(SFPredictor):
    """Variational streaming flow with one trajectory-level latent."""

    def __init__(
        self,
        *,
        latent_dim: int,
        latent_feature_dim: int,
        latent_weight: float,
        beta_kl: float,
        crps_num_samples: int,
        crps_diversity_weight: float,
        crps_beta: float,
        posterior_hidden_dim: int,
        posterior_depth: int,
        posterior_heads: int,
        posterior_dim_head: int,
        posterior_mlp_dim: int,
        posterior_dropout: float = 0.0,
        flow_step_embed_dim: int = 128,
        sin_embedding_scale: float = 1.0,
        **kwargs,
    ) -> None:
        crps_beta = float(crps_beta)
        super().__init__(
            flow_step_embed_dim=flow_step_embed_dim,
            sin_embedding_scale=sin_embedding_scale,
            **kwargs,
        )
        if crps_num_samples < 2:
            raise ValueError("crps_num_samples must be at least 2.")
        if not 0.0 < crps_beta <= 2.0:
            raise ValueError("crps_beta must be in (0, 2].")
        if not 0.0 <= latent_weight <= 1.0:
            raise ValueError("latent_weight must be in [0, 1].")

        hidden_dim = self.transformer.norm.normalized_shape[0]
        self.latent_dim = latent_dim
        self.latent_weight = latent_weight
        self.beta_kl = beta_kl
        self.crps_num_samples = crps_num_samples
        self.crps_diversity_weight = float(crps_diversity_weight)
        self.crps_beta = crps_beta
        self.posterior_encoder = TrajectoryPosteriorEncoder(
            num_frames=self.num_frames,
            input_dim=self.input_dim,
            hidden_dim=posterior_hidden_dim,
            depth=posterior_depth,
            heads=posterior_heads,
            dim_head=posterior_dim_head,
            mlp_dim=posterior_mlp_dim,
            latent_dim=latent_dim,
            dropout=posterior_dropout,
            flow_step_embed_dim=flow_step_embed_dim,
            sin_embedding_scale=sin_embedding_scale,
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_feature_dim),
            nn.SiLU(),
            nn.Linear(latent_feature_dim, hidden_dim),
        )

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def sample_prior(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)

    def _call_velocity(
        self,
        x: Tensor,
        c: Tensor,
        flow_time: Tensor,
        latent: Tensor | None = None,
    ) -> Tensor:
        if latent is None:
            raise ValueError("latent is required by VSFPredictor.")
        latent_condition = self.latent_weight * self.latent_encoder(latent)
        flow_condition = self.flow_step_encoder(flow_time)
        x = self.dropout(x + self.pos_embedding[:, : x.shape[1]])
        return self.transformer(
            x,
            c,
            flow_condition,
            bottleneck_condition=latent_condition[:, None],
        )

    def predict(
        self,
        x: Tensor,
        c: Tensor,
        latent: Tensor | None = None,
    ) -> Tensor:
        if latent is None:
            latent = self.sample_prior(x.shape[0], x.device, x.dtype)
        context_length = x.shape[1]
        flow_time = torch.arange(context_length, device=x.device, dtype=torch.float32)[
            None
        ].expand(x.shape[0], -1) / float(self.horizon)
        return x + self._call_velocity(x, c, flow_time, latent) / float(self.horizon)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        latent: Tensor | None = None,
    ) -> Tensor:
        return self.predict(x, c, latent)

    def train_loss(self, states: Tensor, actions: Tensor) -> Tensor:
        batch_size, horizon, _ = actions.shape
        local_phase = torch.rand(
            batch_size, 1, device=states.device, dtype=torch.float32
        )
        interval = torch.arange(
            horizon, device=states.device, dtype=torch.float32
        ).unsqueeze(0)
        flow_time = (interval + local_phase) / float(horizon)
        interpolated, target_velocity = batched_linear_interpolate_trajectory(
            states, flow_time
        )
        mean, logvar = self.posterior_encoder(
            states,
            actions,
            interpolated,
            flow_time,
        )

        samples = self.crps_num_samples
        latent = self.reparameterize(
            mean[:, None].expand(-1, samples, -1),
            logvar[:, None].expand(-1, samples, -1),
        )
        predicted_velocity = self._call_velocity(
            interpolated[:, None]
            .expand(-1, samples, -1, -1)
            .reshape(batch_size * samples, horizon, -1),
            actions[:, None]
            .expand(-1, samples, -1, -1)
            .reshape(batch_size * samples, horizon, -1),
            flow_time[:, None]
            .expand(-1, samples, -1)
            .reshape(batch_size * samples, horizon),
            latent.reshape(batch_size * samples, -1),
        ).reshape(batch_size, samples, horizon, -1)
        target_difference = predicted_velocity - target_velocity[:, None, :]
        pairwise_difference = (
            predicted_velocity[:, :, None, :]
            - predicted_velocity[:, None, :, :]
        )
        accuracy = target_difference.abs().pow(self.crps_beta).mean()
        diversity = pairwise_difference.abs().pow(self.crps_beta).mean()
        crps_loss = accuracy - self.crps_diversity_weight * diversity

        kl_loss = (
            -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(dim=-1).mean()
        )
        return crps_loss + self.beta_kl * kl_loss


def batched_linear_interpolate_trajectory(
    xi: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Interpolate ``[B, T, D]`` trajectories at ``[B, Q]`` times."""
    batch_size, horizon, _ = xi.shape
    scaled_t = t * (horizon - 1)
    lower = scaled_t.floor().long().clamp(0, horizon - 2)
    upper = lower + 1
    mix = (scaled_t - lower).to(xi.dtype).unsqueeze(-1)
    batch_idx = torch.arange(batch_size, device=xi.device)[:, None]
    xi_lower = xi[batch_idx, lower]
    xi_upper = xi[batch_idx, upper]
    displacement = xi_upper - xi_lower
    xi_t = xi_lower + mix * displacement
    dxi_dt = (horizon - 1) * displacement
    return xi_t, dxi_dt
