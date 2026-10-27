from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.algo.training import (
    BaseForecasterTrainer,
    TrainingLogCallback,
)


def linear_interpolate_trajectory(xi: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Linearly interpolate a discrete trajectory at normalized time t.

    Returns:
        xi_t:   [B, D], interpolated state at continuous time t
        dxi_dt: [B, D], piecewise-constant velocity / derivative with respect to normalized t.
    """
    batch_size, horizon, _ = xi.shape

    if horizon == 1:
        return xi[:, 0, :], torch.zeros_like(xi[:, 0, :])

    scaled_t = t * (horizon - 1)
    lower = scaled_t.floor().long().clamp(0, horizon - 2)
    upper = (lower + 1).clamp(0, horizon - 1)
    mix = (scaled_t - lower.float()).unsqueeze(-1)

    batch_idx = torch.arange(batch_size, device=xi.device)
    xi_lower = xi[batch_idx, lower, :]
    xi_upper = xi[batch_idx, upper, :]
    xi_t = xi_lower + mix * (xi_upper - xi_lower)
    # to do: divide this by physical time.
    dxi_dt = (xi_upper - xi_lower) * (horizon - 1)
    return xi_t, dxi_dt


def sample_cfm_inputs_and_targets(
    xi_t: Tensor,
    dxi_dt: Tensor,
    t: Tensor,
    k: float,
    sigma0: float,
) -> Tuple[Tensor, Tensor]:
    """Add time-dependent Gaussian noise and compute the velocity target."""
    t = t.unsqueeze(-1)
    sampled_error = sigma0 * torch.exp(-k * t) * torch.randn_like(xi_t)
    noisy_state = xi_t + sampled_error
    target_velocity = -k * sampled_error + dxi_dt
    return noisy_state, target_velocity


class StreamingFlowForecasterTrainer(BaseForecasterTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        num_epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 200,
        sigma0: float = 0.4,
        k: float = 10.0,
        device: Optional[Union[str, torch.device]] = None,
        log_callback: Optional[TrainingLogCallback] = None,
    ) -> None:
        super().__init__(
            model=model,
            dataloader=dataloader,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            device=device,
            log_callback=log_callback,
        )
        self.sigma0 = sigma0
        self.k = k

    def _sample_inputs_and_targets(
        self,
        xi_t: Tensor,
        dxi_dt: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return sample_cfm_inputs_and_targets(
            xi_t,
            dxi_dt,
            t,
            k=self.k,
            sigma0=self.sigma0,
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
        """Return optional model inputs and an auxiliary training loss."""
        del trajectory_timestep, timestep, hist_cond, target_trajectory
        return {}, xi_t.new_zeros(())

    def _call_velocity(
        self,
        state: Tensor,
        t: Tensor,
        hist_cond: Tensor,
        model_kwargs: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        output = self.model(
            sample=state.unsqueeze(1),
            timestep=t,
            hist_cond=hist_cond,
            **model_kwargs,
        )
        if output.ndim == 3:
            output = output[:, 0, :]
        return output

    def _velocity_loss(
        self,
        velocity_pred: Tensor,
        target_velocity: Tensor,
    ) -> Tensor:
        return nn.functional.mse_loss(velocity_pred, target_velocity)

    def batch_loss(self, batch) -> Tensor:
        nobs = batch["obs"].to(self.device)
        target = batch["target"].to(self.device)
        hist_cond = nobs.flatten(start_dim=1)

        t = torch.rand(target.shape[0], device=self.device, dtype=torch.float32)
        xi_t, dxi_dt = linear_interpolate_trajectory(target, t)
        noisy_state, physical_velocity = self._sample_inputs_and_targets(
            xi_t,
            dxi_dt,
            t,
        )
        model_kwargs, auxiliary_loss = self._prepare_velocity_model_kwargs(
            xi_t=xi_t,
            trajectory_timestep=t,
            timestep=t,
            hist_cond=hist_cond,
            target_trajectory=target,
        )

        velocity_pred = self._call_velocity(
            noisy_state,
            t,
            hist_cond,
            model_kwargs,
        )
        loss = (
            self._velocity_loss(
                velocity_pred,
                physical_velocity,
            )
            + auxiliary_loss
        )
        return loss

    def extra_artifacts(self) -> Dict[str, object]:
        return {
            "sigma0": self.sigma0,
            "k": self.k,
        }
