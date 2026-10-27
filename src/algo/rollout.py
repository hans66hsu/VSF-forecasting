from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.datasets.data import (
    normalize_data,
    unnormalize_data,
    get_anchor_index_range,
)
from src.algo.common.distr_metrics import (
    DISTRIBUTION_METRICS,
    compute_sample_metrics,
)
from src.utils.utils import get_default_device


TensorLike = Union[np.ndarray, torch.Tensor]

POINT_ROLLOUT_METRICS = frozenset({"rmse", "mae", "crps"})
DISTRIBUTION_ROLLOUT_METRICS = DISTRIBUTION_METRICS


def _as_batched_initial_obs(initial_obs: np.ndarray) -> tuple[np.ndarray, bool]:
    initial_obs = np.asarray(initial_obs, dtype=np.float32)
    if initial_obs.ndim == 2:
        return initial_obs[None], True
    if initial_obs.ndim == 3:
        return initial_obs, False
    raise ValueError(
        "initial_obs must have shape [obs_horizon, state_dim] or "
        f"[batch, obs_horizon, state_dim], got {initial_obs.shape}."
    )


def prepare_rollout_example(
    trajectory: np.ndarray,
    anchor_index: int,
    obs_horizon: int,
    total_future_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    min_anchor_idx, max_anchor_idx = get_anchor_index_range(
        trajectory_length=trajectory.shape[0],
        obs_horizon=obs_horizon,
        future_horizon=total_future_steps,
        future_start_offset=1,
    )
    if not min_anchor_idx <= anchor_index <= max_anchor_idx:
        raise ValueError(
            "anchor_index must allow a full observation context and future window "
            f"without padding; got anchor_index={anchor_index}, valid range is "
            f"[{min_anchor_idx}, {max_anchor_idx}]."
        )
    context_start = anchor_index - obs_horizon + 1
    context_end = anchor_index + 1
    future_start = anchor_index + 1
    future_end = future_start + total_future_steps

    context = trajectory[context_start:context_end]
    future = trajectory[future_start:future_end]
    return context.astype(np.float32), future.astype(np.float32)


def rollout_streaming_flow_forecast(
    model: nn.Module,
    initial_obs: np.ndarray,
    stats: Mapping[str, np.ndarray],
    pred_horizon: int,
    rollout_horizon: int,
    total_future_steps: int,
    initial_noise_scale: float = 0.0,
    add_noise_every_rollout: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    latent_dim: Optional[int] = None,
    resample_latent_each_step: bool = False,
) -> np.ndarray:
    if rollout_horizon > pred_horizon - 1:
        raise ValueError("rollout_horizon must be <= pred_horizon - 1")

    device = torch.device(device) if device is not None else get_default_device()
    obs_history, squeeze_output = _as_batched_initial_obs(initial_obs)
    batch_size = obs_history.shape[0]

    # current_state shape: [B, D]
    current_state = normalize_data(obs_history[:, -1], stats).astype(np.float32)
    state_from_prev_chunk = torch.from_numpy(current_state).to(
        device=device, dtype=torch.float32
    )
    # state_from_prev_chunk shape: [B, 1, D]
    state_from_prev_chunk = state_from_prev_chunk[:, None]
    noise_scale = float(initial_noise_scale)

    model = model.to(device)
    model.eval()
    latent = None
    if latent_dim is not None:
        latent = torch.randn(
            batch_size,
            latent_dim,
            device=device,
            dtype=state_from_prev_chunk.dtype,
        )
    dt_chunk = 1.0 / max(pred_horizon - 1, 1)

    predictions = []
    while len(predictions) < total_future_steps:
        nobs = normalize_data(obs_history, stats).astype(np.float32)
        obs_cond = torch.from_numpy(nobs).to(device=device, dtype=torch.float32)
        obs_cond = obs_cond.flatten(start_dim=1)

        state = state_from_prev_chunk
        # Add noise at the first anchor, or at every rollout anchor if enabled.
        if noise_scale > 0.0 and (add_noise_every_rollout or not predictions):
            state = state + noise_scale * torch.randn_like(state)
        with torch.no_grad():
            for step_idx in range(rollout_horizon):
                if latent is not None and resample_latent_each_step and predictions:
                    latent = torch.randn_like(latent)
                time_value = torch.full(
                    (batch_size,),
                    step_idx * dt_chunk,
                    device=device,
                    dtype=torch.float32,
                )
                dt = torch.full(
                    (batch_size,),
                    dt_chunk,
                    device=device,
                    dtype=torch.float32,
                )
                model_kwargs = dict(
                    sample=state,
                    timestep=time_value,
                    hist_cond=obs_cond,
                )
                if latent is not None:
                    model_kwargs["latent"] = latent
                velocity = model(**model_kwargs)
                if velocity.ndim == 2:
                    velocity = velocity[:, None]
                state = state + velocity * dt.view(batch_size, 1, 1)

                next_state = state.detach().cpu().numpy()[:, 0]
                next_state = unnormalize_data(next_state, stats).astype(np.float32)
                predictions.append(next_state)
                # drop the first history, append one newly predicted state
                obs_history = np.concatenate(
                    [obs_history[:, 1:], next_state[:, None]],
                    axis=1,
                )

                if len(predictions) >= total_future_steps:
                    break

        state_from_prev_chunk = state.detach()

    result = np.stack(predictions, axis=1).astype(np.float32)
    return result[0] if squeeze_output else result


def sample_rollout_forecast(
    rollout_fn,
    num_samples: int,
    **kwargs,
) -> np.ndarray:
    initial_obs = np.asarray(kwargs["initial_obs"], dtype=np.float32)
    if initial_obs.ndim == 2:
        # One context [H, D] is copied S times:
        # [H, D] -> [S, H, D] -> rollout [S, T, D].
        kwargs["initial_obs"] = np.repeat(initial_obs[None], num_samples, axis=0)
        return rollout_fn(**kwargs).astype(np.float32)
    if initial_obs.ndim == 3:
        num_contexts = initial_obs.shape[0]
        # Vectorize S forecast samples for each of C contexts in one rollout:
        # contexts [C, H, D]
        #     -> repeat [C * S, H, D]
        #     -> rollout [C * S, T, D]
        #     -> reshape [C, S, T, D].
        kwargs["initial_obs"] = np.repeat(initial_obs, num_samples, axis=0)
        samples = rollout_fn(**kwargs).astype(np.float32)
        return samples.reshape(num_contexts, num_samples, *samples.shape[1:])
    raise ValueError(
        "initial_obs must have shape [obs_horizon, state_dim] or "
        f"[num_contexts, obs_horizon, state_dim], got {initial_obs.shape}."
    )


def _as_float_tensor(
    x: TensorLike,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if torch.is_tensor(x):
        tensor = x.detach()
    else:
        tensor = torch.as_tensor(x)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.to(dtype=torch.float32)


def _check_state_names(state_names: Sequence[str], state_dim: int) -> None:
    if len(state_names) != state_dim:
        raise ValueError(
            "Expected len(state_names) == state_dim. "
            f"Got len(state_names)={len(state_names)}, state_dim={state_dim}."
        )


def _per_feature_dict(
    state_names: Sequence[str],
    values: torch.Tensor,
) -> Dict[str, float]:
    values = values.detach().cpu().reshape(-1)
    _check_state_names(state_names, int(values.numel()))
    return {
        name: float(value)
        for name, value in zip(state_names, values.tolist())
    }


def _canonicalize_point_inputs(
    model_samples: torch.Tensor,
    real_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return model [B, S, T, D] and real [B, T, D] tensors."""
    usable_steps = min(model_samples.shape[2], real_samples.shape[1])
    return model_samples[:, :, :usable_steps], real_samples[:, :usable_steps]


def _canonicalize_distribution_inputs(
    model_samples: torch.Tensor,
    real_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return real [B, N_real, T, D] and model [B, N_model, T, D]."""
    usable_steps = min(model_samples.shape[2], real_samples.shape[2])
    return real_samples[:, :, :usable_steps], model_samples[:, :, :usable_steps]


def crps_ensemble_torch(
    observations: torch.Tensor,
    ensemble: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CRPS values [B, T, D] and mean CRPS from [B, S, T, D]."""
    if observations.ndim != 3 or ensemble.ndim != 4:
        raise ValueError(
            "Expected observations [B, T, D] and ensemble [B, S, T, D]. "
            f"Got observations={tuple(observations.shape)}, "
            f"ensemble={tuple(ensemble.shape)}."
        )

    num_members = ensemble.shape[1]
    if num_members < 1:
        raise ValueError("CRPS requires at least one ensemble member.")

    term_1 = (ensemble - observations[:, None]).abs().mean(dim=1)
    sorted_ensemble = ensemble.sort(dim=1).values
    coeff = (
        2
        * torch.arange(
            1,
            num_members + 1,
            device=ensemble.device,
            dtype=ensemble.dtype,
        )
        - num_members
        - 1
    )
    term_2 = (
        sorted_ensemble * coeff[None, :, None, None]
    ).sum(dim=1) / (num_members ** 2)
    crps_values = term_1 - term_2
    return crps_values, crps_values.mean()


@torch.no_grad()
def compute_rollout_metrics(
    predicted_samples: TensorLike,
    reference_samples: TensorLike,
    state_names: Sequence[str],
    metric_names: Sequence[str] = ("rmse", "mae"),
    dataset_kind: str = "deterministic",
    normalize_distribution_samples: bool = True,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 100,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, object]:
    """Compute rollout metrics from generated and reference trajectory samples.

    Deterministic datasets use point metrics with reference shape [B, T, D].
    Probabilistic datasets use distribution metrics with reference shape [B, S, T, D].
    """
    requested_metrics = tuple(metric_names)
    if dataset_kind not in {"deterministic", "probabilistic"}:
        raise ValueError("dataset_kind must be 'deterministic' or 'probabilistic'.")
    supported_metrics = (
        DISTRIBUTION_ROLLOUT_METRICS
        if dataset_kind == "probabilistic"
        else POINT_ROLLOUT_METRICS
    )
    unknown_metrics = sorted(set(requested_metrics) - supported_metrics)
    if unknown_metrics:
        raise ValueError(
            f"Unknown rollout metric(s): {unknown_metrics}. "
            f"Valid {dataset_kind} metrics: {sorted(supported_metrics)}."
        )

    names = tuple(state_names)
    model_samples = _as_float_tensor(predicted_samples, device=device)
    real_samples = _as_float_tensor(reference_samples, device=model_samples.device)

    metrics: Dict[str, object] = {}
    if dataset_kind == "deterministic":
        point_model_samples, point_real_samples = _canonicalize_point_inputs(
            model_samples,
            real_samples,
        )
        _check_state_names(names, point_real_samples.shape[-1])
        errors = point_model_samples.mean(dim=1) - point_real_samples

        if "rmse" in requested_metrics:
            squared_errors = errors.square()
            metrics["rmse"] = float(squared_errors.mean().sqrt().detach().cpu())
            metrics["rmse_per_feature"] = _per_feature_dict(
                names,
                squared_errors.mean(dim=(0, 1)).sqrt(),
            )
        if "mae" in requested_metrics:
            abs_errors = errors.abs()
            metrics["mae"] = float(abs_errors.mean().detach().cpu())
            metrics["mae_per_feature"] = _per_feature_dict(
                names,
                abs_errors.mean(dim=(0, 1)),
            )
        if "crps" in requested_metrics:
            crps_values, mean_crps = crps_ensemble_torch(
                observations=point_real_samples,
                ensemble=point_model_samples,
            )
            metrics["crps"] = float(mean_crps.detach().cpu())
            metrics["crps_per_feature"] = _per_feature_dict(
                names,
                crps_values.mean(dim=(0, 1)),
            )
    else:
        real_distribution_samples, model_distribution_samples = (
            _canonicalize_distribution_inputs(model_samples, real_samples)
        )
        _check_state_names(names, real_distribution_samples.shape[-1])

        if normalize_distribution_samples:
            mean = real_distribution_samples.mean(dim=(1, 2), keepdim=True)
            variance = (real_distribution_samples - mean).square().mean(
                dim=(1, 2),
                keepdim=True,
            )
            std = torch.sqrt(torch.clamp(variance, min=1e-12)).clamp_min(1e-6)
            real_distribution_samples = (real_distribution_samples - mean) / std
            model_distribution_samples = (model_distribution_samples - mean) / std

        distribution_metrics = compute_sample_metrics(
            real_distribution_samples,
            model_distribution_samples,
            metric_names=requested_metrics,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
        )
        for metric_name in requested_metrics:
            metrics[metric_name] = float(
                distribution_metrics[metric_name].mean().detach().cpu()
            )
    return metrics
