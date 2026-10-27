import math
from typing import Dict, Optional, Sequence

import torch


DISTRIBUTION_METRICS = frozenset(
    {
        "wasserstein_w1",
        "energy_distance",
        "conditional_mean_error",
        "conditional_std_error",
    }
)


def _with_context_axis(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim < 2:
        raise ValueError(f"Expected trajectory tensor with shape [N, ...], got {x.shape}")
    if x.ndim in (2, 3):
        return x.unsqueeze(0), True
    return x, False


def flatten_trajs(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(
            f"Expected trajectory tensor with shape [B, N, ...], got {x.shape}"
        )
    return x.reshape(x.shape[0], x.shape[1], -1)


def check_trajectory_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError("x and y must be torch.Tensor objects.")

    if x.ndim < 3 or y.ndim < 3:
        raise ValueError("x and y must have shape [B, N, ...] and [B, M, ...].")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same batch size. Got x.shape={x.shape}, y.shape={y.shape}."
        )

    if x.shape[2:] != y.shape[2:]:
        raise ValueError(
            "Trajectory shapes must match after context and sample dimensions. "
            f"Got x.shape={x.shape}, y.shape={y.shape}."
        )

    if x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("Both x and y must contain at least one trajectory per context.")

    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device. Got {x.device} and {y.device}.")

    if not torch.is_floating_point(x) or not torch.is_floating_point(y):
        raise TypeError("x and y must be floating-point tensors.")


def pairwise_traj_l2(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, L, T, D]
    y: [B, K, T, D]
    output: [B, L, K]. Pairwise is computed per context B.
    """
    check_trajectory_inputs(x, y)

    return torch.cdist(flatten_trajs(x), flatten_trajs(y), p=2)


def sinkhorn_ot_cost_from_cost_matrix(
    C: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 100,
) -> torch.Tensor:
    if C.ndim != 3:
        raise ValueError(f"C must have shape [B, N, M], got {C.shape}")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1.")

    batch_size, n, m = C.shape
    dtype = C.dtype
    device = C.device

    log_a = torch.full((batch_size, n), -math.log(n), device=device, dtype=dtype)
    log_b = torch.full((batch_size, m), -math.log(m), device=device, dtype=dtype)

    log_K = -C / epsilon

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(n_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, :, None], dim=1)

    log_P = log_u[:, :, None] + log_K + log_v[:, None, :]
    P = torch.exp(log_P)

    return torch.sum(P * C, dim=(-2, -1))


def sinkhorn_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 100,
) -> torch.Tensor:
    """Compute per-context entropic Sinkhorn Wasserstein-1 distance.

    x shape: [B, N, ...], usually [B, N, T, D].
    y shape: [B, M, ...], usually [B, M, T, D].
    Output shape: [B].
    """
    check_trajectory_inputs(x, y)

    C = pairwise_traj_l2(x, y)
    ot_cost = sinkhorn_ot_cost_from_cost_matrix(
        C,
        epsilon=epsilon,
        n_iters=n_iters,
    )

    return torch.clamp(ot_cost, min=0.0)


def energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    unbiased: bool = True,
) -> torch.Tensor:
    """Compute per-context energy distance between trajectory samples.

    x shape: [B, N, ...], usually [B, N, T, D].
    y shape: [B, M, ...], usually [B, M, T, D].
    Output shape: [B].
    """
    check_trajectory_inputs(x, y)

    n = x.shape[1]
    m = y.shape[1]

    d_xy = pairwise_traj_l2(x, y)
    d_xx = pairwise_traj_l2(x, x)
    d_yy = pairwise_traj_l2(y, y)

    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("Unbiased energy distance requires at least 2 samples from each distribution.")

        sum_xx = (
            d_xx.sum(dim=(-2, -1))
            - d_xx.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        )
        sum_yy = (
            d_yy.sum(dim=(-2, -1))
            - d_yy.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        )

        return (
            2.0 * d_xy.mean(dim=(-2, -1))
            - sum_xx / (n * (n - 1))
            - sum_yy / (m * (m - 1))
        )

    return 2.0 * d_xy.mean(dim=(-2, -1)) - d_xx.mean(dim=(-2, -1)) - d_yy.mean(
        dim=(-2, -1)
    )


def conditional_mean_error(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    check_trajectory_inputs(x, y)
    x_mean = x.mean(dim=1).flatten(1)
    y_mean = y.mean(dim=1).flatten(1)
    return (x_mean - y_mean).norm(dim=1) / x_mean.norm(dim=1).clamp_min(1e-8)


def conditional_std_error(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    check_trajectory_inputs(x, y)
    x_std = x.std(dim=1, correction=0).flatten(1)
    y_std = y.std(dim=1, correction=0).flatten(1)
    return (x_std - y_std).norm(dim=1) / x_std.norm(dim=1).clamp_min(1e-8)


@torch.no_grad()
def compute_sample_metrics(
    real_trajs: torch.Tensor,
    model_trajs: torch.Tensor,
    metric_names: Optional[Sequence[str]] = None,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 100,
) -> Dict[str, torch.Tensor]:
    real_trajs, squeeze_output = _with_context_axis(real_trajs)
    model_trajs, model_was_single = _with_context_axis(model_trajs)
    if squeeze_output != model_was_single:
        raise ValueError(
            "real_trajs and model_trajs must either both include a context dimension "
            "or both omit it."
        )
    check_trajectory_inputs(real_trajs, model_trajs)

    requested = set(DISTRIBUTION_METRICS) if metric_names is None else set(metric_names)
    unknown = requested - DISTRIBUTION_METRICS
    if unknown:
        raise ValueError(f"Unknown distribution metric(s): {sorted(unknown)}.")

    metrics: Dict[str, torch.Tensor] = {}
    if "conditional_mean_error" in requested:
        metrics["conditional_mean_error"] = conditional_mean_error(
            real_trajs, model_trajs
        )

    if "conditional_std_error" in requested:
        metrics["conditional_std_error"] = conditional_std_error(
            real_trajs, model_trajs
        )

    if "wasserstein_w1" in requested:
        metrics["wasserstein_w1"] = sinkhorn_wasserstein(
            real_trajs,
            model_trajs,
            epsilon=sinkhorn_epsilon,
            n_iters=sinkhorn_iters,
        )

    if "energy_distance" in requested:
        metrics["energy_distance"] = energy_distance(real_trajs, model_trajs)

    if squeeze_output:
        return {name: value.squeeze(0) for name, value in metrics.items()}
    return metrics
