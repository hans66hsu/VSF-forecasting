from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import wandb


WANDB_DISABLED_MODE = "disabled"
WANDB_MODES = {"online", "offline", WANDB_DISABLED_MODE}
OVERVIEW_MODULES = {
    "lorenz": "src.datasets.lorenz.data",
    "rossler": "src.datasets.rossler.data",
    "double-pendulum": "src.datasets.double_pendulum.data",
}


def _to_container_if_config(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return value

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def init_wandb_run(cfg: Any, run_dir: Path, seed: int | None = None):
    wandb_cfg = cfg.get("wandb", {})
    mode = str(wandb_cfg.get("mode", WANDB_DISABLED_MODE))
    if mode == WANDB_DISABLED_MODE:
        return None
    if mode not in WANDB_MODES:
        raise ValueError(f"wandb.mode must be one of {sorted(WANDB_MODES)}, got {mode}.")

    run_config = _to_container_if_config(cfg)
    if isinstance(run_config, dict):
        run_config["seed"] = seed
        run_config["run_dir"] = str(run_dir)

    project = wandb_cfg.get("project", None)
    if project is None:
        project = Path.cwd().name

    run_name = wandb_cfg.get("name", None)
    if run_name is None:
        dataset_name = cfg.dataset.get("name", cfg.dataset.get("kind", "dataset"))
        model_names = "+".join(str(model) for model in cfg.get("models", []))
        run_name = f"{dataset_name} | {model_names} ({run_dir.parent.name}/{run_dir.name})"

    init_kwargs: Dict[str, Any] = {
        "name": run_name,
        "project": project,
        "mode": mode,
        "dir": str(run_dir),
        "config": run_config,
    }
    for key in ("entity", "tags", "notes"):
        value = wandb_cfg.get(key, None)
        value = _to_container_if_config(value)
        if value is None or value == []:
            continue
        init_kwargs[key] = value

    resume_id = wandb_cfg.get("resume", None)
    if resume_id:
        init_kwargs["id"] = str(resume_id)
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def finish_wandb_run(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def make_train_loss_logger(
    wandb_run,
    model_name: str,
) -> Optional[Callable[[Dict[str, float]], None]]:
    if wandb_run is None:
        return None

    def _log(metrics: Dict[str, float]) -> None:
        payload = {}
        if "train_loss" in metrics:
            payload[f"train/{model_name}/loss"] = metrics["train_loss"]
        if "lr" in metrics:
            payload[f"train/{model_name}/lr"] = metrics["lr"]
        if "epoch" in metrics:
            payload[f"train/{model_name}/epoch"] = metrics["epoch"]
        if payload:
            wandb_run.log(payload)

    return _log


def _as_trajectory_batch(trajectories: np.ndarray) -> np.ndarray:
    array = np.asarray(trajectories, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            "trajectories must have shape (steps, state_dim) or "
            f"(num_samples, steps, state_dim), got {array.shape}."
        )
    return array


def _resolve_feature_indices(
    state_dim: int,
    feature_indices: Optional[Iterable[int]],
) -> list[int]:
    if feature_indices is None:
        return list(range(state_dim))
    indices = [int(index) for index in feature_indices]
    return [index for index in indices if 0 <= index < state_dim]


def _sample_colors(num_samples: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.linspace(0.0, 1.0, num_samples, endpoint=False) + rng.random()) % 1.0
    rng.shuffle(hues)
    import matplotlib.colors as mcolors

    return np.asarray([mcolors.hsv_to_rgb((hue, 0.72, 0.82)) for hue in hues])


def _overview_fn(dataset_name: str):
    for prefix, module_name in OVERVIEW_MODULES.items():
        if dataset_name.startswith(prefix):
            return importlib.import_module(module_name).overview_trajectory
    return None


def _paired_grid_fn(dataset_name: str):
    for prefix, module_name in OVERVIEW_MODULES.items():
        if dataset_name.startswith(prefix):
            return importlib.import_module(module_name).paired_trajectory_grid
    return None


def log_trajectory_line_plots(
    wandb_run,
    key: str,
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    *,
    dt: Optional[float] = None,
    feature_indices: Optional[Iterable[int]] = None,
    title: Optional[str] = None,
    step: Optional[int] = None,
) -> None:
    if wandb_run is None:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = _as_trajectory_batch(real_trajs)
    model = _as_trajectory_batch(model_trajs)
    usable_steps = min(real.shape[1], model.shape[1])
    real = real[:, :usable_steps]
    model = model[:, :usable_steps]

    state_dim = min(real.shape[-1], model.shape[-1], len(state_names))
    indices = _resolve_feature_indices(state_dim, feature_indices) or list(
        range(state_dim)
    )
    x_axis = np.arange(real.shape[1], dtype=np.float32)
    x_label = "step"
    if dt is not None:
        x_axis = x_axis * float(dt)
        x_label = "time"

    payload = {}
    for state_index in indices:
        state_name = str(state_names[state_index])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, trajectories, label, seed in (
            (axes[0], real, "ground truth", 17),
            (axes[1], model, "generated", 29),
        ):
            colors = _sample_colors(len(trajectories), seed=seed)
            for trajectory, color in zip(trajectories, colors):
                ax.plot(
                    x_axis,
                    trajectory[:, state_index],
                    linewidth=0.8,
                    alpha=0.7,
                    color=color,
                )
            ax.set_title(label)
            ax.set_ylabel(state_name)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel(x_label)
        fig.suptitle(f"{title or 'ground truth vs generated'}: {state_name}")
        fig.tight_layout()
        payload[f"{key}/{state_name}"] = wandb.Image(fig)
        plt.close(fig)

    if payload:
        wandb_run.log(payload, step=step)


def log_overview_trajectory_plots(
    wandb_run,
    dataset_name: str,
    key: str,
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    *,
    max_contexts: int = 1,
    title: Optional[str] = None,
    step: Optional[int] = None,
) -> None:
    if wandb_run is None or max_contexts <= 0:
        return

    overview_fn = _overview_fn(dataset_name)
    if overview_fn is None:
        return

    import matplotlib.pyplot as plt

    figures = overview_fn(
        real_trajs=real_trajs,
        model_trajs=model_trajs,
        state_names=state_names,
        max_contexts=max_contexts,
        title=title,
    )
    payload = {
        f"{key}/{name}": wandb.Image(fig)
        for name, fig in figures.items()
    }
    for fig in figures.values():
        plt.close(fig)
    if payload:
        wandb_run.log(payload, step=step)


def log_paired_trajectory_grid_plots(
    wandb_run,
    dataset_name: str,
    key: str,
    real_trajs: np.ndarray,
    model_trajs: np.ndarray,
    state_names: Sequence[str],
    *,
    max_contexts: int = 3,
    max_trajectories: int = 20,
    title: Optional[str] = None,
    step: Optional[int] = None,
) -> None:
    if wandb_run is None or max_contexts <= 0 or max_trajectories <= 0:
        return

    paired_grid_fn = _paired_grid_fn(dataset_name)
    if paired_grid_fn is None:
        return

    import matplotlib.pyplot as plt

    figures = paired_grid_fn(
        real_trajs=real_trajs,
        model_trajs=model_trajs,
        state_names=state_names,
        max_contexts=max_contexts,
        max_trajectories=max_trajectories,
        title=title,
    )
    payload = {
        f"{key}/{name}": wandb.Image(fig)
        for name, fig in figures.items()
    }
    for fig in figures.values():
        plt.close(fig)
    if payload:
        wandb_run.log(payload, step=step)
