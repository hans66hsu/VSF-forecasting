from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import src.algo.rollout as rollout_module
from src.algo.common.models import ConditionalUnet1D
from src.algo.streaming_flow import StreamingFlowForecasterTrainer
from src.algo.variational_streaming_flow import (
    PosteriorUnetEncoder1D,
    VariationalStreamingFlowForecasterTrainer,
    VariationalStreamingFlowModel,
)
from src.datasets.data import (
    ForecastDataset,
    split_trajectory_bank,
)
from src.datasets.double_pendulum import simulate_trajectory_bank
from src.datasets.lorenz import (
    simulate_trajectory_bank as simulate_lorenz_trajectory_bank,
)
from src.datasets.rossler import (
    simulate_trajectory_bank as simulate_rossler_trajectory_bank,
)
from src.experiments.utils import (
    checkpoint_payload,
    load_model_checkpoints,
    save_model_checkpoint,
)
from src.utils.utils import get_default_device, set_seed
from src.utils.value_utils import as_tuple
from src.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_overview_trajectory_plots,
    log_paired_trajectory_grid_plots,
    log_trajectory_line_plots,
    make_train_loss_logger,
)

SEED = 100


MODEL_LABELS = {
    "streaming_flow": "Streaming flow (fixed history)",
    "variational_streaming_flow": "Variational streaming flow",
}


def _requested_model_names(cfg: DictConfig) -> list[str]:
    requested = list(OmegaConf.to_container(cfg.models, resolve=True))
    if "all" in requested:
        requested = list(MODEL_LABELS)

    unknown = sorted(set(requested) - set(MODEL_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Valid models: {sorted(MODEL_LABELS)}"
        )
    return requested


def _load_or_create_bank(cfg: DictConfig) -> Mapping[str, np.ndarray]:
    data_dir = Path(to_absolute_path(cfg.data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / cfg.dataset.file_name

    if dataset_path.exists() and not cfg.dataset.regenerate:
        print(f"Loading dataset bank from {dataset_path}")
        loaded = np.load(dataset_path)
        bank = {key: loaded[key] for key in loaded.files}
        if (
            str(cfg.dataset.get("kind", "deterministic")) != "probabilistic"
            or np.asarray(bank[cfg.dataset.state_key]).ndim == 4
        ):
            return bank
        print("Existing probabilistic bank is flat; regenerating grouped bank.")

    print(f"Generating dataset bank at {dataset_path}")
    dataset_name = str(cfg.dataset.name)
    if dataset_name.startswith("double-pendulum"):
        bank = simulate_trajectory_bank(
            num_trajectories=cfg.dataset.num_trajectories,
            t_max=cfg.dataset.t_max,
            dt=cfg.dataset.dt,
            theta1_range=as_tuple(cfg.dataset.theta1_range),
            theta2_range=as_tuple(cfg.dataset.theta2_range),
            max_initial_angle_difference=cfg.dataset.get(
                "max_initial_angle_difference",
            ),
            omega1_range=as_tuple(cfg.dataset.omega1_range),
            omega2_range=as_tuple(cfg.dataset.omega2_range),
            m1=cfg.dataset.m1,
            m2=cfg.dataset.m2,
            l1=cfg.dataset.l1,
            l2=cfg.dataset.l2,
            g=cfg.dataset.g,
            seed=cfg.dataset.seed,
        )
    elif dataset_name.startswith("lorenz"):
        bank = simulate_lorenz_trajectory_bank(
            num_initial_conditions=cfg.dataset.get("num_initial_conditions", None),
            num_rhos_per_initial_condition=cfg.dataset.get(
                "num_rhos_per_initial_condition",
                10,
            ),
            num_trajectories=cfg.dataset.get("num_trajectories", None),
            t_max=cfg.dataset.t_max,
            dt=cfg.dataset.dt,
            rho_values=as_tuple(cfg.dataset.rho_values),
            initial_noise_std=cfg.dataset.initial_noise_std,
            sigma=cfg.dataset.sigma,
            beta=cfg.dataset.beta,
            seed=cfg.dataset.seed,
        )
    elif dataset_name.startswith("rossler"):
        bank = simulate_rossler_trajectory_bank(
            num_initial_conditions=cfg.dataset.get("num_initial_conditions", None),
            num_cs_per_initial_condition=cfg.dataset.get(
                "num_cs_per_initial_condition",
                10,
            ),
            num_trajectories=cfg.dataset.get("num_trajectories", None),
            t_max=cfg.dataset.t_max,
            dt=cfg.dataset.dt,
            c_values=as_tuple(cfg.dataset.c_values),
            initial_noise_std=cfg.dataset.initial_noise_std,
            a=cfg.dataset.a,
            b=cfg.dataset.b,
            seed=cfg.dataset.seed,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")
    if dataset_name.endswith("-deter"):
        bank["states"] = bank["states"][:, 0]
    np.savez_compressed(dataset_path, **bank)
    return bank


def _flatten_sample_axis(split: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    split = dict(split)
    states = np.asarray(split["states"])
    if states.ndim == 4:
        split["states"] = states.reshape(-1, *states.shape[2:])
    return split


def _prepare_dataloaders(
    cfg: DictConfig,
    bank: Mapping[str, np.ndarray],
) -> tuple[Dict[str, Dict[str, np.ndarray]], ForecastDataset, DataLoader]:
    splits = split_trajectory_bank(
        bank,
        train_fraction=cfg.dataset.train_fraction,
        val_fraction=cfg.dataset.val_fraction,
    )

    state_key = cfg.dataset.state_key
    for split in splits.values():
        split["states"] = split[state_key]

    state_names = as_tuple(cfg.dataset.state_names)
    state_dim = splits["train"]["states"].shape[-1]
    if len(state_names) != state_dim:
        raise ValueError(
            f"Resolved {len(state_names)} state names for state_dim={state_dim}: "
            f"{state_names}"
        )

    dataset_kwargs = {
        "trajectory_bank": _flatten_sample_axis(splits["train"]),
        "obs_horizon": cfg.dataset.obs_horizon,
        "pred_horizon": cfg.dataset.pred_horizon,
    }
    train_dataset = ForecastDataset(**dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    return splits, train_dataset, train_loader


def _build_unet(
    cfg: DictConfig,
    state_dim: int,
    global_cond_dim: int,
    *,
    sin_embedding_scale: float,
    latent_dim: Optional[int] = None,
    latent_feature_dim: int = 256,
    latent_weight: float = 0.9,
) -> ConditionalUnet1D:
    return ConditionalUnet1D(
        input_dim=state_dim,
        global_cond_dim=global_cond_dim,
        sin_embedding_scale=sin_embedding_scale,
        diffusion_step_embed_dim=cfg.model.unet.diffusion_step_embed_dim,
        down_dims=as_tuple(cfg.model.unet.down_dims),
        kernel_size=cfg.model.unet.kernel_size,
        n_groups=cfg.model.unet.n_groups,
        latent_dim=latent_dim,
        latent_feature_dim=latent_feature_dim,
        latent_weight=latent_weight,
    )


def _common_trainer_kwargs(
    cfg: DictConfig,
    train_loader: DataLoader,
    device: torch.device,
    log_callback,
) -> Dict[str, Any]:
    return {
        "dataloader": train_loader,
        "num_epochs": cfg.train.num_epochs,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "warmup_steps": cfg.train.warmup_steps,
        "device": device,
        "log_callback": log_callback,
    }


def _flow_matching_trainer_kwargs(flow_cfg: DictConfig) -> Dict[str, Any]:
    return {
        "sigma0": flow_cfg.sigma0,
        "k": flow_cfg.k,
    }


def _thick_tube_trainer_kwargs(variational_cfg: DictConfig) -> Dict[str, Any]:
    return {
        "use_thick_tube": variational_cfg.get("use_thick_tube", False),
        "thick_tube_p_original_noise": variational_cfg.get(
            "thick_tube_p_original_noise",
            0.25,
        ),
    }


def _build_trainer(
    model_name: str,
    cfg: DictConfig,
    state_dim: int,
    global_cond_dim: int,
    train_loader: DataLoader,
    device: torch.device,
    log_callback,
):
    common_kwargs = _common_trainer_kwargs(cfg, train_loader, device, log_callback)
    if model_name == "variational_streaming_flow":
        variational_cfg = cfg.variational_streaming_flow
        latent_dim = int(variational_cfg.latent_dim)
        model = VariationalStreamingFlowModel(
            posterior_encoder=PosteriorUnetEncoder1D(
                input_dim=state_dim,
                global_cond_dim=global_cond_dim,
                latent_dim=latent_dim,
                sin_embedding_scale=cfg.streaming_flow.sin_embedding_scale,
                time_embed_dim=cfg.model.unet.diffusion_step_embed_dim,
                down_dims=as_tuple(variational_cfg.posterior_down_dims),
                kernel_size=cfg.model.unet.kernel_size,
                n_groups=cfg.model.unet.n_groups,
            ),
            velocity_unet=_build_unet(
                cfg,
                state_dim,
                global_cond_dim,
                sin_embedding_scale=cfg.streaming_flow.sin_embedding_scale,
                latent_dim=latent_dim,
                latent_feature_dim=int(variational_cfg.latent_feature_dim),
                latent_weight=float(variational_cfg.latent_weight),
            ),
            latent_dim=latent_dim,
        )
        return VariationalStreamingFlowForecasterTrainer(
            model=model,
            beta_kl=float(variational_cfg.beta_kl),
            crps_num_samples=int(variational_cfg.get("crps_num_samples", 8)),
            crps_diversity_weight=float(
                variational_cfg.get("crps_diversity_weight", 0.5)
            ),
            crps_beta=float(variational_cfg.get("crps_beta", 1.99)),
            **_flow_matching_trainer_kwargs(variational_cfg),
            **_thick_tube_trainer_kwargs(variational_cfg),
            **common_kwargs,
        )

    if model_name == "streaming_flow":
        model = _build_unet(
            cfg,
            state_dim,
            global_cond_dim,
            sin_embedding_scale=cfg.streaming_flow.sin_embedding_scale,
        )
        return StreamingFlowForecasterTrainer(
            model=model,
            **_flow_matching_trainer_kwargs(cfg.streaming_flow),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported model: {model_name}.")


def _train_selected_models(
    cfg: DictConfig,
    train_loader: DataLoader,
    train_dataset: ForecastDataset,
    device: torch.device,
    seed: int,
    wandb_run=None,
) -> Dict[str, Mapping[str, Any]]:
    """Load requested checkpoints when available and train any missing models.

    Returns each requested model's training artifacts, saving newly trained
    checkpoints when checkpointing is enabled.
    """
    requested = _requested_model_names(cfg)
    artifacts_by_model: Dict[str, Mapping[str, Any]] = {}
    models_to_train = list(requested)
    if cfg.checkpoint.get("load_dir", None) is not None:
        models_to_train = []
        for model_name in requested:
            try:
                artifacts_by_model.update(
                    load_model_checkpoints(
                        cfg,
                        train_loader,
                        train_dataset,
                        device,
                        [model_name],
                        seed,
                        _build_trainer,
                        MODEL_LABELS,
                    )
                )
            except FileNotFoundError as error:
                print(f"\n{error}\nCheckpoint missing; falling back to training.")
                models_to_train.append(model_name)

    state_dim = train_dataset.state_dim
    global_cond_dim = cfg.dataset.obs_horizon * state_dim
    checkpoint_root = Path(to_absolute_path("model"))
    checkpoint_dir = checkpoint_root / str(cfg.dataset.name)

    for model_name in models_to_train:
        print(f"\nTraining {MODEL_LABELS[model_name]}")
        wandb_cfg = cfg.get("wandb", {})
        log_callback = (
            make_train_loss_logger(wandb_run, model_name)
            if wandb_cfg.get("log_train_loss", True)
            else None
        )

        trainer = _build_trainer(
            model_name,
            cfg,
            state_dim,
            global_cond_dim,
            train_loader,
            device,
            log_callback,
        )
        artifacts = trainer.fit()

        artifacts_by_model[model_name] = artifacts
        if cfg.checkpoint.enabled:
            checkpoint_path = save_model_checkpoint(
                checkpoint_dir,
                model_name,
                checkpoint_payload(model_name, artifacts, train_dataset, cfg),
                cfg,
                seed,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    return {model_name: artifacts_by_model[model_name] for model_name in requested}


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_rollout(
    label: str,
    rollout_fn,
    sync_device: torch.device,
    timing_steps: int,
    num_samples: int,
    **kwargs,
) -> np.ndarray:
    """Generate vectorized forecast samples and report synchronized runtime.

    The result has shape [S, T, D] for one context or [C, S, T, D] for a
    batch of contexts.
    """
    _sync_device_for_timing(sync_device)
    start_time = time.perf_counter()
    # Vectorized sampling performed inside sample_rollout_forecast:
    # [C, H, D] -> [C * S, H, D] -> [C * S, T, D]
    # -> [C, S, T, D].
    samples = rollout_module.sample_rollout_forecast(
        rollout_fn,
        num_samples=num_samples,
        **kwargs,
    )
    _sync_device_for_timing(sync_device)
    elapsed_s = time.perf_counter() - start_time
    initial_obs = np.asarray(kwargs["initial_obs"])
    num_contexts = initial_obs.shape[0] if initial_obs.ndim == 3 else 1
    sample_steps = timing_steps * num_samples * num_contexts
    print(
        f"{label} inference time: {elapsed_s:.3f} s "
        f"({elapsed_s / sample_steps * 1000:.2f} ms/sample-step)"
    )
    return samples


def _sample_model_rollout(
    model_name: str,
    label: str,
    artifacts: Mapping[str, Any],
    cfg: DictConfig,
    train_dataset: ForecastDataset,
    device: torch.device,
    contexts: np.ndarray,
    total_future_steps: int,
    num_samples: int,
) -> np.ndarray:
    """Sample timed SF or VSF rollouts from a model's EMA checkpoint artifacts.

    The returned array has shape [C, S, T, D], where C is the number of
    contexts and S is the number of forecast samples per context.
    """
    stats = artifacts.get("stats", train_dataset.stats)
    if model_name not in {
        "streaming_flow",
        "variational_streaming_flow",
    }:
        raise ValueError(f"Unsupported rollout model: {model_name}.")

    flow_cfg = cfg.streaming_flow
    return _measure_rollout(
        label,
        rollout_module.rollout_streaming_flow_forecast,
        device,
        total_future_steps,
        num_samples,
        model=artifacts["ema_model"],
        initial_obs=contexts,
        stats=stats,
        pred_horizon=cfg.dataset.pred_horizon,
        rollout_horizon=cfg.dataset.rollout_horizon,
        total_future_steps=total_future_steps,
        initial_noise_scale=float(flow_cfg.initial_noise_state),
        add_noise_every_rollout=bool(flow_cfg.get("add_noise_every_rollout", False)),
        latent_dim=(
            int(cfg.variational_streaming_flow.latent_dim)
            if model_name == "variational_streaming_flow"
            else None
        ),
        resample_latent_each_step=bool(
            cfg.eval.get("vsf_resample_latent_each_step", False)
        ),
        device=device,
    )


def _prepare_eval_contexts(
    test_states: np.ndarray,
    anchor_index: int,
    obs_horizon: int,
    total_future_steps: int,
    start_index: int,
    num_contexts: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract observation contexts and matching futures for evaluation.

    Deterministic input [B, T, D] produces futures [C, T_future, D]. For
    probabilistic input [B, S, T, D], contexts are averaged over S and the
    futures retain that sample dimension as [C, S, T_future, D].
    """
    if not 0 <= start_index < test_states.shape[0]:
        raise ValueError(
            f"eval.test_trajectory_index must be in [0, {test_states.shape[0] - 1}], "
            f"got {start_index}."
        )
    test_states = test_states[start_index:]
    if num_contexts is None:
        num_contexts = test_states.shape[0]
    num_contexts = min(int(num_contexts), test_states.shape[0])
    if num_contexts <= 0:
        raise ValueError(
            f"eval.num_contexts must be positive or null, got {num_contexts}."
        )

    selected = test_states[:num_contexts]
    contexts = []
    futures = []
    if selected.ndim == 3:
        for trajectory in selected:
            context, future = rollout_module.prepare_rollout_example(
                trajectory=trajectory,
                anchor_index=anchor_index,
                obs_horizon=obs_horizon,
                total_future_steps=total_future_steps,
            )
            contexts.append(context)
            futures.append(future)
    elif selected.ndim == 4:
        for trajectory_samples in selected:
            prepared = [
                rollout_module.prepare_rollout_example(
                    trajectory=trajectory,
                    anchor_index=anchor_index,
                    obs_horizon=obs_horizon,
                    total_future_steps=total_future_steps,
                )
                for trajectory in trajectory_samples
            ]
            contexts.append(np.stack([context for context, _ in prepared]).mean(axis=0))
            futures.append(np.stack([future for _, future in prepared]))
    else:
        raise ValueError(
            "test_states must have shape [B, T, D] or [B, S, T, D], "
            f"got {selected.shape}."
        )

    return (
        np.stack(contexts).astype(np.float32),
        np.stack(futures).astype(np.float32),
    )


def _evaluate_models(
    cfg: DictConfig,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    train_dataset: ForecastDataset,
    artifacts_by_model: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    wandb_run=None,
) -> Dict[str, Dict[str, Any]]:
    if not cfg.eval.enabled:
        return {}

    state_names = as_tuple(cfg.dataset.state_names)
    total_future_steps = (
        cfg.eval.total_future_steps
        if cfg.eval.total_future_steps is not None
        else cfg.dataset.eval_total_future_steps
    )
    anchor_index = (
        cfg.eval.anchor_index
        if cfg.eval.anchor_index is not None
        else cfg.dataset.eval_anchor_index
    )
    # contexts: [B, obs_horizon, D]
    # reference_futures: deterministic [B, T, D], probabilistic [B, S_env, T, D]
    contexts, reference_futures = _prepare_eval_contexts(
        test_states=splits["test"]["states"],
        anchor_index=anchor_index,
        obs_horizon=cfg.dataset.obs_horizon,
        total_future_steps=total_future_steps,
        start_index=cfg.eval.test_trajectory_index,
        num_contexts=cfg.eval.get("num_contexts", 1),
    )
    metric_names = as_tuple(cfg.dataset.get("metrics", ("rmse", "mae")))
    pred_num_samples = max(1, int(cfg.eval.pred_num_samples))
    dataset_kind = str(cfg.dataset.get("kind", "deterministic"))
    is_probabilistic = dataset_kind == "probabilistic"
    plot_reference_future = (
        reference_futures[0, 0] if is_probabilistic else reference_futures[0]
    )

    print(
        f"\nEvaluating {total_future_steps}-step rollout over "
        f"{len(contexts)} context(s)"
    )
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    # Loop over trained models
    for model_name, artifacts in artifacts_by_model.items():
        label = MODEL_LABELS[model_name]

        # model_samples_by_context:
        # (num_contexts, num_samples, total_future_steps, state_dim)
        model_samples_by_context = _sample_model_rollout(
            model_name,
            label,
            artifacts,
            cfg,
            train_dataset,
            device,
            contexts,
            total_future_steps,
            pred_num_samples,
        )
        # plot_model_samples: first context's generated sample bank,
        # (num_samples, total_future_steps, state_dim)
        plot_model_samples = model_samples_by_context[0]

        metrics = rollout_module.compute_rollout_metrics(
            predicted_samples=model_samples_by_context,
            reference_samples=reference_futures,
            state_names=state_names,
            metric_names=metric_names,
            dataset_kind=dataset_kind,
        )

        metrics_by_model[model_name] = metrics
        for metric_name in metric_names:
            if metric_name in metrics:
                print(f"{label} {metric_name.upper()}: {metrics[metric_name]}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{model_name}/{metric_name}": float(metrics[metric_name])
                    for metric_name in metric_names
                    if metric_name in metrics
                }
            )
        wandb_cfg = cfg.get("wandb", {})
        if wandb_run is not None and wandb_cfg.get("log_trajectory_plots", True):
            plot_real_samples = (
                reference_futures[0] if is_probabilistic else plot_reference_future
            )

            feature_indices = wandb_cfg.get("trajectory_feature_indices", None)
            if OmegaConf.is_config(feature_indices):
                feature_indices = OmegaConf.to_container(feature_indices, resolve=True)

            # only the first context is uploaded to W&B for trajectory plots.
            log_trajectory_line_plots(
                wandb_run,
                key=f"rollout/{model_name}/trajectory",
                real_trajs=plot_real_samples,
                model_trajs=plot_model_samples,
                state_names=state_names,
                dt=cfg.dataset.get("dt", None),
                feature_indices=feature_indices,
                title=f"{label}: ground truth vs generated",
            )
        if wandb_run is not None and wandb_cfg.get("log_overview_trajectory", True):
            log_overview_trajectory_plots(
                wandb_run,
                dataset_name=str(cfg.dataset.name),
                key=f"overview_trajectory/{model_name}",
                real_trajs=reference_futures,
                model_trajs=model_samples_by_context,
                state_names=state_names,
                max_contexts=int(wandb_cfg.get("overview_num_contexts", 1)),
                title=f"{label}: ground truth vs generated",
            )
        if wandb_run is not None and wandb_cfg.get("log_paired_trajectory_grid", True):
            log_paired_trajectory_grid_plots(
                wandb_run,
                dataset_name=str(cfg.dataset.name),
                key=f"paired_trajectory_grid/{model_name}",
                real_trajs=reference_futures,
                model_trajs=model_samples_by_context,
                state_names=state_names,
                max_contexts=int(wandb_cfg.get("paired_grid_num_contexts", 3)),
                max_trajectories=int(wandb_cfg.get("paired_grid_num_trajectories", 20)),
                title=f"{label}: paired ground truth vs generated",
            )

    return metrics_by_model


def run_forecasting(
    cfg: DictConfig,
    run_dir: Path | None = None,
    seed: int = SEED,
) -> Dict[str, Dict[str, Any]]:
    if run_dir is None:
        run_dir = Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = get_default_device(cfg.device)
    print(f"Using seed: {seed}")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    wandb_run = init_wandb_run(cfg, run_dir, seed)
    try:
        bank = _load_or_create_bank(cfg)
        splits, train_dataset, train_loader = _prepare_dataloaders(cfg, bank)
        print("train trajectories:", splits["train"]["states"].shape[0])
        print("test trajectories:", splits["test"]["states"].shape[0])
        print("state_dim:", train_dataset.state_dim)
        print("state names:", as_tuple(cfg.dataset.state_names))
        print("training windows:", len(train_dataset))

        artifacts_by_model = _train_selected_models(
            cfg,
            train_loader,
            train_dataset,
            device,
            seed,
            wandb_run=wandb_run,
        )
        return _evaluate_models(
            cfg,
            splits,
            train_dataset,
            artifacts_by_model,
            device,
            wandb_run=wandb_run,
        )
    finally:
        finish_wandb_run(wandb_run)


@hydra.main(version_base=None, config_path="../../configs", config_name="forecast")
def main(cfg: DictConfig) -> Dict[str, Dict[str, Any]]:
    return run_forecasting(cfg, seed=SEED)


if __name__ == "__main__":
    main()
