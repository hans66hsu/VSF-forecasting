from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.datasets.data import ForecastDataset
from src.utils.value_utils import as_tuple


_FORECAST_THICK_TUBE_DEFAULTS = {
    "use_thick_tube": False,
    "thick_tube_p_original_noise": 0.0,
}
_FIXED_FORECAST_THICK_TUBE = {
    "thick_tube_sigma_min": 0.005,
    "thick_tube_sigma_max": 0.5,
    "thick_tube_noise_path": "contracting",
}
_LEGACY_STREAMING_FLOW_THICK_TUBE = {
    **_FORECAST_THICK_TUBE_DEFAULTS,
    **_FIXED_FORECAST_THICK_TUBE,
}
_FORECAST_FLOW_SCHEDULE_KEYS = ("sigma0", "k")
_ROLLOUT_ONLY_FLOW_KEYS = (
    "initial_noise_state",
    "add_noise_every_rollout",
)


def checkpoint_payload(
    model_name: str,
    artifacts: Mapping[str, Any],
    train_dataset: ForecastDataset,
    cfg: DictConfig,
) -> Dict[str, Any]:
    payload = {
        "model_name": model_name,
        "model_state_dict": artifacts["model"].state_dict(),
        "ema_model_state_dict": artifacts["ema_model"].state_dict(),
        "history": artifacts["history"],
        "stats": train_dataset.stats,
        "state_names": as_tuple(cfg.dataset.state_names),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    for key in ("sigma0", "k"):
        if key in artifacts:
            payload[key] = artifacts[key]
    return payload


def _without_keys(values: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if key not in keys}


def _flow_checkpoint_identity(cfg: DictConfig) -> Dict[str, Any]:
    return _without_keys(
        OmegaConf.to_container(cfg.streaming_flow, resolve=True),
        _ROLLOUT_ONLY_FLOW_KEYS,
    )


def _identity_checkpoint_filename(
    model_name: str,
    seed: int,
    obs_horizon: int,
    pred_horizon: int | None,
    identity: Mapping[str, Any],
    k: Any = None,
    epoch: int | None = None,
) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]

    k_tag = ""
    if k is not None:
        k_value = f"{float(k):g}".replace("-", "m").replace(".", "p")
        k_tag = f"_k{k_value}"
    pred_tag = "" if pred_horizon is None else f"_pred{pred_horizon}"
    epoch_tag = "" if epoch is None else f"_epoch{int(epoch)}"
    return (
        f"{model_name}_seed{seed}{k_tag}_obs{obs_horizon}"
        f"{pred_tag}_{digest}{epoch_tag}.pt"
    )


def model_checkpoint_identity(cfg: DictConfig, model_name: str) -> Dict[str, Any]:
    architecture = OmegaConf.to_container(cfg.model, resolve=True)
    transformer_models = {
        "SFPredictor",
        "VSFPredictor",
    }
    architecture_key = "transformer" if model_name in transformer_models else "unet"

    if model_name in {"variational_streaming_flow", "VSFPredictor"}:
        variational = dict(
            OmegaConf.to_container(cfg.variational_streaming_flow, resolve=True)
        )
        flow = None
        if model_name == "variational_streaming_flow":
            flow = _flow_checkpoint_identity(cfg)
            for key in _FORECAST_FLOW_SCHEDULE_KEYS:
                flow[key] = variational.pop(key, flow[key])
            for key, default in _FORECAST_THICK_TUBE_DEFAULTS.items():
                flow[key] = variational.pop(key, default)
            for key, value in _FIXED_FORECAST_THICK_TUBE.items():
                variational.pop(key, None)
                flow[key] = value

        # Preserve identities of checkpoints created with the former selectors.
        variational["velocity_loss"] = "crps"
        variational["posterior_type"] = (
            "local_velocity"
            if model_name == "variational_streaming_flow"
            else "trajectory"
        )
        variational["latent_conditioning"] = "bottleneck_sum"
        algorithm = {
            "variational": variational,
        }
        if flow is not None:
            algorithm["flow"] = flow
    elif model_name == "streaming_flow":
        algorithm = _flow_checkpoint_identity(cfg)
        algorithm.update(_LEGACY_STREAMING_FLOW_THICK_TUBE)
    else:
        algorithm = {}

    return {
        "architecture": architecture[architecture_key],
        "algorithm": algorithm,
    }


def _checkpoint_filename(
    cfg: DictConfig,
    model_name: str,
    seed: int,
    *,
    identity: Mapping[str, Any] | None = None,
    obs_horizon: int | None = None,
    pred_horizon: int | None = None,
    epoch: int | None = None,
) -> str:
    if identity is not None:
        return _identity_checkpoint_filename(
            model_name,
            seed,
            int(obs_horizon),
            None if pred_horizon is None else int(pred_horizon),
            identity,
            epoch=epoch,
        )

    resolved = OmegaConf.to_container(cfg, resolve=True)
    model_identity = model_checkpoint_identity(cfg, model_name)
    identity = {
        "model_name": model_name,
        "dataset": _without_keys(
            resolved["dataset"],
            (
                "metrics",
                "regenerate",
                "rollout_horizon",
                "eval_anchor_index",
                "eval_total_future_steps",
            ),
        ),
        "train": _without_keys(
            resolved["train"],
            ("num_workers", "pin_memory"),
        ),
        **model_identity,
    }
    return _identity_checkpoint_filename(
        model_name,
        seed,
        int(cfg.dataset.obs_horizon),
        int(cfg.dataset.pred_horizon),
        identity,
        model_identity["algorithm"].get("k"),
        epoch,
    )


def save_model_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    payload: Mapping[str, Any],
    cfg: DictConfig,
    seed: int,
    **naming,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / _checkpoint_filename(
        cfg, model_name, seed, **naming
    )
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    return checkpoint_path


def model_checkpoint_path(
    load_dir: str | Path,
    seed: int,
    model_name: str,
    cfg: DictConfig,
    *,
    allow_legacy: bool = True,
    **naming,
) -> Path:
    load_root = Path(to_absolute_path(str(load_dir)))
    dataset_name = str(cfg.dataset.name)
    model_dir = load_root if load_root.name == dataset_name else load_root / dataset_name
    model_path = model_dir / _checkpoint_filename(
        cfg, model_name, seed, **naming
    )
    if model_path.exists() or not allow_legacy:
        return model_path

    legacy_path = (
        load_root
        / f"seed_{seed}"
        / "checkpoints"
        / f"{model_name}.pt"
    )
    return legacy_path if legacy_path.exists() else model_path


def load_model_checkpoint(
    load_dir: str | Path,
    model_name: str,
    cfg: DictConfig,
    seed: int,
    *,
    allow_legacy: bool = True,
    weights_only: bool = False,
    **naming,
) -> tuple[Dict[str, Any], Path]:
    checkpoint_path = model_checkpoint_path(
        load_dir,
        seed,
        model_name,
        cfg,
        allow_legacy=allow_legacy,
        **naming,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=weights_only,
    )
    return checkpoint, checkpoint_path


def _artifacts_from_checkpoint(
    model_name: str,
    checkpoint: Mapping[str, Any],
    trainer,
    train_dataset: ForecastDataset,
    device: torch.device,
) -> Dict[str, Any]:
    saved_model_name = checkpoint.get("model_name", model_name)
    if saved_model_name != model_name:
        raise ValueError(
            f"Checkpoint model_name={saved_model_name!r} does not match "
            f"requested model {model_name!r}."
        )

    model = trainer.model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ema_model = copy.deepcopy(model).to(device)
    ema_model.load_state_dict(
        checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    )
    ema_model.eval()

    artifacts = {
        "model": model,
        "ema_model": ema_model,
        "history": checkpoint.get("history", []),
        "stats": checkpoint.get("stats", train_dataset.stats),
    }
    artifacts.update(trainer.extra_artifacts())
    for key in ("sigma0", "k"):
        if key in checkpoint:
            artifacts[key] = checkpoint[key]
    return artifacts


def load_model_checkpoints(
    cfg: DictConfig,
    train_loader,
    train_dataset: ForecastDataset,
    device: torch.device,
    model_names: Sequence[str],
    seed: int,
    build_trainer: Callable[..., Any],
    model_labels: Mapping[str, str],
) -> Dict[str, Mapping[str, Any]]:
    state_dim = train_dataset.state_dim
    global_cond_dim = cfg.dataset.obs_horizon * state_dim
    artifacts_by_model: Dict[str, Mapping[str, Any]] = {}

    for model_name in model_names:
        checkpoint, checkpoint_path = load_model_checkpoint(
            cfg.checkpoint.load_dir,
            model_name,
            cfg,
            seed,
        )
        print(f"\nLoading {model_labels[model_name]} checkpoint from {checkpoint_path}")
        trainer = build_trainer(
            model_name,
            cfg,
            state_dim,
            global_cond_dim,
            train_loader,
            device,
            log_callback=None,
        )
        artifacts_by_model[model_name] = _artifacts_from_checkpoint(
            model_name,
            checkpoint,
            trainer,
            train_dataset,
            device,
        )

    return artifacts_by_model
