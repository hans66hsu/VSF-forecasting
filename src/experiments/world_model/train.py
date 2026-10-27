from __future__ import annotations

import json
from pathlib import Path

import hydra
import stable_pretraining as spt
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.algo.common.jepa import FrozenJEPA, load_lewm_checkpoint
from src.algo.common.models import EMAModel
from src.algo.vsf_predictor import SFPredictor, VSFPredictor
from src.experiments.utils import (
    model_checkpoint_identity,
    model_checkpoint_path,
    save_model_checkpoint,
)
from src.experiments.world_model.embed import load_embedding_dataset
from src.utils.utils import get_default_device, set_seed

SF_PREDICTOR_NAME = "SFPredictor"
VSF_PREDICTOR_NAME = "VSFPredictor"
FLOW_PREDICTOR_NAMES = (SF_PREDICTOR_NAME, VSF_PREDICTOR_NAME)
PREDICTOR_NAMES = (
    "ARPredictor",
    SF_PREDICTOR_NAME,
    VSF_PREDICTOR_NAME,
)
TRAINABLE_PREDICTOR_NAMES = (
    SF_PREDICTOR_NAME,
    VSF_PREDICTOR_NAME,
)
EMA_DECAY = 0.75
CHECKPOINT_INTERVAL = 5
SEED = 100


def _num_frames(cfg: DictConfig) -> int:
    return int(cfg.dataset.num_steps) - 1


def predictor_names(cfg: DictConfig) -> list[str]:
    names = [str(name) for name in cfg.predictor.models]
    unknown = sorted(set(names) - set(PREDICTOR_NAMES))
    if not names or unknown:
        raise ValueError(
            f"predictor.models must use {list(PREDICTOR_NAMES)}; got {names}."
        )
    return names


def validate_world_model_config(cfg: DictConfig, *, evaluation: bool = False) -> None:
    if int(cfg.dataset.num_steps) < 2:
        raise ValueError("Push-T num_steps must be at least 2.")
    if int(cfg.planning.action_block) != int(cfg.dataset.frameskip):
        raise ValueError("planning.action_block must equal the Push-T frameskip.")
    if "ARPredictor" in cfg.predictor.models and (
        int(cfg.dataset.frameskip) != 5 or _num_frames(cfg) != 3
    ):
        raise ValueError(
            "The pretrained ARPredictor requires frameskip=5 and num_steps=4."
        )
    if evaluation and (
        int(cfg.planning.horizon) * int(cfg.planning.action_block)
        > int(cfg.eval.eval_budget)
    ):
        raise ValueError("The planned action horizon cannot exceed eval_budget.")


def _wrap_predictor(cfg: DictConfig, predictor: torch.nn.Module) -> FrozenJEPA:
    checkpoint_dir = Path(to_absolute_path(cfg.jepa.checkpoint_dir))
    pretrained = load_lewm_checkpoint(checkpoint_dir)
    frameskip = int(cfg.dataset.frameskip)
    if frameskip == 5:
        action_encoder = pretrained.action_encoder
    else:
        with (checkpoint_dir / "config.json").open() as stream:
            lewm_config = json.load(stream)
        action_encoder = instantiate(
            lewm_config["action_encoder"], input_dim=2 * frameskip
        )
    model = FrozenJEPA(
        encoder=pretrained.encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=pretrained.projector,
        freeze_action_encoder=frameskip == 5,
    )
    if getattr(model.action_encoder, "emb_dim", None) != int(cfg.jepa.embed_dim):
        raise ValueError(
            "The frozen action encoder output dimension does not match embed_dim."
        )
    return model


def _flow_predictor_kwargs(cfg: DictConfig) -> dict:
    transformer_cfg = cfg.model.transformer
    embed_dim = int(cfg.jepa.embed_dim)
    return {
        "num_frames": _num_frames(cfg),
        "input_dim": embed_dim,
        "hidden_dim": int(transformer_cfg.hidden_dim),
        "output_dim": embed_dim,
        "depth": int(transformer_cfg.num_layers),
        "heads": int(transformer_cfg.num_heads),
        "dim_head": int(transformer_cfg.dim_head),
        "mlp_dim": int(transformer_cfg.mlp_dim),
        "dropout": float(transformer_cfg.dropout),
        "flow_step_embed_dim": int(transformer_cfg.diffusion_step_embed_dim),
        "sin_embedding_scale": float(transformer_cfg.sin_embedding_scale),
    }


def build_sf_model(cfg: DictConfig) -> FrozenJEPA:
    return _wrap_predictor(cfg, SFPredictor(**_flow_predictor_kwargs(cfg)))


def build_vsf_model(cfg: DictConfig) -> FrozenJEPA:
    variational_cfg = cfg.variational_streaming_flow
    posterior_cfg = variational_cfg.posterior
    predictor = VSFPredictor(
        **_flow_predictor_kwargs(cfg),
        latent_dim=int(variational_cfg.latent_dim),
        latent_feature_dim=int(variational_cfg.latent_feature_dim),
        latent_weight=float(variational_cfg.latent_weight),
        beta_kl=float(variational_cfg.beta_kl),
        crps_num_samples=int(variational_cfg.crps_num_samples),
        crps_diversity_weight=float(variational_cfg.crps_diversity_weight),
        crps_beta=float(variational_cfg.crps_beta),
        posterior_hidden_dim=int(posterior_cfg.hidden_dim),
        posterior_depth=int(posterior_cfg.num_layers),
        posterior_heads=int(posterior_cfg.num_heads),
        posterior_dim_head=int(posterior_cfg.dim_head),
        posterior_mlp_dim=int(posterior_cfg.mlp_dim),
        posterior_dropout=float(posterior_cfg.dropout),
    )
    predictor.resample_latent_each_step = bool(
        cfg.eval.get("vsf_resample_latent_each_step", False)
    )
    return _wrap_predictor(cfg, predictor)


def _predictor_config(source: DictConfig, predictor_name: str) -> dict:
    config = {
        "transformer": OmegaConf.to_container(source.model.transformer, resolve=True),
    }
    if predictor_name == VSF_PREDICTOR_NAME:
        config["variational"] = OmegaConf.to_container(
            source.variational_streaming_flow, resolve=True
        )
    return config


def _predictor_checkpoint_identity(cfg: DictConfig, predictor_name: str) -> dict:
    model_identity = model_checkpoint_identity(cfg, predictor_name)
    return {
        "model_name": predictor_name,
        "dataset": {
            key: cfg.dataset[key]
            for key in ("name", "path", "num_steps", "frameskip", "image_size")
        },
        "train": {
            key: cfg.train[key]
            for key in (
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "train_fraction",
            )
        },
        "ema_decay": EMA_DECAY,
        **model_identity,
    }


def _checkpoint_naming(
    cfg: DictConfig,
    predictor_name: str,
    epoch: int | None = None,
) -> dict:
    return {
        "identity": _predictor_checkpoint_identity(cfg, predictor_name),
        "obs_horizon": _num_frames(cfg),
        "epoch": epoch,
    }


def predictor_checkpoint_path(
    cfg: DictConfig,
    predictor_name: str,
    seed: int,
    *,
    epoch: int | None = None,
) -> Path:
    return model_checkpoint_path(
        cfg.checkpoint.get("load_dir") or "model",
        int(seed),
        predictor_name,
        cfg,
        allow_legacy=False,
        **_checkpoint_naming(cfg, predictor_name, epoch),
    )


def load_predictor_checkpoint(
    model: FrozenJEPA,
    cfg: DictConfig,
    predictor_name: str,
    seed: int,
    *,
    use_ema: bool = False,
    epoch: int | None = None,
) -> tuple[dict, Path]:
    path = predictor_checkpoint_path(cfg, predictor_name, seed, epoch=epoch)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    predictor = checkpoint["predictor"]
    if use_ema:
        predictor = checkpoint.get("ema_predictor", predictor)
    model.predictor.load_state_dict(predictor, strict=True)
    if "pred_proj" in checkpoint:
        model.pred_proj.load_state_dict(checkpoint["pred_proj"], strict=True)
    if not model.freeze_action_encoder:
        model.action_encoder.load_state_dict(checkpoint["action_encoder"], strict=True)
    return checkpoint, path


def _batch_loss(
    model: FrozenJEPA,
    batch,
    device: torch.device,
) -> torch.Tensor:
    emb, actions = (value.to(device, non_blocking=True) for value in batch)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        num_frames = int(model.predictor.num_frames)
        act_emb = (
            actions
            if model.freeze_action_encoder
            else model.action_encoder(torch.nan_to_num(actions, 0.0))
        )
        return model.predictor.train_loss(
            emb[:, : num_frames + 1], act_emb[:, :num_frames]
        )


def _mean_validation_loss(
    model: FrozenJEPA,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss = _batch_loss(model, batch, device)
            batch_size = batch[0].shape[0]
            total += float(loss) * batch_size
            count += batch_size
    return total / max(count, 1)


def train_world_model(
    cfg: DictConfig,
    seed: int = SEED,
    predictor_name: str | None = None,
) -> Path | None:
    validate_world_model_config(cfg)
    if predictor_name is None:
        selected = [
            name for name in predictor_names(cfg) if name in TRAINABLE_PREDICTOR_NAMES
        ]
        if not selected:
            print("ARPredictor is loaded from the pretrained LeWM checkpoint.")
            return None
        checkpoint_path = None
        for name in selected:
            checkpoint_path = train_world_model(cfg, seed, name)
        return checkpoint_path
    if predictor_name not in TRAINABLE_PREDICTOR_NAMES:
        raise ValueError(f"Cannot train predictor {predictor_name!r}.")

    seed = int(seed)
    device = get_default_device(cfg.device)
    set_seed(seed)
    run_dir = Path.cwd()
    print(f"Training predictor: {predictor_name}")
    print(f"Using seed: {seed}")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    if predictor_name == SF_PREDICTOR_NAME:
        model = build_sf_model(cfg)
    else:
        model = build_vsf_model(cfg)
    model.to(device)

    loader_kwargs = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": int(cfg.train.num_workers),
        "pin_memory": bool(cfg.train.pin_memory),
        "persistent_workers": int(cfg.train.num_workers) > 0,
    }
    if int(cfg.train.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 3

    dataset = load_embedding_dataset(cfg, model, device, loader_kwargs)
    data_generator = torch.Generator().manual_seed(seed)
    splits = spt.data.random_split(
        dataset,
        [float(cfg.train.train_fraction), 1 - float(cfg.train.train_fraction)],
        generator=data_generator,
    )
    train_loader = DataLoader(
        splits[0],
        shuffle=True,
        drop_last=True,
        generator=data_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(splits[1], shuffle=False, drop_last=False, **loader_kwargs)

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    checkpoint_path = predictor_checkpoint_path(cfg, predictor_name, seed)
    checkpoint = None
    start_epoch = 0
    if bool(cfg.train.resume) and checkpoint_path.is_file():
        checkpoint, checkpoint_path = load_predictor_checkpoint(
            model, cfg, predictor_name, seed
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
    ema = EMAModel(model.predictor, decay=EMA_DECAY)
    if checkpoint is not None and "ema_predictor" in checkpoint:
        for name, value in checkpoint["ema_predictor"].items():
            ema.shadow_state[name].copy_(value)

    print(
        f"Push-T clips: {len(dataset):,}; train: {len(splits[0]):,}; "
        f"validation: {len(splits[1]):,}"
    )
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}")
    global_step = 0
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        model.train()
        running = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            ema.update(model.predictor)
            running += float(loss.detach())
            global_step += 1
            if batch_index % int(cfg.train.log_every) == 0:
                print(
                    f"epoch {epoch + 1} step {global_step}: "
                    f"loss={running / int(cfg.train.log_every):.6f}"
                )
                running = 0.0

        val_loss = _mean_validation_loss(model, val_loader, device)
        print(f"epoch {epoch + 1}: validation_loss={val_loss:.6f}")
        if (epoch + 1) % CHECKPOINT_INTERVAL:
            continue
        payload = {
            "model_name": predictor_name,
            "epoch": epoch + 1,
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        payload["ema_predictor"] = ema.shadow_state
        payload["predictor_config"] = _predictor_config(cfg, predictor_name)
        if not model.freeze_action_encoder:
            payload["action_encoder"] = model.action_encoder.state_dict()
        checkpoint_path = save_model_checkpoint(
            checkpoint_path.parent,
            predictor_name,
            payload,
            cfg,
            seed,
            **_checkpoint_naming(cfg, predictor_name, epoch + 1),
        )
        latest_path = predictor_checkpoint_path(cfg, predictor_name, seed)
        temporary_link = latest_path.with_suffix(latest_path.suffix + ".link")
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(checkpoint_path.name)
        temporary_link.replace(latest_path)
        print(f"saved {checkpoint_path}")

    return checkpoint_path


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="world_model",
)
def run(cfg: DictConfig) -> Path | None:
    return train_world_model(cfg, seed=SEED)


if __name__ == "__main__":
    run()
