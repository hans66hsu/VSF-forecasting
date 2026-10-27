"""Frozen LeWM visual representation with a replaceable dynamics predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn


def load_lewm_checkpoint(checkpoint_dir: str | Path) -> nn.Module:
    """Hydra-instantiate and strictly load an official LeWM checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "weights.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected config.json and weights.pt in {checkpoint_dir}"
        )

    from hydra.utils import instantiate

    with config_path.open() as stream:
        config = json.load(stream)
    model = instantiate(config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


class FrozenJEPA(nn.Module):
    """LeWM-compatible planning wrapper with frozen representation modules."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        freeze_pred_proj: bool = False,
        freeze_action_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.freeze_pred_proj = bool(freeze_pred_proj)
        self.freeze_action_encoder = bool(freeze_action_encoder)
        self._freeze_representation()

    def _freeze_representation(self) -> None:
        modules = [self.encoder, self.projector]
        if self.freeze_action_encoder:
            modules.append(self.action_encoder)
        if self.freeze_pred_proj:
            modules.append(self.pred_proj)
        for module in modules:
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> "FrozenJEPA":
        super().train(mode)
        # In particular, keep the projector's BatchNorm running statistics fixed.
        self._freeze_representation()
        return self

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        """Encode ``pixels`` with the frozen ViT CLS tower and, if present, actions."""
        pixels = info["pixels"]
        parameter = next(self.encoder.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        pixels = rearrange(pixels.to(dtype=dtype), "b t ... -> (b t) ...")
        batch_size = info["pixels"].shape[0]

        with torch.no_grad():
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            embedding = self.projector(output.last_hidden_state[:, 0])
            info["emb"] = rearrange(embedding, "(b t) d -> b t d", b=batch_size)

        if "action" in info:
            # Frozen parameters do not prevent gradients with respect to actions.
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(
        self,
        emb: Tensor,
        act_emb: Tensor,
        latent: Tensor | None = None,
    ) -> Tensor:
        prediction = (
            self.predictor(emb, act_emb)
            if latent is None
            else self.predictor(emb, act_emb, latent=latent)
        )
        prediction = self.pred_proj(rearrange(prediction, "b t d -> (b t) d"))
        return rearrange(prediction, "(b t) d -> b t d", b=emb.shape[0])

    def rollout(
        self,
        info: dict[str, Any],
        action_sequence: Tensor,
        history_size: int | None = None,
    ) -> dict[str, Any]:
        """Autoregress over each candidate action sequence."""
        if "pixels" not in info:
            raise KeyError("pixels not in info")
        if action_sequence.ndim != 4:
            raise ValueError("action_sequence must have shape (B, S, T, A)")

        batch_size, num_samples, horizon = action_sequence.shape[:3]
        initial_steps = info["pixels"].shape[2]
        if horizon < initial_steps:
            raise ValueError("action horizon cannot be shorter than pixel history")

        initial_actions = action_sequence[:, :, :initial_steps]
        info["action"] = initial_actions

        cached = info.get("emb")
        if cached is None or cached.shape[:2] != (batch_size, num_samples):
            initial = {
                key: value[:, 0]
                for key, value in info.items()
                if torch.is_tensor(value) and key != "action"
            }
            initial = self.encode(initial)
            info["emb"] = (
                initial["emb"]
                .detach()
                .unsqueeze(1)
                .expand(batch_size, num_samples, -1, -1)
            )

        # emb_init: (S, 1, D), all_action_emb: (S, T, D)
        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        all_action_emb = self.action_encoder(
            rearrange(action_sequence, "b s ... -> (b s) ...")
        )
        sample_prior = getattr(self.predictor, "sample_prior", None)
        latent = (
            sample_prior(emb_init.shape[0], emb_init.device, emb_init.dtype)
            if sample_prior is not None
            and not getattr(self.predictor, "resample_latent_each_step", False)
            else None
        )

        max_history = (
            getattr(self.predictor, "num_frames", 3)
            if history_size is None
            else history_size
        )
        embeddings = list(emb_init.unbind(dim=1))
        num_future_steps = horizon - initial_steps
        for step in range(num_future_steps + 1):
            lower = max(0, initial_steps + step - max_history)
            emb_context = torch.stack(embeddings[lower:], dim=1)
            act_context = all_action_emb[:, lower : initial_steps + step]
            # ARpred: self.predict(emb_context, act_context) [S, T, D]
            # VarSF pred: self.predict(emb_context, act_context) [S, 1, D]
            next_embedding = self.predict(
                emb_context, act_context, latent=latent
            )[:, -1]
            embeddings.append(next_embedding)

        rollout = torch.stack(embeddings, dim=1)
        info["predicted_emb"] = rearrange(
            rollout,
            "(b s) ... -> b s ...",
            b=batch_size,
            s=num_samples,
        )
        return info

    def criterion(self, info: dict[str, Any]) -> Tensor:
        """Compute the cost between predicted embeddings and goal embeddings."""
        prediction = info["predicted_emb"]  # (B,S, T+1, dim)
        goal = info["goal_emb"]  # (B, S, 1, dim)
        if goal.ndim == prediction.ndim - 1:
            goal = goal.unsqueeze(1)
        if goal.ndim != prediction.ndim:
            raise ValueError("goal_emb must have shape (B, T, D) or (B, S, T, D)")
        if goal.shape[0] != prediction.shape[0] or goal.shape[1] not in (
            1,
            prediction.shape[1],
        ):
            raise ValueError("goal_emb cannot be broadcast over action candidates")

        goal_final = goal[..., -1, :].expand(
            prediction.shape[0], prediction.shape[1], -1
        )
        return (prediction[..., -1, :] - goal_final.detach()).square().sum(dim=-1)

    def get_cost(self, info: dict[str, Any], action_candidates: Tensor) -> Tensor:
        if "goal" not in info:
            raise KeyError("goal not in info")
        batch_size, num_samples = action_candidates.shape[:2]

        cached_goal = info.get("goal_emb")
        if (
            cached_goal is None
            or cached_goal.ndim != 4
            or cached_goal.shape[:2] != (batch_size, num_samples)
        ):
            goal_info = {}
            for key, value in info.items():
                if not torch.is_tensor(value):
                    continue
                if value.ndim >= 2 and value.shape[:2] == (
                    batch_size,
                    num_samples,
                ):
                    value = value[:, 0]
                goal_info[key] = value

            goal_info["pixels"] = goal_info["goal"]
            for key in list(goal_info):
                if key.startswith("goal_"):
                    goal_info[key.removeprefix("goal_")] = goal_info.pop(key)
            goal_info.pop("action", None)
            goal_embedding = self.encode(goal_info)["emb"]
            info["goal_emb"] = goal_embedding.unsqueeze(1).expand(
                batch_size, num_samples, -1, -1
            )
        # action_candidates: (B, S, T, A)
        # T: planning.horizon, A: action dimension (action block * action dimension)
        return self.criterion(self.rollout(info, action_candidates))

    @classmethod
    def from_ar_checkpoint(
        cls,
        checkpoint_dir: str | Path,
    ) -> "FrozenJEPA":
        """Wrap a pretrained AR predictor in the common FrozenJEPA pipeline."""
        pretrained = load_lewm_checkpoint(checkpoint_dir)

        modules = {}
        # Unlike a replacement flow predictor, the pretrained AR predictor must
        # retain its learned output projection to reproduce LeWM predictions.
        for name in (
            "encoder",
            "predictor",
            "action_encoder",
            "projector",
            "pred_proj",
        ):
            module = getattr(pretrained, name, None)
            modules[name] = module
        return cls(**modules)
