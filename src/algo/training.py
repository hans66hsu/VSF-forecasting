import math
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from tqdm.auto import tqdm

from src.algo.common.models import EMAModel
from src.utils.utils import get_default_device


TrainingLogCallback = Callable[[Dict[str, float]], None]


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = min(warmup_steps, self.total_steps)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_idx = 0

    def step(self) -> None:
        self.step_idx += 1

        if self.warmup_steps > 0 and self.step_idx <= self.warmup_steps:
            scale = self.step_idx / self.warmup_steps
        else:
            denom = max(self.total_steps - self.warmup_steps, 1)
            progress = (self.step_idx - self.warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _call_log_callback(
    log_callback: Optional[TrainingLogCallback],
    metrics: Dict[str, float],
) -> None:
    if log_callback is not None:
        log_callback(metrics)


class BaseForecasterTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        num_epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 200,
        device: Optional[Union[str, torch.device]] = None,
        log_callback: Optional[TrainingLogCallback] = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else get_default_device()
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.log_callback = log_callback

        self.ema = EMAModel(self.model, decay=0.75)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            total_steps=max(len(dataloader) * num_epochs, 1),
            warmup_steps=warmup_steps,
        )
        self.epoch_losses = []

    def batch_loss(self, batch) -> Tensor:
        raise NotImplementedError

    def extra_artifacts(self) -> Dict[str, object]:
        return {}

    def artifacts(self) -> Dict[str, object]:
        ema_model = self.ema.clone_model(self.model).to(self.device)
        ema_model.eval()
        artifacts = {
            "model": self.model,
            "ema_model": ema_model,
            "history": self.epoch_losses,
        }
        artifacts.update(self.extra_artifacts())
        return artifacts

    def fit(self) -> Dict[str, object]:
        self.model.train()
        with tqdm(range(self.num_epochs), desc="Epoch") as tglobal:
            for epoch_idx in tglobal:
                batch_losses = []
                for batch in self.dataloader:
                    loss = self.batch_loss(batch)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.ema.update(self.model)

                    batch_losses.append(loss.item())

                mean_epoch_loss = float(np.mean(batch_losses))
                self.epoch_losses.append(mean_epoch_loss)
                tglobal.set_postfix(loss=mean_epoch_loss)
                _call_log_callback(
                    self.log_callback,
                    {
                        "epoch": float(epoch_idx + 1),
                        "train_loss": mean_epoch_loss,
                        "lr": _optimizer_lr(self.optimizer),
                    },
                )
        return self.artifacts()
