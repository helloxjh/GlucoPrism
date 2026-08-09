"""Optimizer and scheduler builders with warmup."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no weight decay on bias/norm parameters."""
    no_decay: List[nn.Parameter] = []
    decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in ("bias", "norm", "Norm")):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine decay after warmup to reduce late-epoch validation oscillation."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=min_lr,
    )


class WarmupScheduler:
    """Linear warmup wrapper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        self._set_lr(self.base_lr / max(self.warmup_epochs, 1))

    def step(self) -> None:
        self.current_epoch += 1
        if self.current_epoch < self.warmup_epochs:
            scale = (self.current_epoch + 1) / max(self.warmup_epochs, 1)
            self._set_lr(self.base_lr * scale)
        elif self.current_epoch == self.warmup_epochs:
            self._set_lr(self.base_lr)

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
