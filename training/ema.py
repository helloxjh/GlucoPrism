"""Exponential moving average for stable LOSO evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import Tensor, nn


class ModelEMA:
    """Maintain an exponential moving average of floating-point model states.

    EMA is used only for validation/testing weight averaging. It does not add
    new model layers or alter the forward architecture.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {}
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name] = value.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        backup: Dict[str, Tensor] = {}
        with torch.no_grad():
            state = model.state_dict()
            for name, shadow_value in self.shadow.items():
                if name in state:
                    backup[name] = state[name].detach().clone()
                    state[name].copy_(shadow_value.to(device=state[name].device, dtype=state[name].dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                state = model.state_dict()
                for name, value in backup.items():
                    state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
