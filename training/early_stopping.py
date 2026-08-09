"""Early stopping utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    best_loss: float = float("inf")
    counter: int = 0
    best_state: Optional[Dict[str, Tensor]] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update state and return True when training should stop."""
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module, device: torch.device) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            model.to(device)
