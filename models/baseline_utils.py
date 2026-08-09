"""Shared utilities for paper baseline forecasting models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


def load_json_yaml(filename: str) -> Mapping[str, Any]:
    """Load a JSON-compatible YAML 1.2 configuration from configs/."""
    path = Path(__file__).resolve().parents[1] / "configs" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model configuration: {path}")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON-compatible YAML in {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"Model configuration must be an object: {path}")
    return values


class MultiHorizonTargetMixin:
    """Provide the target-selection contract required by the shared Trainer."""

    horizon_minutes: tuple[int, ...]
    horizon_steps: tuple[int, ...]
    required_horizon_steps: int
    target_indices: tuple[int, ...]

    def configure_horizons(
        self,
        horizon_minutes: Sequence[int],
        sampling_interval_minutes: int,
    ) -> None:
        if sampling_interval_minutes <= 0:
            raise ValueError("sampling_interval_minutes must be positive.")
        self.horizon_minutes = tuple(int(value) for value in horizon_minutes)
        if any(
            value <= 0 or value % sampling_interval_minutes != 0
            for value in self.horizon_minutes
        ):
            raise ValueError("Every horizon must be a positive sampling multiple.")
        self.horizon_steps = tuple(
            value // sampling_interval_minutes for value in self.horizon_minutes
        )
        self.required_horizon_steps = max(self.horizon_steps)
        self.target_indices = tuple(step - 1 for step in self.horizon_steps)

    def select_targets(self, future_glucose: Tensor) -> Tensor:
        if future_glucose.ndim != 2:
            raise ValueError(f"future_glucose must be [B,S], got {tuple(future_glucose.shape)}")
        if future_glucose.shape[1] < self.required_horizon_steps:
            raise ValueError(
                f"future_glucose needs at least {self.required_horizon_steps} steps, "
                f"got {future_glucose.shape[1]}"
            )
        indices = torch.as_tensor(self.target_indices, device=future_glucose.device)
        return future_glucose.index_select(1, indices)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal positions for batch-first temporal representations."""

    def __init__(self, hidden_dim: int, max_steps: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_steps, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds positional capacity.")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)
