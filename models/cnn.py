"""Standard 1D-CNN baseline for multi-horizon glucose forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CNNConfig:
    """Architecture-only settings loaded from configs/cnn.yaml."""

    num_conv_layers: int
    kernel_size: int
    activation: str
    pooling: str


def load_cnn_config(path: str | Path | None = None) -> CNNConfig:
    """Load the JSON-compatible YAML configuration without extra dependencies."""
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "configs" / "cnn.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing CNN configuration: {config_path}")
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON-compatible YAML in {config_path}: {exc}") from exc
    return CNNConfig(
        num_conv_layers=int(values["num_conv_layers"]),
        kernel_size=int(values["kernel_size"]),
        activation=str(values["activation"]).lower(),
        pooling=str(values["pooling"]).lower(),
    )


class CNNBaseline(nn.Module):
    """A classical temporal Conv1d encoder followed by a linear output layer."""

    def __init__(
        self,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        pooling: str = "adaptive_average",
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if cgm_dim <= 0 or num_physio_nodes < 0 or hidden_dim <= 0:
            raise ValueError("Input and hidden dimensions must be positive.")
        if num_conv_layers <= 0:
            raise ValueError("num_conv_layers must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if activation != "relu":
            raise ValueError("The paper baseline currently supports activation='relu' only.")
        if pooling != "adaptive_average":
            raise ValueError("The paper baseline currently supports adaptive average pooling only.")
        if sampling_interval_minutes <= 0:
            raise ValueError("sampling_interval_minutes must be positive.")

        self.horizon_minutes = tuple(int(value) for value in horizon_minutes)
        if any(
            value <= 0 or value % sampling_interval_minutes != 0
            for value in self.horizon_minutes
        ):
            raise ValueError("Every prediction horizon must be a positive sampling multiple.")
        self.horizon_steps = tuple(
            value // sampling_interval_minutes for value in self.horizon_minutes
        )
        self.required_horizon_steps = max(self.horizon_steps)
        self.target_indices = tuple(step - 1 for step in self.horizon_steps)

        layers: list[nn.Module] = []
        input_channels = cgm_dim + num_physio_nodes
        for layer_index in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    input_channels if layer_index == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(hidden_dim, len(self.horizon_minutes))

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        """Predict four horizons from aligned CGM and wearable sequences."""
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        encoded = self.encoder(sequence.transpose(1, 2))
        pooled = self.pool(encoded).squeeze(-1)
        return self.output(pooled)

    def select_targets(self, future_glucose: Tensor) -> Tensor:
        """Select labels at 15/30/45/60 min from the 5-min future sequence."""
        if future_glucose.ndim != 2:
            raise ValueError(f"future_glucose must be [B,S], got {tuple(future_glucose.shape)}")
        if future_glucose.shape[1] < self.required_horizon_steps:
            raise ValueError(
                f"future_glucose needs at least {self.required_horizon_steps} steps, "
                f"got {future_glucose.shape[1]}"
            )
        indices = torch.as_tensor(self.target_indices, device=future_glucose.device)
        return future_glucose.index_select(1, indices)
