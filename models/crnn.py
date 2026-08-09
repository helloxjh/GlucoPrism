"""Convolutional recurrent neural network baseline for glucose forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .baseline_utils import MultiHorizonTargetMixin, load_json_yaml


@dataclass(frozen=True)
class CRNNConfig:
    conv_layers: int
    kernel_size: int
    recurrent_layers: int
    activation: str
    prediction_steps: int


def load_crnn_config() -> CRNNConfig:
    values = load_json_yaml("crnn.yaml")
    return CRNNConfig(
        conv_layers=int(values["conv_layers"]),
        kernel_size=int(values["kernel_size"]),
        recurrent_layers=int(values["recurrent_layers"]),
        activation=str(values["activation"]).lower(),
        prediction_steps=int(values["prediction_steps"]),
    )


class CRNNBaseline(nn.Module, MultiHorizonTargetMixin):
    """Temporal Conv1d feature extractor followed by an LSTM and linear head."""

    def __init__(
        self,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        conv_layers: int = 2,
        kernel_size: int = 3,
        recurrent_layers: int = 1,
        activation: str = "relu",
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if conv_layers <= 0 or recurrent_layers <= 0:
            raise ValueError("CRNN layer counts must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if activation != "relu":
            raise ValueError("The paper baseline currently supports activation='relu' only.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")
        self.prediction_steps = prediction_steps

        layers: list[nn.Module] = []
        input_channels = cgm_dim + num_physio_nodes
        for layer_index in range(conv_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        input_channels if layer_index == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.convolution = nn.Sequential(*layers)
        self.recurrent = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=recurrent_layers,
            dropout=dropout if recurrent_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        local_features = self.convolution(sequence.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.recurrent(local_features)
        full_forecast = self.output(hidden[-1])
        indices = torch.as_tensor(self.target_indices, device=full_forecast.device)
        return full_forecast.index_select(1, indices)

    @staticmethod
    def _validate_inputs(cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
