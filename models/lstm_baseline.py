"""Standard LSTM baseline for multi-horizon glucose forecasting."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class LSTMBaseline(nn.Module):
    """A classical LSTM -> Linear baseline with direct multi-horizon output."""

    def __init__(
        self,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if cgm_dim <= 0 or num_physio_nodes < 0:
            raise ValueError("Input dimensions must be positive.")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if sampling_interval_minutes <= 0:
            raise ValueError("sampling_interval_minutes must be positive.")

        self.horizon_minutes = tuple(int(value) for value in horizon_minutes)
        if any(value <= 0 or value % sampling_interval_minutes != 0 for value in self.horizon_minutes):
            raise ValueError("Every prediction horizon must be a positive multiple of the sampling interval.")
        self.horizon_steps = tuple(
            value // sampling_interval_minutes for value in self.horizon_minutes
        )
        self.required_horizon_steps = max(self.horizon_steps)
        self.target_indices = tuple(step - 1 for step in self.horizon_steps)

        self.lstm = nn.LSTM(
            input_size=cgm_dim + num_physio_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, len(self.horizon_minutes))

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        """Predict four horizons from aligned CGM and wearable sequences."""
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        _, (hidden, _) = self.lstm(sequence)
        return self.output(hidden[-1])

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
