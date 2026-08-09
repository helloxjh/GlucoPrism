"""Channel-independent PatchTST baseline for glucose forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .baseline_utils import MultiHorizonTargetMixin, load_json_yaml


@dataclass(frozen=True)
class PatchTSTConfig:
    patch_length: int
    stride: int
    encoder_layers: int
    feedforward_multiplier: int
    revin: bool
    prediction_steps: int


def load_patchtst_config() -> PatchTSTConfig:
    values = load_json_yaml("patchtst.yaml")
    return PatchTSTConfig(
        patch_length=int(values["patch_length"]),
        stride=int(values["stride"]),
        encoder_layers=int(values["encoder_layers"]),
        feedforward_multiplier=int(values["feedforward_multiplier"]),
        revin=bool(values["revin"]),
        prediction_steps=int(values["prediction_steps"]),
    )


class PatchTSTBaseline(nn.Module, MultiHorizonTargetMixin):
    """PatchTST with channel-independent encoding and a shared forecast head."""

    def __init__(
        self,
        history_steps: int = 24,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        patch_length: int = 6,
        stride: int = 3,
        encoder_layers: int = 2,
        feedforward_multiplier: int = 4,
        revin: bool = True,
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if patch_length <= 0 or patch_length > history_steps:
            raise ValueError("patch_length must be within the history window.")
        if stride <= 0 or encoder_layers <= 0 or prediction_steps <= 0:
            raise ValueError("stride, encoder_layers, and prediction_steps must be positive.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")

        self.history_steps = history_steps
        self.input_channels = cgm_dim + num_physio_nodes
        self.patch_length = patch_length
        self.stride = stride
        self.revin = revin
        self.prediction_steps = prediction_steps
        self.num_patches = 1 + (history_steps - patch_length) // stride

        self.patch_projection = nn.Linear(patch_length, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(self.num_patches * hidden_dim, prediction_steps)

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        channels_first = sequence.transpose(1, 2)
        if self.revin:
            mean = channels_first.mean(dim=-1, keepdim=True).detach()
            std = torch.sqrt(
                channels_first.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            channels_first = (channels_first - mean) / std
        else:
            mean = channels_first.new_zeros(channels_first.shape[0], self.input_channels, 1)
            std = channels_first.new_ones(channels_first.shape[0], self.input_channels, 1)

        patches = channels_first.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.stride,
        )
        batch_size, channels, num_patches, _ = patches.shape
        if num_patches != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {num_patches}; "
                "the runtime history length differs from model configuration."
            )
        tokens = patches.reshape(batch_size * channels, num_patches, self.patch_length)
        tokens = self.patch_projection(tokens) + self.position_embedding
        encoded = self.encoder(tokens)
        normalized_forecast = self.head(encoded.flatten(start_dim=1))
        normalized_forecast = normalized_forecast.reshape(
            batch_size, channels, self.prediction_steps
        )
        forecast = normalized_forecast * std + mean
        cgm_forecast = forecast[:, 0]
        indices = torch.as_tensor(self.target_indices, device=cgm_forecast.device)
        return cgm_forecast.index_select(1, indices)

    def _validate_inputs(self, cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
        if cgm.shape[1] != self.history_steps:
            raise ValueError(
                f"Expected history length {self.history_steps}, got {cgm.shape[1]}."
            )
