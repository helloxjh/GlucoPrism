"""Autoformer baseline with series decomposition and FFT autocorrelation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baseline_utils import MultiHorizonTargetMixin, load_json_yaml


@dataclass(frozen=True)
class AutoformerConfig:
    encoder_layers: int
    decoder_layers: int
    feedforward_multiplier: int
    moving_average_kernel: int
    autocorrelation_factor: int
    label_steps: int
    prediction_steps: int


def load_autoformer_config() -> AutoformerConfig:
    values = load_json_yaml("autoformer.yaml")
    return AutoformerConfig(
        encoder_layers=int(values["encoder_layers"]),
        decoder_layers=int(values["decoder_layers"]),
        feedforward_multiplier=int(values["feedforward_multiplier"]),
        moving_average_kernel=int(values["moving_average_kernel"]),
        autocorrelation_factor=int(values["autocorrelation_factor"]),
        label_steps=int(values["label_steps"]),
        prediction_steps=int(values["prediction_steps"]),
    )


class SeriesDecomposition(nn.Module):
    """Separate a sequence into seasonal residual and moving-average trend."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel must be a positive odd integer.")
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        radius = self.kernel_size // 2
        channels_first = x.transpose(1, 2)
        padded = F.pad(channels_first, (radius, radius), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        trend = trend.transpose(1, 2)
        return x - trend, trend


class AutoCorrelationLayer(nn.Module):
    """Frequency-domain period discovery followed by time-delay aggregation."""

    def __init__(self, hidden_dim: int, num_heads: int, factor: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.factor = max(1, int(factor))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_size, query_steps, hidden_dim = query.shape
        key = self._match_length(key, query_steps)
        value = self._match_length(value, query_steps)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        correlation = torch.fft.irfft(q_fft * torch.conj(k_fft), n=query_steps, dim=2)
        delay_scores = correlation.mean(dim=-1)
        top_k = min(
            query_steps,
            max(1, self.factor * int(math.ceil(math.log(query_steps + 1)))),
        )
        top_values, top_delays = delay_scores.topk(top_k, dim=-1)
        delay_weights = self.dropout(torch.softmax(top_values, dim=-1))

        base = torch.arange(query_steps, device=query.device).view(1, 1, query_steps)
        aggregated = torch.zeros_like(v)
        for delay_index in range(top_k):
            delays = top_delays[:, :, delay_index].unsqueeze(-1)
            gather_steps = (base + delays) % query_steps
            gather_index = gather_steps.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            )
            shifted = v.gather(dim=2, index=gather_index)
            weight = delay_weights[:, :, delay_index].unsqueeze(-1).unsqueeze(-1)
            aggregated = aggregated + weight * shifted

        output = aggregated.transpose(1, 2).reshape(batch_size, query_steps, hidden_dim)
        return self.out_proj(output)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, steps, _ = x.shape
        return x.reshape(batch_size, steps, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _match_length(x: Tensor, target_steps: int) -> Tensor:
        if x.shape[1] == target_steps:
            return x
        if x.shape[1] > target_steps:
            return x[:, :target_steps]
        padding = x.new_zeros(x.shape[0], target_steps - x.shape[1], x.shape[2])
        return torch.cat([x, padding], dim=1)


class AutoformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        moving_average_kernel: int,
        autocorrelation_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.autocorrelation = AutoCorrelationLayer(
            hidden_dim, num_heads, autocorrelation_factor, dropout
        )
        self.decomposition1 = SeriesDecomposition(moving_average_kernel)
        self.decomposition2 = SeriesDecomposition(moving_average_kernel)
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.decomposition1(
            x + self.dropout(self.autocorrelation(x, x, x))
        )
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x, _ = self.decomposition2(x + self.dropout(feedforward))
        return x


class AutoformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        moving_average_kernel: int,
        autocorrelation_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_correlation = AutoCorrelationLayer(
            hidden_dim, num_heads, autocorrelation_factor, dropout
        )
        self.cross_correlation = AutoCorrelationLayer(
            hidden_dim, num_heads, autocorrelation_factor, dropout
        )
        self.decomposition1 = SeriesDecomposition(moving_average_kernel)
        self.decomposition2 = SeriesDecomposition(moving_average_kernel)
        self.decomposition3 = SeriesDecomposition(moving_average_kernel)
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.trend_projection = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, trend: Tensor) -> tuple[Tensor, Tensor]:
        x, trend1 = self.decomposition1(
            x + self.dropout(self.self_correlation(x, x, x))
        )
        x, trend2 = self.decomposition2(
            x + self.dropout(self.cross_correlation(x, memory, memory))
        )
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x, trend3 = self.decomposition3(x + self.dropout(feedforward))
        trend = trend + self.trend_projection(trend1 + trend2 + trend3)
        return x, trend


class AutoformerBaseline(nn.Module, MultiHorizonTargetMixin):
    """Encoder-decoder Autoformer for direct 15/30/45/60-min CGM output."""

    def __init__(
        self,
        history_steps: int = 24,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        feedforward_multiplier: int = 4,
        moving_average_kernel: int = 5,
        autocorrelation_factor: int = 3,
        label_steps: int = 12,
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if label_steps <= 0 or label_steps > history_steps:
            raise ValueError("label_steps must be within the history window.")
        if encoder_layers <= 0 or decoder_layers <= 0 or prediction_steps <= 0:
            raise ValueError("Autoformer layer counts and prediction_steps must be positive.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")
        self.label_steps = label_steps
        self.prediction_steps = prediction_steps
        input_dim = cgm_dim + num_physio_nodes
        feedforward_dim = hidden_dim * feedforward_multiplier

        self.raw_decomposition = SeriesDecomposition(moving_average_kernel)
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                AutoformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    feedforward_dim,
                    moving_average_kernel,
                    autocorrelation_factor,
                    dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                AutoformerDecoderLayer(
                    hidden_dim,
                    num_heads,
                    feedforward_dim,
                    moving_average_kernel,
                    autocorrelation_factor,
                    dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.seasonal_projection = nn.Linear(hidden_dim, 1)

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        seasonal, trend = self.raw_decomposition(sequence)

        memory = self.encoder_embedding(sequence)
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
        memory = self.encoder_norm(memory)

        future_zeros = sequence.new_zeros(
            sequence.shape[0], self.prediction_steps, sequence.shape[2]
        )
        seasonal_init = torch.cat(
            [seasonal[:, -self.label_steps :], future_zeros], dim=1
        )
        cgm_mean = cgm.mean(dim=1, keepdim=True)
        trend_init = torch.cat(
            [trend[:, -self.label_steps :, :1], cgm_mean.expand(-1, self.prediction_steps, -1)],
            dim=1,
        )
        decoded = self.decoder_embedding(seasonal_init)
        for decoder_layer in self.decoder_layers:
            decoded, trend_init = decoder_layer(decoded, memory, trend_init)
        decoded = self.decoder_norm(decoded)
        full_forecast = (
            self.seasonal_projection(decoded[:, -self.prediction_steps :])
            + trend_init[:, -self.prediction_steps :]
        ).squeeze(-1)
        indices = torch.as_tensor(self.target_indices, device=full_forecast.device)
        return full_forecast.index_select(1, indices)

    @staticmethod
    def _validate_inputs(cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
