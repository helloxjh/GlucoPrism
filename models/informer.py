"""Informer baseline with ProbSparse attention and encoder distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .baseline_utils import (
    MultiHorizonTargetMixin,
    SinusoidalPositionEncoding,
    load_json_yaml,
)


@dataclass(frozen=True)
class InformerConfig:
    encoder_layers: int
    decoder_layers: int
    feedforward_multiplier: int
    attention_factor: int
    distil: bool
    prediction_steps: int


def load_informer_config() -> InformerConfig:
    values = load_json_yaml("informer.yaml")
    return InformerConfig(
        encoder_layers=int(values["encoder_layers"]),
        decoder_layers=int(values["decoder_layers"]),
        feedforward_multiplier=int(values["feedforward_multiplier"]),
        attention_factor=int(values["attention_factor"]),
        distil=bool(values["distil"]),
        prediction_steps=int(values["prediction_steps"]),
    )


class ProbSparseSelfAttention(nn.Module):
    """ProbSparse query selection with full keys for the selected queries."""

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
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, steps, hidden_dim = x.shape
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        sparsity = scores.max(dim=-1).values - scores.mean(dim=-1)
        selected_queries = min(
            steps,
            max(1, self.factor * int(math.ceil(math.log(steps + 1)))),
        )
        top_indices = sparsity.topk(selected_queries, dim=-1, sorted=False).indices
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, steps)
        selected_scores = scores.gather(dim=2, index=gather_index)
        attention = self.attention_dropout(torch.softmax(selected_scores, dim=-1))
        updates = torch.matmul(attention, value)

        context = value.mean(dim=2, keepdim=True).expand(-1, -1, steps, -1).clone()
        scatter_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        context.scatter_(dim=2, index=scatter_index, src=updates)
        context = context.transpose(1, 2).reshape(batch_size, steps, hidden_dim)
        return self.out_proj(context)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, steps, _ = x.shape
        return x.reshape(batch_size, steps, self.num_heads, self.head_dim).transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        attention_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = ProbSparseSelfAttention(
            hidden_dim, num_heads, attention_factor, dropout
        )
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(feedforward))


class InformerDistillingLayer(nn.Module):
    """Temporal convolution and max pooling used between Informer encoders."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pool(self.activation(self.norm(self.conv(x))))
        return x.transpose(1, 2)


class InformerBaseline(nn.Module, MultiHorizonTargetMixin):
    """Encoder-decoder Informer adapted to direct CGM multi-horizon output."""

    def __init__(
        self,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        feedforward_multiplier: int = 4,
        attention_factor: int = 3,
        distil: bool = True,
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if encoder_layers <= 0 or decoder_layers <= 0 or prediction_steps <= 0:
            raise ValueError("Informer layer counts and prediction_steps must be positive.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")
        self.prediction_steps = prediction_steps

        input_dim = cgm_dim + num_physio_nodes
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = SinusoidalPositionEncoding(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        feedforward_dim = hidden_dim * feedforward_multiplier
        self.encoder_layers = nn.ModuleList(
            [
                InformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    feedforward_dim,
                    attention_factor,
                    dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.distilling_layers = nn.ModuleList(
            [InformerDistillingLayer(hidden_dim) for _ in range(encoder_layers - 1)]
            if distil
            else []
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.future_queries = nn.Parameter(torch.empty(1, prediction_steps, hidden_dim))
        nn.init.normal_(self.future_queries, std=0.02)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        memory = self.input_dropout(self.position(self.input_projection(sequence)))
        distil_index = 0
        for layer_index, encoder_layer in enumerate(self.encoder_layers):
            memory = encoder_layer(memory)
            if layer_index < len(self.encoder_layers) - 1 and self.distilling_layers:
                memory = self.distilling_layers[distil_index](memory)
                distil_index += 1

        queries = self.future_queries.expand(cgm.shape[0], -1, -1)
        queries = self.position(queries)
        causal_mask = torch.triu(
            torch.ones(
                self.prediction_steps,
                self.prediction_steps,
                dtype=torch.bool,
                device=cgm.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(queries, memory, tgt_mask=causal_mask)
        full_forecast = self.output(decoded).squeeze(-1)
        indices = torch.as_tensor(self.target_indices, device=full_forecast.device)
        return full_forecast.index_select(1, indices)

    @staticmethod
    def _validate_inputs(cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
