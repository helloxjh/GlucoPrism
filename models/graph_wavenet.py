"""Graph WaveNet baseline for multivariate glucose forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .baseline_utils import MultiHorizonTargetMixin, load_json_yaml


@dataclass(frozen=True)
class GraphWaveNetConfig:
    blocks: int
    layers_per_block: int
    temporal_kernel_size: int
    diffusion_order: int
    node_embedding_dim: int
    skip_multiplier: int
    end_multiplier: int
    prediction_steps: int


def load_graphwavenet_config() -> GraphWaveNetConfig:
    values = load_json_yaml("graphwavenet.yaml")
    return GraphWaveNetConfig(
        blocks=int(values["blocks"]),
        layers_per_block=int(values["layers_per_block"]),
        temporal_kernel_size=int(values["temporal_kernel_size"]),
        diffusion_order=int(values["diffusion_order"]),
        node_embedding_dim=int(values["node_embedding_dim"]),
        skip_multiplier=int(values["skip_multiplier"]),
        end_multiplier=int(values["end_multiplier"]),
        prediction_steps=int(values["prediction_steps"]),
    )


class DiffusionGraphConv(nn.Module):
    """Graph WaveNet diffusion convolution over forward and reverse supports."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        diffusion_order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.diffusion_order = diffusion_order
        support_terms = 1 + 2 * diffusion_order
        self.projection = nn.Conv2d(
            input_channels * support_terms,
            output_channels,
            kernel_size=(1, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        outputs = [x]
        for support in (adjacency, adjacency.transpose(0, 1)):
            first = self._propagate(x, support)
            outputs.append(first)
            previous = first
            for _ in range(2, self.diffusion_order + 1):
                previous = self._propagate(previous, support)
                outputs.append(previous)
        return self.dropout(self.projection(torch.cat(outputs, dim=1)))

    @staticmethod
    def _propagate(x: Tensor, adjacency: Tensor) -> Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adjacency)


class GraphWaveNetBaseline(nn.Module, MultiHorizonTargetMixin):
    """Adaptive-graph WaveNet with direct future CGM output."""

    def __init__(
        self,
        history_steps: int = 24,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        blocks: int = 2,
        layers_per_block: int = 2,
        temporal_kernel_size: int = 2,
        diffusion_order: int = 2,
        node_embedding_dim: int = 10,
        skip_multiplier: int = 2,
        end_multiplier: int = 4,
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if blocks <= 0 or layers_per_block <= 0 or temporal_kernel_size <= 1:
            raise ValueError("Graph WaveNet temporal configuration is invalid.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")
        self.history_steps = history_steps
        self.num_nodes = cgm_dim + num_physio_nodes
        self.prediction_steps = prediction_steps
        skip_channels = hidden_dim * skip_multiplier
        end_channels = hidden_dim * end_multiplier

        receptive_field = 1
        for _ in range(blocks):
            for layer_index in range(layers_per_block):
                receptive_field += (temporal_kernel_size - 1) * (2**layer_index)
        if history_steps < receptive_field:
            raise ValueError(
                f"History length {history_steps} is shorter than receptive field {receptive_field}."
            )

        self.node_embedding_source = nn.Parameter(
            torch.randn(self.num_nodes, node_embedding_dim) * 0.1
        )
        self.node_embedding_target = nn.Parameter(
            torch.randn(node_embedding_dim, self.num_nodes) * 0.1
        )
        self.start_conv = nn.Conv2d(1, hidden_dim, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(blocks):
            for layer_index in range(layers_per_block):
                dilation = 2**layer_index
                self.filter_convs.append(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=(1, temporal_kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=(1, temporal_kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(hidden_dim, skip_channels, kernel_size=(1, 1))
                )
                self.graph_convs.append(
                    DiffusionGraphConv(
                        hidden_dim,
                        hidden_dim,
                        diffusion_order,
                        dropout,
                    )
                )
                self.norms.append(nn.BatchNorm2d(hidden_dim))
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, prediction_steps, kernel_size=(1, 1))

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1)
        x = self.start_conv(sequence.transpose(1, 2).unsqueeze(1))
        adjacency = torch.softmax(
            torch.relu(self.node_embedding_source @ self.node_embedding_target),
            dim=-1,
        )
        skip: Tensor | None = None
        for layer_index in range(len(self.filter_convs)):
            residual = x
            filtered = torch.tanh(self.filter_convs[layer_index](x))
            gated = torch.sigmoid(self.gate_convs[layer_index](x))
            x = filtered * gated
            layer_skip = self.skip_convs[layer_index](x)
            if skip is None:
                skip = layer_skip
            else:
                skip = skip[..., -layer_skip.shape[-1] :] + layer_skip
            x = self.graph_convs[layer_index](x, adjacency)
            x = x + self.residual_convs[layer_index](
                residual[..., -x.shape[-1] :]
            )
            x = self.norms[layer_index](x)
        if skip is None:
            raise RuntimeError("Graph WaveNet produced no skip representation.")
        output = self.end_conv2(torch.relu(self.end_conv1(torch.relu(skip))))
        full_forecast = output[:, :, 0, -1]
        indices = torch.as_tensor(self.target_indices, device=full_forecast.device)
        return full_forecast.index_select(1, indices)

    def _validate_inputs(self, cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
        if cgm.shape[1] != self.history_steps:
            raise ValueError(
                f"Expected history length {self.history_steps}, got {cgm.shape[1]}."
            )
