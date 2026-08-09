"""Diffusion Convolutional Recurrent Neural Network baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .baseline_utils import MultiHorizonTargetMixin, load_json_yaml


@dataclass(frozen=True)
class DCRNNConfig:
    encoder_layers: int
    decoder_layers: int
    diffusion_order: int
    adjacency: str
    prediction_steps: int


def load_dcrnn_config() -> DCRNNConfig:
    values = load_json_yaml("dcrnn.yaml")
    return DCRNNConfig(
        encoder_layers=int(values["encoder_layers"]),
        decoder_layers=int(values["decoder_layers"]),
        diffusion_order=int(values["diffusion_order"]),
        adjacency=str(values["adjacency"]),
        prediction_steps=int(values["prediction_steps"]),
    )


class DiffusionGraphLinear(nn.Module):
    """Linear projection over forward/reverse random-walk diffusion powers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        diffusion_order: int,
    ) -> None:
        super().__init__()
        self.diffusion_order = diffusion_order
        self.projection = nn.Linear(
            input_dim * (1 + 2 * diffusion_order),
            output_dim,
        )

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        outputs = [x]
        for support in (adjacency, adjacency.transpose(0, 1)):
            diffused = x
            for _ in range(self.diffusion_order):
                diffused = torch.einsum("bnf,nm->bmf", diffused, support)
                outputs.append(diffused)
        return self.projection(torch.cat(outputs, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose affine transformations are diffusion graph convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, diffusion_order: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        combined_dim = input_dim + hidden_dim
        self.gates = DiffusionGraphLinear(
            combined_dim,
            2 * hidden_dim,
            diffusion_order,
        )
        self.candidate = DiffusionGraphLinear(
            combined_dim,
            hidden_dim,
            diffusion_order,
        )

    def forward(self, x: Tensor, state: Tensor, adjacency: Tensor) -> Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat([x, state], dim=-1), adjacency)
        ).chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * state], dim=-1), adjacency)
        )
        return update * state + (1.0 - update) * candidate


class DCRNNBaseline(nn.Module, MultiHorizonTargetMixin):
    """DCRNN encoder-decoder with free-running multi-step graph decoding."""

    def __init__(
        self,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        diffusion_order: int = 2,
        adjacency: str = "fully_connected_normalized",
        prediction_steps: int = 12,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
    ) -> None:
        super().__init__()
        if encoder_layers <= 0 or decoder_layers <= 0 or diffusion_order <= 0:
            raise ValueError("DCRNN layer counts and diffusion_order must be positive.")
        if adjacency != "fully_connected_normalized":
            raise ValueError("Only the documented fully connected baseline graph is supported.")
        self.configure_horizons(horizon_minutes, sampling_interval_minutes)
        if prediction_steps < self.required_horizon_steps:
            raise ValueError("prediction_steps does not cover all requested horizons.")
        self.num_nodes = cgm_dim + num_physio_nodes
        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps
        adjacency_tensor = torch.full(
            (self.num_nodes, self.num_nodes),
            1.0 / self.num_nodes,
            dtype=torch.float32,
        )
        self.register_buffer("adjacency", adjacency_tensor)

        self.encoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    input_dim=1 if layer_index == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    diffusion_order=diffusion_order,
                )
                for layer_index in range(encoder_layers)
            ]
        )
        self.decoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    input_dim=1 if layer_index == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    diffusion_order=diffusion_order,
                )
                for layer_index in range(decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cgm: Tensor, physio: Tensor) -> Tensor:
        self._validate_inputs(cgm, physio)
        sequence = torch.cat([cgm, physio.transpose(1, 2)], dim=-1).transpose(1, 2)
        sequence = sequence.unsqueeze(-1)
        batch_size = sequence.shape[0]
        encoder_states = [
            sequence.new_zeros(batch_size, self.num_nodes, self.hidden_dim)
            for _ in self.encoder_cells
        ]
        for time_index in range(sequence.shape[2]):
            layer_input = sequence[:, :, time_index]
            for layer_index, cell in enumerate(self.encoder_cells):
                encoder_states[layer_index] = cell(
                    layer_input,
                    encoder_states[layer_index],
                    self.adjacency,
                )
                layer_input = encoder_states[layer_index]
                if layer_index < len(self.encoder_cells) - 1:
                    layer_input = self.dropout(layer_input)

        decoder_states = [
            encoder_states[min(index, len(encoder_states) - 1)].clone()
            for index in range(len(self.decoder_cells))
        ]
        decoder_input = sequence[:, :, -1]
        forecasts = []
        for _ in range(self.prediction_steps):
            layer_input = decoder_input
            for layer_index, cell in enumerate(self.decoder_cells):
                decoder_states[layer_index] = cell(
                    layer_input,
                    decoder_states[layer_index],
                    self.adjacency,
                )
                layer_input = decoder_states[layer_index]
                if layer_index < len(self.decoder_cells) - 1:
                    layer_input = self.dropout(layer_input)
            decoder_input = self.output_projection(layer_input)
            forecasts.append(decoder_input.squeeze(-1))
        full_forecast = torch.stack(forecasts, dim=1)[:, :, 0]
        indices = torch.as_tensor(self.target_indices, device=full_forecast.device)
        return full_forecast.index_select(1, indices)

    def _validate_inputs(self, cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or physio.ndim != 3:
            raise ValueError("cgm and physio must be [B,T,C] and [B,N,T].")
        if cgm.shape[0] != physio.shape[0] or cgm.shape[1] != physio.shape[2]:
            raise ValueError("cgm and physio must share batch and time dimensions.")
        if cgm.shape[2] + physio.shape[1] != self.num_nodes:
            raise ValueError("Runtime node count differs from DCRNN configuration.")
