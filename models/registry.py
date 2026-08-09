"""Model registry used by the shared benchmark training pipeline."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from .autoformer import AutoformerBaseline, load_autoformer_config
from .baseline_utils import load_json_yaml
from .cnn import CNNBaseline, load_cnn_config
from .crnn import CRNNBaseline, load_crnn_config
from .dcrnn import DCRNNBaseline, load_dcrnn_config
from .graph_wavenet import GraphWaveNetBaseline, load_graphwavenet_config
from .informer import InformerBaseline, load_informer_config
from .lstm_baseline import LSTMBaseline
from .patchtst import PatchTSTBaseline, load_patchtst_config
from .st_msffnet import ST_MSFFNet


AVAILABLE_MODELS = (
    "glucoprism",
    "lstm",
    "cnn",
    "informer",
    "autoformer",
    "patchtst",
    "graphwavenet",
    "dcrnn",
    "crnn",
)


def build_forecasting_model(
    model_name: str,
    *,
    history_steps: int,
    num_physio_nodes: int,
    hidden_dim: int,
    num_heads: int,
    graph_layers: int,
    dropout: float,
    ablation_mode: str = "full",
    enable_horizon_refinement: bool = False,
    lstm_num_layers: int = 1,
    horizon_minutes: Sequence[int] = (15, 30, 45, 60),
) -> nn.Module:
    """Construct a registered model without changing the shared training loop."""
    normalized = model_name.lower()
    if normalized == "glucoprism":
        return ST_MSFFNet(
            history_steps=history_steps,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            graph_layers=graph_layers,
            dropout=dropout,
            ablation_mode=ablation_mode,
            enable_horizon_refinement=enable_horizon_refinement,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "lstm":
        return LSTMBaseline(
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_layers=lstm_num_layers,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "cnn":
        config = load_cnn_config()
        return CNNBaseline(
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_conv_layers=config.num_conv_layers,
            kernel_size=config.kernel_size,
            dropout=dropout,
            activation=config.activation,
            pooling=config.pooling,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "informer":
        config = load_informer_config()
        return InformerBaseline(
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            feedforward_multiplier=config.feedforward_multiplier,
            attention_factor=config.attention_factor,
            distil=config.distil,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "autoformer":
        config = load_autoformer_config()
        return AutoformerBaseline(
            history_steps=history_steps,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            feedforward_multiplier=config.feedforward_multiplier,
            moving_average_kernel=config.moving_average_kernel,
            autocorrelation_factor=config.autocorrelation_factor,
            label_steps=config.label_steps,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "patchtst":
        config = load_patchtst_config()
        return PatchTSTBaseline(
            history_steps=history_steps,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            patch_length=config.patch_length,
            stride=config.stride,
            encoder_layers=config.encoder_layers,
            feedforward_multiplier=config.feedforward_multiplier,
            revin=config.revin,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "graphwavenet":
        config = load_graphwavenet_config()
        return GraphWaveNetBaseline(
            history_steps=history_steps,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            temporal_kernel_size=config.temporal_kernel_size,
            diffusion_order=config.diffusion_order,
            node_embedding_dim=config.node_embedding_dim,
            skip_multiplier=config.skip_multiplier,
            end_multiplier=config.end_multiplier,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "dcrnn":
        config = load_dcrnn_config()
        return DCRNNBaseline(
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            diffusion_order=config.diffusion_order,
            adjacency=config.adjacency,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    if normalized == "crnn":
        config = load_crnn_config()
        return CRNNBaseline(
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            conv_layers=config.conv_layers,
            kernel_size=config.kernel_size,
            recurrent_layers=config.recurrent_layers,
            activation=config.activation,
            prediction_steps=config.prediction_steps,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
        )
    raise ValueError(f"Unknown model {model_name!r}; available models: {AVAILABLE_MODELS}")


def get_horizon_minutes(model: nn.Module) -> tuple[int, ...]:
    """Read the shared horizon contract from a registered forecasting model."""
    values = getattr(model, "horizon_minutes", None)
    if values is None and hasattr(model, "prediction_head"):
        values = getattr(model.prediction_head, "horizon_minutes", None)
    if values is None:
        raise AttributeError("Forecasting model must expose horizon_minutes.")
    return tuple(int(value) for value in values)


def get_required_horizon_steps(model: nn.Module) -> int:
    """Read target step indices required by a registered forecasting model."""
    values = getattr(model, "required_horizon_steps", None)
    if values is None and hasattr(model, "prediction_head"):
        values = getattr(model.prediction_head, "required_horizon_steps", None)
    if values is None:
        raise AttributeError("Forecasting model must expose required_horizon_steps.")
    return int(values)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def get_registered_model_config(model_name: str) -> dict[str, object]:
    """Return the paper-facing architecture configuration when one exists."""
    normalized = model_name.lower()
    config_names = {
        "cnn": "cnn.yaml",
        "informer": "informer.yaml",
        "autoformer": "autoformer.yaml",
        "patchtst": "patchtst.yaml",
        "graphwavenet": "graphwavenet.yaml",
        "dcrnn": "dcrnn.yaml",
        "crnn": "crnn.yaml",
    }
    filename = config_names.get(normalized)
    return dict(load_json_yaml(filename)) if filename is not None else {}
