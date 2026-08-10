"""Paper-facing GlucoPrism model interface.

This module keeps the论文命名 ``GlucoPrism`` while reusing the complete
implementation in ``ST_MSFFNet``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from torch import Tensor

from .st_msffnet import DEFAULT_NODE_NAMES, ST_MSFFNet


class GlucoPrism(ST_MSFFNet):
    """
    Complete GlucoPrism model for multimodal wearable glucose forecasting.

    Input contract:
        cgm: Tensor [batch_size, history_steps=24, 1]
        physio: Tensor [batch_size, num_physio_nodes, history_steps=24]

    Output contract:
        pred: Tensor [batch_size, 4]
        columns correspond to [15min, 30min, 45min, 60min].
    """

    def __init__(
        self,
        history_steps: int = 24,
        cgm_dim: int = 1,
        num_physio_nodes: int = len(DEFAULT_NODE_NAMES),
        hidden_dim: int = 64,
        num_heads: int = 4,
        graph_layers: int = 2,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
        node_names: Sequence[str] = DEFAULT_NODE_NAMES,
        A_prior: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            history_steps=history_steps,
            cgm_dim=cgm_dim,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            graph_layers=graph_layers,
            dropout=dropout,
            horizon_minutes=horizon_minutes,
            sampling_interval_minutes=sampling_interval_minutes,
            node_names=node_names,
            A_prior=A_prior,
        )
