"""Complete GlucoPrism / ST-MSFFNet model assembly."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bidirectional_cross_attention import BidirectionalCrossAttention
from .multi_horizon_prediction_head import MultiHorizonPredictionHead
from .multiscale_time_encoder import MultiScaleTimeEncoder
from .physio_graph_encoder import PhysioGraphEncoder


DEFAULT_NODE_NAMES = ("acc_l2", "eda", "temp", "hr", "bvp", "ibi")
ABLATION_MODES = {
    "full",
    "cgm_only",
    "no_graph",
    "no_cross_attention",
    "single_scale_temporal",
    "fixed_prior_graph",
}


def build_physiology_prior(node_names: Sequence[str] = DEFAULT_NODE_NAMES) -> Tensor:
    """Build a non-negative physiology-prior adjacency matrix for wearable nodes."""
    names = [name.lower() for name in node_names]
    n = len(names)
    prior = torch.zeros(n, n, dtype=torch.float32)
    edges = {
        ("hr", "bvp"): 0.85,
        ("hr", "ibi"): 0.85,
        ("bvp", "ibi"): 0.75,
        ("eda", "hr"): 0.45,
        ("eda", "temp"): 0.40,
        ("acc_l2", "hr"): 0.50,
        ("acc", "hr"): 0.50,
        ("acc_l2", "eda"): 0.30,
        ("acc", "eda"): 0.30,
        ("temp", "hr"): 0.25,
        ("temp", "bvp"): 0.25,
    }
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for (src, dst), weight in edges.items():
        if src in name_to_idx and dst in name_to_idx:
            i, j = name_to_idx[src], name_to_idx[dst]
            prior[i, j] = weight
            prior[j, i] = weight
    return prior


class ST_MSFFNet(nn.Module):
    """End-to-end GlucoPrism network for 15/30/45/60min glucose prediction.

    Key design: the prediction head receives the last observed CGM value as a
    residual, so even an untrained model matches the persistence baseline
    (predict the last value for all horizons). The model learns to predict
    *deviations* (deltas) from the last observed glucose.
    """

    def __init__(
        self,
        history_steps: int = 24,
        cgm_dim: int = 1,
        num_physio_nodes: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        graph_layers: int = 2,
        dropout: float = 0.1,
        horizon_minutes: Sequence[int] = (15, 30, 45, 60),
        sampling_interval_minutes: int = 5,
        node_names: Sequence[str] = DEFAULT_NODE_NAMES,
        A_prior: Optional[Tensor] = None,
        ablation_mode: str = "full",
        enable_horizon_refinement: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(
                f"ablation_mode must be one of {sorted(ABLATION_MODES)}, got {ablation_mode!r}."
            )
        self.history_steps = history_steps
        self.cgm_dim = cgm_dim
        self.num_physio_nodes = num_physio_nodes
        self.hidden_dim = hidden_dim
        self.ablation_mode = ablation_mode
        if A_prior is None:
            selected_names = tuple(node_names[:num_physio_nodes])
            A_prior = build_physiology_prior(selected_names)

        self.cgm_aug_dim = cgm_dim * 4
        self.physio_aug_dim = 4
        temporal_mode = (
            "single_scale"
            if ablation_mode == "single_scale_temporal"
            else "multi_scale"
        )
        self.cgm_time_encoder = MultiScaleTimeEncoder(
            self.cgm_aug_dim,
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_mode=temporal_mode,
        )
        self.physio_time_encoder = MultiScaleTimeEncoder(
            self.physio_aug_dim,
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_mode=temporal_mode,
        )
        self.physio_time_pool = nn.Linear(hidden_dim, 1)
        self.physio_graph_encoder = PhysioGraphEncoder(
            num_nodes=num_physio_nodes,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_gcn_layers=graph_layers,
            dropout=dropout,
            A_prior=A_prior,
            edge_dropout=0.0 if ablation_mode == "fixed_prior_graph" else 0.05,
            use_residual_graph=ablation_mode != "fixed_prior_graph",
            use_edge_importance=ablation_mode != "fixed_prior_graph",
            learn_prior_weight=ablation_mode != "fixed_prior_graph",
        )
        self.cross_attention_fusion = BidirectionalCrossAttention(
            dim_a=hidden_dim,
            dim_b=hidden_dim,
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.prediction_head = MultiHorizonPredictionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            horizon_minutes=horizon_minutes,
            sampling_interval_minutes=sampling_interval_minutes,
            dropout=dropout,
            enable_horizon_refinement=enable_horizon_refinement,
        )
        if ablation_mode == "cgm_only":
            self._freeze_modules(
                self.physio_time_encoder,
                self.physio_time_pool,
                self.physio_graph_encoder,
                self.cross_attention_fusion,
            )
        elif ablation_mode == "no_graph":
            self._freeze_modules(self.physio_graph_encoder)
        elif ablation_mode == "no_cross_attention":
            self._freeze_modules(self.cross_attention_fusion)

    def forward(
        self,
        cgm: Tensor,
        physio: Tensor,
        return_dict: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        """
        Args:
            cgm: [B, T=24, 1] standardized CGM history.
            physio: [B, N=6, T=24] standardized physiological signals.
        Returns:
            pred: [B, 4] standardized predictions for [15, 30, 45, 60] min.
        """
        self._check_inputs(cgm, physio)
        b = cgm.shape[0]

        # --- Extract last observed CGM value for residual connection ---
        # Shape: [B, 1] – the anchor for all horizon predictions.
        last_cgm = cgm[:, -1, 0:1]  # [B, 1]

        # --- CGM temporal encoding ---
        cgm_aug = self._augment_cgm(cgm)  # [B, T, 3]
        cgm_feature = self.cgm_time_encoder(cgm_aug)  # [B, T, H]

        physio_node_feature = cgm_feature.new_zeros(
            b, self.num_physio_nodes, self.hidden_dim
        )
        physio_graph_feature = physio_node_feature
        if self.ablation_mode == "cgm_only":
            fused = cgm_feature
        else:
            # --- Physio temporal encoding (per-channel, then graph) ---
            physio_flat = physio.unsqueeze(-1).reshape(
                b * self.num_physio_nodes, self.history_steps, 1
            )
            physio_aug = self._augment_physio(physio_flat)
            physio_temporal = self.physio_time_encoder(physio_aug)
            physio_temporal = physio_temporal.reshape(
                b, self.num_physio_nodes, self.history_steps, self.hidden_dim
            )
            pool_logits = self.physio_time_pool(physio_temporal).squeeze(-1)
            pool_weights = torch.softmax(pool_logits, dim=-1)
            physio_node_feature = torch.sum(
                physio_temporal * pool_weights.unsqueeze(-1), dim=2
            )
            if self.ablation_mode == "no_graph":
                physio_graph_feature = physio_node_feature
            else:
                physio_graph_feature = self.physio_graph_encoder(physio_node_feature)

            # Parameter-free additive fusion isolates cross-attention's contribution.
            if self.ablation_mode == "no_cross_attention":
                physio_context = physio_graph_feature.mean(dim=1, keepdim=True)
                fused = F.layer_norm(
                    cgm_feature + physio_context,
                    normalized_shape=(self.hidden_dim,),
                )
            else:
                fused = self.cross_attention_fusion(cgm_feature, physio_graph_feature)

        # --- Multi-horizon prediction with CGM residual ---
        pred = self.prediction_head(fused, last_cgm)  # [B, 4]

        if return_dict:
            return {
                "pred": pred,
                "cgm_feature": cgm_feature,
                "physio_node_feature": physio_node_feature,
                "physio_graph_feature": physio_graph_feature,
                "fused_feature": fused,
            }
        return pred

    def select_targets(self, future_glucose: Tensor) -> Tensor:
        """future_glucose [B, 12] -> labels [B, 4]."""
        return self.prediction_head.select_targets(future_glucose)

    def regularization_loss(self) -> Tensor:
        """Small structure regularization for graph interpretability and fusion stability."""
        device = next(self.parameters()).device
        loss = torch.zeros((), device=device)
        graph_is_active = self.ablation_mode not in {"cgm_only", "no_graph"}
        fusion_is_active = self.ablation_mode not in {"cgm_only", "no_cross_attention"}
        if graph_is_active and hasattr(self.physio_graph_encoder, "prior_consistency_loss"):
            loss = loss + self.physio_graph_encoder.prior_consistency_loss().to(device)
        if fusion_is_active and hasattr(self.cross_attention_fusion, "modality_balance_loss"):
            loss = loss + 0.01 * self.cross_attention_fusion.modality_balance_loss().to(device)
        return loss

    @torch.no_grad()
    def diagnostic_metrics(self) -> Dict[str, float]:
        """Return lightweight graph/fusion diagnostics for interpretability logs."""
        metrics: Dict[str, float] = {}
        graph = self.physio_graph_encoder
        graph_is_active = self.ablation_mode not in {"cgm_only", "no_graph"}
        fusion_is_active = self.ablation_mode not in {"cgm_only", "no_cross_attention"}
        if graph_is_active and getattr(graph, "learn_prior_weight", False):
            prior_weight = 0.5 + torch.sigmoid(graph.prior_weight_logit.detach())
            metrics["graph_prior_weight"] = float(prior_weight.cpu())
        if graph_is_active and getattr(graph, "use_residual_graph", False):
            mask = graph.prior_mask.to(
                device=graph.edge_residual_logits.device,
                dtype=graph.edge_residual_logits.dtype,
            )
            residual = torch.nn.functional.softplus(graph.edge_residual_logits.detach()) * mask
            metrics["graph_residual_mean"] = float(residual.mean().cpu())
        weights = getattr(self.cross_attention_fusion, "last_modality_weights", None)
        if fusion_is_active and weights is not None:
            mean_weights = weights.detach().float().mean(dim=(0, 1)).cpu()
            if mean_weights.numel() >= 3:
                metrics["fusion_gate_cgm"] = float(mean_weights[0])
                metrics["fusion_gate_cgm_guided"] = float(mean_weights[1])
                metrics["fusion_gate_physio"] = float(mean_weights[2])
        if hasattr(self.prediction_head, "diagnostic_metrics"):
            metrics.update(self.prediction_head.diagnostic_metrics())
        return metrics

    @staticmethod
    def _augment_cgm(cgm: Tensor) -> Tensor:
        """CGM [B,T,1] -> [B,T,4] with value, velocity, smooth value, acceleration."""
        delta = torch.zeros_like(cgm)
        delta[:, 1:, :] = cgm[:, 1:, :] - cgm[:, :-1, :]
        accel = torch.zeros_like(cgm)
        accel[:, 1:, :] = delta[:, 1:, :] - delta[:, :-1, :]
        smooth = ST_MSFFNet._moving_average(cgm, kernel_size=3)
        return torch.cat([cgm, delta, smooth, accel], dim=-1)

    @staticmethod
    def _augment_physio(physio_flat: Tensor) -> Tensor:
        """Physio [B*N,T,1] -> [B*N,T,4] with value, delta, smooth, local variability."""
        delta = torch.zeros_like(physio_flat)
        delta[:, 1:, :] = physio_flat[:, 1:, :] - physio_flat[:, :-1, :]
        smooth = ST_MSFFNet._moving_average(physio_flat, kernel_size=5)
        second_moment = ST_MSFFNet._moving_average(physio_flat.square(), kernel_size=5)
        variability = (second_moment - smooth.square()).clamp_min(0.0).sqrt()
        return torch.cat([physio_flat, delta, smooth, variability], dim=-1)

    @staticmethod
    def _moving_average(x: Tensor, kernel_size: int) -> Tensor:
        """Replicate-padded moving average for [B,T,C] sequences."""
        if kernel_size <= 1:
            return x
        left = kernel_size // 2
        right = kernel_size - 1 - left
        channels_first = x.transpose(1, 2)  # [B,C,T]
        padded = torch.nn.functional.pad(channels_first, (left, right), mode="replicate")
        smoothed = torch.nn.functional.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
        return smoothed.transpose(1, 2)

    @staticmethod
    def _freeze_modules(*modules: nn.Module) -> None:
        for module in modules:
            module.requires_grad_(False)

    def _check_inputs(self, cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3 or cgm.shape[1:] != (self.history_steps, self.cgm_dim):
            raise ValueError(
                f"cgm expected [B,{self.history_steps},{self.cgm_dim}], "
                f"got {tuple(cgm.shape)}"
            )
        if physio.ndim != 3 or physio.shape[1:] != (self.num_physio_nodes, self.history_steps):
            raise ValueError(
                f"physio expected [B,{self.num_physio_nodes},{self.history_steps}], "
                f"got {tuple(physio.shape)}"
            )
        if cgm.shape[0] != physio.shape[0]:
            raise ValueError("cgm and physio must share batch size.")
        if not torch.isfinite(cgm).all() or not torch.isfinite(physio).all():
            raise ValueError("Model inputs contain NaN/Inf.")
