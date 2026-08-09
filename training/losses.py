"""Loss functions for glucose forecasting."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class HorizonWeightedLoss(nn.Module):
    """SmoothL1 loss with per-horizon weights.

    Long-horizon outputs receive slightly higher weight to reduce 45/60min drift.
    All computation happens in standardized glucose space.
    """

    def __init__(
        self,
        horizon_weights: Sequence[float] = (1.0, 1.05, 1.25, 1.45),
        curriculum_start_weights: Sequence[float] | None = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        target_weights = self._validate_weights(horizon_weights, "horizon_weights")
        if curriculum_start_weights is None:
            curriculum_start_weights = horizon_weights
        start_weights = self._validate_weights(
            curriculum_start_weights, "curriculum_start_weights"
        )
        if start_weights.numel() != target_weights.numel():
            raise ValueError("curriculum and target horizon weights must have equal length")
        self.register_buffer("target_weights", target_weights / target_weights.sum())
        self.register_buffer("start_weights", start_weights / start_weights.sum())
        self.register_buffer("weights", self.target_weights.clone())
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred: Tensor, target: Tensor, last_cgm: Tensor | None = None) -> Tensor:
        """Compute weighted SmoothL1 loss.

        Args:
            pred: [B, K] standardized predictions.
            target: [B, K] standardized targets.
            last_cgm: optional [B, 1], accepted for a shared loss interface.
        """
        del last_cgm
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
        if pred.ndim != 2:
            raise ValueError(f"pred/target must be [B,K], got {tuple(pred.shape)}")
        if pred.shape[1] != len(self.weights):
            raise ValueError(f"horizon_weights length {len(self.weights)} != prediction dim {pred.shape[1]}")
        per_element = self.smooth_l1(pred, target)  # [B, K]
        weights = self.weights.to(device=pred.device, dtype=pred.dtype)
        return (per_element * weights.unsqueeze(0)).sum(dim=1).mean()

    @torch.no_grad()
    def set_curriculum_progress(self, progress: float) -> None:
        """Interpolate from easy-to-hard weights to final long-horizon weights."""
        value = float(min(1.0, max(0.0, progress)))
        interpolated = torch.lerp(self.start_weights, self.target_weights, value)
        self.weights.copy_(interpolated / interpolated.sum())

    @staticmethod
    def _validate_weights(values: Sequence[float], name: str) -> Tensor:
        weights = torch.as_tensor(values, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
        if (weights < 0).any() or float(weights.sum()) <= 0.0:
            raise ValueError(f"{name} must be non-negative with a positive sum")
        return weights


class GlucoPrismForecastingLoss(nn.Module):
    """Prediction + trend + Pearson + smoothness + horizon consistency loss.

    All terms are computed in standardized glucose space:
        L = L_pred + lambda_trend * L_trend
                   + lambda_corr * L_corr
                   + lambda_smooth * L_smooth
                   + lambda_horizon * L_horizon

    Shapes:
        pred: [B, K] predictions for 15/30/45/60 min.
        target: [B, K] labels for 15/30/45/60 min.
        last_cgm: [B, 1] last observed CGM value.
    """

    def __init__(
        self,
        horizon_weights: Sequence[float] = (1.0, 1.05, 1.25, 1.45),
        curriculum_start_weights: Sequence[float] = (1.0, 0.85, 0.65, 0.50),
        beta: float = 1.0,
        trend_weight: float = 0.25,
        corr_weight: float = 0.08,
        temporal_smooth_weight: float = 0.06,
        horizon_consistency_weight: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            trend_weight < 0
            or corr_weight < 0
            or temporal_smooth_weight < 0
            or horizon_consistency_weight < 0
        ):
            raise ValueError("loss weights must be non-negative.")
        self.prediction_loss = HorizonWeightedLoss(
            horizon_weights=horizon_weights,
            curriculum_start_weights=curriculum_start_weights,
            beta=beta,
        )
        self.trend_weight = float(trend_weight)
        self.corr_weight = float(corr_weight)
        self.temporal_smooth_weight = float(temporal_smooth_weight)
        self.horizon_consistency_weight = float(horizon_consistency_weight)
        self.current_horizon_consistency_scale = 1.0
        self.eps = eps
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred: Tensor, target: Tensor, last_cgm: Tensor | None = None) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
        if pred.ndim != 2:
            raise ValueError(f"pred/target must be [B,K], got {tuple(pred.shape)}")
        pred_loss = self.prediction_loss(pred, target)
        if last_cgm is None:
            return pred_loss
        if last_cgm.ndim != 2 or last_cgm.shape != (pred.shape[0], 1):
            raise ValueError(f"last_cgm must be [B,1], got {tuple(last_cgm.shape)}")

        trend_loss = self._trend_consistency_loss(pred, target, last_cgm)
        corr_loss = self._pearson_correlation_loss(pred, target)
        smooth_loss = self._temporal_smoothness_loss(pred, target, last_cgm)
        horizon_loss = self._horizon_consistency_loss(pred, target)
        return (
            pred_loss
            + self.trend_weight * trend_loss
            + self.corr_weight * corr_loss
            + self.temporal_smooth_weight * smooth_loss
            + self.horizon_consistency_weight
            * self.current_horizon_consistency_scale
            * horizon_loss
        )

    @torch.no_grad()
    def set_curriculum_progress(self, progress: float) -> None:
        """Update training-only horizon emphasis without changing validation loss."""
        value = float(min(1.0, max(0.0, progress)))
        self.prediction_loss.set_curriculum_progress(value)
        self.current_horizon_consistency_scale = 0.25 + 0.75 * value

    def horizon_weight_metrics(self) -> dict[str, float]:
        """Return active weights for reproducible training diagnostics."""
        weights = self.prediction_loss.weights.detach().cpu()
        return {
            f"loss_horizon_weight_{index}": float(weight)
            for index, weight in enumerate(weights)
        }

    def _trend_consistency_loss(self, pred: Tensor, target: Tensor, last_cgm: Tensor) -> Tensor:
        # [B,1+K] -> first-order glucose velocity across forecast horizons.
        pred_path = torch.cat([last_cgm, pred], dim=1)
        target_path = torch.cat([last_cgm, target], dim=1)
        pred_delta = pred_path[:, 1:] - pred_path[:, :-1]  # [B,K]
        target_delta = target_path[:, 1:] - target_path[:, :-1]  # [B,K]
        weights = self.prediction_loss.weights.to(device=pred.device, dtype=pred.dtype)
        per_element = self.smooth_l1(pred_delta, target_delta)
        return (per_element * weights.unsqueeze(0)).sum(dim=1).mean()

    def _temporal_smoothness_loss(self, pred: Tensor, target: Tensor, last_cgm: Tensor) -> Tensor:
        # Match second-order glucose dynamics instead of merely penalizing
        # rough predictions. This preserves true rises/falls while reducing
        # long-horizon drift and unrealistic curvature.
        pred_path = torch.cat([last_cgm, pred], dim=1)  # [B,1+K]
        target_path = torch.cat([last_cgm, target], dim=1)  # [B,1+K]
        if pred_path.shape[1] < 3:
            return pred.new_tensor(0.0)
        pred_accel = pred_path[:, 2:] - 2.0 * pred_path[:, 1:-1] + pred_path[:, :-2]
        target_accel = target_path[:, 2:] - 2.0 * target_path[:, 1:-1] + target_path[:, :-2]
        return self.smooth_l1(pred_accel, target_accel).mean()

    def _pearson_correlation_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        # Long-horizon weighted correlation discourages regression-to-the-mean
        # behavior at 45/60 min while retaining all horizon contributions.
        if pred.shape[0] < 2:
            return pred.new_tensor(0.0)
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=0)
        denominator = torch.sqrt(
            pred_centered.square().sum(dim=0).clamp_min(self.eps)
            * target_centered.square().sum(dim=0).clamp_min(self.eps)
        )
        corr = (numerator / denominator.clamp_min(self.eps)).clamp(min=-1.0, max=1.0)
        weights = self.prediction_loss.weights.to(device=pred.device, dtype=pred.dtype)
        return ((1.0 - corr) * weights).sum()

    def _horizon_consistency_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Match pairwise trajectory displacements across all horizon pairs.

        Adjacent trend loss only constrains 15-minute increments. Pairwise
        consistency additionally supervises long spans such as 15->60 min,
        directly limiting long-horizon drift without forcing a flat trajectory.
        """
        num_horizons = pred.shape[1]
        if num_horizons < 2:
            return pred.new_tensor(0.0)
        weighted_losses: list[Tensor] = []
        pair_weights: list[float] = []
        max_span = float(num_horizons - 1)
        for start in range(num_horizons - 1):
            for end in range(start + 1, num_horizons):
                pred_displacement = pred[:, end] - pred[:, start]
                target_displacement = target[:, end] - target[:, start]
                pair_loss = self.smooth_l1(
                    pred_displacement, target_displacement
                ).mean()
                span_weight = (end - start) / max_span
                terminal_weight = (end + 1) / num_horizons
                weighted_losses.append(pair_loss)
                pair_weights.append(span_weight * terminal_weight)
        weights = pred.new_tensor(pair_weights)
        losses = torch.stack(weighted_losses)
        return torch.sum(losses * weights) / weights.sum().clamp_min(self.eps)


class BasicRegressionLoss(nn.Module):
    """MSE/MAE wrappers with the same optional last_cgm interface."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        normalized = kind.lower()
        if normalized in {"mse", "mse_loss"}:
            self.loss = nn.MSELoss()
        elif normalized in {"mae", "l1"}:
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown basic loss: {kind}")

    def forward(self, pred: Tensor, target: Tensor, last_cgm: Tensor | None = None) -> Tensor:
        del last_cgm
        return self.loss(pred, target)


def build_loss(
    name: str = "glucoprism",
    horizon_weights: Sequence[float] | None = None,
    curriculum_start_weights: Sequence[float] | None = None,
    horizon_consistency_weight: float | None = None,
) -> nn.Module:
    """Build regression loss for glucose forecasting."""
    normalized = name.lower()
    if normalized in {"glucoprism", "composite", "forecasting"}:
        kwargs = {}
        if horizon_weights is not None:
            kwargs["horizon_weights"] = horizon_weights
        if curriculum_start_weights is not None:
            kwargs["curriculum_start_weights"] = curriculum_start_weights
        if horizon_consistency_weight is not None:
            kwargs["horizon_consistency_weight"] = horizon_consistency_weight
        return GlucoPrismForecastingLoss(**kwargs)
    if normalized in {"smooth_l1", "huber"}:
        kwargs = {}
        if horizon_weights is not None:
            kwargs["horizon_weights"] = horizon_weights
        if curriculum_start_weights is not None:
            kwargs["curriculum_start_weights"] = curriculum_start_weights
        return HorizonWeightedLoss(**kwargs)
    if normalized in {"mse", "mse_loss"}:
        return BasicRegressionLoss("mse")
    if normalized in {"mae", "l1"}:
        return BasicRegressionLoss("mae")
    raise ValueError(f"Unknown loss: {name}")
