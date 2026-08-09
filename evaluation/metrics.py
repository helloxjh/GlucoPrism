"""Regression metrics for glucose prediction."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor

from .clarke_ega import clarke_error_grid_percentages


def _compute_flat_metrics(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Compute metrics after flattening all samples and horizons."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    p = pred.detach().float().reshape(-1).cpu()
    y = target.detach().float().reshape(-1).cpu()
    err = p - y
    mae = err.abs().mean()
    rmse = torch.sqrt(err.square().mean())
    mard = (err.abs() / y.abs().clamp_min(eps)).mean() * 100.0
    mape = mard.clone()
    ss_res = err.square().sum()
    ss_tot = (y - y.mean()).square().sum().clamp_min(eps)
    r2 = 1.0 - ss_res / ss_tot
    p_centered = p - p.mean()
    y_centered = y - y.mean()
    pearson = (p_centered * y_centered).sum() / torch.sqrt(
        p_centered.square().sum().clamp_min(eps) * y_centered.square().sum().clamp_min(eps)
    )
    metrics = {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "mard": float(mard.item()),
        "mape": float(mape.item()),
        "r2": float(r2.item()),
        "pearson": float(pearson.clamp(min=-1.0, max=1.0).item()),
    }
    metrics.update(clarke_error_grid_percentages(reference=y, prediction=p))
    return metrics


def compute_metrics(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-6,
    horizon_minutes: Iterable[int] = (15, 30, 45, 60),
) -> Dict[str, float]:
    """
    Compute overall and per-horizon glucose prediction metrics.

    Args:
        pred: Tensor [num_samples, num_horizons]
        target: Tensor [num_samples, num_horizons]
        horizon_minutes: Labels for each output column, e.g. [15, 30, 45, 60].

    Returns:
        Dict with pooled metrics plus keys like horizon_30_mae.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 2:
        raise ValueError(f"pred and target must be [N, K], got {tuple(pred.shape)}")

    horizons = tuple(int(value) for value in horizon_minutes)
    if len(horizons) != pred.shape[1]:
        raise ValueError(f"horizon_minutes length must match output dim {pred.shape[1]}, got {len(horizons)}")

    metrics = _compute_flat_metrics(pred, target, eps=eps)
    for col, horizon in enumerate(horizons):
        one_horizon = _compute_flat_metrics(pred[:, col], target[:, col], eps=eps)
        for key, value in one_horizon.items():
            metrics[f"horizon_{horizon}_{key}"] = value
    return metrics


def format_metrics(prefix: str, loss: float, metrics: Dict[str, float]) -> str:
    """Compact metrics line for logs."""
    return (
        f"{prefix}_loss={loss:.4f} "
        f"{prefix}_mae={metrics['mae']:.4f} "
        f"{prefix}_rmse={metrics['rmse']:.4f} "
        f"{prefix}_mard={metrics['mard']:.2f}% "
        f"{prefix}_mape={metrics['mape']:.2f}% "
        f"{prefix}_r2={metrics['r2']:.4f} "
        f"{prefix}_pearson={metrics['pearson']:.4f} "
        f"{prefix}_EGA[A/B/C/D/E]="
        f"{metrics['ega_zone_a_pct']:.2f}/"
        f"{metrics['ega_zone_b_pct']:.2f}/"
        f"{metrics['ega_zone_c_pct']:.2f}/"
        f"{metrics['ega_zone_d_pct']:.2f}/"
        f"{metrics['ega_zone_e_pct']:.2f}%"
    )


def format_horizon_metrics(prefix: str, metrics: Dict[str, float], horizons: Iterable[int] = (15, 30, 45, 60)) -> str:
    """Compact per-horizon metrics for final test logs."""
    parts = []
    for horizon in horizons:
        key = f"horizon_{int(horizon)}"
        parts.append(
            f"{int(horizon)}min: "
            f"MAE={metrics[f'{key}_mae']:.4f}, "
            f"RMSE={metrics[f'{key}_rmse']:.4f}, "
            f"MARD={metrics[f'{key}_mard']:.2f}%, "
            f"MAPE={metrics[f'{key}_mape']:.2f}%, "
            f"R2={metrics[f'{key}_r2']:.4f}, "
            f"Pearson={metrics[f'{key}_pearson']:.4f}, "
            f"EGA[A/B/C/D/E]="
            f"{metrics[f'{key}_ega_zone_a_pct']:.2f}/"
            f"{metrics[f'{key}_ega_zone_b_pct']:.2f}/"
            f"{metrics[f'{key}_ega_zone_c_pct']:.2f}/"
            f"{metrics[f'{key}_ega_zone_d_pct']:.2f}/"
            f"{metrics[f'{key}_ega_zone_e_pct']:.2f}%"
        )
    return f"{prefix}_per_horizon | " + " | ".join(parts)
