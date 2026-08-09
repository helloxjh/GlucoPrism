"""Validation and testing loops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.normalization import DataStandardizer
from models.registry import get_horizon_minutes, get_required_horizon_steps
from .metrics import compute_metrics


@dataclass(frozen=True)
class EvaluationOutputs:
    """Predictions and timing collected from one validation/test pass."""

    loss: float
    metrics: Dict[str, float]
    predictions: Tensor
    targets: Tensor
    subject_ids: List[str]
    inference_time_seconds: float
    num_samples: int


def move_batch_to_device(batch: Dict[str, Tensor | list[str]], device: torch.device) -> Dict[str, Tensor | list[str]]:
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}


def validate_batch_shapes(batch: Dict[str, Tensor | list[str]], required_future_steps: int) -> None:
    """Defensive batch shape checks."""
    cgm, physio, y = batch["cgm"], batch["physio"], batch["future_glucose"]
    if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(y, Tensor):
        raise TypeError("Batch must include tensor cgm, physio, and future_glucose.")
    if cgm.ndim != 3:
        raise ValueError(f"cgm must be [B,T,1], got {tuple(cgm.shape)}")
    if physio.ndim != 3:
        raise ValueError(f"physio must be [B,N,T], got {tuple(physio.shape)}")
    if y.ndim != 2 or y.shape[1] < required_future_steps:
        raise ValueError(f"future_glucose must be [B,>={required_future_steps}], got {tuple(y.shape)}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    standardizer: Optional[DataStandardizer] = None,
) -> Tuple[float, Dict[str, float]]:
    """Run validation/test and return loss plus metrics in mg/dL."""
    outputs = evaluate_detailed(
        model, loader, criterion, device, standardizer, measure_inference=False
    )
    return outputs.loss, outputs.metrics


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    standardizer: Optional[DataStandardizer] = None,
    measure_inference: bool = True,
) -> EvaluationOutputs:
    """Run evaluation and retain predictions for benchmark artifacts."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds: List[Tensor] = []
    targets: List[Tensor] = []
    subject_ids: List[str] = []
    inference_time_seconds = 0.0
    required_steps = get_required_horizon_steps(model)
    for batch_idx, raw_batch in enumerate(loader):
        try:
            batch = move_batch_to_device(raw_batch, device)
            validate_batch_shapes(batch, required_steps)
            cgm = batch["cgm"]
            physio = batch["physio"]
            future = batch["future_glucose"]
            assert isinstance(cgm, Tensor) and isinstance(physio, Tensor) and isinstance(future, Tensor)
            if measure_inference:
                synchronize_device(device)
                inference_start = time.perf_counter()
                pred = model(cgm, physio)  # [B, 4]
                synchronize_device(device)
                inference_time_seconds += time.perf_counter() - inference_start
            else:
                pred = model(cgm, physio)  # [B, 4]
            target = model.select_targets(future)  # [B, 4]
            last_cgm = cgm[:, -1, 0:1]  # [B, 1]
            loss = criterion(pred, target, last_cgm=last_cgm)
            if standardizer is not None:
                raw_future = batch.get("future_glucose_raw")
                if not isinstance(raw_future, Tensor):
                    raise TypeError("standardized evaluation requires future_glucose_raw.")
                metric_pred = standardizer.inverse_glucose(pred)
                metric_target = model.select_targets(raw_future)
            else:
                metric_pred = pred
                metric_target = target
            total_loss += float(loss.item()) * cgm.shape[0]
            total_samples += cgm.shape[0]
            preds.append(metric_pred.detach().cpu())
            targets.append(metric_target.detach().cpu())
            batch_subjects = raw_batch.get("subject_id", [])
            subject_ids.extend(str(value) for value in batch_subjects)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Evaluation failed at batch {batch_idx}: {exc}") from exc
    if not preds:
        raise RuntimeError("Evaluation loader produced no batches.")
    horizon_minutes = get_horizon_minutes(model)
    all_preds = torch.cat(preds)
    all_targets = torch.cat(targets)
    metrics = compute_metrics(all_preds, all_targets, horizon_minutes=horizon_minutes)
    return EvaluationOutputs(
        loss=total_loss / max(total_samples, 1),
        metrics=metrics,
        predictions=all_preds,
        targets=all_targets,
        subject_ids=subject_ids,
        inference_time_seconds=inference_time_seconds,
        num_samples=total_samples,
    )


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerators before wall-clock timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def test(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, standardizer: Optional[DataStandardizer]) -> Tuple[float, Dict[str, float]]:
    """Alias for final testing."""
    return evaluate(model, loader, criterion, device, standardizer)
