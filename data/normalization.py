"""Fold-wise normalization utilities without subject leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataStandardizer:
    """Statistics fitted on training subjects only for one LOSO fold."""

    glucose_mean: Tensor
    glucose_std: Tensor
    physio_mean: Tensor
    physio_std: Tensor
    eps: float = 1e-6

    def transform_cgm(self, cgm: Tensor) -> Tensor:
        """CGM [T, 1] or [B, T, 1] -> standardized same shape."""
        return (cgm - self.glucose_mean) / self.glucose_std.clamp_min(self.eps)

    def transform_future_glucose(self, y: Tensor) -> Tensor:
        """Future glucose [S] or [B, S] -> standardized same shape."""
        return (y - self.glucose_mean) / self.glucose_std.clamp_min(self.eps)

    def inverse_glucose(self, y: Tensor) -> Tensor:
        """Standardized glucose -> mg/dL."""
        mean = self.glucose_mean.to(device=y.device, dtype=y.dtype)
        std = self.glucose_std.to(device=y.device, dtype=y.dtype).clamp_min(self.eps)
        return y * std + mean

    def transform_physio(self, x: Tensor) -> Tensor:
        """Physio [N, T] or [B, N, T] -> standardized per physiological node."""
        if x.ndim == 2:
            mean = self.physio_mean
            std = self.physio_std.clamp_min(self.eps)
        elif x.ndim == 3:
            mean = self.physio_mean.unsqueeze(0)
            std = self.physio_std.unsqueeze(0).clamp_min(self.eps)
        else:
            raise ValueError(f"physio must be [N, T] or [B, N, T], got {tuple(x.shape)}")
        return (x - mean) / std


class StandardizedDataset(Dataset):
    """Apply fold-specific normalization while preserving raw labels for metrics."""

    def __init__(self, base_dataset: Dataset, standardizer: DataStandardizer) -> None:
        self.base_dataset = base_dataset
        self.standardizer = standardizer
        self.subject_ids = getattr(base_dataset, "subject_ids")
        self.history_steps = getattr(base_dataset, "history_steps")
        self.num_physio_nodes = getattr(base_dataset, "num_physio_nodes")
        self.future_steps = getattr(base_dataset, "future_steps")
        self.node_names = getattr(base_dataset, "node_names", ())
        self.dataset_name = getattr(base_dataset, "dataset_name", "unknown")
        self.A_prior = getattr(base_dataset, "A_prior", None)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        item = self.base_dataset[index]
        cgm = item["cgm"]
        physio = item["physio"]
        future = item["future_glucose"]
        if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(future, Tensor):
            raise TypeError("Base dataset must return tensor cgm, physio, and future_glucose.")
        return {
            "cgm": self.standardizer.transform_cgm(cgm),  # [T, 1]
            "physio": self.standardizer.transform_physio(physio),  # [N, T]
            "future_glucose": self.standardizer.transform_future_glucose(future),  # [S]
            "future_glucose_raw": future,  # [S], mg/dL
            "subject_id": item["subject_id"],
        }


def fit_standardizer(dataset: Dataset, train_indices: Sequence[int], eps: float = 1e-6) -> DataStandardizer:
    """Fit normalization statistics using training samples only."""
    if not train_indices:
        raise ValueError("train_indices must not be empty.")
    cgm_values: List[Tensor] = []
    future_values: List[Tensor] = []
    physio_values: List[Tensor] = []
    for index in train_indices:
        item = dataset[int(index)]
        cgm = item["cgm"]
        physio = item["physio"]
        future = item["future_glucose"]
        if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(future, Tensor):
            raise TypeError("Dataset must return tensor cgm, physio, and future_glucose.")
        cgm_values.append(cgm.reshape(-1))
        future_values.append(future.reshape(-1))
        physio_values.append(physio)

    glucose = torch.cat([torch.cat(cgm_values), torch.cat(future_values)]).float()
    physio_stack = torch.stack(physio_values).float()  # [B_train, N, T]
    if not torch.isfinite(glucose).all() or not torch.isfinite(physio_stack).all():
        raise ValueError("Cannot fit standardizer on NaN/Inf values.")
    glucose_mean = glucose.mean()
    glucose_std = glucose.std(unbiased=False).clamp_min(eps)
    physio_mean = physio_stack.mean(dim=(0, 2)).unsqueeze(-1)  # [N, 1]
    physio_std = physio_stack.std(dim=(0, 2), unbiased=False).unsqueeze(-1).clamp_min(eps)
    return DataStandardizer(glucose_mean, glucose_std, physio_mean, physio_std, eps)
