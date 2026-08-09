#!/usr/bin/env python3
"""Single-file ST_MSFFNet model assembly and LOSO-CV training pipeline.

This script is intentionally runnable with dummy data:

    .venv_torch/bin/python train_st_msffnet_loso.py

It assembles the previously implemented modules:
    - MultiScaleTimeEncoder
    - PhysioGraphEncoder
    - BidirectionalCrossAttention
    - MultiHorizonPredictionHead

Tensor convention:
    cgm:            [batch_size, history_steps=24, cgm_dim=1]
    physio:         [batch_size, num_physio_nodes=6, history_steps=24]
    future_glucose: [batch_size, future_steps=12]
    pred:           [batch_size, 4] for 15/30/45/60 minutes
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from models import (
    BidirectionalCrossAttention,
    MultiHorizonPredictionHead,
    MultiScaleTimeEncoder,
    PhysioGraphEncoder,
)


@dataclass(frozen=True)
class LOSOSplit:
    """Subject-independent fold definition for LOSO-CV."""

    fold_id: int
    test_subject: str
    val_subjects: List[str]
    train_subjects: List[str]
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


@dataclass(frozen=True)
class DataStandardizer:
    """Fold-specific standardizer fitted only on training subjects."""

    glucose_mean: Tensor
    glucose_std: Tensor
    physio_mean: Tensor
    physio_std: Tensor
    eps: float = 1e-6

    def transform_cgm(self, cgm: Tensor) -> Tensor:
        """Standardize CGM history: [T, 1] or [B, T, 1]."""
        return (cgm - self.glucose_mean) / self.glucose_std.clamp_min(self.eps)

    def transform_future_glucose(self, future_glucose: Tensor) -> Tensor:
        """Standardize future glucose labels: [future_steps] or [B, future_steps]."""
        return (future_glucose - self.glucose_mean) / self.glucose_std.clamp_min(self.eps)

    def inverse_glucose(self, glucose: Tensor) -> Tensor:
        """Map standardized glucose predictions/targets back to mg/dL."""
        mean = self.glucose_mean.to(device=glucose.device, dtype=glucose.dtype)
        std = self.glucose_std.to(device=glucose.device, dtype=glucose.dtype).clamp_min(self.eps)
        return glucose * std + mean

    def transform_physio(self, physio: Tensor) -> Tensor:
        """Standardize physio features per signal node: [N, T] or [B, N, T]."""
        if physio.ndim == 2:
            mean = self.physio_mean
            std = self.physio_std.clamp_min(self.eps)
        elif physio.ndim == 3:
            mean = self.physio_mean.unsqueeze(0)
            std = self.physio_std.unsqueeze(0).clamp_min(self.eps)
        else:
            raise ValueError(f"physio must be [N, T] or [B, N, T], got {tuple(physio.shape)}")
        return (physio - mean) / std


class ST_MSFFNet(nn.Module):
    """
    Final ST_MSFFNet model assembled from temporal, graph, fusion, and prediction modules.

    Forward inputs:
        cgm: Tensor [B, T, 1]
            Historical CGM sequence.
        physio: Tensor [B, N, T]
            Multi-source wearable signals. N is the number of physiological nodes.

    Forward output:
        pred: Tensor [B, 4]
            Glucose predictions at [15min, 30min, 45min, 60min].
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
        A_prior: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self._validate_config(
            history_steps=history_steps,
            cgm_dim=cgm_dim,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            graph_layers=graph_layers,
            dropout=dropout,
        )

        self.history_steps = history_steps
        self.cgm_dim = cgm_dim
        self.num_physio_nodes = num_physio_nodes
        self.hidden_dim = hidden_dim

        # CGM temporal branch:
        # cgm [B, T, 1] -> cgm_feature [B, T, H]
        self.cgm_time_encoder = MultiScaleTimeEncoder(
            feature_dim=cgm_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Physio temporal branch is applied per physiological node:
        # physio [B, N, T] -> reshape [B*N, T, 1] -> [B*N, T, H]
        self.physio_time_encoder = MultiScaleTimeEncoder(
            feature_dim=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.physio_time_pool = nn.Linear(hidden_dim, 1)

        # Physiological topology branch:
        # physio_node_feature [B, N, H] -> graph_feature [B, N, H]
        self.physio_graph_encoder = PhysioGraphEncoder(
            num_nodes=num_physio_nodes,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_gcn_layers=graph_layers,
            dropout=dropout,
            A_prior=A_prior,
        )

        # Cross-modal fusion:
        # cgm_feature [B, T, H], graph_feature [B, N, H] -> fused [B, T, H]
        self.cross_attention_fusion = BidirectionalCrossAttention(
            dim_a=hidden_dim,
            dim_b=hidden_dim,
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Multi-horizon regression head:
        # fused [B, T, H] -> pred [B, 4]
        self.prediction_head = MultiHorizonPredictionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            horizon_minutes=horizon_minutes,
            sampling_interval_minutes=sampling_interval_minutes,
            dropout=dropout,
        )

    def forward(self, cgm: Tensor, physio: Tensor, return_dict: bool = False) -> Tensor | Dict[str, Tensor]:
        """
        Args:
            cgm: Tensor [batch_size, history_steps, cgm_dim]
            physio: Tensor [batch_size, num_physio_nodes, history_steps]
            return_dict: Whether to return intermediate features for debugging.

        Returns:
            Tensor [batch_size, 4] or a dict containing pred and intermediate features.
        """
        self._check_forward_inputs(cgm=cgm, physio=physio)
        batch_size = cgm.shape[0]

        cgm_feature = self.cgm_time_encoder(cgm)
        assert cgm_feature.shape == (batch_size, self.history_steps, self.hidden_dim)

        # [B, N, T] -> [B, N, T, 1] -> [B*N, T, 1]
        physio_flat = physio.unsqueeze(-1).reshape(
            batch_size * self.num_physio_nodes,
            self.history_steps,
            1,
        )
        assert physio_flat.shape == (
            batch_size * self.num_physio_nodes,
            self.history_steps,
            1,
        )

        physio_temporal = self.physio_time_encoder(physio_flat)
        assert physio_temporal.shape == (
            batch_size * self.num_physio_nodes,
            self.history_steps,
            self.hidden_dim,
        )

        # [B*N, T, H] -> [B, N, T, H]
        physio_temporal = physio_temporal.reshape(
            batch_size,
            self.num_physio_nodes,
            self.history_steps,
            self.hidden_dim,
        )
        assert physio_temporal.shape == (
            batch_size,
            self.num_physio_nodes,
            self.history_steps,
            self.hidden_dim,
        )

        # Attention pooling over time for each physiological node.
        pool_logits = self.physio_time_pool(physio_temporal).squeeze(-1)
        assert pool_logits.shape == (batch_size, self.num_physio_nodes, self.history_steps)

        pool_weights = torch.softmax(pool_logits, dim=-1)
        assert pool_weights.shape == (batch_size, self.num_physio_nodes, self.history_steps)

        physio_node_feature = torch.sum(
            physio_temporal * pool_weights.unsqueeze(-1),
            dim=2,
        )
        assert physio_node_feature.shape == (
            batch_size,
            self.num_physio_nodes,
            self.hidden_dim,
        )

        physio_graph_feature = self.physio_graph_encoder(physio_node_feature)
        assert physio_graph_feature.shape == (
            batch_size,
            self.num_physio_nodes,
            self.hidden_dim,
        )

        fused_feature = self.cross_attention_fusion(cgm_feature, physio_graph_feature)
        assert fused_feature.shape == (batch_size, self.history_steps, self.hidden_dim)

        pred = self.prediction_head(fused_feature)
        assert pred.shape == (batch_size, self.prediction_head.num_horizons)

        if return_dict:
            return {
                "pred": pred,
                "cgm_feature": cgm_feature,
                "physio_node_feature": physio_node_feature,
                "physio_graph_feature": physio_graph_feature,
                "fused_feature": fused_feature,
            }
        return pred

    def select_targets(self, future_glucose: Tensor) -> Tensor:
        """Select [15min, 30min, 45min, 60min] labels from future glucose sequence."""
        return self.prediction_head.select_targets(future_glucose)

    def _check_forward_inputs(self, cgm: Tensor, physio: Tensor) -> None:
        if cgm.ndim != 3:
            raise ValueError(f"cgm must be [B, T, C], got {tuple(cgm.shape)}")
        if physio.ndim != 3:
            raise ValueError(f"physio must be [B, N, T], got {tuple(physio.shape)}")
        if cgm.shape[1] != self.history_steps or cgm.shape[2] != self.cgm_dim:
            raise ValueError(
                f"cgm expected [B, {self.history_steps}, {self.cgm_dim}], "
                f"got {tuple(cgm.shape)}"
            )
        if physio.shape[1] != self.num_physio_nodes or physio.shape[2] != self.history_steps:
            raise ValueError(
                f"physio expected [B, {self.num_physio_nodes}, {self.history_steps}], "
                f"got {tuple(physio.shape)}"
            )
        if cgm.shape[0] != physio.shape[0]:
            raise ValueError("cgm and physio must have the same batch_size.")
        if not torch.isfinite(cgm).all():
            raise ValueError("cgm contains NaN or Inf values.")
        if not torch.isfinite(physio).all():
            raise ValueError("physio contains NaN or Inf values.")

    @staticmethod
    def _validate_config(
        history_steps: int,
        cgm_dim: int,
        num_physio_nodes: int,
        hidden_dim: int,
        num_heads: int,
        graph_layers: int,
        dropout: float,
    ) -> None:
        if history_steps <= 0:
            raise ValueError("history_steps must be positive.")
        if cgm_dim <= 0:
            raise ValueError("cgm_dim must be positive.")
        if num_physio_nodes <= 0:
            raise ValueError("num_physio_nodes must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class DummyPhysioNetDataset(Dataset):
    """Dummy dataset that mimics preprocessed BIG IDEAs windows."""

    def __init__(
        self,
        num_subjects: int = 16,
        samples_per_subject: int = 16,
        history_steps: int = 24,
        num_physio_nodes: int = 6,
        future_steps: int = 12,
        seed: int = 42,
    ) -> None:
        if num_subjects < 5:
            raise ValueError("num_subjects should be at least 5 for LOSO train/val/test.")
        if samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be positive.")
        if history_steps <= 0 or future_steps <= 0:
            raise ValueError("history_steps and future_steps must be positive.")

        generator = torch.Generator().manual_seed(seed)
        cgm_list: List[Tensor] = []
        physio_list: List[Tensor] = []
        future_list: List[Tensor] = []
        subject_ids: List[str] = []

        future_axis = torch.arange(1, future_steps + 1, dtype=torch.float32)

        for subject_idx in range(num_subjects):
            subject_id = f"{subject_idx + 1:03d}"
            subject_baseline = 95.0 + subject_idx * 0.8

            for _ in range(samples_per_subject):
                cgm_noise = torch.randn(history_steps, generator=generator) * 1.5
                cgm_trend = torch.cumsum(cgm_noise, dim=0)
                cgm = subject_baseline + cgm_trend
                cgm = cgm.unsqueeze(-1)

                physio = torch.randn(
                    num_physio_nodes,
                    history_steps,
                    generator=generator,
                )

                recent_slope = cgm[-1, 0] - cgm[-6:, 0].mean()
                physio_effect = 0.4 * physio[0].mean() - 0.2 * physio[1].mean()
                future_noise = torch.randn(future_steps, generator=generator) * 1.0
                future = (
                    cgm[-1, 0]
                    + 0.08 * future_axis * recent_slope
                    + physio_effect
                    + future_noise
                )

                cgm_list.append(cgm.float())
                physio_list.append(physio.float())
                future_list.append(future.float())
                subject_ids.append(subject_id)

        self.cgm = torch.stack(cgm_list, dim=0)
        self.physio = torch.stack(physio_list, dim=0)
        self.future_glucose = torch.stack(future_list, dim=0)
        self.subject_ids = subject_ids
        self.history_steps = history_steps
        self.num_physio_nodes = num_physio_nodes
        self.future_steps = future_steps

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        return {
            "cgm": self.cgm[index],
            "physio": self.physio[index],
            "future_glucose": self.future_glucose[index],
            "subject_id": self.subject_ids[index],
        }


class ProcessedBigIdeasDataset(Dataset):
    """Dataset for real preprocessed BIG IDEAs windows."""

    def __init__(self, data_dir: str | Path, required_future_steps: int = 12) -> None:
        self.data_dir = Path(data_dir)
        self.npz_path = self.data_dir / "big_ideas_windows.npz"
        self.metadata_path = self.data_dir / "window_metadata.csv"

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ file: {self.npz_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}")

        arrays = np.load(self.npz_path)
        self.cgm = torch.from_numpy(arrays["X_cgm"]).float()
        self.physio = torch.from_numpy(arrays["X_physio"]).float()
        self.future_glucose = torch.from_numpy(arrays["Y"]).float()
        metadata = pd.read_csv(self.metadata_path, dtype={"subject_id": str})

        self._validate_storage(metadata=metadata, required_future_steps=required_future_steps)

        self.subject_ids = metadata["subject_id"].tolist()
        self.history_steps = int(self.cgm.shape[1])
        self.num_physio_nodes = int(self.physio.shape[1])
        self.future_steps = int(self.future_glucose.shape[1])

    def _validate_storage(self, metadata: pd.DataFrame, required_future_steps: int) -> None:
        if self.cgm.ndim != 3:
            raise ValueError(f"X_cgm must be [N, T, 1], got {tuple(self.cgm.shape)}")
        if self.physio.ndim != 3:
            raise ValueError(f"X_physio must be [N, nodes, T], got {tuple(self.physio.shape)}")
        if self.future_glucose.ndim != 2:
            raise ValueError(f"Y must be [N, future_steps], got {tuple(self.future_glucose.shape)}")
        if self.cgm.shape[0] != self.physio.shape[0] or self.cgm.shape[0] != self.future_glucose.shape[0]:
            raise ValueError("X_cgm, X_physio, and Y must have the same sample count.")
        if self.cgm.shape[0] != len(metadata):
            raise ValueError("Tensor sample count and metadata row count do not match.")
        if self.cgm.shape[2] != 1:
            raise ValueError(f"Expected CGM feature dim 1, got {self.cgm.shape[2]}.")
        if self.physio.shape[2] != self.cgm.shape[1]:
            raise ValueError("X_physio time dimension must match X_cgm history_steps.")
        if self.future_glucose.shape[1] < required_future_steps:
            raise ValueError(
                "Y does not contain enough future steps for 15/30/45/60min prediction. "
                f"Need at least {required_future_steps}, got {self.future_glucose.shape[1]}. "
                "Regenerate with: .venv/bin/python preprocess_big_ideas.py "
                "--horizon-steps 12 --output-dir processed_big_ideas_60min"
            )
        if torch.isnan(self.cgm).any() or torch.isnan(self.physio).any() or torch.isnan(self.future_glucose).any():
            raise ValueError("Processed tensors contain NaN values.")
        required_columns = {"subject_id", "start_time", "target_start_time"}
        missing = required_columns.difference(metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        return {
            "cgm": self.cgm[index],
            "physio": self.physio[index],
            "future_glucose": self.future_glucose[index],
            "subject_id": self.subject_ids[index],
        }


class StandardizedGlucoseDataset(Dataset):
    """Dataset wrapper that applies fold-specific standardization without data leakage."""

    def __init__(self, base_dataset: Dataset, standardizer: DataStandardizer) -> None:
        self.base_dataset = base_dataset
        self.standardizer = standardizer
        self.subject_ids = getattr(base_dataset, "subject_ids")
        self.history_steps = getattr(base_dataset, "history_steps")
        self.num_physio_nodes = getattr(base_dataset, "num_physio_nodes")
        self.future_steps = getattr(base_dataset, "future_steps")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        item = self.base_dataset[index]
        cgm = item["cgm"]
        physio = item["physio"]
        future_glucose = item["future_glucose"]
        if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(future_glucose, Tensor):
            raise TypeError("Base dataset must return Tensor values for cgm, physio, and future_glucose.")

        return {
            "cgm": self.standardizer.transform_cgm(cgm),
            "physio": self.standardizer.transform_physio(physio),
            "future_glucose": self.standardizer.transform_future_glucose(future_glucose),
            "future_glucose_raw": future_glucose,
            "subject_id": item["subject_id"],
        }


def fit_standardizer(dataset: Dataset, train_indices: Sequence[int], eps: float = 1e-6) -> DataStandardizer:
    """Fit fold-specific statistics using training samples only."""
    if not train_indices:
        raise ValueError("train_indices must not be empty.")

    cgm_values: List[Tensor] = []
    future_values: List[Tensor] = []
    physio_values: List[Tensor] = []

    for index in train_indices:
        item = dataset[int(index)]
        cgm = item["cgm"]
        physio = item["physio"]
        future_glucose = item["future_glucose"]
        if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(future_glucose, Tensor):
            raise TypeError("Dataset must return Tensor values for cgm, physio, and future_glucose.")
        cgm_values.append(cgm.reshape(-1))
        future_values.append(future_glucose.reshape(-1))
        physio_values.append(physio)

    glucose_flat = torch.cat([torch.cat(cgm_values), torch.cat(future_values)], dim=0).float()
    physio_stack = torch.stack(physio_values, dim=0).float()

    if not torch.isfinite(glucose_flat).all() or not torch.isfinite(physio_stack).all():
        raise ValueError("Cannot fit standardizer on NaN/Inf values.")

    glucose_mean = glucose_flat.mean()
    glucose_std = glucose_flat.std(unbiased=False).clamp_min(eps)

    # Per-node statistics across training samples and time: physio [B_train, N, T] -> [N, 1].
    physio_mean = physio_stack.mean(dim=(0, 2), keepdim=False).unsqueeze(-1)
    physio_std = physio_stack.std(dim=(0, 2), unbiased=False, keepdim=False).unsqueeze(-1).clamp_min(eps)

    return DataStandardizer(
        glucose_mean=glucose_mean,
        glucose_std=glucose_std,
        physio_mean=physio_mean,
        physio_std=physio_std,
        eps=eps,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loso_splits(
    subject_ids: Sequence[str],
    num_val_subjects: int = 3,
    seed: int = 42,
) -> List[LOSOSplit]:
    """Build LOSO-CV folds without subject-level leakage."""
    unique_subjects = sorted(set(subject_ids))
    if len(unique_subjects) < num_val_subjects + 2:
        raise ValueError("Not enough unique subjects for LOSO train/val/test splits.")

    splits: List[LOSOSplit] = []
    for fold_id, test_subject in enumerate(unique_subjects, start=1):
        remaining = [subject for subject in unique_subjects if subject != test_subject]
        rng = random.Random(seed + fold_id)
        val_subjects = sorted(rng.sample(remaining, k=num_val_subjects))
        val_set = set(val_subjects)
        train_subjects = [subject for subject in remaining if subject not in val_set]

        train_set = set(train_subjects)
        train_indices = [
            index for index, subject in enumerate(subject_ids) if subject in train_set
        ]
        val_indices = [
            index for index, subject in enumerate(subject_ids) if subject in val_set
        ]
        test_indices = [
            index for index, subject in enumerate(subject_ids) if subject == test_subject
        ]

        splits.append(
            LOSOSplit(
                fold_id=fold_id,
                test_subject=test_subject,
                val_subjects=val_subjects,
                train_subjects=train_subjects,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )
    return splits


def validate_batch_shapes(batch: Dict[str, Tensor | List[str]], model: ST_MSFFNet) -> None:
    cgm = batch["cgm"]
    physio = batch["physio"]
    future_glucose = batch["future_glucose"]
    if not isinstance(cgm, Tensor) or not isinstance(physio, Tensor) or not isinstance(future_glucose, Tensor):
        raise TypeError("Batch must contain Tensor values for cgm, physio, and future_glucose.")
    if cgm.ndim != 3:
        raise ValueError(f"Batch cgm must be [B, T, 1], got {tuple(cgm.shape)}")
    if physio.ndim != 3:
        raise ValueError(f"Batch physio must be [B, N, T], got {tuple(physio.shape)}")
    if future_glucose.ndim != 2:
        raise ValueError(
            f"Batch future_glucose must be [B, future_steps], got {tuple(future_glucose.shape)}"
        )
    if future_glucose.shape[1] < model.prediction_head.required_horizon_steps:
        raise ValueError(
            "future_glucose is too short for 60min prediction: "
            f"need {model.prediction_head.required_horizon_steps}, got {future_glucose.shape[1]}"
        )
    if "future_glucose_raw" in batch:
        future_glucose_raw = batch["future_glucose_raw"]
        if not isinstance(future_glucose_raw, Tensor):
            raise TypeError("future_glucose_raw must be a Tensor when present.")
        if future_glucose_raw.shape != future_glucose.shape:
            raise ValueError(
                "future_glucose_raw shape must match future_glucose shape: "
                f"{tuple(future_glucose_raw.shape)} vs {tuple(future_glucose.shape)}"
            )


def move_batch_to_device(
    batch: Dict[str, Tensor | List[str]],
    device: torch.device,
) -> Dict[str, Tensor | List[str]]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: ST_MSFFNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, raw_batch in enumerate(loader):
        try:
            batch = move_batch_to_device(raw_batch, device)
            validate_batch_shapes(batch, model)

            cgm = batch["cgm"]
            physio = batch["physio"]
            future_glucose = batch["future_glucose"]
            assert isinstance(cgm, Tensor)
            assert isinstance(physio, Tensor)
            assert isinstance(future_glucose, Tensor)

            pred = model(cgm, physio)
            target = model.select_targets(future_glucose)
            assert pred.shape == target.shape

            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            batch_size = cgm.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Training failed at batch {batch_idx}: {exc}") from exc

    return total_loss / max(total_samples, 1)


def compute_regression_metrics(pred: Tensor, target: Tensor, mape_eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute regression metrics for glucose prediction.

    Args:
        pred: Tensor [num_samples, num_horizons]
        target: Tensor [num_samples, num_horizons]

    Returns:
        MAE, MAPE(%), RMSE, R2, and Clarke EGA zone percentages.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.numel() == 0:
        raise ValueError("Cannot compute metrics on empty tensors.")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("pred or target contains NaN/Inf values.")

    pred_flat = pred.detach().float().reshape(-1).cpu()
    target_flat = target.detach().float().reshape(-1).cpu()
    error = pred_flat - target_flat

    mae = error.abs().mean()
    rmse = torch.sqrt(error.square().mean())
    denominator = target_flat.abs().clamp_min(mape_eps)
    mape = (error.abs() / denominator).mean() * 100.0

    ss_res = error.square().sum()
    target_centered = target_flat - target_flat.mean()
    ss_tot = target_centered.square().sum()
    r2 = 1.0 - ss_res / ss_tot.clamp_min(mape_eps)

    metrics = {
        "mae": float(mae.item()),
        "mape": float(mape.item()),
        "rmse": float(rmse.item()),
        "r2": float(r2.item()),
    }
    metrics.update(clarke_error_grid_percentages(reference=target_flat, prediction=pred_flat))
    return metrics


def clarke_error_grid_percentages(reference: Tensor, prediction: Tensor) -> Dict[str, float]:
    """
    Clarke Error Grid Analysis (EGA) zone percentages.

    Args:
        reference: Tensor [M], reference glucose values in mg/dL.
        prediction: Tensor [M], predicted glucose values in mg/dL.

    Returns:
        Percentages and counts for Zone A-E.

    Zone convention:
        x-axis is reference glucose, y-axis is predicted glucose.
        Zone A is clinically accurate, Zone B is benign error, and Zone C-E
        indicate increasing clinical risk.
    """
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference and prediction shape mismatch: {tuple(reference.shape)} vs {tuple(prediction.shape)}"
        )
    if reference.ndim != 1:
        raise ValueError(f"reference and prediction must be flattened 1D tensors, got {reference.ndim}D")
    if reference.numel() == 0:
        raise ValueError("Cannot compute Clarke EGA on empty tensors.")
    if not torch.isfinite(reference).all() or not torch.isfinite(prediction).all():
        raise ValueError("reference or prediction contains NaN/Inf values.")

    ref = reference.float()
    pred = prediction.float()
    total = int(ref.numel())
    zones = torch.full((total,), fill_value=1, dtype=torch.long)  # default Zone B -> index 1

    zone_a = ((ref <= 70.0) & (pred <= 70.0)) | ((pred - ref).abs() <= 0.20 * ref)
    zones[zone_a] = 0

    unassigned = ~zone_a
    zone_e = (((ref <= 70.0) & (pred >= 180.0)) | ((ref >= 180.0) & (pred <= 70.0))) & unassigned
    zones[zone_e] = 4

    unassigned = unassigned & ~zone_e
    zone_c = (
        ((ref >= 70.0) & (ref <= 290.0) & (pred >= ref + 110.0))
        | ((ref >= 130.0) & (ref <= 180.0) & (pred <= (7.0 / 5.0) * ref - 182.0))
    ) & unassigned
    zones[zone_c] = 2

    unassigned = unassigned & ~zone_c
    zone_d = (
        ((ref >= 240.0) & (pred >= 70.0) & (pred <= 180.0))
        | ((ref <= 175.0 / 3.0) & (pred >= 70.0) & (pred <= 180.0))
        | ((ref >= 175.0 / 3.0) & (ref <= 70.0) & (pred >= (6.0 / 5.0) * ref))
    ) & unassigned
    zones[zone_d] = 3

    counts = torch.bincount(zones, minlength=5).float()
    percentages = counts / float(total) * 100.0

    zone_names = ["a", "b", "c", "d", "e"]
    metrics: Dict[str, float] = {}
    for idx, name in enumerate(zone_names):
        metrics[f"ega_zone_{name}_pct"] = float(percentages[idx].item())
        metrics[f"ega_zone_{name}_count"] = float(counts[idx].item())
    return metrics


def format_metrics(prefix: str, loss: float, metrics: Dict[str, float]) -> str:
    """Format regression and Clarke EGA metrics for compact logging."""
    return (
        f"{prefix}_loss={loss:.4f} "
        f"{prefix}_mae={metrics['mae']:.4f} "
        f"{prefix}_mape={metrics['mape']:.2f}% "
        f"{prefix}_rmse={metrics['rmse']:.4f} "
        f"{prefix}_r2={metrics['r2']:.4f} "
        f"{prefix}_EGA[A/B/C/D/E]="
        f"{metrics['ega_zone_a_pct']:.2f}/"
        f"{metrics['ega_zone_b_pct']:.2f}/"
        f"{metrics['ega_zone_c_pct']:.2f}/"
        f"{metrics['ega_zone_d_pct']:.2f}/"
        f"{metrics['ega_zone_e_pct']:.2f}%"
    )


@torch.no_grad()
def evaluate(
    model: ST_MSFFNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    standardizer: Optional[DataStandardizer] = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_pred: List[Tensor] = []
    all_target: List[Tensor] = []

    for batch_idx, raw_batch in enumerate(loader):
        try:
            batch = move_batch_to_device(raw_batch, device)
            validate_batch_shapes(batch, model)

            cgm = batch["cgm"]
            physio = batch["physio"]
            future_glucose = batch["future_glucose"]
            assert isinstance(cgm, Tensor)
            assert isinstance(physio, Tensor)
            assert isinstance(future_glucose, Tensor)

            pred = model(cgm, physio)
            target = model.select_targets(future_glucose)
            assert pred.shape == target.shape

            loss = criterion(pred, target)
            if standardizer is not None:
                future_glucose_raw = batch.get("future_glucose_raw")
                if not isinstance(future_glucose_raw, Tensor):
                    raise TypeError("Standardized evaluation requires future_glucose_raw in the batch.")
                pred_for_metrics = standardizer.inverse_glucose(pred)
                target_for_metrics = model.select_targets(future_glucose_raw)
            else:
                pred_for_metrics = pred
                target_for_metrics = target
            assert pred_for_metrics.shape == target_for_metrics.shape

            batch_size = cgm.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_pred.append(pred_for_metrics.detach().cpu())
            all_target.append(target_for_metrics.detach().cpu())
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Evaluation failed at batch {batch_idx}: {exc}") from exc

    if not all_pred:
        raise RuntimeError("Evaluation loader produced no batches.")

    pred_tensor = torch.cat(all_pred, dim=0)
    target_tensor = torch.cat(all_target, dim=0)
    metrics = compute_regression_metrics(pred=pred_tensor, target=target_tensor)
    return total_loss / max(total_samples, 1), metrics


def make_loader(dataset: Dataset, indices: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def run_loso_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    if args.data_dir is None:
        dataset = DummyPhysioNetDataset(
            num_subjects=args.num_subjects,
            samples_per_subject=args.samples_per_subject,
            history_steps=args.history_steps,
            num_physio_nodes=args.num_physio_nodes,
            future_steps=args.future_steps,
            seed=args.seed,
        )
        print("[INFO] Using dummy data. Pass --data-dir processed_big_ideas_60min for real data.")
    else:
        dataset = ProcessedBigIdeasDataset(
            data_dir=args.data_dir,
            required_future_steps=12,
        )
        print(
            f"[INFO] Using real processed data from {args.data_dir}: "
            f"samples={len(dataset)}, subjects={len(set(dataset.subject_ids))}, "
            f"history_steps={dataset.history_steps}, future_steps={dataset.future_steps}"
        )

    history_steps = int(dataset.history_steps)
    num_physio_nodes = int(dataset.num_physio_nodes)
    splits = build_loso_splits(
        subject_ids=dataset.subject_ids,
        num_val_subjects=args.num_val_subjects,
        seed=args.seed,
    )
    if args.fold_limit is not None:
        splits = splits[: args.fold_limit]

    criterion = nn.SmoothL1Loss()
    fold_metrics: List[Dict[str, float]] = []

    for split in splits:
        print(
            f"\n[Fold {split.fold_id:02d}] test={split.test_subject}, "
            f"val={split.val_subjects}, train_subjects={len(split.train_subjects)}"
        )

        if args.no_standardize:
            fold_dataset: Dataset = dataset
            standardizer = None
            print("  [INFO] Standardization disabled.")
        else:
            standardizer = fit_standardizer(dataset, split.train_indices)
            fold_dataset = StandardizedGlucoseDataset(dataset, standardizer)
            print(
                "  [INFO] Standardization fitted on train subjects only: "
                f"glucose_mean={float(standardizer.glucose_mean):.4f}, "
                f"glucose_std={float(standardizer.glucose_std):.4f}"
            )

        train_loader = make_loader(fold_dataset, split.train_indices, args.batch_size, shuffle=True)
        val_loader = make_loader(fold_dataset, split.val_indices, args.batch_size, shuffle=False)
        test_loader = make_loader(fold_dataset, split.test_indices, args.batch_size, shuffle=False)

        model = ST_MSFFNet(
            history_steps=history_steps,
            cgm_dim=1,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
            horizon_minutes=(15, 30, 45, 60),
            sampling_interval_minutes=5,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Tensor]] = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
            )
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                standardizer=standardizer,
            )
            scheduler.step()

            print(
                f"  epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} "
                f"{format_metrics('val', val_loss, val_metrics)} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  early_stop at epoch={epoch:02d}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        test_loss, test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            standardizer=standardizer,
        )
        print(f"  {format_metrics('test', test_loss, test_metrics)}")
        fold_metrics.append(
            {
                "loss": test_loss,
                **test_metrics,
            }
        )

    if fold_metrics:
        avg_metrics = {
            key: sum(item[key] for item in fold_metrics) / len(fold_metrics)
            for key in fold_metrics[0].keys()
        }
        avg_loss = avg_metrics.pop("loss")
        print(f"\n[LOSO Summary] folds={len(fold_metrics)} {format_metrics('avg', avg_loss, avg_metrics)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ST_MSFFNet LOSO-CV training pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Processed data directory containing big_ideas_windows.npz and window_metadata.csv.",
    )
    parser.add_argument("--num-subjects", type=int, default=16)
    parser.add_argument("--samples-per-subject", type=int, default=16)
    parser.add_argument("--num-val-subjects", type=int, default=3)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--num-physio-nodes", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable fold-wise train-only standardization. Standardization is enabled by default.",
    )
    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
        help="Run only the first K LOSO folds for quick smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_loso_training(parse_args())
