"""Dataset implementations for GlucoPrism experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ProcessedBigIdeasDataset(Dataset):
    """Load preprocessed BIG IDEAs windows from NPZ and metadata CSV files."""

    def __init__(self, data_dir: str | Path, required_future_steps: int = 12) -> None:
        self.data_dir = Path(data_dir)
        self.npz_path = self.data_dir / "big_ideas_windows.npz"
        self.metadata_path = self.data_dir / "window_metadata.csv"

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ file: {self.npz_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata CSV: {self.metadata_path}")

        arrays = np.load(self.npz_path)
        self.cgm = torch.from_numpy(arrays["X_cgm"]).float()
        self.physio = torch.from_numpy(arrays["X_physio"]).float()
        self.future_glucose = torch.from_numpy(arrays["Y"]).float()
        self.metadata = pd.read_csv(self.metadata_path, dtype={"subject_id": str})
        self._validate(required_future_steps)

        self.subject_ids: List[str] = self.metadata["subject_id"].tolist()
        self.history_steps = int(self.cgm.shape[1])
        self.num_physio_nodes = int(self.physio.shape[1])
        self.future_steps = int(self.future_glucose.shape[1])

    def _validate(self, required_future_steps: int) -> None:
        if self.cgm.ndim != 3:
            raise ValueError(f"X_cgm must be [N, T, 1], got {tuple(self.cgm.shape)}")
        if self.physio.ndim != 3:
            raise ValueError(f"X_physio must be [N, nodes, T], got {tuple(self.physio.shape)}")
        if self.future_glucose.ndim != 2:
            raise ValueError(f"Y must be [N, future_steps], got {tuple(self.future_glucose.shape)}")
        if self.cgm.shape[2] != 1:
            raise ValueError(f"CGM feature dim must be 1, got {self.cgm.shape[2]}")
        if self.physio.shape[2] != self.cgm.shape[1]:
            raise ValueError("X_physio time dimension must match X_cgm history length.")
        if self.future_glucose.shape[1] < required_future_steps:
            raise ValueError(
                "Y does not contain enough future steps for 15/30/45/60min prediction. "
                f"Need at least {required_future_steps}, got {self.future_glucose.shape[1]}."
            )
        if not (len(self.cgm) == len(self.physio) == len(self.future_glucose) == len(self.metadata)):
            raise ValueError("X_cgm, X_physio, Y, and metadata must have equal sample counts.")
        if torch.isnan(self.cgm).any() or torch.isnan(self.physio).any() or torch.isnan(self.future_glucose).any():
            raise ValueError("Processed tensors contain NaN values.")
        required = {"subject_id", "start_time", "target_start_time"}
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        return {
            "cgm": self.cgm[index],  # [history_steps, 1]
            "physio": self.physio[index],  # [num_nodes, history_steps]
            "future_glucose": self.future_glucose[index],  # [future_steps]
            "subject_id": self.subject_ids[index],
        }


class DummyPhysioNetDataset(Dataset):
    """Small synthetic dataset that mimics aligned PhysioNet windows for smoke tests."""

    def __init__(
        self,
        num_subjects: int = 16,
        samples_per_subject: int = 8,
        history_steps: int = 24,
        num_physio_nodes: int = 6,
        future_steps: int = 12,
        seed: int = 42,
    ) -> None:
        if num_subjects < 5:
            raise ValueError("num_subjects must be at least 5 for LOSO-CV.")
        generator = torch.Generator().manual_seed(seed)
        cgm_list: List[Tensor] = []
        physio_list: List[Tensor] = []
        future_list: List[Tensor] = []
        subject_ids: List[str] = []
        future_axis = torch.arange(1, future_steps + 1, dtype=torch.float32)

        for subject_idx in range(num_subjects):
            subject_id = f"{subject_idx + 1:03d}"
            baseline = 95.0 + 0.8 * subject_idx
            for _ in range(samples_per_subject):
                cgm_noise = torch.randn(history_steps, generator=generator) * 1.5
                cgm = (baseline + torch.cumsum(cgm_noise, dim=0)).unsqueeze(-1)
                physio = torch.randn(num_physio_nodes, history_steps, generator=generator)
                recent_slope = cgm[-1, 0] - cgm[-6:, 0].mean()
                physio_effect = 0.4 * physio[0].mean() - 0.2 * physio[1].mean()
                future = cgm[-1, 0] + 0.08 * future_axis * recent_slope + physio_effect
                future = future + torch.randn(future_steps, generator=generator)
                cgm_list.append(cgm.float())
                physio_list.append(physio.float())
                future_list.append(future.float())
                subject_ids.append(subject_id)

        self.cgm = torch.stack(cgm_list)
        self.physio = torch.stack(physio_list)
        self.future_glucose = torch.stack(future_list)
        self.subject_ids = subject_ids
        self.history_steps = history_steps
        self.num_physio_nodes = num_physio_nodes
        self.future_steps = future_steps

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        return {
            "cgm": self.cgm[index],  # [history_steps, 1]
            "physio": self.physio[index],  # [num_nodes, history_steps]
            "future_glucose": self.future_glucose[index],  # [future_steps]
            "subject_id": self.subject_ids[index],
        }
