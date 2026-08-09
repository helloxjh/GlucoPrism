"""Dataset definitions for processed BIG IDEAs glucose-wearable windows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GlucoseWearableDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed GlucoPrism windows.

    Expected NPZ keys:
    - X_cgm: Tensor-like array [num_samples, 24, 1]
    - X_physio: Tensor-like array [num_samples, 6, 24]
    - Y: Tensor-like array [num_samples, 6]

    Expected metadata CSV columns:
    - subject_id
    - start_time
    - target_start_time
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        metadata_path: Union[str, Path],
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.metadata_path = Path(metadata_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path}")

        arrays = np.load(self.npz_path)
        self.x_cgm = torch.from_numpy(arrays["X_cgm"]).float()
        self.x_physio = torch.from_numpy(arrays["X_physio"]).float()
        self.y = torch.from_numpy(arrays["Y"]).float()
        self.metadata = pd.read_csv(self.metadata_path, dtype={"subject_id": str})

        self._validate_storage()

        if indices is None:
            self.indices = torch.arange(len(self.y), dtype=torch.long)
        else:
            self.indices = torch.as_tensor(list(indices), dtype=torch.long)
            if self.indices.numel() == 0:
                raise ValueError("Dataset indices must not be empty.")
            if int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.y):
                raise IndexError("Dataset indices contain values outside the sample range.")

    def _validate_storage(self) -> None:
        if self.x_cgm.ndim != 3:
            raise ValueError(f"X_cgm must be 3D [N, 24, 1], got {tuple(self.x_cgm.shape)}")
        if self.x_physio.ndim != 3:
            raise ValueError(
                f"X_physio must be 3D [N, 6, 24], got {tuple(self.x_physio.shape)}"
            )
        if self.y.ndim != 2:
            raise ValueError(f"Y must be 2D [N, 6], got {tuple(self.y.shape)}")
        if not (len(self.x_cgm) == len(self.x_physio) == len(self.y) == len(self.metadata)):
            raise ValueError("X_cgm, X_physio, Y, and metadata must have the same length.")
        if torch.isnan(self.x_cgm).any() or torch.isnan(self.x_physio).any() or torch.isnan(self.y).any():
            raise ValueError("Dataset tensors contain NaN values.")
        required = {"subject_id", "start_time", "target_start_time"}
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        storage_index = int(self.indices[item])
        return {
            "cgm": self.x_cgm[storage_index],  # Tensor [history_steps=24, cgm_dim=1]
            "physio": self.x_physio[storage_index],  # Tensor [physio_channels=6, history_steps=24]
            "target": self.y[storage_index],  # Tensor [horizon_steps=6]
            "index": torch.tensor(storage_index, dtype=torch.long),
        }
