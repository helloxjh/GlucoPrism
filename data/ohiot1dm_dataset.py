"""Processed OhioT1DM dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ProcessedOhioT1DMDataset(Dataset):
    """Load preprocessed OhioT1DM windows with configurable physiological nodes."""

    dataset_name = "ohiot1dm"

    def __init__(self, data_dir: str | Path, required_future_steps: int = 12) -> None:
        self.data_dir = Path(data_dir)
        self.npz_path = self.data_dir / "ohiot1dm_windows.npz"
        self.metadata_path = self.data_dir / "window_metadata.csv"

        if not self.npz_path.is_file():
            raise FileNotFoundError(f"Missing NPZ file: {self.npz_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata CSV: {self.metadata_path}")

        arrays = np.load(self.npz_path, allow_pickle=False)
        self.cgm = torch.from_numpy(arrays["X_cgm"]).float()
        self.physio = torch.from_numpy(arrays["X_physio"]).float()
        self.future_glucose = torch.from_numpy(arrays["Y"]).float()
        self.node_names: Tuple[str, ...] = tuple(
            str(value) for value in arrays["physio_features"].tolist()
        )
        self.metadata = pd.read_csv(self.metadata_path, dtype={"subject_id": str})
        self._validate(required_future_steps)

        self.subject_ids: List[str] = self.metadata["subject_id"].tolist()
        self.history_steps = int(self.cgm.shape[1])
        self.num_physio_nodes = int(self.physio.shape[1])
        self.future_steps = int(self.future_glucose.shape[1])
        self.A_prior = None

    def _validate(self, required_future_steps: int) -> None:
        if self.cgm.ndim != 3 or self.cgm.shape[2] != 1:
            raise ValueError(f"X_cgm must be [N,T,1], got {tuple(self.cgm.shape)}")
        if self.physio.ndim != 3:
            raise ValueError(f"X_physio must be [N,nodes,T], got {tuple(self.physio.shape)}")
        if self.future_glucose.ndim != 2:
            raise ValueError(f"Y must be [N,future_steps], got {tuple(self.future_glucose.shape)}")
        if self.physio.shape[2] != self.cgm.shape[1]:
            raise ValueError("X_physio time dimension must match X_cgm history length.")
        if self.future_glucose.shape[1] < required_future_steps:
            raise ValueError(
                f"Y requires at least {required_future_steps} steps, "
                f"got {self.future_glucose.shape[1]}."
            )
        if len(self.node_names) != self.physio.shape[1]:
            raise ValueError(
                f"physio_features contains {len(self.node_names)} names, "
                f"but X_physio has {self.physio.shape[1]} nodes."
            )
        lengths = {
            len(self.cgm),
            len(self.physio),
            len(self.future_glucose),
            len(self.metadata),
        }
        if len(lengths) != 1:
            raise ValueError("X_cgm, X_physio, Y, and metadata must have equal sample counts.")
        tensors = (self.cgm, self.physio, self.future_glucose)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Processed OhioT1DM tensors contain NaN or Inf values.")
        required_columns = {
            "subject_id",
            "source_split",
            "source_file",
            "start_time",
            "target_start_time",
        }
        missing = required_columns.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        return {
            "cgm": self.cgm[index],
            "physio": self.physio[index],
            "future_glucose": self.future_glucose[index],
            "subject_id": self.subject_ids[index],
        }
