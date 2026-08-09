"""Sliding-window builders for multimodal time series."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def make_sliding_windows(
    aligned: pd.DataFrame,
    cgm_col: str,
    physio_cols: Sequence[str],
    history_steps: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create X_cgm [N,T,1], X_physio [N,M,T], Y [N,H] windows."""
    x_cgm, x_physio, y = [], [], []
    total = history_steps + horizon_steps
    for start in range(len(aligned) - total + 1):
        hist = slice(start, start + history_steps)
        fut = slice(start + history_steps, start + total)
        x_cgm.append(aligned[[cgm_col]].iloc[hist].to_numpy(dtype=np.float32))
        x_physio.append(aligned[list(physio_cols)].iloc[hist].to_numpy(dtype=np.float32).T)
        y.append(aligned[cgm_col].iloc[fut].to_numpy(dtype=np.float32))
    return np.stack(x_cgm), np.stack(x_physio), np.stack(y)
