"""Physiological signal feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_acc_l2(acc: pd.DataFrame, x_col: str = "acc_x", y_col: str = "acc_y", z_col: str = "acc_z") -> pd.Series:
    """Compute ACC movement intensity as L2 norm."""
    value = np.sqrt(acc[x_col].astype(float) ** 2 + acc[y_col].astype(float) ** 2 + acc[z_col].astype(float) ** 2)
    return pd.Series(value, index=acc.index, name="acc_l2")


def aggregate_signal_mean(signal: pd.Series, freq: str = "5min") -> pd.Series:
    """Aggregate one wearable signal to 5-minute mean values."""
    return signal.sort_index().resample(freq).mean()


def expected_physio_order() -> tuple[str, ...]:
    """Channel order used by preprocess_big_ideas.py."""
    return ("acc_l2", "eda", "temp", "hr", "bvp", "ibi")
