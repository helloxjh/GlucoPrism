"""Missing value handling for aligned time series."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def fill_short_missing_values(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Forward-fill then backward-fill short gaps."""
    out = df.copy()
    out.loc[:, list(columns)] = out.loc[:, list(columns)].ffill().bfill()
    return out


def mark_long_missing_runs(df: pd.DataFrame, columns: Sequence[str], max_gap_steps: int) -> pd.Series:
    """Mark positions that belong to missing runs longer than max_gap_steps."""
    if max_gap_steps <= 0:
        raise ValueError("max_gap_steps must be positive.")
    long_gap = pd.Series(False, index=df.index)
    for col in columns:
        missing = df[col].isna()
        if not missing.any():
            continue
        run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
        run_len = missing.groupby(run_id).transform("sum")
        long_gap |= missing & (run_len > max_gap_steps)
    return long_gap
