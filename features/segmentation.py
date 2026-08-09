"""Time-series segmentation helpers."""

from __future__ import annotations

from typing import List

import pandas as pd


def split_by_time_gaps(df: pd.DataFrame, max_gap: pd.Timedelta) -> List[pd.DataFrame]:
    """Split a time-indexed DataFrame into contiguous segments."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must use a DatetimeIndex.")
    if df.empty:
        return []
    gaps = df.index.to_series().diff().fillna(pd.Timedelta(0)) > max_gap
    segment_id = gaps.cumsum()
    return [part for _, part in df.groupby(segment_id)]
