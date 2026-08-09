"""Time alignment helpers for multimodal wearable and CGM signals."""

from __future__ import annotations

import pandas as pd


def align_to_cgm_axis(cgm: pd.Series, features: dict[str, pd.Series], freq: str = "5min") -> pd.DataFrame:
    """Align all features to the CGM 5-minute axis."""
    if cgm.empty:
        raise ValueError("cgm series must not be empty.")
    index = pd.date_range(cgm.index.min(), cgm.index.max(), freq=freq)
    out = pd.DataFrame(index=index)
    out["glucose"] = cgm.reindex(index)
    for name, series in features.items():
        out[name] = series.reindex(index)
    return out


def resample_mean(series: pd.Series, freq: str = "5min") -> pd.Series:
    """Resample one continuous signal to a fixed time grid using mean aggregation."""
    return series.sort_index().resample(freq).mean()
