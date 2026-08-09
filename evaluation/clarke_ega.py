"""Clarke Error Grid Analysis."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def clarke_error_grid_zones(reference: Tensor, prediction: Tensor) -> Tensor:
    """Classify paired values into Clarke zones encoded as A=0 through E=4."""
    ref = reference.detach().float().reshape(-1).cpu()
    pred = prediction.detach().float().reshape(-1).cpu()
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction shapes must match.")
    if ref.numel() == 0:
        raise ValueError("Cannot compute EGA on empty tensors.")
    zones = torch.full((ref.numel(),), 1, dtype=torch.long)
    zone_a = ((ref <= 70.0) & (pred <= 70.0)) | ((pred - ref).abs() <= 0.20 * ref)
    zones[zone_a] = 0
    remaining = ~zone_a
    zone_e = (((ref <= 70.0) & (pred >= 180.0)) | ((ref >= 180.0) & (pred <= 70.0))) & remaining
    zones[zone_e] = 4
    remaining = remaining & ~zone_e
    zone_c = (
        ((ref >= 70.0) & (ref <= 290.0) & (pred >= ref + 110.0))
        | ((ref >= 130.0) & (ref <= 180.0) & (pred <= (7.0 / 5.0) * ref - 182.0))
    ) & remaining
    zones[zone_c] = 2
    remaining = remaining & ~zone_c
    zone_d = (
        ((ref >= 240.0) & (pred >= 70.0) & (pred <= 180.0))
        | ((ref <= 175.0 / 3.0) & (pred >= 70.0) & (pred <= 180.0))
        | ((ref >= 175.0 / 3.0) & (ref <= 70.0) & (pred >= (6.0 / 5.0) * ref))
    ) & remaining
    zones[zone_d] = 3
    return zones


def clarke_error_grid_percentages(reference: Tensor, prediction: Tensor) -> Dict[str, float]:
    """Return Zone A-E percentages using reference glucose as x-axis."""
    zones = clarke_error_grid_zones(reference, prediction)
    counts = torch.bincount(zones, minlength=5).float()
    pct = counts / float(zones.numel()) * 100.0
    names = ["a", "b", "c", "d", "e"]
    out: Dict[str, float] = {}
    for idx, name in enumerate(names):
        out[f"ega_zone_{name}_pct"] = float(pct[idx].item())
        out[f"ega_zone_{name}_count"] = float(counts[idx].item())
    return out
