"""Thin API wrapper around the robust BIG IDEAs preprocessing script."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from preprocess_big_ideas import PreprocessConfig, process_all_subjects, save_outputs


def preprocess_big_ideas_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    horizon_steps: int = 12,
    history_steps: int = 24,
    subjects: Optional[Sequence[str]] = None,
) -> None:
    """Run raw CSV -> aligned sliding-window preprocessing."""
    cfg = PreprocessConfig(
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    x_cgm, x_physio, y, meta, summary = process_all_subjects(cfg, selected_subjects=subjects)
    save_outputs(cfg, x_cgm, x_physio, y, meta, summary)
