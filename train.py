"""Minimal training entry point placeholder for GlucoPrism experiments."""

from __future__ import annotations

from pathlib import Path

from data import build_loso_splits
from models import GlucoPrism


def main() -> None:
    metadata_path = Path("processed_big_ideas/window_metadata.csv")
    splits = build_loso_splits(metadata_path=metadata_path, num_val_subjects=3, seed=42)
    model = GlucoPrism(
        history_steps=24,
        horizon_steps=6,
        cgm_input_dim=1,
        physio_channels=6,
        hidden_dim=64,
        dropout=0.1,
    )
    print(f"Prepared {len(splits)} LOSO folds.")
    print(model)


if __name__ == "__main__":
    main()
