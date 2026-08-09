"""Experiment logging utilities with TensorBoard fallback to CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


class ExperimentLogger:
    """Log metrics to TensorBoard when available and always to CSV."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self._fieldnames: Optional[list[str]] = None
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            print(f"[INFO] TensorBoard logging enabled: {self.output_dir / 'tensorboard'}")
        except Exception as exc:
            print(f"[INFO] TensorBoard unavailable ({exc}); using CSV logging only.")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, float(value), step)
        row = {"step": step, **metrics}
        fieldnames = list(row.keys())
        write_header = not self.csv_path.exists() or self._fieldnames != fieldnames
        self._fieldnames = fieldnames
        with self.csv_path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def write_table(self, filename: str, rows: Iterable[Dict[str, object]]) -> Path:
        """Write a rectangular CSV table under the experiment output directory."""
        path = self.output_dir / filename
        rows = list(rows)
        if not rows:
            return path
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
