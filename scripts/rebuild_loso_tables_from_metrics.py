#!/usr/bin/env python3
"""Rebuild LOSO fold and summary CSV files from an appended metrics.csv log."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latest fold*/test_* rows from metrics.csv and rebuild LOSO tables."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics.csv.")
    parser.add_argument("--expected-folds", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    fold_records: Dict[int, Dict[str, float]] = {}
    metric_order: List[str] = []
    current_header: List[str] | None = None

    with metrics_path.open(newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            if row[0] == "step":
                current_header = row
                continue
            if current_header is None:
                continue

            values = dict(zip(current_header, row))
            fold_id = None
            record: Dict[str, float] = {}
            for key, value in values.items():
                if not key.startswith("fold") or "/test_" not in key:
                    continue
                prefix, metric_name = key.split("/test_", 1)
                try:
                    parsed_fold = int(prefix.removeprefix("fold"))
                    parsed_value = float(value)
                except ValueError:
                    continue
                if fold_id is None:
                    fold_id = parsed_fold
                elif fold_id != parsed_fold:
                    continue
                record[metric_name] = parsed_value
                if metric_name not in metric_order:
                    metric_order.append(metric_name)

            if fold_id is not None and record:
                # Keep the latest test row if the log contains interrupted/retried runs.
                fold_records[fold_id] = record

    missing = [fold for fold in range(1, args.expected_folds + 1) if fold not in fold_records]
    if missing:
        print(f"[WARN] Missing completed test folds: {missing}")
    else:
        print(f"[INFO] Found all {args.expected_folds} completed test folds.")

    rows = []
    for fold_id in sorted(fold_records):
        rows.append({"fold": fold_id, **fold_records[fold_id]})

    fold_csv = args.run_dir / "loso_test_metrics.csv"
    write_table(fold_csv, rows, ["fold", *metric_order])
    print(f"[INFO] Wrote {fold_csv}")

    if rows:
        summary_rows = build_summary(rows, metric_order)
        summary_csv = args.run_dir / "loso_summary_metrics.csv"
        write_table(summary_csv, summary_rows, ["stat", *metric_order])
        print(f"[INFO] Wrote {summary_csv}")


def build_summary(rows: List[Dict[str, float]], metric_order: List[str]) -> List[Dict[str, float | str]]:
    mean_row: Dict[str, float | str] = {"stat": "mean"}
    std_row: Dict[str, float | str] = {"stat": "std"}
    for metric in metric_order:
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        mean_row[metric] = mean
        std_row[metric] = math.sqrt(variance)
    return [mean_row, std_row]


def write_table(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
