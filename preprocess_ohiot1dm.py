#!/usr/bin/env python3
"""Preprocess OhioT1DM XML recordings into aligned forecasting windows."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.missing import mark_long_missing_runs


DEFAULT_DATA_ROOT = Path("OhioT1DM")
DEFAULT_OUTPUT_DIR = Path("processed_ohiot1dm_60min")
DEFAULT_FREQ = "5min"
DEFAULT_NODE_NAMES = ("activity", "gsr", "skin_temperature", "heart_rate")
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"
GLUCOSE_COL = "glucose"
OBSERVED_CGM_COL = "__observed_glucose__"
LONG_GAP_COL = "__long_gap__"
RECORDING_PATTERN = re.compile(
    r"^(?P<subject>\d+)-ws-(?P<split>training|testing)\.xml$"
)


@dataclass(frozen=True)
class NodeSpec:
    xml_section: str
    aggregation: str
    convert_fahrenheit_to_celsius: bool = False


NODE_SPECS: Dict[str, NodeSpec] = {
    "activity": NodeSpec("basis_steps", "sum"),
    "gsr": NodeSpec("basis_gsr", "mean"),
    "skin_temperature": NodeSpec(
        "basis_skin_temperature", "mean", convert_fahrenheit_to_celsius=True
    ),
    "heart_rate": NodeSpec("basis_heart_rate", "mean"),
}


@dataclass(frozen=True)
class PreprocessConfig:
    data_root: Path
    output_dir: Path
    freq: str = DEFAULT_FREQ
    history_steps: int = 24
    horizon_steps: int = 12
    max_gap_hours: float = 2.0
    node_names: Tuple[str, ...] = DEFAULT_NODE_NAMES
    require_observed_cgm_history: bool = True
    save_aligned_csv: bool = False

    @property
    def max_gap_steps(self) -> int:
        minutes = pd.Timedelta(self.freq).total_seconds() / 60.0
        if minutes <= 0:
            raise ValueError(f"Invalid frequency: {self.freq}")
        return max(1, int(math.floor(self.max_gap_hours * 60.0 / minutes)))


@dataclass(frozen=True)
class Recording:
    subject_id: str
    source_split: str
    path: Path


def discover_recordings(
    data_root: Path,
    selected_subjects: Optional[Sequence[str]] = None,
) -> List[Recording]:
    """Discover complete training/testing pairs and ignore editor temporary files."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"OhioT1DM directory does not exist: {data_root}")
    selected = None if selected_subjects is None else {str(value) for value in selected_subjects}
    recordings: List[Recording] = []
    splits_by_subject: Dict[str, set[str]] = {}
    for path in sorted(data_root.iterdir()):
        if not path.is_file():
            continue
        match = RECORDING_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        subject_id = match.group("subject")
        if selected is not None and subject_id not in selected:
            continue
        source_split = match.group("split")
        recordings.append(Recording(subject_id, source_split, path))
        splits_by_subject.setdefault(subject_id, set()).add(source_split)

    if not recordings:
        raise FileNotFoundError(f"No OhioT1DM XML recordings found under {data_root}")
    if selected is not None:
        missing_subjects = selected.difference(splits_by_subject)
        if missing_subjects:
            raise FileNotFoundError(f"Requested subjects not found: {sorted(missing_subjects)}")
    incomplete = {
        subject: sorted({"training", "testing"}.difference(splits))
        for subject, splits in splits_by_subject.items()
        if splits != {"training", "testing"}
    }
    if incomplete:
        raise ValueError(f"Incomplete OhioT1DM recording pairs: {incomplete}")
    return sorted(recordings, key=lambda item: (item.subject_id, item.source_split))


def _section_series(root: ET.Element, section_name: str) -> pd.Series:
    section = root.find(section_name)
    if section is None:
        raise ValueError(f"XML is missing required section {section_name!r}.")
    timestamps: List[str] = []
    values: List[str] = []
    for event in section.findall("event"):
        timestamp = event.attrib.get("ts")
        value = event.attrib.get("value")
        if timestamp is None or value is None:
            continue
        timestamps.append(timestamp)
        values.append(value)
    if not timestamps:
        raise ValueError(f"XML section {section_name!r} contains no usable events.")

    index = pd.to_datetime(
        pd.Series(timestamps, dtype="string"),
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )
    numeric = pd.to_numeric(pd.Series(values, dtype="string"), errors="coerce")
    frame = pd.DataFrame({"value": numeric.to_numpy()}, index=index)
    frame = frame.loc[frame.index.notna()].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["value"]).sort_index()
    frame = frame.groupby(level=0)["value"].mean().to_frame()
    if frame.empty:
        raise ValueError(f"XML section {section_name!r} has no finite numeric values.")
    return frame["value"]


def _resample(series: pd.Series, freq: str, aggregation: str) -> pd.Series:
    ordered = series.sort_index()
    if aggregation == "mean":
        return ordered.resample(freq).mean()
    if aggregation == "sum":
        return ordered.resample(freq).sum(min_count=1)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def align_recording(recording: Recording, cfg: PreprocessConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse and align one XML recording without crossing its file boundary."""
    root = ET.parse(recording.path).getroot()
    xml_subject = root.attrib.get("id")
    if xml_subject is not None and xml_subject != recording.subject_id:
        raise ValueError(
            f"Subject mismatch for {recording.path}: filename={recording.subject_id}, "
            f"xml={xml_subject}."
        )

    raw_counts: Dict[str, int] = {}
    glucose_raw = _section_series(root, "glucose_level")
    raw_counts[GLUCOSE_COL] = int(len(glucose_raw))
    glucose = _resample(glucose_raw, cfg.freq, "mean")
    full_index = pd.date_range(glucose.index.min(), glucose.index.max(), freq=cfg.freq)
    aligned = pd.DataFrame(index=full_index)
    aligned[GLUCOSE_COL] = glucose.reindex(full_index)
    aligned[OBSERVED_CGM_COL] = aligned[GLUCOSE_COL].notna()

    for node_name in cfg.node_names:
        if node_name not in NODE_SPECS:
            raise ValueError(f"Unknown OhioT1DM physiological node: {node_name}")
        spec = NODE_SPECS[node_name]
        raw = _section_series(root, spec.xml_section)
        raw_counts[node_name] = int(len(raw))
        values = _resample(raw, cfg.freq, spec.aggregation)
        if spec.convert_fahrenheit_to_celsius:
            values = (values - 32.0) * (5.0 / 9.0)
        aligned[node_name] = values.reindex(full_index)

    value_columns = [GLUCOSE_COL, *cfg.node_names]
    aligned[LONG_GAP_COL] = mark_long_missing_runs(
        aligned,
        columns=value_columns,
        max_gap_steps=cfg.max_gap_steps,
    )
    aligned.loc[:, list(cfg.node_names)] = aligned.loc[:, list(cfg.node_names)].ffill(
        limit=cfg.max_gap_steps
    )
    if not cfg.require_observed_cgm_history:
        aligned[GLUCOSE_COL] = aligned[GLUCOSE_COL].ffill(limit=cfg.max_gap_steps)
    aligned["subject_id"] = recording.subject_id
    aligned["source_split"] = recording.source_split
    aligned["source_file"] = recording.path.name
    return aligned, raw_counts


def make_windows(
    aligned: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create leakage-controlled windows from one aligned recording."""
    cgm_values = aligned[[GLUCOSE_COL]].to_numpy(dtype=np.float32)
    physio_values = aligned[list(cfg.node_names)].to_numpy(dtype=np.float32)
    target_values = aligned[GLUCOSE_COL].to_numpy(dtype=np.float32)
    observed_cgm = aligned[OBSERVED_CGM_COL].astype(bool).to_numpy()
    long_gap = aligned[LONG_GAP_COL].astype(bool).to_numpy()

    x_cgm: List[np.ndarray] = []
    x_physio: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    total_steps = cfg.history_steps + cfg.horizon_steps
    for start in range(max(0, len(aligned) - total_steps + 1)):
        hist = slice(start, start + cfg.history_steps)
        fut = slice(start + cfg.history_steps, start + total_steps)
        whole = slice(start, start + total_steps)
        if long_gap[whole].any():
            continue
        if cfg.require_observed_cgm_history and not observed_cgm[hist].all():
            continue
        if not observed_cgm[fut].all():
            continue
        if not np.isfinite(cgm_values[hist]).all():
            continue
        if not np.isfinite(physio_values[hist]).all():
            continue
        if not np.isfinite(target_values[fut]).all():
            continue

        x_cgm.append(cgm_values[hist])
        x_physio.append(physio_values[hist].T)
        targets.append(target_values[fut])
        metadata.append(
            {
                "subject_id": str(aligned["subject_id"].iloc[0]),
                "source_split": str(aligned["source_split"].iloc[0]),
                "source_file": str(aligned["source_file"].iloc[0]),
                "start_time": aligned.index[start],
                "target_start_time": aligned.index[start + cfg.history_steps],
            }
        )

    if not targets:
        return (
            np.empty((0, cfg.history_steps, 1), dtype=np.float32),
            np.empty((0, len(cfg.node_names), cfg.history_steps), dtype=np.float32),
            np.empty((0, cfg.horizon_steps), dtype=np.float32),
            pd.DataFrame(
                columns=[
                    "subject_id",
                    "source_split",
                    "source_file",
                    "start_time",
                    "target_start_time",
                ]
            ),
        )
    return (
        np.stack(x_cgm).astype(np.float32),
        np.stack(x_physio).astype(np.float32),
        np.stack(targets).astype(np.float32),
        pd.DataFrame(metadata),
    )


def process_all_recordings(
    cfg: PreprocessConfig,
    selected_subjects: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    recordings = discover_recordings(cfg.data_root, selected_subjects)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = cfg.output_dir / "aligned_5min"
    if cfg.save_aligned_csv:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    x_cgm_parts: List[np.ndarray] = []
    x_physio_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    metadata_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for recording in recordings:
        print(f"[INFO] Processing {recording.path.name} ...", flush=True)
        aligned, raw_counts = align_recording(recording, cfg)
        windows = make_windows(aligned, cfg)
        x_cgm, x_physio, y, metadata = windows
        if cfg.save_aligned_csv:
            aligned.to_csv(aligned_dir / f"aligned_{recording.path.stem}.csv", index_label="datetime")
        if len(y):
            x_cgm_parts.append(x_cgm)
            x_physio_parts.append(x_physio)
            y_parts.append(y)
            metadata_parts.append(metadata)
        summary_rows.append(
            {
                "subject_id": recording.subject_id,
                "source_split": recording.source_split,
                "source_file": recording.path.name,
                "aligned_steps": len(aligned),
                "long_gap_steps": int(aligned[LONG_GAP_COL].sum()),
                "num_samples": len(y),
                **{f"raw_events_{name}": count for name, count in raw_counts.items()},
                **{
                    f"remaining_nan_{name}": int(aligned[name].isna().sum())
                    for name in [GLUCOSE_COL, *cfg.node_names]
                },
            }
        )
        print(
            f"[INFO] {recording.path.name}: aligned_steps={len(aligned)}, "
            f"samples={len(y)}, long_gap_steps={int(aligned[LONG_GAP_COL].sum())}",
            flush=True,
        )

    if not y_parts:
        raise RuntimeError("No valid OhioT1DM windows were generated.")
    x_cgm = np.concatenate(x_cgm_parts, axis=0)
    x_physio = np.concatenate(x_physio_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    if x_cgm.shape[1:] != (cfg.history_steps, 1):
        raise AssertionError(f"Unexpected X_cgm shape: {x_cgm.shape}")
    if x_physio.shape[1:] != (len(cfg.node_names), cfg.history_steps):
        raise AssertionError(f"Unexpected X_physio shape: {x_physio.shape}")
    if y.shape[1] != cfg.horizon_steps:
        raise AssertionError(f"Unexpected Y shape: {y.shape}")
    if not all(np.isfinite(array).all() for array in (x_cgm, x_physio, y)):
        raise ValueError("Final OhioT1DM arrays contain NaN or Inf values.")
    return x_cgm, x_physio, y, metadata, summary


def save_outputs(
    cfg: PreprocessConfig,
    x_cgm: np.ndarray,
    x_physio: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / "ohiot1dm_windows.npz"
    np.savez_compressed(
        output_path,
        X_cgm=x_cgm,
        X_physio=x_physio,
        Y=y,
        physio_features=np.asarray(cfg.node_names),
        freq=np.asarray(cfg.freq),
        history_steps=np.asarray(cfg.history_steps),
        horizon_steps=np.asarray(cfg.horizon_steps),
    )
    metadata.to_csv(cfg.output_dir / "window_metadata.csv", index=False)
    summary.to_csv(cfg.output_dir / "preprocess_summary.csv", index=False)
    config_payload = asdict(cfg)
    config_payload["data_root"] = str(cfg.data_root)
    config_payload["output_dir"] = str(cfg.output_dir)
    config_payload["node_names"] = list(cfg.node_names)
    (cfg.output_dir / "preprocess_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] X_cgm shape: {x_cgm.shape}")
    print(f"[DONE] X_physio shape: {x_physio.shape}")
    print(f"[DONE] Y shape: {y.shape}")
    print(f"[DONE] Subjects: {sorted(metadata['subject_id'].astype(str).unique())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OhioT1DM XML files into 5-minute forecasting windows."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--max-gap-hours", type=float, default=2.0)
    parser.add_argument(
        "--allow-imputed-cgm-history",
        action="store_true",
        help="Allow short forward-filled CGM gaps in the historical input only.",
    )
    parser.add_argument("--save-aligned-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        freq=args.freq,
        history_steps=args.history_steps,
        horizon_steps=args.horizon_steps,
        max_gap_hours=args.max_gap_hours,
        require_observed_cgm_history=not args.allow_imputed_cgm_history,
        save_aligned_csv=args.save_aligned_csv,
    )
    outputs = process_all_recordings(cfg, args.subjects)
    save_outputs(cfg, *outputs)


if __name__ == "__main__":
    main()
