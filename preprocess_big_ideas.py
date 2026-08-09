#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIG IDEAs Lab 血糖-可穿戴多模态数据预处理脚本。

功能：
1. 自动读取 001-016 受试者目录下的 Dexcom、ACC、EDA、TEMP、HR、BVP、IBI CSV。
2. 将日期字符串或 Unix 时间戳统一转为 pandas datetime64[ns] 索引。
3. 以 CGM 的 5 分钟尺度为主轴，对高频外周信号下采样聚合。
4. 对短缺失执行 ffill + bfill；对连续超过 2 小时的严重缺失片段打标并在滑窗生成时剔除。
5. 生成 PyTorch 友好的数组：
   X_cgm:    [num_samples, 24, 1]
   X_physio: [num_samples, 6, 24]
   Y:        [num_samples, 6]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(
    "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3"
)
DEFAULT_OUTPUT_DIR = Path("processed_big_ideas")
DEFAULT_FREQ = "5min"  # 与论文/需求中常写的 "5T" 等价。

GLUCOSE_COL = "glucose"
LONG_GAP_COL = "__long_gap__"
OBSERVED_CGM_COL = "__observed_glucose__"

# 默认 6 个外周生理通道，保证 X_physio 输出为 [N, 6, 24]。
DEFAULT_PHYSIO_ORDER = ["acc_l2", "eda", "temp", "hr", "bvp", "ibi"]


@dataclass(frozen=True)
class PreprocessConfig:
    data_root: Path
    output_dir: Path
    freq: str = DEFAULT_FREQ
    history_steps: int = 24
    horizon_steps: int = 6
    max_gap_hours: float = 2.0
    chunksize: int = 1_000_000
    physio_order: Tuple[str, ...] = tuple(DEFAULT_PHYSIO_ORDER)
    require_observed_targets: bool = True
    require_observed_cgm_history: bool = False
    save_aligned_csv: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "freq", canonicalize_frequency(self.freq))

    @property
    def max_gap_steps(self) -> int:
        step_minutes = pd.Timedelta(self.freq).total_seconds() / 60.0
        if step_minutes <= 0:
            raise ValueError(f"非法重采样频率: {self.freq}")
        return int(math.floor(self.max_gap_hours * 60.0 / step_minutes))


def canonicalize_frequency(freq: str) -> str:
    """兼容用户传入 5T，同时在 pandas 内部使用无弃用警告的 5min。"""
    cleaned = str(freq).strip()
    if cleaned.lower().endswith("t"):
        prefix = cleaned[:-1]
        return f"{prefix}min" if prefix else "min"
    return cleaned


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉 BOM、首尾空格，避免 ' datetime' 之类表头导致列识别失败。"""
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """按候选名查找列；先精确匹配，再忽略大小写匹配。"""
    normalized = {str(col).strip().lower(): col for col in columns}
    for name in candidates:
        if name in columns:
            return name
        found = normalized.get(name.strip().lower())
        if found is not None:
            return found
    return None


def detect_time_column(columns: Sequence[str]) -> str:
    """自动识别时间列，兼容 datetime、Timestamp、Unix timestamp 等常见命名。"""
    candidates = [
        "datetime",
        "timestamp",
        "Timestamp",
        "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "time",
        "Time",
        "date",
        "Date",
    ]
    found = find_column(columns, candidates)
    if found is not None:
        return found

    for col in columns:
        lower = str(col).strip().lower()
        if "time" in lower or "date" in lower:
            return col
    raise ValueError(f"无法识别时间列，可用列为: {list(columns)}")


def parse_datetime_series(values: pd.Series) -> pd.Series:
    """
    将一列时间值解析为 pandas datetime64[ns]。

    兼容：
    - 字符串时间：2020-02-13 15:28:50.000、2/13/20 15:29 等；
    - Unix 秒/毫秒/微秒/纳秒时间戳。
    """
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(values) else 0.0

    if numeric_ratio > 0.9:
        median_abs = float(numeric.dropna().abs().median())
        if median_abs > 1e17:
            unit = "ns"
        elif median_abs > 1e14:
            unit = "us"
        elif median_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")

    text = values.astype("string").str.strip()
    try:
        return pd.to_datetime(text, format="mixed", errors="coerce")
    except TypeError:
        # pandas<2.0 不支持 format="mixed" 时退回普通解析。
        return pd.to_datetime(text, errors="coerce")


def read_cgm_file(path: Path, freq: str) -> pd.Series:
    """读取 Dexcom CGM 文件，筛选 EGV 血糖记录并聚合到 5 分钟主轴。"""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = normalize_columns(df)

    time_col = detect_time_column(df.columns)
    glucose_col = find_column(
        df.columns,
        [
            "Glucose Value (mg/dL)",
            "Glucose",
            "glucose",
            "CGM",
            "cgm",
            "Value",
            "value",
        ],
    )
    if glucose_col is None:
        raise ValueError(f"{path} 无法识别血糖列，可用列为: {list(df.columns)}")

    event_col = find_column(df.columns, ["Event Type", "event_type", "event"])
    if event_col is not None:
        # Dexcom 导出文件中可能包含 Alert、Calibration、元数据等非血糖行。
        event = df[event_col].astype("string").str.strip().str.upper()
        if (event == "EGV").any():
            df = df.loc[event == "EGV"].copy()

    ts = parse_datetime_series(df[time_col])
    glucose = pd.to_numeric(df[glucose_col], errors="coerce")

    out = pd.DataFrame({GLUCOSE_COL: glucose.to_numpy()}, index=ts)
    out = out.loc[out.index.notna()]
    out = out.dropna(subset=[GLUCOSE_COL])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if out.empty:
        raise ValueError(f"{path} 没有有效 CGM 血糖记录")

    return out[GLUCOSE_COL].resample(freq).mean().dropna()


def value_column_for_signal(columns: Sequence[str], signal: str) -> str:
    """识别单通道外周信号的数值列。"""
    candidates = {
        "eda": ["eda", "EDA", "Value", "value"],
        "temp": ["temp", "TEMP", "temperature", "Temperature", "Value", "value"],
        "hr": ["hr", "HR", "heart_rate", "Heart Rate", "Value", "value"],
        "bvp": ["bvp", "BVP", "Value", "value"],
        "ibi": ["ibi", "IBI", "Value", "value"],
    }
    found = find_column(columns, candidates[signal])
    if found is not None:
        return found

    time_col = detect_time_column(columns)
    numeric_candidates = [col for col in columns if col != time_col]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise ValueError(f"无法识别 {signal} 数值列，可用列为: {list(columns)}")


def acc_columns(columns: Sequence[str]) -> Tuple[str, str, str]:
    """识别 ACC 三轴列。"""
    x_col = find_column(columns, ["acc_x", "x", "X", "ACC_X"])
    y_col = find_column(columns, ["acc_y", "y", "Y", "ACC_Y"])
    z_col = find_column(columns, ["acc_z", "z", "Z", "ACC_Z"])
    if not all([x_col, y_col, z_col]):
        raise ValueError(f"无法识别 ACC 三轴列，可用列为: {list(columns)}")
    return x_col, y_col, z_col


def aggregate_signal_file(
    path: Path,
    signal: str,
    feature_name: str,
    freq: str,
    chunksize: int,
) -> pd.Series:
    """
    将一个外周信号文件聚合到 5 分钟尺度。

    采用分块 sum/count 聚合，避免 ACC/BVP 等高频文件过大时内存爆炸。
    """
    sum_parts: List[pd.Series] = []
    count_parts: List[pd.Series] = []

    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        chunk = normalize_columns(chunk)
        time_col = detect_time_column(chunk.columns)
        ts = parse_datetime_series(chunk[time_col])

        if signal == "acc":
            x_col, y_col, z_col = acc_columns(chunk.columns)
            x = pd.to_numeric(chunk[x_col], errors="coerce")
            y = pd.to_numeric(chunk[y_col], errors="coerce")
            z = pd.to_numeric(chunk[z_col], errors="coerce")
            value = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        else:
            value_col = value_column_for_signal(chunk.columns, signal)
            value = pd.to_numeric(chunk[value_col], errors="coerce")

        tmp = pd.DataFrame({"value": value.to_numpy()}, index=ts)
        tmp = tmp.loc[tmp.index.notna()]
        tmp = tmp.dropna(subset=["value"])
        if tmp.empty:
            continue

        bins = tmp.index.floor(freq)
        grouped = tmp["value"].groupby(bins)
        sum_parts.append(grouped.sum())
        count_parts.append(grouped.count())

    if not sum_parts:
        raise ValueError(f"{path} 没有有效 {signal} 记录")

    total_sum = pd.concat(sum_parts).groupby(level=0).sum()
    total_count = pd.concat(count_parts).groupby(level=0).sum()
    mean = (total_sum / total_count).sort_index()
    mean.name = feature_name
    return mean


def find_subject_dirs(data_root: Path, selected_subjects: Optional[Sequence[str]]) -> List[Path]:
    """查找 001-016 这类受试者目录。"""
    if selected_subjects:
        subject_ids = [str(s).zfill(3) for s in selected_subjects]
        dirs = [data_root / subject_id for subject_id in subject_ids]
    else:
        dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit())

    missing = [str(p) for p in dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"以下受试者目录不存在: {missing}")
    return dirs


def find_signal_file(subject_dir: Path, prefix: str) -> Optional[Path]:
    """在受试者目录中查找 ACC_001.csv、Dexcom_001.csv 等文件。"""
    matches = sorted(subject_dir.glob(f"{prefix}_*.csv"))
    if matches:
        return matches[0]

    # 兼容 Dexcom_Challenge2.csv、EDA.csv 等不带受试者编号的文件名。
    matches = sorted(subject_dir.glob(f"{prefix}*.csv"))
    return matches[0] if matches else None


def mark_long_missing_runs(
    df: pd.DataFrame,
    columns: Sequence[str],
    max_gap_steps: int,
) -> pd.Series:
    """
    标记连续缺失超过阈值的时间点。

    例如 5 分钟频率、2 小时阈值时，max_gap_steps=24。
    任一关键列出现超过 24 个连续 NaN，该缺失段都会被标为 True。
    """
    if max_gap_steps <= 0:
        raise ValueError("max_gap_steps 必须为正数")

    long_gap = pd.Series(False, index=df.index)
    for col in columns:
        missing = df[col].isna()
        if not missing.any():
            continue
        run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
        run_len = missing.groupby(run_id).transform("sum")
        long_gap |= missing & (run_len > max_gap_steps)
    return long_gap


def fill_short_missing_values(df: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    """
    先前向填充，再后向填充。

    注意：严重长缺失段不会在这里直接删除，而是在滑窗生成时通过 LONG_GAP_COL 剔除，
    这样可以避免窗口跨越断连片段。
    """
    filled = df.copy()
    filled.loc[:, value_columns] = filled.loc[:, value_columns].ffill().bfill()
    return filled


def align_subject(subject_dir: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    """读取并对齐单个受试者的 CGM 与外周生理信号。"""
    subject_id = subject_dir.name
    cgm_path = find_signal_file(subject_dir, "Dexcom")
    if cgm_path is None:
        raise FileNotFoundError(f"{subject_dir} 缺少 Dexcom CSV")

    cgm = read_cgm_file(cgm_path, cfg.freq)
    full_index = pd.date_range(cgm.index.min(), cgm.index.max(), freq=cfg.freq)
    aligned = pd.DataFrame(index=full_index)
    aligned[GLUCOSE_COL] = cgm.reindex(full_index)
    aligned[OBSERVED_CGM_COL] = aligned[GLUCOSE_COL].notna()

    feature_specs = {
        "acc_l2": ("ACC", "acc", "acc_l2"),
        "eda": ("EDA", "eda", "eda"),
        "temp": ("TEMP", "temp", "temp"),
        "hr": ("HR", "hr", "hr"),
        "bvp": ("BVP", "bvp", "bvp"),
        "ibi": ("IBI", "ibi", "ibi"),
    }

    for feature in cfg.physio_order:
        if feature not in feature_specs:
            raise ValueError(f"未知生理特征: {feature}")
        prefix, signal, feature_name = feature_specs[feature]
        path = find_signal_file(subject_dir, prefix)
        if path is None:
            raise FileNotFoundError(f"{subject_dir} 缺少 {prefix} CSV，无法生成 {feature}")
        series = aggregate_signal_file(
            path=path,
            signal=signal,
            feature_name=feature_name,
            freq=cfg.freq,
            chunksize=cfg.chunksize,
        )
        aligned[feature] = series.reindex(full_index)

    value_columns = [GLUCOSE_COL, *cfg.physio_order]
    aligned[LONG_GAP_COL] = mark_long_missing_runs(
        aligned,
        columns=value_columns,
        max_gap_steps=cfg.max_gap_steps,
    )
    aligned["subject_id"] = subject_id

    filled = fill_short_missing_values(aligned, value_columns=value_columns)
    remaining_nan = filled[value_columns].isna().sum()
    if int(remaining_nan.sum()) > 0:
        raise ValueError(
            f"{subject_id} ffill+bfill 后仍存在 NaN: {remaining_nan[remaining_nan > 0].to_dict()}"
        )
    return filled


def make_windows(
    aligned: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """根据单个受试者对齐后的 5 分钟数据生成滑窗样本。"""
    value_columns = [GLUCOSE_COL, *cfg.physio_order]
    if aligned[value_columns].isna().any().any():
        nan_counts = aligned[value_columns].isna().sum()
        raise ValueError(f"滑窗前仍存在 NaN: {nan_counts[nan_counts > 0].to_dict()}")

    cgm_values = aligned[[GLUCOSE_COL]].to_numpy(dtype=np.float32)
    physio_values = aligned[list(cfg.physio_order)].to_numpy(dtype=np.float32)
    y_values = aligned[GLUCOSE_COL].to_numpy(dtype=np.float32)

    # 原始观测标记：经过 fill 后的 NaN 已消失，但目标 Y 默认必须来自真实 CGM 观测。
    observed_cgm = aligned[OBSERVED_CGM_COL].astype(bool).to_numpy()
    long_gap = aligned[LONG_GAP_COL].astype(bool).to_numpy()

    x_cgm_list: List[np.ndarray] = []
    x_physio_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    total_steps = cfg.history_steps + cfg.horizon_steps
    max_start = len(aligned) - total_steps + 1
    if max_start <= 0:
        empty_meta = pd.DataFrame(columns=["subject_id", "start_time", "target_start_time"])
        return (
            np.empty((0, cfg.history_steps, 1), dtype=np.float32),
            np.empty((0, len(cfg.physio_order), cfg.history_steps), dtype=np.float32),
            np.empty((0, cfg.horizon_steps), dtype=np.float32),
            empty_meta,
        )

    for start in range(max_start):
        hist = slice(start, start + cfg.history_steps)
        fut = slice(start + cfg.history_steps, start + total_steps)
        whole = slice(start, start + total_steps)

        if long_gap[whole].any():
            continue
        if cfg.require_observed_targets and not observed_cgm[fut].all():
            continue
        if cfg.require_observed_cgm_history and not observed_cgm[hist].all():
            continue

        x_cgm_list.append(cgm_values[hist])
        # PyTorch 常见 Conv/RNN 输入可按 [通道, 时间] 组织，因此转置为 [6, 24]。
        x_physio_list.append(physio_values[hist].T)
        y_list.append(y_values[fut])
        meta_rows.append(
            {
                "subject_id": aligned["subject_id"].iloc[0],
                "start_time": aligned.index[start],
                "target_start_time": aligned.index[start + cfg.history_steps],
            }
        )

    if not x_cgm_list:
        empty_meta = pd.DataFrame(columns=["subject_id", "start_time", "target_start_time"])
        return (
            np.empty((0, cfg.history_steps, 1), dtype=np.float32),
            np.empty((0, len(cfg.physio_order), cfg.history_steps), dtype=np.float32),
            np.empty((0, cfg.horizon_steps), dtype=np.float32),
            empty_meta,
        )

    return (
        np.stack(x_cgm_list).astype(np.float32),
        np.stack(x_physio_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        pd.DataFrame(meta_rows),
    )


def process_all_subjects(
    cfg: PreprocessConfig,
    selected_subjects: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """处理所有受试者并合并样本。"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = cfg.output_dir / "aligned_5min"
    if cfg.save_aligned_csv:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    x_cgm_all: List[np.ndarray] = []
    x_physio_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    meta_all: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    subject_dirs = find_subject_dirs(cfg.data_root, selected_subjects)
    for subject_dir in subject_dirs:
        print(f"[INFO] Processing subject {subject_dir.name} ...", flush=True)
        aligned = align_subject(subject_dir, cfg)

        if cfg.save_aligned_csv:
            aligned.to_csv(aligned_dir / f"aligned_{subject_dir.name}.csv", index_label="datetime")

        x_cgm, x_physio, y, meta = make_windows(aligned, cfg)
        x_cgm_all.append(x_cgm)
        x_physio_all.append(x_physio)
        y_all.append(y)
        meta_all.append(meta)

        value_columns = [GLUCOSE_COL, *cfg.physio_order]
        summary_rows.append(
            {
                "subject_id": subject_dir.name,
                "aligned_steps": len(aligned),
                "long_gap_steps": int(aligned[LONG_GAP_COL].sum()),
                "num_samples": int(len(y)),
                **{f"nan_after_fill_{col}": int(aligned[col].isna().sum()) for col in value_columns},
            }
        )
        print(
            f"[INFO] Subject {subject_dir.name}: aligned_steps={len(aligned)}, "
            f"samples={len(y)}, long_gap_steps={int(aligned[LONG_GAP_COL].sum())}",
            flush=True,
        )

    if not y_all:
        raise RuntimeError("没有发现任何可处理的受试者")

    x_cgm_final = np.concatenate(x_cgm_all, axis=0)
    x_physio_final = np.concatenate(x_physio_all, axis=0)
    y_final = np.concatenate(y_all, axis=0)
    meta_final = pd.concat(meta_all, ignore_index=True) if meta_all else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    assert x_cgm_final.ndim == 3 and x_cgm_final.shape[1:] == (cfg.history_steps, 1)
    assert x_physio_final.ndim == 3 and x_physio_final.shape[1:] == (
        len(cfg.physio_order),
        cfg.history_steps,
    )
    assert y_final.ndim == 2 and y_final.shape[1] == cfg.horizon_steps
    if np.isnan(x_cgm_final).any() or np.isnan(x_physio_final).any() or np.isnan(y_final).any():
        raise ValueError("最终数组中仍存在 NaN，请检查缺失值处理逻辑")

    return x_cgm_final, x_physio_final, y_final, meta_final, summary


def save_outputs(
    cfg: PreprocessConfig,
    x_cgm: np.ndarray,
    x_physio: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """保存 numpy 数据集与元数据。"""
    out_npz = cfg.output_dir / "big_ideas_windows.npz"
    np.savez_compressed(
        out_npz,
        X_cgm=x_cgm,
        X_physio=x_physio,
        Y=y,
        physio_features=np.array(cfg.physio_order),
        freq=np.array(cfg.freq),
        history_steps=np.array(cfg.history_steps),
        horizon_steps=np.array(cfg.horizon_steps),
    )
    meta.to_csv(cfg.output_dir / "window_metadata.csv", index=False)
    summary.to_csv(cfg.output_dir / "preprocess_summary.csv", index=False)

    print("[DONE] Saved:", out_npz)
    print("[DONE] X_cgm shape:   ", x_cgm.shape)
    print("[DONE] X_physio shape:", x_physio.shape)
    print("[DONE] Y shape:       ", y.shape)
    print("[DONE] Summary CSV:   ", cfg.output_dir / "preprocess_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BIG IDEAs glycemic wearable dataset into aligned PyTorch arrays."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="*", default=None, help="例如: --subjects 001 002 016")
    parser.add_argument("--freq", default=DEFAULT_FREQ, help="默认 5min；也兼容 5T")
    parser.add_argument("--history-steps", type=int, default=24, help="过去 2 小时=24 个 5分钟步")
    parser.add_argument("--horizon-steps", type=int, default=6, help="未来 30 分钟=6 个 5分钟步")
    parser.add_argument("--max-gap-hours", type=float, default=2.0, help="超过该连续缺失时长的片段会被剔除")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="高频 CSV 分块读取行数")
    parser.add_argument(
        "--physio-features",
        nargs="+",
        default=DEFAULT_PHYSIO_ORDER,
        choices=DEFAULT_PHYSIO_ORDER,
        help="默认使用 6 个通道: acc_l2 eda temp hr bvp ibi",
    )
    parser.add_argument(
        "--allow-imputed-targets",
        action="store_true",
        help="默认 Y 必须是实际观测 CGM；打开后允许用 ffill/bfill 后的 CGM 作为 Y。",
    )
    parser.add_argument(
        "--require-observed-cgm-history",
        action="store_true",
        help="打开后要求历史 CGM 输入也必须全部来自实际观测，而不是填充值。",
    )
    parser.add_argument(
        "--save-aligned-csv",
        action="store_true",
        help="额外保存每个受试者对齐后的 5分钟 DataFrame，便于人工检查。",
    )
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
        chunksize=args.chunksize,
        physio_order=tuple(args.physio_features),
        require_observed_targets=not args.allow_imputed_targets,
        require_observed_cgm_history=args.require_observed_cgm_history,
        save_aligned_csv=args.save_aligned_csv,
    )

    x_cgm, x_physio, y, meta, summary = process_all_subjects(
        cfg,
        selected_subjects=args.subjects,
    )
    save_outputs(cfg, x_cgm, x_physio, y, meta, summary)


if __name__ == "__main__":
    main()
