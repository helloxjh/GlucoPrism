"""Cross-model benchmark statistics and publication-ready LaTeX tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


HORIZONS = (15, 30, 45, 60)
METRICS = {
    "mae": ("MAE", "lower"),
    "rmse": ("RMSE", "lower"),
    "mard": ("MARD", "lower"),
    "mape": ("MAPE", "lower"),
    "r2": ("R2", "higher"),
    "pearson": ("Pearson", "higher"),
    "ega_zone_a_pct": ("EGA_A", "higher"),
    "ega_zone_b_pct": ("EGA_B", "lower"),
    "ega_zone_c_pct": ("EGA_C", "lower"),
    "ega_zone_d_pct": ("EGA_D", "lower"),
    "ega_zone_e_pct": ("EGA_E", "lower"),
}
RANK_METRICS = ("mae", "rmse", "mard", "r2", "pearson", "ega_zone_a_pct")


def rebuild_benchmark_reports(output_root: str | Path) -> tuple[Path, Path]:
    """Rebuild CSV statistics and LaTeX output from completed model fold files."""
    root = Path(output_root)
    model_runs = _load_model_runs(root)
    if not model_runs:
        raise RuntimeError(f"No model LOSO metrics found under {root}.")
    rows = [_summarize_model(model_name, records) for model_name, records in model_runs]
    _attach_average_ranks(rows)
    summary_path = root / "benchmark_summary.csv"
    latex_path = root / "benchmark_table.tex"
    _write_csv(summary_path, rows)
    latex_path.write_text(_build_latex_table(rows), encoding="utf-8")
    return summary_path, latex_path


def _load_model_runs(root: Path) -> list[tuple[str, list[Dict[str, float]]]]:
    runs: list[tuple[str, list[Dict[str, float]]]] = []
    for metrics_path in sorted(root.glob("*/loso_test_metrics.csv")):
        records: list[Dict[str, float]] = []
        with metrics_path.open(newline="", encoding="utf-8") as fp:
            for raw in csv.DictReader(fp):
                if not raw.get("fold"):
                    continue
                record: Dict[str, float] = {}
                for key, value in raw.items():
                    if key in {"test_subject"} or value in {None, ""}:
                        continue
                    try:
                        record[key] = float(value)
                    except ValueError:
                        continue
                records.append(record)
        if not records:
            continue
        config_path = metrics_path.parent / "run_config.json"
        model_name = metrics_path.parent.name
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                architecture = config.get("model_architecture", {})
                model_name = architecture.get("display_name", model_name)
            except (json.JSONDecodeError, AttributeError):
                pass
        runs.append((str(model_name), records))
    return runs


def _summarize_model(model_name: str, records: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    row: Dict[str, object] = {"Model": model_name, "LOSO_Folds": len(records)}
    for horizon in HORIZONS:
        for metric, (label, direction) in METRICS.items():
            key = f"horizon_{horizon}_{metric}"
            values = [float(record[key]) for record in records if key in record]
            if len(values) != len(records):
                raise ValueError(f"Missing {key} for model {model_name}.")
            mean, std = _mean_std(values)
            best = min(values) if direction == "lower" else max(values)
            prefix = f"{horizon}min_{label}"
            row[prefix] = mean
            row[f"{prefix}_Mean"] = mean
            row[f"{prefix}_Std"] = std
            row[f"{prefix}_Best"] = best

    for metric, (label, direction) in METRICS.items():
        fold_averages = [
            sum(float(record[f"horizon_{horizon}_{metric}"]) for horizon in HORIZONS)
            / len(HORIZONS)
            for record in records
        ]
        mean, std = _mean_std(fold_averages)
        row[f"Mean_{label}"] = mean
        row[f"Std_{label}"] = std
        row[f"Best_{label}"] = (
            min(fold_averages) if direction == "lower" else max(fold_averages)
        )

    row["Mean_Train_Time_s"] = _record_mean(records, "train_time_seconds")
    row["Std_Train_Time_s"] = _record_std(records, "train_time_seconds")
    row["Best_Train_Time_s"] = min(float(record["train_time_seconds"]) for record in records)
    row["Mean_Inference_ms_per_sample"] = _record_mean(
        records, "inference_ms_per_sample"
    )
    row["Std_Inference_ms_per_sample"] = _record_std(
        records, "inference_ms_per_sample"
    )
    row["Best_Inference_ms_per_sample"] = min(
        float(record["inference_ms_per_sample"]) for record in records
    )
    row["Parameter_Count"] = int(records[0]["parameter_count"])
    return row


def _attach_average_ranks(rows: list[Dict[str, object]]) -> None:
    all_rank_values: Dict[str, list[float]] = {str(row["Model"]): [] for row in rows}
    mae_rank_values: Dict[str, list[float]] = {str(row["Model"]): [] for row in rows}
    for horizon in HORIZONS:
        for metric in RANK_METRICS:
            label, direction = METRICS[metric]
            key = f"{horizon}min_{label}_Mean"
            values = [float(row[key]) for row in rows]
            ranks = _average_tie_ranks(values, direction)
            for row, rank in zip(rows, ranks):
                model = str(row["Model"])
                all_rank_values[model].append(rank)
                if metric == "mae":
                    mae_rank_values[model].append(rank)
    for row in rows:
        model = str(row["Model"])
        row["Average_Rank"] = sum(all_rank_values[model]) / len(all_rank_values[model])
        row["Average_Rank_MAE"] = sum(mae_rank_values[model]) / len(mae_rank_values[model])
    best_rank = min(float(row["Average_Rank"]) for row in rows)
    for row in rows:
        row["Overall_Best"] = "Yes" if math.isclose(float(row["Average_Rank"]), best_rank) else "No"


def _average_tie_ranks(values: Sequence[float], direction: str) -> list[float]:
    order = sorted(
        range(len(values)),
        key=lambda index: values[index],
        reverse=direction == "higher",
    )
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[position]], rel_tol=1e-9, abs_tol=1e-12
        ):
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    return ranks


def _build_latex_table(rows: Sequence[Mapping[str, object]]) -> str:
    best_values: Dict[tuple[int, str], float] = {}
    table_metrics = ("MAE", "RMSE", "MARD", "R2", "Pearson", "EGA_A")
    directions = {label: direction for _, (label, direction) in METRICS.items()}
    for horizon in HORIZONS:
        for label in table_metrics:
            values = [float(row[f"{horizon}min_{label}_Mean"]) for row in rows]
            best_values[(horizon, label)] = (
                min(values) if directions[label] == "lower" else max(values)
            )
    best_rank = min(float(row["Average_Rank"]) for row in rows)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{LOSO-CV benchmark results across prediction horizons. Values are mean $\pm$ standard deviation across test subjects.}",
        r"\label{tab:benchmark_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & PH & MAE (mg/dL) & RMSE (mg/dL) & MARD (\%) & $R^2$ & Pearson $r$ & EGA-A (\%) & Avg. Rank \\",
        r"\midrule",
    ]
    for model_index, row in enumerate(rows):
        for horizon_index, horizon in enumerate(HORIZONS):
            cells = [_escape_latex(str(row["Model"])), f"{horizon} min"]
            for label in table_metrics:
                mean = float(row[f"{horizon}min_{label}_Mean"])
                std = float(row[f"{horizon}min_{label}_Std"])
                formatted = f"{mean:.3f} $\\pm$ {std:.3f}"
                if math.isclose(mean, best_values[(horizon, label)], rel_tol=1e-9, abs_tol=1e-12):
                    formatted = r"\textbf{" + formatted + "}"
                cells.append(formatted)
            rank = float(row["Average_Rank"])
            rank_text = f"{rank:.3f}" if horizon_index == 0 else "--"
            if horizon_index == 0 and math.isclose(rank, best_rank):
                rank_text = r"\textbf{" + rank_text + "}"
            cells.append(rank_text)
            lines.append(" & ".join(cells) + r" \\")
        if model_index < len(rows) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _record_mean(records: Sequence[Mapping[str, float]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def _record_std(records: Sequence[Mapping[str, float]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return _mean_std(values)[1]


def _escape_latex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
