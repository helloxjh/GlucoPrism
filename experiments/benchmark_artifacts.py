"""Paper-facing artifact generation for the shared benchmark pipeline."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from data.normalization import DataStandardizer
from data.splits import LOSOSplit
from evaluation.clarke_ega import clarke_error_grid_zones
from evaluation.evaluator import EvaluationOutputs


MODEL_DISPLAY_NAMES = {
    "glucoprism": "GlucoPrism",
    "lstm": "LSTM",
    "cnn": "1D-CNN",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "patchtst": "PatchTST",
    "graphwavenet": "Graph WaveNet",
    "dcrnn": "DCRNN",
    "crnn": "CRNN",
}

METRIC_NAMES = (
    "mae",
    "rmse",
    "mard",
    "mape",
    "r2",
    "pearson",
    "ega_zone_a_pct",
    "ega_zone_b_pct",
    "ega_zone_c_pct",
    "ega_zone_d_pct",
    "ega_zone_e_pct",
)

RESUME_CONFIG_IGNORED_KEYS = {
    "device_resolved",
    "fold_limit",
    "fold_start",
    "overwrite",
    "resume",
    "selected_fold_ids",
}


class BenchmarkArtifactWriter:
    """Persist checkpoints, fold predictions, figures, and benchmark summaries."""

    def __init__(
        self,
        output_root: str | Path,
        model_name: str,
        config: Mapping[str, Any],
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        if overwrite and resume:
            raise ValueError("--overwrite and --resume cannot be used together.")
        normalized = model_name.lower()
        if normalized not in MODEL_DISPLAY_NAMES:
            raise ValueError(f"Unsupported benchmark artifact model: {model_name!r}")
        self.output_root = Path(output_root)
        self.model_name = normalized
        self.display_name = MODEL_DISPLAY_NAMES[normalized]
        self.model_dir = self.output_root / self.display_name
        self.config = dict(config)
        self.fold_records: list[Dict[str, Any]] = []
        self.histories: list[Dict[str, Any]] = []
        self.predictions: Dict[int, list[Dict[str, Any]]] = {
            horizon: [] for horizon in (15, 30, 45, 60)
        }
        if self.model_dir.exists():
            if resume:
                self._load_existing_run()
                return
            if not overwrite:
                raise FileExistsError(
                    f"Benchmark output already exists: {self.model_dir}. "
                    "Use --resume to continue it or --overwrite to replace it."
                )
            shutil.rmtree(self.model_dir)
        elif resume:
            raise FileNotFoundError(
                f"Cannot resume because benchmark output does not exist: {self.model_dir}"
            )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "checkpoints").mkdir()
        (self.model_dir / "logs").mkdir()
        self._write_json(self.model_dir / "run_config.json", self.config)

    @property
    def completed_fold_ids(self) -> set[int]:
        """Return fold identifiers that have complete recoverable artifacts."""
        return {int(record["fold"]) for record in self.fold_records}

    def metric_records(self) -> list[Dict[str, float]]:
        """Return completed-fold metrics in the shape used by the console summary."""
        return [
            {
                "loss": float(record["loss"]),
                **{metric: float(record[metric]) for metric in METRIC_NAMES},
            }
            for record in self.fold_records
        ]

    def _load_existing_run(self) -> None:
        config_path = self.model_dir / "run_config.json"
        metrics_path = self.model_dir / "loso_test_metrics.csv"
        if not config_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(
                f"Cannot resume incomplete benchmark metadata under {self.model_dir}."
            )
        stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        self._validate_resume_config(stored_config)
        self.fold_records = self._read_csv(metrics_path)
        if not self.fold_records:
            raise RuntimeError(f"No completed folds found in {metrics_path}.")

        history_path = self.model_dir / "logs" / "training_history.csv"
        self.histories = self._read_csv(history_path) if history_path.is_file() else []
        for horizon in self.predictions:
            prediction_path = self.model_dir / f"{horizon}min" / "prediction.csv"
            if not prediction_path.is_file():
                raise RuntimeError(
                    f"Cannot resume because prediction artifacts are missing: {prediction_path}"
                )
            self.predictions[horizon] = self._read_csv(prediction_path)

        completed = self.completed_fold_ids
        if len(completed) != len(self.fold_records):
            raise RuntimeError("Cannot resume because loso_test_metrics.csv contains duplicate folds.")
        for fold_id in completed:
            checkpoint = self.model_dir / "checkpoints" / f"fold_{fold_id:02d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"Cannot resume because the checkpoint for fold {fold_id:02d} is missing."
                )
        for horizon, rows in self.predictions.items():
            prediction_folds = {int(row["fold"]) for row in rows}
            if prediction_folds != completed:
                raise RuntimeError(
                    f"Cannot resume because {horizon}min predictions cover folds "
                    f"{sorted(prediction_folds)}, expected {sorted(completed)}."
                )
        self._sort_state()
        print(
            f"[INFO] Resuming {self.display_name}: preserved completed folds "
            f"{sorted(completed)}"
        )

    def _validate_resume_config(self, stored: Mapping[str, Any]) -> None:
        mismatches = []
        for key, current_value in self.config.items():
            if key in RESUME_CONFIG_IGNORED_KEYS or key not in stored:
                continue
            stored_value = stored[key]
            if _normalize_json_value(stored_value) != _normalize_json_value(current_value):
                mismatches.append(f"{key}: stored={stored_value!r}, current={current_value!r}")
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(f"Resume configuration does not match the existing run: {details}")

    def _sort_state(self) -> None:
        self.fold_records.sort(key=lambda row: int(row["fold"]))
        self.histories.sort(key=lambda row: (int(row["fold"]), float(row["epoch"])))
        for rows in self.predictions.values():
            rows.sort(
                key=lambda row: (
                    int(row["fold"]),
                    int(row["sample_index_within_fold"]),
                )
            )

    def record_fold(
        self,
        *,
        split: LOSOSplit,
        model: nn.Module,
        standardizer: DataStandardizer | None,
        evaluation: EvaluationOutputs,
        history: Sequence[Mapping[str, float]],
        train_time_seconds: float,
        total_parameters: int,
        trainable_parameters: int,
        horizon_minutes: Sequence[int],
    ) -> None:
        """Record one completed LOSO fold and immediately flush recoverable files."""
        if split.fold_id in self.completed_fold_ids:
            raise RuntimeError(f"Fold {split.fold_id:02d} has already been recorded.")
        inference_ms_per_sample = (
            1000.0 * evaluation.inference_time_seconds / max(evaluation.num_samples, 1)
        )
        fold_record: Dict[str, Any] = {
            "fold": split.fold_id,
            "test_subject": split.test_subject,
            "loss": evaluation.loss,
            "train_time_seconds": train_time_seconds,
            "inference_time_seconds": evaluation.inference_time_seconds,
            "inference_ms_per_sample": inference_ms_per_sample,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            **evaluation.metrics,
        }
        self.fold_records.append(fold_record)
        for row in history:
            self.histories.append({"fold": split.fold_id, **dict(row)})

        for horizon_index, horizon in enumerate(horizon_minutes):
            horizon = int(horizon)
            for sample_index in range(evaluation.num_samples):
                target = float(evaluation.targets[sample_index, horizon_index])
                prediction = float(evaluation.predictions[sample_index, horizon_index])
                subject = (
                    evaluation.subject_ids[sample_index]
                    if sample_index < len(evaluation.subject_ids)
                    else split.test_subject
                )
                self.predictions[horizon].append(
                    {
                        "fold": split.fold_id,
                        "test_subject": subject,
                        "sample_index_within_fold": sample_index,
                        "target_mg_dl": target,
                        "prediction_mg_dl": prediction,
                        "error_mg_dl": prediction - target,
                    }
                )

        self._sort_state()
        checkpoint = {
            "model": self.model_name,
            "fold": split.fold_id,
            "test_subject": split.test_subject,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "standardizer": _serialize_standardizer(standardizer),
            "horizon_minutes": tuple(int(value) for value in horizon_minutes),
            "config": self.config,
        }
        torch.save(
            checkpoint,
            self.model_dir / "checkpoints" / f"fold_{split.fold_id:02d}.pt",
        )
        self._flush_tables()
        self._append_training_log(fold_record)

    def finalize(self) -> Path:
        """Generate figures and update the cross-model benchmark summary."""
        if not self.fold_records:
            raise RuntimeError("No completed folds are available for benchmark finalization.")
        self._flush_tables()
        self._write_visualizations()
        summary_path = self._update_benchmark_summary()
        return summary_path

    def _flush_tables(self) -> None:
        self._write_csv(self.model_dir / "loso_test_metrics.csv", self.fold_records)
        summary_rows = _mean_std_rows(self.fold_records, excluded={"fold", "test_subject"})
        self._write_csv(self.model_dir / "loso_summary_metrics.csv", summary_rows)
        self._write_csv(self.model_dir / "logs" / "training_history.csv", self.histories)

        for horizon in self.predictions:
            horizon_dir = self.model_dir / f"{horizon}min"
            horizon_dir.mkdir(exist_ok=True)
            rows = []
            prefix = f"horizon_{horizon}_"
            for fold_record in self.fold_records:
                row = {
                    "fold": fold_record["fold"],
                    "test_subject": fold_record["test_subject"],
                    "train_time_seconds": fold_record["train_time_seconds"],
                    "inference_time_seconds": fold_record["inference_time_seconds"],
                    "inference_ms_per_sample": fold_record["inference_ms_per_sample"],
                    "parameter_count": fold_record["parameter_count"],
                }
                row.update(
                    {
                        metric: fold_record[f"{prefix}{metric}"]
                        for metric in METRIC_NAMES
                    }
                )
                rows.append(row)
            self._write_csv(horizon_dir / "metrics.csv", rows)
            self._write_csv(horizon_dir / "prediction.csv", self.predictions[horizon])

    def _write_visualizations(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Benchmark visualization requires matplotlib. Install requirements-experiment.txt."
            ) from exc

        for horizon, rows in self.predictions.items():
            horizon_dir = self.model_dir / f"{horizon}min"
            targets = torch.tensor([row["target_mg_dl"] for row in rows], dtype=torch.float32)
            predictions = torch.tensor(
                [row["prediction_mg_dl"] for row in rows], dtype=torch.float32
            )
            _plot_loss_curves(plt, self.histories, horizon_dir / "loss_curve.png")
            _plot_scatter(plt, targets, predictions, horizon, horizon_dir / "scatter.png")
            _plot_ega(plt, targets, predictions, horizon, horizon_dir / "ega.png")

    def _update_benchmark_summary(self) -> Path:
        from .benchmark_reporting import rebuild_benchmark_reports

        summary_path, latex_path = rebuild_benchmark_reports(self.output_root)
        print(f"[INFO] Saved LaTeX benchmark table: {latex_path}")
        return summary_path

    def _append_training_log(self, record: Mapping[str, Any]) -> None:
        path = self.model_dir / "logs" / "training.log"
        with path.open("a", encoding="utf-8") as fp:
            fp.write(
                f"fold={int(record['fold']):02d} test_subject={record['test_subject']} "
                f"mae={float(record['mae']):.6f} rmse={float(record['rmse']):.6f} "
                f"train_time_s={float(record['train_time_seconds']):.3f} "
                f"inference_time_s={float(record['inference_time_seconds']):.6f}\n"
            )

    @staticmethod
    def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = [dict(row) for row in rows]
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)

    @staticmethod
    def _read_csv(path: Path) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as fp:
            for raw_row in csv.DictReader(fp):
                row: Dict[str, Any] = {}
                for key, value in raw_row.items():
                    if key == "test_subject":
                        row[key] = value
                    elif key in {"fold", "sample_index_within_fold"}:
                        row[key] = int(float(value))
                    else:
                        try:
                            row[key] = float(value)
                        except (TypeError, ValueError):
                            row[key] = value
                rows.append(row)
        return rows

    @staticmethod
    def _write_json(path: Path, data: Mapping[str, Any]) -> None:
        serializable = {
            key: str(value) if isinstance(value, Path) else value for key, value in data.items()
        }
        path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True), encoding="utf-8")


def _serialize_standardizer(standardizer: DataStandardizer | None) -> Dict[str, Tensor] | None:
    if standardizer is None:
        return None
    return {
        "glucose_mean": standardizer.glucose_mean.detach().cpu(),
        "glucose_std": standardizer.glucose_std.detach().cpu(),
        "physio_mean": standardizer.physio_mean.detach().cpu(),
        "physio_std": standardizer.physio_std.detach().cpu(),
    }


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    return value


def _mean_std_rows(
    records: Sequence[Mapping[str, Any]], excluded: set[str]
) -> list[Dict[str, Any]]:
    numeric_keys = [
        key
        for key, value in records[0].items()
        if key not in excluded and isinstance(value, (int, float))
    ]
    mean_row: Dict[str, Any] = {"stat": "mean"}
    std_row: Dict[str, Any] = {"stat": "std"}
    for key in numeric_keys:
        values = [float(record[key]) for record in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        mean_row[key] = mean
        std_row[key] = math.sqrt(variance)
    return [mean_row, std_row]


def _plot_loss_curves(plt: Any, histories: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    folds = sorted({int(row["fold"]) for row in histories})
    for fold in folds:
        rows = [row for row in histories if int(row["fold"]) == fold]
        epochs = [int(row["epoch"]) for row in rows]
        axis.plot(epochs, [float(row["train_loss"]) for row in rows], color="#4477AA", alpha=0.22)
        axis.plot(epochs, [float(row["val_loss"]) for row in rows], color="#CC6677", alpha=0.22)
    axis.plot([], [], color="#4477AA", label="Training loss")
    axis.plot([], [], color="#CC6677", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_scatter(plt: Any, target: Tensor, prediction: Tensor, horizon: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 5.2), dpi=160)
    low = float(torch.min(torch.cat([target, prediction])).floor())
    high = float(torch.max(torch.cat([target, prediction])).ceil())
    padding = max(5.0, 0.04 * (high - low))
    low, high = low - padding, high + padding
    axis.scatter(target.numpy(), prediction.numpy(), s=9, alpha=0.28, color="#336699", edgecolors="none")
    axis.plot([low, high], [low, high], color="#222222", linewidth=1.2)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Reference glucose (mg/dL)")
    axis.set_ylabel("Predicted glucose (mg/dL)")
    axis.set_title(f"{horizon}-min prediction")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_ega(plt: Any, target: Tensor, prediction: Tensor, horizon: int, path: Path) -> None:
    zones = clarke_error_grid_zones(target, prediction)
    colors = ("#2E8B57", "#4C78A8", "#F2CF5B", "#E6863B", "#C44E52")
    labels = ("A", "B", "C", "D", "E")
    figure, axis = plt.subplots(figsize=(5.6, 5.2), dpi=160)
    upper = max(400.0, float(torch.max(torch.cat([target, prediction])).ceil()))
    for index, (label, color) in enumerate(zip(labels, colors)):
        mask = zones == index
        count = int(mask.sum())
        percentage = 100.0 * count / max(1, zones.numel())
        axis.scatter(
            target[mask].numpy(),
            prediction[mask].numpy(),
            s=9,
            alpha=0.34,
            color=color,
            edgecolors="none",
            label=f"Zone {label}: {percentage:.1f}%",
        )
    axis.plot([0.0, upper], [0.0, upper], color="#222222", linewidth=1.0)
    reference_axis = torch.linspace(0.0, upper, 300)
    axis.plot(reference_axis, 0.8 * reference_axis, color="#777777", linewidth=0.8, linestyle="--")
    axis.plot(reference_axis, 1.2 * reference_axis, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(0.0, upper)
    axis.set_ylim(0.0, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Reference glucose (mg/dL)")
    axis.set_ylabel("Predicted glucose (mg/dL)")
    axis.set_title(f"Clarke error grid, {horizon} min")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.grid(alpha=0.12)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
