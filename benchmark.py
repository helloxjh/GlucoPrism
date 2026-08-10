#!/usr/bin/env python3
"""Unified LOSO benchmark entry point for GlucoPrism and paper baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.datasets import DummyPhysioNetDataset, ProcessedBigIdeasDataset
from data.ohiot1dm_dataset import ProcessedOhioT1DMDataset
from data.splits import build_loso_splits
from evaluation.metrics import format_metrics
from experiments.benchmark_artifacts import BenchmarkArtifactWriter
from experiments.logging import ExperimentLogger
from experiments.seeds import resolve_device, set_seed
from models.registry import AVAILABLE_MODELS, get_registered_model_config
from training.trainer import run_loso_training


DATASET_DEFAULTS = {
    "big_ideas": {
        "data_dir": Path("processed_big_ideas_60min"),
        "output_root": Path("results"),
        "num_val_subjects": 3,
    },
    "ohiot1dm": {
        "data_dir": Path("processed_ohiot1dm_60min"),
        "output_root": Path("results") / "OhioT1DM",
        "num_val_subjects": 1,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified LOSO-CV benchmark for continuous glucose prediction."
    )
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DEFAULTS),
        default="big_ideas",
        help="Dataset protocol to use; defaults preserve the original BIG IDEAs benchmark.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--use-dummy", action="store_true")
    parser.add_argument("--num-subjects", type=int, default=16)
    parser.add_argument("--samples-per-subject", type=int, default=8)
    parser.add_argument("--num-val-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--graph-reg-weight", type=float, default=2e-3)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--loss",
        default="glucoprism",
        choices=("glucoprism", "composite", "smooth_l1", "huber", "mse", "mae"),
    )
    parser.add_argument("--loss-horizon-weights", default="1.0,1.05,1.25,1.45")
    parser.add_argument("--curriculum-start-weights", default="1.0,0.85,0.65,0.50")
    parser.add_argument("--no-horizon-curriculum", action="store_true")
    parser.add_argument("--horizon-curriculum-fraction", type=float, default=0.5)
    parser.add_argument("--horizon-consistency-weight", type=float, default=0.0)
    parser.add_argument("--horizon-head-mode", choices=("baseline", "refined"), default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected model's complete result directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve completed folds and continue the selected model's interrupted run.",
    )
    return parser


def parse_float_tuple(value: str, argument_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{argument_name} must be a comma-separated float list.") from exc
    if len(values) != 4:
        raise ValueError(f"{argument_name} must contain four weights for 15/30/45/60 min.")
    return values


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_defaults = DATASET_DEFAULTS[args.dataset]
    data_dir = args.data_dir or dataset_defaults["data_dir"]
    output_root = args.output_root or dataset_defaults["output_root"]
    num_val_subjects = (
        args.num_val_subjects
        if args.num_val_subjects is not None
        else int(dataset_defaults["num_val_subjects"])
    )
    if num_val_subjects <= 0:
        raise ValueError("--num-val-subjects must be positive.")

    if args.use_dummy:
        dataset = DummyPhysioNetDataset(
            num_subjects=args.num_subjects,
            samples_per_subject=args.samples_per_subject,
            history_steps=24,
            num_physio_nodes=6,
            future_steps=12,
            seed=args.seed,
        )
        print(f"[INFO] Using dummy data: samples={len(dataset)}")
    elif args.dataset == "big_ideas":
        dataset = ProcessedBigIdeasDataset(data_dir, required_future_steps=12)
        print(
            f"[INFO] Using BIG IDEAs data: samples={len(dataset)}, "
            f"subjects={len(set(dataset.subject_ids))}, history_steps={dataset.history_steps}"
        )
    else:
        dataset = ProcessedOhioT1DMDataset(data_dir, required_future_steps=12)
        print(
            f"[INFO] Using OhioT1DM data: samples={len(dataset)}, "
            f"subjects={len(set(dataset.subject_ids))}, history_steps={dataset.history_steps}, "
            f"physio_nodes={dataset.node_names}"
        )

    splits = build_loso_splits(
        dataset.subject_ids,
        num_val_subjects=num_val_subjects,
        seed=args.seed,
    )
    if args.fold_start < 1 or args.fold_start > len(splits):
        raise ValueError(f"--fold-start must be between 1 and {len(splits)}.")
    fold_end = args.fold_limit if args.fold_limit is not None else len(splits)
    if fold_end < args.fold_start or fold_end > len(splits):
        raise ValueError(
            f"--fold-limit must be between --fold-start and {len(splits)}."
        )
    selected_splits = splits[args.fold_start - 1 : fold_end]

    config = vars(args).copy()
    config.update(
        {
            "data_dir": str(data_dir),
            "output_root": str(output_root),
            "num_val_subjects": num_val_subjects,
            "dataset_resolved": "dummy" if args.use_dummy else args.dataset,
            "data_dir_resolved": str(data_dir),
            "output_root_resolved": str(output_root),
            "num_val_subjects_resolved": num_val_subjects,
            "device_resolved": str(device),
            "history_steps": dataset.history_steps,
            "num_physio_nodes": dataset.num_physio_nodes,
            "node_names": tuple(getattr(dataset, "node_names", ())),
            "prediction_horizons_minutes": (15, 30, 45, 60),
            "selected_fold_ids": tuple(split.fold_id for split in selected_splits),
            "model_architecture": get_registered_model_config(args.model),
        }
    )
    artifacts = BenchmarkArtifactWriter(
        output_root,
        args.model,
        config,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    completed_fold_ids = artifacts.completed_fold_ids
    unknown_completed_folds = completed_fold_ids.difference(
        split.fold_id for split in splits
    )
    if unknown_completed_folds:
        raise RuntimeError(
            f"Existing benchmark contains unknown folds: {sorted(unknown_completed_folds)}"
        )
    if completed_fold_ids:
        selected_splits = [
            split for split in selected_splits if split.fold_id not in completed_fold_ids
        ]
        print(
            f"[INFO] Resume skipped {len(completed_fold_ids)} completed folds; "
            f"remaining folds: {[split.fold_id for split in selected_splits]}"
        )
    logger = ExperimentLogger(artifacts.model_dir / "logs" / "runtime")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Benchmark output: {artifacts.model_dir}")

    try:
        if selected_splits:
            run_loso_training(
                dataset=dataset,
                splits=selected_splits,
                device=device,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                graph_layers=args.graph_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                min_lr=args.min_lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                graph_reg_weight=args.graph_reg_weight,
                ema_decay=args.ema_decay,
                patience=args.patience,
                loss_name=args.loss,
                ablation_mode="full",
                horizon_head_mode=args.horizon_head_mode,
                horizon_consistency_weight=args.horizon_consistency_weight,
                loss_horizon_weights=parse_float_tuple(
                    args.loss_horizon_weights, "--loss-horizon-weights"
                ),
                curriculum_start_weights=parse_float_tuple(
                    args.curriculum_start_weights, "--curriculum-start-weights"
                ),
                enable_horizon_curriculum=not args.no_horizon_curriculum,
                horizon_curriculum_fraction=args.horizon_curriculum_fraction,
                standardize=not args.no_standardize,
                logger=logger,
                model_name=args.model,
                lstm_num_layers=args.lstm_num_layers,
                artifact_writer=artifacts,
                random_seed=args.seed,
            )
        else:
            print("[INFO] No folds remain; rebuilding final artifacts from saved results.")
        summary_path = artifacts.finalize()
        fold_metrics = artifacts.metric_records()
        if fold_metrics:
            mean_metrics = {
                key: sum(record[key] for record in fold_metrics) / len(fold_metrics)
                for key in fold_metrics[0]
            }
            mean_loss = mean_metrics.pop("loss")
            print(
                f"\n[Benchmark Summary] model={args.model} folds={len(fold_metrics)} "
                f"{format_metrics('mean', mean_loss, mean_metrics)}"
            )
        print(f"[INFO] Saved benchmark summary: {summary_path}")
        print(f"[INFO] Saved benchmark LaTeX table: {output_root / 'benchmark_table.tex'}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
