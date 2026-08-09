#!/usr/bin/env python3
"""GlucoPrism end-to-end LOSO-CV experiment entry point."""

from __future__ import annotations

import math

from data.datasets import DummyPhysioNetDataset, ProcessedBigIdeasDataset
from data.splits import build_loso_splits
from evaluation.metrics import format_metrics
from experiments import ExperimentLogger, parse_args, resolve_device, set_seed
from training import run_loso_training


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger = ExperimentLogger(cfg.output_dir)
    print(f"[INFO] Device: {device}")

    if cfg.use_dummy or cfg.data_dir is None:
        dataset = DummyPhysioNetDataset(
            num_subjects=cfg.num_subjects,
            samples_per_subject=cfg.samples_per_subject,
            history_steps=24,
            num_physio_nodes=6,
            future_steps=12,
            seed=cfg.seed,
        )
        print("[INFO] Using dummy data. For real data pass --data-dir processed_big_ideas_60min.")
    else:
        dataset = ProcessedBigIdeasDataset(cfg.data_dir, required_future_steps=12)
        print(
            f"[INFO] Using real data: samples={len(dataset)}, subjects={len(set(dataset.subject_ids))}, "
            f"history_steps={dataset.history_steps}, future_steps={dataset.future_steps}"
        )

    splits = build_loso_splits(dataset.subject_ids, num_val_subjects=cfg.num_val_subjects, seed=cfg.seed)
    if cfg.fold_start < 1:
        raise ValueError("--fold-start must be >= 1.")
    if cfg.fold_start > len(splits):
        raise ValueError(f"--fold-start {cfg.fold_start} exceeds total folds {len(splits)}.")
    fold_end = cfg.fold_limit if cfg.fold_limit is not None else len(splits)
    if fold_end < cfg.fold_start:
        raise ValueError("--fold-limit must be >= --fold-start when both are provided.")
    splits = splits[cfg.fold_start - 1 : fold_end]

    fold_metrics = run_loso_training(
        dataset=dataset,
        splits=splits,
        device=device,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        graph_layers=cfg.graph_layers,
        dropout=cfg.dropout,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        graph_reg_weight=cfg.graph_reg_weight,
        ema_decay=cfg.ema_decay,
        patience=cfg.patience,
        loss_name=cfg.loss,
        ablation_mode=cfg.ablation_mode,
        horizon_head_mode=cfg.horizon_head_mode,
        horizon_consistency_weight=cfg.horizon_consistency_weight,
        loss_horizon_weights=cfg.loss_horizon_weights,
        curriculum_start_weights=cfg.curriculum_start_weights,
        enable_horizon_curriculum=cfg.enable_horizon_curriculum,
        horizon_curriculum_fraction=cfg.horizon_curriculum_fraction,
        standardize=cfg.standardize,
        logger=logger,
    )

    if fold_metrics:
        fold_rows = [{"fold": split.fold_id, **metrics} for split, metrics in zip(splits, fold_metrics)]
        fold_csv = logger.write_table("loso_test_metrics.csv", fold_rows)

        avg = {key: sum(item[key] for item in fold_metrics) / len(fold_metrics) for key in fold_metrics[0]}
        std = {
            key: math.sqrt(sum((item[key] - avg[key]) ** 2 for item in fold_metrics) / len(fold_metrics))
            for key in fold_metrics[0]
        }
        summary_csv = logger.write_table(
            "loso_summary_metrics.csv",
            [
                {"stat": "mean", **avg},
                {"stat": "std", **std},
            ],
        )

        loss = avg.pop("loss")
        print(f"\n[LOSO Summary] folds={len(fold_metrics)} {format_metrics('avg', loss, avg)}")
        print(f"[INFO] Saved fold metrics: {fold_csv}")
        print(f"[INFO] Saved summary metrics: {summary_csv}")
        logger.log_metrics({f"summary/{key}": value for key, value in {"loss": loss, **avg}.items()}, step=0)
    logger.close()


if __name__ == "__main__":
    main()
