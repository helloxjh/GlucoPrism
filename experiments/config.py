"""Argparse configuration for GlucoPrism experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path | None
    output_dir: Path
    use_dummy: bool
    num_subjects: int
    samples_per_subject: int
    num_val_subjects: int
    epochs: int
    batch_size: int
    hidden_dim: int
    num_heads: int
    graph_layers: int
    dropout: float
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    graph_reg_weight: float
    ema_decay: float
    patience: int
    loss: str
    ablation_mode: str
    horizon_head_mode: str
    horizon_consistency_weight: float
    loss_horizon_weights: Tuple[float, ...]
    curriculum_start_weights: Tuple[float, ...]
    enable_horizon_curriculum: bool
    horizon_curriculum_fraction: float
    seed: int
    device: str
    fold_start: int
    fold_limit: int | None
    standardize: bool


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="GlucoPrism LOSO-CV experiment runner.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with preprocessed NPZ/metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/glucoprism"))
    parser.add_argument("--use-dummy", action="store_true", help="Use synthetic data instead of --data-dir.")
    parser.add_argument("--num-subjects", type=int, default=16)
    parser.add_argument("--samples-per-subject", type=int, default=8)
    parser.add_argument("--num-val-subjects", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
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
        choices=["glucoprism", "composite", "smooth_l1", "huber", "mse", "mae"],
    )
    parser.add_argument(
        "--ablation-mode",
        default="full",
        choices=[
            "full",
            "cgm_only",
            "no_graph",
            "no_cross_attention",
            "single_scale_temporal",
            "fixed_prior_graph",
        ],
        help="Controlled core-module ablation; full preserves the complete GlucoPrism model.",
    )
    parser.add_argument(
        "--horizon-head-mode",
        default="baseline",
        choices=["baseline", "refined"],
        help="baseline recovers the original query-attention head; refined enables horizon-aware refinements.",
    )
    parser.add_argument("--horizon-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--loss-horizon-weights",
        default="1.0,1.05,1.25,1.45",
        help="Comma-separated final loss weights for 15/30/45/60 min.",
    )
    parser.add_argument(
        "--curriculum-start-weights",
        default="1.0,0.85,0.65,0.50",
        help="Comma-separated initial curriculum weights for 15/30/45/60 min.",
    )
    parser.add_argument("--no-horizon-curriculum", action="store_true")
    parser.add_argument("--horizon-curriculum-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold-start", type=int, default=1, help="1-based first LOSO fold to run.")
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_dummy=args.use_dummy,
        num_subjects=args.num_subjects,
        samples_per_subject=args.samples_per_subject,
        num_val_subjects=args.num_val_subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        graph_reg_weight=args.graph_reg_weight,
        ema_decay=args.ema_decay,
        patience=args.patience,
        loss=args.loss,
        ablation_mode=args.ablation_mode,
        horizon_head_mode=args.horizon_head_mode,
        horizon_consistency_weight=args.horizon_consistency_weight,
        loss_horizon_weights=_parse_float_tuple(args.loss_horizon_weights, "loss-horizon-weights"),
        curriculum_start_weights=_parse_float_tuple(args.curriculum_start_weights, "curriculum-start-weights"),
        enable_horizon_curriculum=not args.no_horizon_curriculum,
        horizon_curriculum_fraction=args.horizon_curriculum_fraction,
        seed=args.seed,
        device=args.device,
        fold_start=args.fold_start,
        fold_limit=args.fold_limit,
        standardize=not args.no_standardize,
    )


def _parse_float_tuple(value: str, name: str) -> Tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"--{name} must be a comma-separated float list, got {value!r}") from exc
    if not parsed:
        raise ValueError(f"--{name} must not be empty")
    return parsed
