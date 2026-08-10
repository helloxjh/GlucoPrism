"""Training loops and LOSO-CV runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from data.normalization import DataStandardizer, StandardizedDataset, fit_standardizer
from data.splits import LOSOSplit
from evaluation.evaluator import (
    evaluate,
    evaluate_detailed,
    move_batch_to_device,
    synchronize_device,
    validate_batch_shapes,
)
from evaluation.metrics import format_horizon_metrics, format_metrics
from experiments.logging import ExperimentLogger
from experiments.seeds import set_seed
from models.registry import (
    build_forecasting_model,
    count_parameters,
    get_horizon_minutes,
    get_required_horizon_steps,
)
from .early_stopping import EarlyStopping
from .ema import ModelEMA
from .losses import build_loss
from .optim import WarmupScheduler, build_optimizer, build_scheduler

if TYPE_CHECKING:
    from experiments.benchmark_artifacts import BenchmarkArtifactWriter


WARMUP_EPOCHS = 3
HORIZON_CURRICULUM_FRACTION = 0.5


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    graph_reg_weight: float,
    ema: Optional[ModelEMA] = None,
) -> float:
    """Train for one epoch. Loss in standardized glucose space."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    required_steps = get_required_horizon_steps(model)
    for batch_idx, raw_batch in enumerate(loader):
        try:
            batch = move_batch_to_device(raw_batch, device)
            validate_batch_shapes(batch, required_steps)
            cgm = batch["cgm"]
            physio = batch["physio"]
            future = batch["future_glucose"]
            assert isinstance(cgm, Tensor)
            assert isinstance(physio, Tensor)
            assert isinstance(future, Tensor)

            pred = model(cgm, physio)  # [B, K]
            target = model.select_targets(future)  # [B, K]
            last_cgm = cgm[:, -1, 0:1]  # [B, 1]
            loss = criterion(pred, target, last_cgm=last_cgm)
            if graph_reg_weight > 0.0 and hasattr(model, "regularization_loss"):
                reg_loss = model.regularization_loss()
                loss = loss + graph_reg_weight * reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            total_loss += float(loss.item()) * cgm.shape[0]
            total_samples += cgm.shape[0]
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Training failed at batch {batch_idx}: {exc}") from exc
    return total_loss / max(total_samples, 1)


def run_loso_training(
    dataset: Dataset,
    splits: Sequence[LOSOSplit],
    device: torch.device,
    hidden_dim: int,
    num_heads: int,
    graph_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    grad_clip: float,
    graph_reg_weight: float,
    ema_decay: float,
    patience: int,
    loss_name: str,
    ablation_mode: str,
    horizon_head_mode: str,
    horizon_consistency_weight: float,
    loss_horizon_weights: Sequence[float],
    curriculum_start_weights: Sequence[float],
    enable_horizon_curriculum: bool,
    horizon_curriculum_fraction: float,
    standardize: bool,
    logger: ExperimentLogger,
    model_name: str = "glucoprism",
    lstm_num_layers: int = 1,
    artifact_writer: Optional["BenchmarkArtifactWriter"] = None,
    random_seed: int | None = None,
) -> List[Dict[str, float]]:
    """Run LOSO-CV and return per-fold test metrics."""
    if horizon_head_mode not in {"baseline", "refined"}:
        raise ValueError("horizon_head_mode must be 'baseline' or 'refined'.")
    if not 0.0 <= horizon_curriculum_fraction <= 1.0:
        raise ValueError("horizon_curriculum_fraction must be in [0, 1].")
    criterion = build_loss(
        loss_name,
        horizon_weights=loss_horizon_weights,
        curriculum_start_weights=curriculum_start_weights,
        horizon_consistency_weight=horizon_consistency_weight,
    )
    fold_metrics: List[Dict[str, float]] = []
    history_steps = int(getattr(dataset, "history_steps"))
    num_physio_nodes = int(getattr(dataset, "num_physio_nodes"))
    node_names = tuple(getattr(dataset, "node_names", ()))
    A_prior = getattr(dataset, "A_prior", None)
    if node_names and len(node_names) != num_physio_nodes:
        raise ValueError(
            f"dataset node_names has {len(node_names)} entries, "
            f"expected {num_physio_nodes}."
        )
    print(f"[INFO] Benchmark model: {model_name}")
    if model_name.lower() == "glucoprism":
        print(f"[INFO] Core-module ablation mode: {ablation_mode}")

    for split in splits:
        fold_seed = random_seed + split.fold_id if random_seed is not None else None
        if fold_seed is not None:
            set_seed(fold_seed)
        print(
            f"\n[Fold {split.fold_id:02d}] test={split.test_subject}, "
            f"val={split.val_subjects}, train_subjects={len(split.train_subjects)}"
        )
        # --- Fold-wise standardization (fit on train only) ---
        if standardize:
            standardizer: Optional[DataStandardizer] = fit_standardizer(
                dataset, split.train_indices
            )
            fold_dataset: Dataset = StandardizedDataset(dataset, standardizer)
            print(
                f"  [INFO] Standardization: glucose_mean={float(standardizer.glucose_mean):.2f}, "
                f"glucose_std={float(standardizer.glucose_std):.2f}"
            )
        else:
            standardizer = None
            fold_dataset = dataset
            print("  [INFO] Standardization disabled.")

        train_loader = make_loader(
            fold_dataset, split.train_indices, batch_size, True, seed=fold_seed
        )
        val_loader = make_loader(fold_dataset, split.val_indices, batch_size, False)
        test_loader = make_loader(fold_dataset, split.test_indices, batch_size, False)

        model = build_forecasting_model(
            model_name,
            history_steps=history_steps,
            num_physio_nodes=num_physio_nodes,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            graph_layers=graph_layers,
            dropout=dropout,
            ablation_mode=ablation_mode,
            enable_horizon_refinement=(horizon_head_mode == "refined"),
            lstm_num_layers=lstm_num_layers,
            node_names=node_names or None,
            A_prior=A_prior,
        ).to(device)
        total_parameters, trainable_parameters = count_parameters(model)
        print(
            f"  [INFO] Parameters: total={total_parameters:,}, "
            f"trainable={trainable_parameters:,}"
        )

        optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
        base_scheduler = build_scheduler(
            optimizer,
            epochs=max(1, epochs - WARMUP_EPOCHS),
            min_lr=min_lr,
        )
        warmup = WarmupScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, base_lr=lr)
        early_stopping = EarlyStopping(patience=patience)
        ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
        curriculum_epochs = max(
            WARMUP_EPOCHS + 1,
            int(round(epochs * horizon_curriculum_fraction)),
        )
        history: List[Dict[str, float]] = []
        synchronize_device(device)
        train_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            if enable_horizon_curriculum:
                curriculum_progress = min(
                    1.0,
                    (epoch - 1) / max(1, curriculum_epochs - 1),
                )
            else:
                curriculum_progress = 1.0
            if hasattr(criterion, "set_curriculum_progress"):
                criterion.set_curriculum_progress(curriculum_progress)
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                grad_clip,
                graph_reg_weight,
                ema,
            )
            loss_diagnostics = (
                criterion.horizon_weight_metrics()
                if hasattr(criterion, "horizon_weight_metrics")
                else {}
            )

            # Validation/early stopping always use the fixed final objective so
            # losses remain comparable across epochs despite training curriculum.
            if hasattr(criterion, "set_curriculum_progress"):
                criterion.set_curriculum_progress(1.0)
            if ema is not None:
                with ema.average_parameters(model):
                    val_loss, val_metrics = evaluate(
                        model, val_loader, criterion, device, standardizer
                    )
                    diagnostics = model.diagnostic_metrics() if hasattr(model, "diagnostic_metrics") else {}
            else:
                val_loss, val_metrics = evaluate(
                    model, val_loader, criterion, device, standardizer
                )
                diagnostics = model.diagnostic_metrics() if hasattr(model, "diagnostic_metrics") else {}

            warmup.step()
            if epoch > WARMUP_EPOCHS:
                base_scheduler.step()
            current_lr = warmup.get_last_lr()[0]

            logger.log_metrics(
                {
                    f"fold{split.fold_id}/train_loss": train_loss,
                    **{f"fold{split.fold_id}/val_{k}": v for k, v in val_metrics.items()},
                    **{f"fold{split.fold_id}/diag_{k}": v for k, v in diagnostics.items()},
                    f"fold{split.fold_id}/diag_horizon_curriculum_progress": curriculum_progress,
                    **{
                        f"fold{split.fold_id}/diag_{k}": v
                        for k, v in loss_diagnostics.items()
                    },
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
            )
            print(
                f"  epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"{format_metrics('val', val_loss, val_metrics)} lr={current_lr:.6f}"
            )

            if ema is not None:
                with ema.average_parameters(model):
                    should_stop = early_stopping.step(val_loss, model)
            else:
                should_stop = early_stopping.step(val_loss, model)
            if should_stop:
                print(f"  early_stop at epoch={epoch:02d}")
                break

        synchronize_device(device)
        train_time_seconds = time.perf_counter() - train_start
        early_stopping.restore(model, device)

        test_outputs = evaluate_detailed(
            model, test_loader, criterion, device, standardizer
        )
        test_loss, test_metrics = test_outputs.loss, test_outputs.metrics
        print(f"  {format_metrics('test', test_loss, test_metrics)}")
        print(
            f"  {format_horizon_metrics('test', test_metrics, get_horizon_minutes(model))}"
        )
        fold_record = {"loss": test_loss, **test_metrics}
        fold_metrics.append(fold_record)
        logger.log_metrics(
            {f"fold{split.fold_id}/test_{k}": v for k, v in fold_record.items()}, step=0
        )
        if artifact_writer is not None:
            artifact_writer.record_fold(
                split=split,
                model=model,
                standardizer=standardizer,
                evaluation=test_outputs,
                history=history,
                train_time_seconds=train_time_seconds,
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                horizon_minutes=get_horizon_minutes(model),
            )

    return fold_metrics
