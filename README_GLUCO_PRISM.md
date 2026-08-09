# GlucoPrism Experiment Scaffold

Recommended project structure:

```text
code_3/
├── data/
│   ├── alignment.py        # Multimodal 5-minute time alignment helpers
│   ├── datasets.py         # Real and dummy datasets
│   ├── missing.py          # Missing-value handling
│   ├── normalization.py    # Fold-wise train-only normalization
│   ├── preprocessing.py    # Raw CSV -> aligned window preprocessing wrapper
│   └── splits.py           # Subject-level LOSO-CV split builder
├── features/
│   ├── physio_features.py  # HR/EDA/TEMP/ACC/CGM feature helpers
│   ├── segmentation.py     # Time-series segmentation
│   └── windowing.py        # Sliding-window construction
├── models/
│   ├── multiscale_time_encoder.py
│   ├── physio_graph_encoder.py
│   ├── bidirectional_cross_attention.py
│   ├── multi_horizon_prediction_head.py
│   └── st_msffnet.py       # Complete GlucoPrism/ST-MSFFNet model
├── training/
│   ├── trainer.py          # Train/val/test LOSO runner
│   ├── losses.py
│   ├── optim.py
│   └── early_stopping.py
├── evaluation/
│   ├── metrics.py          # MAE/RMSE/MARD/R2
│   ├── clarke_ega.py       # Clarke EGA Zone A-E
│   └── evaluator.py
├── experiments/
│   ├── config.py           # argparse config
│   ├── logging.py          # TensorBoard + CSV logging
│   └── seeds.py            # Reproducibility helper
├── processed_big_ideas/    # Generated preprocessing output
├── preprocess_big_ideas.py # Raw CSV -> aligned window dataset
└── main.py                 # End-to-end experiment entry point
```

Current tensor contract:

```text
X_cgm:    [N, 24, 1]
X_physio: [N, 6, 24]
Y:        [N, 12]  # required for 15/30/45/60min targets
```

`GlucoPrism.forward` expects:

```text
cgm:    Tensor [batch_size, 24, 1]
physio: Tensor [batch_size, 6, 24]
```

and returns:

```text
pred: Tensor [batch_size, 4]  # [15min, 30min, 45min, 60min]
```

LOSO-CV design:

For each of the 16 folds, one subject is used as test set, 3 of the remaining
15 subjects are selected as validation subjects, and the other 12 subjects are
used for training.

Validated experiment environment:

```bash
.venv_torch/bin/python train.py
```

For the full 15/30/45/60min experiment, regenerate labels with:

```bash
.venv/bin/python preprocess_big_ideas.py --horizon-steps 12 --output-dir processed_big_ideas_60min
```

The LOSO training pipeline reports pooled and per-horizon MAE, RMSE, MARD, R2,
and Clarke EGA Zone A-E percentages:

```bash
.venv_torch/bin/python main.py --data-dir processed_big_ideas_60min
```

By default, each LOSO fold fits standardization statistics on training subjects only,
then applies them to train/validation/test samples. Metrics are inverse-transformed
back to mg/dL before reporting. Use `--no-standardize` only for ablation.

After training, the output directory contains:

```text
metrics.csv               # train/val/test scalar logs
loso_test_metrics.csv     # one row per LOSO test fold
loso_summary_metrics.csv  # mean/std across folds
tensorboard/              # TensorBoard event files when tensorboard is installed
```

Core dependency versions are recorded in `requirements-experiment.txt`.
