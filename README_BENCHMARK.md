# GlucoPrism Benchmark Framework

The benchmark entry point runs every registered model through the same
subject-level LOSO split, fold-wise normalization, data loaders, optimizer,
loss, scheduler, early stopping, EMA, evaluator, metrics, and artifact writer.

## Commands

Run the standard LSTM baseline on the preprocessed BIG IDEAs dataset:

```bash
source .venv_torch/bin/activate
python benchmark.py --model lstm
```

Run GlucoPrism through the same benchmark path:

```bash
python benchmark.py --model glucoprism
```

Registered paper baselines use the same command form:

```bash
python benchmark.py --model cnn
python benchmark.py --model informer
python benchmark.py --model autoformer
python benchmark.py --model patchtst
python benchmark.py --model graphwavenet
python benchmark.py --model dcrnn
python benchmark.py --model crnn
```

## OhioT1DM External Validation

OhioT1DM uses four configurable physiological nodes in the order `activity`,
`gsr`, `skin_temperature`, and `heart_rate`. Preprocessing aggregates all
signals to 5-minute intervals, creates 24-step histories and 12-step targets
inside each XML recording, and never creates a window across the original
training/testing file boundary.

Prepare the six-subject OhioT1DM dataset:

```bash
source .venv_torch/bin/activate
python preprocess_ohiot1dm.py \
  --data-root OhioT1DM \
  --output-dir processed_ohiot1dm_60min
```

Run a one-fold, one-epoch integration check in an isolated result directory:

```bash
python benchmark.py \
  --dataset ohiot1dm \
  --model glucoprism \
  --epochs 1 \
  --fold-limit 1 \
  --output-root results/OhioT1DM_smoke \
  --overwrite
```

Run the formal six-fold LOSO experiment with the same model and training
hyperparameters used by the primary experiment:

```bash
bash scripts/run_ohiot1dm_glucoprism.sh
```

Formal outputs are isolated under `results/OhioT1DM/GlucoPrism`. If the run is
interrupted after one or more folds have completed, preserve those folds with:

```bash
bash scripts/run_ohiot1dm_glucoprism.sh --resume
```

Do not combine `--resume` with `--overwrite`. An interrupted fold is retrained;
fully completed folds are read from the existing artifacts and skipped.

An existing model result directory is never replaced implicitly. Use
`--overwrite` only when intentionally replacing that model's complete run.

## Result Layout

Each model writes fold checkpoints and shared training logs once because one
network jointly predicts all four horizons. Horizon-specific metrics,
predictions, and figures are written under `15min`, `30min`, `45min`, and
`60min`. Cross-model LOSO macro averages are maintained in
`results/benchmark_summary.csv`.

After every completed model run, the framework rebuilds
`results/benchmark_summary.csv` and `results/benchmark_table.tex` from the
saved LOSO fold files. The CSV contains per-horizon mean, population standard
deviation, best fold, overall statistics, parameter/timing summaries, and
average ranks. `Average_Rank` is computed over 24 comparison tasks: six primary
metrics (MAE, RMSE, MARD, R2, Pearson correlation, and Clarke Zone A) at four
prediction horizons, with tied models assigned their average rank.

## Adding a Baseline

Add the model implementation under `models/` with the shared forward contract
`model(cgm, physio) -> [batch, 4]` and a `select_targets` method. Register its
constructor and display name in `models/registry.py` and
`experiments/benchmark_artifacts.py`. Dataset, trainer, evaluator, metrics,
checkpointing, and visualization code must not be duplicated.
