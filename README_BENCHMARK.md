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
