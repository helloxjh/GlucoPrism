#!/usr/bin/env bash
set -euo pipefail

# Resume only the missing LOSO folds for:
# runs/ablation_04_head_plus_loss_no_curriculum
#
# This preserves the ablation_04 settings from run_long_horizon_ablation.sh and
# runs folds 13-16 only. After it finishes, rebuild the full LOSO CSV tables with
# scripts/rebuild_loso_tables_from_metrics.py.

PYTHON_BIN="${PYTHON_BIN:-.venv_torch/bin/python}"
DATA_DIR="${DATA_DIR:-processed_big_ideas_60min}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ablation_04_head_plus_loss_no_curriculum}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-4}"
GRAPH_LAYERS="${GRAPH_LAYERS:-2}"
SEED="${SEED:-42}"
BASE_WEIGHTS="1.0,1.05,1.25,1.45"

"${PYTHON_BIN}" main.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-heads "${NUM_HEADS}" \
  --graph-layers "${GRAPH_LAYERS}" \
  --seed "${SEED}" \
  --loss glucoprism \
  --loss-horizon-weights "${BASE_WEIGHTS}" \
  --horizon-head-mode refined \
  --horizon-consistency-weight 0.10 \
  --curriculum-start-weights "${BASE_WEIGHTS}" \
  --no-horizon-curriculum \
  --fold-start 13 \
  --fold-limit 16
