#!/usr/bin/env bash
set -euo pipefail

# Five controlled LOSO ablations for long-horizon prediction.
# Run from the project root:
#   bash scripts/run_long_horizon_ablation.sh
#
# Each run uses the same data, split seed, optimizer defaults, model width, and
# original graph_loss horizon weights. Only the named factor is changed.

PYTHON_BIN="${PYTHON_BIN:-.venv_torch/bin/python}"
DATA_DIR="${DATA_DIR:-processed_big_ideas_60min}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-4}"
GRAPH_LAYERS="${GRAPH_LAYERS:-2}"
SEED="${SEED:-42}"
BASE_WEIGHTS="1.0,1.05,1.25,1.45"
EASY_TO_HARD_WEIGHTS="1.0,0.85,0.65,0.50"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-heads "${NUM_HEADS}"
  --graph-layers "${GRAPH_LAYERS}"
  --seed "${SEED}"
  --loss glucoprism
  --loss-horizon-weights "${BASE_WEIGHTS}"
)

run_exp() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  if [[ -e "runs/${name}" ]]; then
    echo "Output directory runs/${name} already exists. Use a new name or clean it manually before rerunning." >&2
    exit 1
  fi
  "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" --output-dir "runs/${name}" "$@"
}

run_exp "ablation_00_graph_loss_original" \
  --horizon-head-mode baseline \
  --horizon-consistency-weight 0.0 \
  --curriculum-start-weights "${BASE_WEIGHTS}" \
  --no-horizon-curriculum

run_exp "ablation_01_horizon_head_only" \
  --horizon-head-mode refined \
  --horizon-consistency-weight 0.0 \
  --curriculum-start-weights "${BASE_WEIGHTS}" \
  --no-horizon-curriculum

run_exp "ablation_02_horizon_consistency_only" \
  --horizon-head-mode baseline \
  --horizon-consistency-weight 0.10 \
  --curriculum-start-weights "${BASE_WEIGHTS}" \
  --no-horizon-curriculum

run_exp "ablation_03_curriculum_only" \
  --horizon-head-mode baseline \
  --horizon-consistency-weight 0.0 \
  --curriculum-start-weights "${EASY_TO_HARD_WEIGHTS}" \
  --horizon-curriculum-fraction 0.5

run_exp "ablation_04_head_plus_loss_no_curriculum" \
  --horizon-head-mode refined \
  --horizon-consistency-weight 0.10 \
  --curriculum-start-weights "${BASE_WEIGHTS}" \
  --no-horizon-curriculum
