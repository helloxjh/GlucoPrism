#!/usr/bin/env bash
set -euo pipefail

# Controlled 16-fold LOSO ablations for GlucoPrism's core modules. All runs use
# the selected curriculum-only training strategy and differ only in ablation mode.

PYTHON_BIN="${PYTHON_BIN:-.venv_torch/bin/python}"
DATA_DIR="${DATA_DIR:-processed_big_ideas_60min}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-4}"
GRAPH_LAYERS="${GRAPH_LAYERS:-2}"
SEED="${SEED:-42}"
FINAL_WEIGHTS="1.0,1.05,1.25,1.45"
CURRICULUM_START_WEIGHTS="1.0,0.85,0.65,0.50"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-heads "${NUM_HEADS}"
  --graph-layers "${GRAPH_LAYERS}"
  --seed "${SEED}"
  --loss glucoprism
  --loss-horizon-weights "${FINAL_WEIGHTS}"
  --curriculum-start-weights "${CURRICULUM_START_WEIGHTS}"
  --horizon-curriculum-fraction 0.5
  --horizon-head-mode baseline
  --horizon-consistency-weight 0.0
)

run_exp() {
  local name="$1"
  local mode="$2"
  echo
  echo "========== ${name} (${mode}) =========="
  if [[ -e "runs/${name}" ]]; then
    echo "Output directory runs/${name} already exists. Use a new name or remove it before rerunning." >&2
    exit 1
  fi
  "${PYTHON_BIN}" main.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "runs/${name}" \
    --ablation-mode "${mode}"
}

run_exp "core_ablation_00_full" "full"
run_exp "core_ablation_01_cgm_only" "cgm_only"
run_exp "core_ablation_02_no_graph" "no_graph"
run_exp "core_ablation_03_no_cross_attention" "no_cross_attention"
run_exp "core_ablation_04_single_scale_temporal" "single_scale_temporal"
run_exp "core_ablation_05_fixed_prior_graph" "fixed_prior_graph"
