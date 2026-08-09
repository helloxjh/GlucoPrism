#!/usr/bin/env bash
set -euo pipefail

# Final GlucoPrism LOSO run selected by the long-horizon ablation study.
# It keeps the model backbone fixed and uses curriculum-only horizon training:
# baseline prediction head, no horizon consistency loss, and easy-to-hard
# horizon weights for 15/30/45/60 min.

PYTHON_BIN="${PYTHON_BIN:-.venv_torch/bin/python}"
DATA_DIR="${DATA_DIR:-processed_big_ideas_60min}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/glucoprism_final_curriculum}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-4}"
GRAPH_LAYERS="${GRAPH_LAYERS:-2}"
SEED="${SEED:-42}"
FINAL_WEIGHTS="1.0,1.05,1.25,1.45"
CURRICULUM_START_WEIGHTS="1.0,0.85,0.65,0.50"

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
  --loss-horizon-weights "${FINAL_WEIGHTS}" \
  --curriculum-start-weights "${CURRICULUM_START_WEIGHTS}" \
  --horizon-curriculum-fraction 0.5 \
  --horizon-head-mode baseline \
  --horizon-consistency-weight 0.0
