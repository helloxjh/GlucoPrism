#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${OHIOT1DM_PROCESSED_DIR:-processed_ohiot1dm_60min}"

if [[ ! -f "${DATA_DIR}/ohiot1dm_windows.npz" ]]; then
  "${PYTHON_BIN}" preprocess_ohiot1dm.py \
    --data-root "${OHIOT1DM_RAW_DIR:-OhioT1DM}" \
    --output-dir "${DATA_DIR}"
fi

"${PYTHON_BIN}" benchmark.py \
  --dataset ohiot1dm \
  --model glucoprism \
  --data-dir "${DATA_DIR}" \
  --output-root results/OhioT1DM \
  "$@"
