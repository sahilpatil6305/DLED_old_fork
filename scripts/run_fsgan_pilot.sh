#!/usr/bin/env bash
# Run a small binary DLED pilot only after the authorized data passes checks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
PILOT_DIR="${DLED_FSGAN_PILOT_DIR:-$ROOT/results/fsgan_pilot/preparation_$RUN_STAMP}"
DATA_ROOT="${DF40_DATA_ROOT:-}"
EPOCHS="${DLED_FSGAN_PILOT_EPOCHS:-5}"
MAX_SAMPLES="${DLED_FSGAN_PILOT_MAX_SAMPLES:-256}"

if [[ -z "$DATA_ROOT" ]]; then
    echo "Set DF40_DATA_ROOT to the authorized extracted dataset root." >&2
    exit 2
fi
if [[ -e "$PILOT_DIR" ]]; then
    echo "Refusing to reuse existing run preparation directory: $PILOT_DIR" >&2
    exit 2
fi

"$PYTHON_BIN" scripts/prepare_fsgan_pilot.py \
    --source-json "$ROOT/dataset_json/fsgan_ff.json" \
    --dataset-root "$DATA_ROOT" \
    --old-root "${DF40_OLD_ROOT:-deepfakes_detection_datasets}" \
    --output-dir "$PILOT_DIR/manifests"

mkdir -p "$PILOT_DIR/logs"
"$PYTHON_BIN" training/train.py \
    --detector_path "$ROOT/training/config/detector/dled.yaml" \
    --train_dataset fsgan_pilot_train \
    --test_dataset fsgan_pilot_val \
    --dataset_json_folder "$PILOT_DIR/manifests" \
    --num_classes 2 \
    --training_mode single \
    --num_epochs "$EPOCHS" \
    --test_fre 1 \
    --max_samples_per_class "$MAX_SAMPLES" \
    --experiment_type fsgan_pilot \
    --experiment_type fsgan_pilot \
    --save_model \
    --avu_reg \
    2>&1 | tee "$PILOT_DIR/logs/launcher_console.log"
