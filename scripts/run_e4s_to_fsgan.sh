#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUT="$ROOT/results/e4s_to_fsgan_pilot"
DATA_ROOT="${DF40_DATA_ROOT:-/Volumes/pendrive/DF40/deepfakes_detection_datasets}"
export DF40_OLD_ROOT="${DF40_OLD_ROOT:-deepfakes_detection_datasets}"
export DF40_DATA_ROOT="$DATA_ROOT"

if [[ ! -f "$OUT/manifests/e4s_to_fsgan_train.json" || ! -f "$OUT/manifests/fsgan_to_e4s_test.json" ]]; then
    echo "Run scripts/prepare_e4s_to_fsgan.py --stage first." >&2
    exit 2
fi
if [[ -e "$OUT/checkpoints/ckpt_best.pth" || -e "$OUT/fsgan_test_metrics.json" ]]; then
    echo "Experiment outputs already exist; refusing to overwrite them." >&2
    exit 2
fi

mkdir -p "$OUT/logs" "$OUT/checkpoints"
"$PYTHON_BIN" scripts/preflight_e4s_to_fsgan.py \
    --manifest-dir "$OUT/manifests" \
    --dataset-root "$DATA_ROOT" \
    --output "$OUT/pilot_preflight.json"

TRAIN_START="$(date +%s)"
"$PYTHON_BIN" training/train.py \
    --detector_path "$ROOT/training/config/detector/dled.yaml" \
    --train_dataset e4s_to_fsgan_train \
    --test_dataset e4s_to_fsgan_val \
    --dataset_json_folder "$OUT/manifests" \
    --num_classes 2 \
    --training_mode single \
    --num_epochs 5 \
    --test_fre 1 \
    --max_samples_per_class 256 \
    --workers 0 \
    --save_path "$OUT/checkpoints" \
    --log_dir "$OUT/logs" \
    --task_target e4s_to_fsgan \
    --save_model \
    --avu_reg 2>&1 | tee "$OUT/logs/training_console.log"
TRAIN_END="$(date +%s)"
echo "training_wall_seconds=$((TRAIN_END - TRAIN_START))" > "$OUT/training_runtime.txt"

TRAIN_LOG="$(find "$OUT/logs" -type f -name training.log -print | sort | tail -n 1)"
if [[ -z "$TRAIN_LOG" ]]; then
    echo "Training log was not found after training." >&2
    exit 3
fi
VAL_METRICS="$(find "$OUT/logs" -type f -path '*/test/e4s_to_fsgan_val/metrics_epoch_005.json' -print | sort | tail -n 1)"
if [[ -z "$VAL_METRICS" ]]; then
    echo "Five-epoch validation metrics were not found after training." >&2
    exit 3
fi
VALIDATION_DIR="$(dirname "$VAL_METRICS")"
BEST_FROM_RUN="$VALIDATION_DIR/ckpt_best.pth"
if [[ ! -f "$BEST_FROM_RUN" || ! -f "$OUT/checkpoints/ckpt_latest.pth" ]]; then
    echo "Best validation or latest checkpoint is missing." >&2
    exit 3
fi
FINAL_SOURCE="$(find "$OUT/checkpoints" -maxdepth 1 -type f -name 'dled*.pth' -print | sort | tail -n 1)"
if [[ -z "$FINAL_SOURCE" ]]; then
    echo "Final timestamped checkpoint was not found." >&2
    exit 3
fi
cp "$BEST_FROM_RUN" "$OUT/checkpoints/best_validation.pth"
cp "$FINAL_SOURCE" "$OUT/checkpoints/final.pth"

"$PYTHON_BIN" scripts/export_epoch_metrics.py \
    --training-log "$TRAIN_LOG" \
    --validation-dir "$VALIDATION_DIR" \
    --output "$OUT/epoch_metrics.csv"

EVAL_START="$(date +%s)"
"$PYTHON_BIN" training/test.py \
    --model_path "$OUT/checkpoints/best_validation.pth" \
    --train_dataset e4s_to_fsgan_train \
    --test_dataset fsgan_to_e4s_test \
    --dataset_json_folder "$OUT/manifests" \
    --log_dir "$OUT/evaluation_logs" \
    --metrics_output "$OUT/fsgan_test_metrics.json" \
    --max_samples_per_class 256 \
    --num_classes 2 \
    --training_mode single \
    --workers 0 \
    --no_balance_test_data 2>&1 | tee "$OUT/logs/fsgan_evaluation_console.log"
EVAL_END="$(date +%s)"
echo "evaluation_wall_seconds=$((EVAL_END - EVAL_START))" >> "$OUT/training_runtime.txt"

echo "Training log: $TRAIN_LOG"
echo "Best validation checkpoint: $OUT/checkpoints/best_validation.pth"
echo "Latest checkpoint: $OUT/checkpoints/ckpt_latest.pth"
echo "Final checkpoint: $OUT/checkpoints/final.pth"
echo "FSGAN metrics: $OUT/fsgan_test_metrics.json"
