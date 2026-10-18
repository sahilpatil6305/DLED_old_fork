#!/usr/bin/env bash
# Unified training launcher for DLED (Dual-Level Evidence Detector).
#
# This single script replaces the original train_protocol{1..8}.sh files.
# Only the protocols that train the DLED method itself (dled detector +
# Trainer class) are exposed here; baseline scripts have been removed.
#
# Usage:
#   ./train.sh <protocol> [extra_train.py_args...]
#
# Protocols:
#   NOEFS      — 3-class same-domain training (Real / Swapping / Reenactment),
#                Face-Editing + EFS held out. (Original train_protocol1.sh)
#   NOFS       — 3-class same-domain training (Real / EFS / Reenactment),
#                Face-Swapping held out.
#   NOFR       — 3-class same-domain training (Real / Swapping / EFS),
#                Face-Reenactment held out.
#   open       — 4-class open-test training (Real / Swap / Reenact / EFS),
#                Face-Editing held out. (Original train_protocol3.sh)
#
# Environment variables (override the defaults to match your machine):
#   DLED_LOG_DIR              — where to write the .log file
#                               (default: ./logs/$PROTOCOL.log)
#   DLED_LOCAL_RANK           — CUDA device index (default: 0)
#   DLED_MAX_SAMPLES_PER_CLS  — cap samples per training class (default: 13504)
#   DLED_NUM_EPOCHS           — number of epochs (default: 50)
#   DLED_TEST_FRE             — test frequency in epochs (default: 50)
#   DLED_PRETRAINED_ROOT      — directory used to cache CLIP weights
#   DLED_DATASET_JSON_DIR     — directory containing protocol JSON files
#   DLED_OUTPUT_DIR           — persistent root for logs and checkpoints
#   DF40_DATA_ROOT / DF40_OLD_ROOT — see README for path-rewriting setup

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <NOEFS|NOFS|NOFR|open> [extra_train.py_args...]" >&2
    exit 1
fi

PROTOCOL="$1"
shift

LOCAL_RANK="${DLED_LOCAL_RANK:-0}"
MAX_SAMPLES_PER_CLS="${DLED_MAX_SAMPLES_PER_CLS:-13504}"
NUM_EPOCHS="${DLED_NUM_EPOCHS:-50}"
TEST_FRE="${DLED_TEST_FRE:-50}"
OUTPUT_DIR="${DLED_OUTPUT_DIR:-./outputs}"
LOG_DIR="${DLED_LOG_DIR:-$OUTPUT_DIR/logs}"
SAVE_DIR="${DLED_SAVE_DIR:-$OUTPUT_DIR/checkpoints/$PROTOCOL}"
DATASET_JSON_DIR="${DLED_DATASET_JSON_DIR:-./datasets/DF40/json_file}"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

case "$PROTOCOL" in
    NOEFS)
        TRAIN_DATASET="DF40_FF_train_NOEFS"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    NOFS)
        TRAIN_DATASET="DF40_FF_train_NOFS"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    NOFR)
        TRAIN_DATASET="DF40_FF_train_NOFR"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    open)
        TRAIN_DATASET="DF40_FF_train"
        TEST_DATASET="DF40_open_test"
        NUM_CLASSES=4
        ;;
    *)
        echo "Unknown protocol: $PROTOCOL" >&2
        echo "Valid protocols: NOEFS, NOFS, NOFR, open" >&2
        exit 1
        ;;
esac

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

LOG_FILE="$LOG_DIR/DLED_${PROTOCOL}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Logging to $LOG_FILE"

python3 training/train.py \
    --detector_path ./training/config/detector/dled.yaml \
    --train_dataset "$TRAIN_DATASET" \
    --test_dataset  "$TEST_DATASET" \
    --local_rank "$LOCAL_RANK" \
    --num_classes "$NUM_CLASSES" \
    --test_fre "$TEST_FRE" \
    --num_epochs "$NUM_EPOCHS" \
    --dataset_json_folder "$DATASET_JSON_DIR" \
    --log_dir "$LOG_DIR" \
    --save_path "$SAVE_DIR" \
    --save_model \
    --avu_reg \
    --max_samples_per_class "$MAX_SAMPLES_PER_CLS" \
    --experiment_type "df40_${PROTOCOL}" \
    "$@" \
    > "$LOG_FILE" 2>&1
