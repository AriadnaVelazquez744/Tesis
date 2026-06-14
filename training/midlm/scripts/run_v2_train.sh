#!/usr/bin/env bash
# =============================================================================
# Train the v2 (Smart Pooling + Hard Negative Mining) configuration
# for any of the four supported base models.
# =============================================================================
# Usage:
#   bash run_v2_train.sh <model_id> [extra args...]
#
# model_id ∈ {qwen3b, qwen7b, mistral7b, phi3}
#
# Notes:
#   * Stops the running MIDLM server (port 1235) for the duration of training
#     and restarts it afterwards (only the Qwen3B one — it is the only one
#     whose v2 checkpoint is currently packaged).
#   * Writes outputs to a per-model subfolder of trained_models_bidirectional_v2.
#   * Mirrors exactly the hyperparameters that produced 69.3% EM on the
#     Qwen2.5-3B-Instruct v2 noisy test set.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

MODEL_ID="${1:-qwen3b}"
shift || true

BASE_MODELS_DIR="$PROJECT_ROOT/training/base_models"
NOISY_DATASET="data/WeaveClinc150_mixed_noisy.json"
OUTPUT_DIR="adapters/trained_models_bidirectional_v2"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

declare -A BASE_PATHS=(
    [qwen3b]="$BASE_MODELS_DIR/Qwen2.5-3B-Instruct"
    [qwen7b]="$BASE_MODELS_DIR/Qwen2.5-7B-Instruct"
    [mistral7b]="$BASE_MODELS_DIR/Mistral-7B-Instruct-v0.3"
    [phi3]="$BASE_MODELS_DIR/Phi-3-mini-4k-instruct"
)

declare -A MODEL_DIRS=(
    [qwen3b]="Qwen2.5-3B-Instruct_midlm_bidirectional"
    [qwen7b]="Qwen2.5-7B-Instruct_midlm_bidirectional"
    [mistral7b]="Mistral-7B-Instruct-v0.3_midlm_bidirectional"
    [phi3]="Phi-3-mini-4k-instruct_midlm_bidirectional"
)

declare -A BATCH=(
    [qwen3b]=2
    [qwen7b]=1
    [mistral7b]=1
    [phi3]=2
)
declare -A GRAD_ACCUM=(
    [qwen3b]=4
    [qwen7b]=8
    [mistral7b]=8
    [phi3]=4
)
declare -A MAX_SEQ=(
    [qwen3b]=384
    [qwen7b]=384
    [mistral7b]=384
    [phi3]=384
)

if [[ -z "${BASE_PATHS[$MODEL_ID]:-}" ]]; then
    echo "Unknown model_id: $MODEL_ID"
    echo "Valid: ${!BASE_PATHS[*]}"
    exit 1
fi

BASE_PATH="${BASE_PATHS[$MODEL_ID]}"
CKPT_DIR="${OUTPUT_DIR}/${MODEL_DIRS[$MODEL_ID]}"
LOG_FILE="${LOG_DIR}/training_v2_${MODEL_ID}.log"

if [ ! -d "$BASE_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_PATH"
    exit 1
fi
if [ ! -f "$NOISY_DATASET" ]; then
    echo "ERROR: Noisy dataset not found: $NOISY_DATASET"
    echo "Run:  bash run_v2_train.sh setup  (not implemented; just call build_noisy_dataset.py)"
    exit 1
fi
if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/midlm_heads.pt" ]; then
    echo "Checkpoint already exists: $CKPT_DIR"
    echo "Delete it manually to retrain."
    exit 0
fi

SERVER_PID_FILE="$LOG_DIR/midlm_server.pid"
SERVER_PID=""
if [ -f "$SERVER_PID_FILE" ]; then
    SERVER_PID=$(cat "$SERVER_PID_FILE")
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping MIDLM server (PID $SERVER_PID) to free VRAM..."
        kill "$SERVER_PID"
        sleep 3
    fi
    rm -f "$SERVER_PID_FILE"
fi

trap 'echo "Training interrupted. Restart server with: bash run_midlm_server.sh"; exit 1' INT TERM

echo "============================================================"
echo "  Training $MODEL_ID  with v2 (Smart Pooling + HNM)"
echo "  Base     : $BASE_PATH"
echo "  Output   : $CKPT_DIR"
echo "  Data     : $NOISY_DATASET"
echo "  Batch    : ${BATCH[$MODEL_ID]}  GradAccum: ${GRAD_ACCUM[$MODEL_ID]}"
echo "  Log      : $LOG_FILE"
echo "============================================================"

uv run python train_midlm_bidirectional.py \
    --model_path "$BASE_PATH" \
    --data_json "$NOISY_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --max_k 3 \
    --epochs 1 \
    --batch_size_per_device "${BATCH[$MODEL_ID]}" \
    --gradient_accumulation_steps "${GRAD_ACCUM[$MODEL_ID]}" \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --use_attention_pool \
    --bf16 \
    --load_in_4bit \
    --max_seq_length "${MAX_SEQ[$MODEL_ID]}" \
    "$@" \
    > "$LOG_FILE" 2>&1

echo
echo "Training done. Checkpoint: $CKPT_DIR"
echo "Log: $LOG_FILE"

if [ -n "$SERVER_PID" ] && [ "$MODEL_ID" = "qwen3b" ]; then
    echo "Restarting MIDLM server..."
    bash run_midlm_server.sh
fi
