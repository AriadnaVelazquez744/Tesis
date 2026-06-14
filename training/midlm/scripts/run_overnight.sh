#!/usr/bin/env bash
# =============================================================================
# Overnight Batch: Train + Eval the remaining 3 models with v2 config
# =============================================================================
# Runs sequentially:
#   1. Qwen2.5-7B-Instruct  (~1h40m train + 5m eval each dataset)
#   2. Mistral-7B-v0.3      (~1h40m train + 5m eval)
#   3. Phi-3-mini-4k        (~1h   train + 5m eval)
#
# Total: ~6 hours — perfect for overnight.
#
# Each training uses the EXACT config from
#   trained_models_bidirectional_v2/Qwen2.5-3B-Instruct_midlm_bidirectional
# adjusted ONLY for GPU memory limits:
#   - 7B models: batch_size=1, grad_accum=8
#   - 3B/Phi models: batch_size=2, grad_accum=4
# =============================================================================

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

BASE="$PROJECT_ROOT/training/base_models"
DATA="data/WeaveClinc150_mixed_noisy.json"
CLEAN_DATA="data/WeaveClinc150_mixed.json"
OUT_DIR="adapters/trained_models_bidirectional_v2"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# --- Free up VRAM: kill running server if any --------------------------------
if [ -f "$LOG_DIR/midlm_server.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/midlm_server.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date) Killing MIDLM server (PID $OLD_PID) to free VRAM..."
        kill "$OLD_PID"
        sleep 3
    fi
    rm -f "$LOG_DIR/midlm_server.pid"
fi

# --- Kill any lingering process on port 1235 ----------------------------------
PORT_PID=$(ss -tlnp 2>/dev/null | grep ':1235' | grep -oP 'pid=\K[0-9]+' || true)
if [ -n "$PORT_PID" ]; then
    echo "$(date) Killing process on port 1235 (PID $PORT_PID)..."
    kill "$PORT_PID" 2>/dev/null || true
    sleep 2
fi

TRAIN_LOG="$LOG_DIR/overnight_training.log"
RESULTS_LOG="$LOG_DIR/overnight_results.txt"

echo "============================================================" | tee -a "$TRAIN_LOG"
echo "$(date) Overnight batch started" | tee -a "$TRAIN_LOG"
echo "============================================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

# --- Shared training flags (from v2 Qwen3B best config) ----------------------
SHARED_FLAGS=(
    --data_json "$DATA"
    --output_dir "$OUT_DIR"
    --max_k 3
    --epochs 1
    --lr 2e-4
    --lora_r 16
    --lora_alpha 32
    --target_modules all-linear
    --use_attention_pool
    --bf16
    --load_in_4bit
    --max_seq_length 384
)

# --- Per-model configuration -------------------------------------------------
declare -A MODEL_DIRS=(
    [qwen7b]="Qwen2.5-7B-Instruct_midlm_bidirectional"
    [mistral7b]="Mistral-7B-Instruct-v0.3_midlm_bidirectional"
    [phi3]="Phi-3-mini-4k-instruct_midlm_bidirectional"
)

declare -A MODEL_PATHS=(
    [qwen7b]="$BASE/Qwen2.5-7B-Instruct"
    [mistral7b]="$BASE/Mistral-7B-Instruct-v0.3"
    [phi3]="$BASE/Phi-3-mini-4k-instruct"
)

declare -A MODEL_BATCH=(
    [qwen7b]=1
    [mistral7b]=1
    [phi3]=2
)

declare -A MODEL_GA=(
    [qwen7b]=8
    [mistral7b]=8
    [phi3]=4
)

MODEL_ORDER=(qwen7b mistral7b phi3)

# --- Train + Eval loop -------------------------------------------------------
for MODEL in "${MODEL_ORDER[@]}"; do
    CKPT_DIR="$OUT_DIR/${MODEL_DIRS[$MODEL]}"
    MODEL_LOG="$LOG_DIR/training_${MODEL}.log"

    echo "" | tee -a "$TRAIN_LOG"
    echo "============================================================" | tee -a "$TRAIN_LOG"
    echo "$(date) STARTING: $MODEL" | tee -a "$TRAIN_LOG"
    echo "  Model path: ${MODEL_PATHS[$MODEL]}" | tee -a "$TRAIN_LOG"
    echo "  Batch size: ${MODEL_BATCH[$MODEL]}" | tee -a "$TRAIN_LOG"
    echo "  Grad accum: ${MODEL_GA[$MODEL]}" | tee -a "$TRAIN_LOG"
    echo "  Checkpoint: $CKPT_DIR" | tee -a "$TRAIN_LOG"
    echo "  Log file  : $MODEL_LOG" | tee -a "$TRAIN_LOG"
    echo "============================================================" | tee -a "$TRAIN_LOG"

    if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/midlm_heads.pt" ]; then
        echo "$(date) SKIPPING $MODEL — checkpoint already exists." | tee -a "$TRAIN_LOG"
    else
        echo "$(date) Training $MODEL ..." | tee -a "$TRAIN_LOG"
        uv run python train_midlm_bidirectional.py \
            --model_path "${MODEL_PATHS[$MODEL]}" \
            --batch_size_per_device "${MODEL_BATCH[$MODEL]}" \
            --gradient_accumulation_steps "${MODEL_GA[$MODEL]}" \
            "${SHARED_FLAGS[@]}" \
            > "$MODEL_LOG" 2>&1

        if [ $? -ne 0 ]; then
            echo "$(date) ERROR: $MODEL training FAILED. Check $MODEL_LOG" | tee -a "$TRAIN_LOG"
            continue
        fi
        echo "$(date) $MODEL training complete." | tee -a "$TRAIN_LOG"
    fi

    # --- Evaluation on both test sets ----------------------------------------
    for DATASET_LABEL in CLEAN NOISY; do
        if [ "$DATASET_LABEL" = "CLEAN" ]; then
            EVAL_DATA="$CLEAN_DATA"
            EXP_DIR="experiments/${MODEL}_v2_clean"
        else
            EVAL_DATA="$DATA"
            EXP_DIR="experiments/${MODEL}_v2_noisy"
        fi

        echo "$(date) Evaluating $MODEL on $DATASET_LABEL ..." | tee -a "$TRAIN_LOG"
        uv run python "$PROJECT_ROOT/experiments/midlm/eval_midlm_bidirectional.py" \
            --checkpoint_dir "$CKPT_DIR" \
            --data_json "$EVAL_DATA" \
            --split test \
            --max_k 3 \
            --experiments_dir "$EXP_DIR" \
            >> "$MODEL_LOG" 2>&1
    done
done

# --- Summary -----------------------------------------------------------------
echo "" | tee -a "$TRAIN_LOG"
echo "============================================================" | tee -a "$TRAIN_LOG"
echo "$(date) OVERNIGHT BATCH COMPLETE" | tee -a "$TRAIN_LOG"
echo "============================================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

PrintResults() {
    echo "  MODEL          DATASET    EXAMPLES    EM       K-ACC    MICRO-F1  MACRO-F1"
    echo "  -------------  ---------  ----------  -------  -------  --------  --------"
    for MODEL in "${MODEL_ORDER[@]}"; do
        for SPLIT in v2_clean v2_noisy; do
            METRICS=$(ls -t experiments/${MODEL}_${SPLIT}/*/metrics.json 2>/dev/null | head -1)
            if [ -n "$METRICS" ]; then
                uv run python -c "
import json
with open('$METRICS') as f:
    m = json.load(f)
print(f'  {MODEL:13s} {SPLIT:9s} {m[\"num_examples\"]:>10}  {m[\"exact_match_accuracy\"]*100:>6.2f}%  {m[\"k_accuracy\"]*100:>6.2f}%  {m[\"micro_f1\"]*100:>7.2f}%  {m[\"macro_f1\"]*100:>7.2f}%')
" 2>/dev/null
            fi
        done
    done
} > "$RESULTS_LOG"

PrintResults

cat "$RESULTS_LOG" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Detailed logs: ls -lh $LOG_DIR/training_*.log" | tee -a "$TRAIN_LOG"
echo "Results file : $RESULTS_LOG" | tee -a "$TRAIN_LOG"
