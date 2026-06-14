#!/usr/bin/env bash
# =============================================================================
# Qwen2.5-3B-Instruct — Noise-Robust Training & Evaluation Pipeline (v2)
# =============================================================================
#
# This script reproduces the BEST-RESULT configuration:
#   - Smart Pooling (Learnable Attention)
#   - Hard Negative Mining (Non-Intent Pool)
#   - Dropout (0.1) and Label Smoothing (0.1)
#
# Expected results on the Noisy Test Set (4,968 examples):
#   Exact Match : ~69.3%
#   K-Accuracy  : ~96.9%
#   Micro-F1    : ~84.3%
#
# Usage:
#   bash run_qwen3b_best.sh           # Full pipeline (setup + train + eval)
#   bash run_qwen3b_best.sh eval      # Skip training, just run evaluation
#   bash run_qwen3b_best.sh setup     # Only generate noise pool and dataset
#
# Hardware: Single 12GB GPU (e.g. RTX 3060) — uses 4-bit quantization.
# Time    : ~55 minutes for training, ~3 minutes for evaluation.
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

BASE_MODEL="$PROJECT_ROOT/training/base_models/Qwen2.5-3B-Instruct"
NOISE_POOL="data/noise_pool.json"
MIXED_DATASET="data/WeaveClinc150_mixed.json"
NOISY_DATASET="data/WeaveClinc150_mixed_noisy.json"
OUTPUT_DIR="adapters/trained_models_bidirectional_v2"
CHECKPOINT_DIR="${OUTPUT_DIR}/Qwen2.5-3B-Instruct_midlm_bidirectional"
TRAIN_LOG="training_v2.log"
EVAL_CLEAN_DIR="experiments/qwen3b_v2_clean"
EVAL_NOISY_DIR="experiments/qwen3b_v2_noisy"

# --- Helpers -----------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
hr()   { echo "------------------------------------------------------------"; }
banner() { hr; echo "  $*"; hr; }

# --- Step 1: Data Setup -------------------------------------------------------
setup_data() {
    banner "Step 1: Data Setup"

    # 1a. Generate the non-intent noise pool (19,698 unique statements)
    if [ ! -f "$NOISE_POOL" ]; then
        log "Generating noise pool..."
        uv run python build_noise_pool.py
    else
        log "Noise pool already exists: $NOISE_POOL"
    fi

    # 1b. Generate the balanced mixed dataset (k=1, k=2, k=3 equal)
    if [ ! -f "$MIXED_DATASET" ]; then
        log "Building balanced mixed dataset..."
        uv run python build_mixed_dataset.py
    else
        log "Mixed dataset already exists: $MIXED_DATASET"
    fi

    # 1c. Build the noise-augmented training set
    if [ ! -f "$NOISY_DATASET" ]; then
        log "Building noisy dataset with non-intent pool..."
        uv run python build_noisy_dataset.py
    else
        log "Noisy dataset already exists: $NOISY_DATASET"
    fi

    log "Data setup complete."
    uv run python -c "
import json
d = json.load(open('$NOISY_DATASET'))
for split in ('train','validation','test'):
    rows = d[split]
    noisy = sum(1 for r in rows if r.get('metadata',{}).get('is_noisy'))
    print(f'  {split:11s}: {len(rows):>6} total, {noisy:>6} noisy ({100*noisy/len(rows):.1f}%)')
"
}

# --- Step 2: Training ---------------------------------------------------------
run_training() {
    banner "Step 2: Training Qwen2.5-3B with Smart Pooling + HNM"

    if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/midlm_heads.pt" ]; then
        log "Checkpoint already exists: $CHECKPOINT_DIR"
        log "Delete it if you want to retrain from scratch."
        return 0
    fi

    log "Launching training in background..."
    log "Log file: $TRAIN_LOG"

    uv run python train_midlm_bidirectional.py \
        --model_path "$BASE_MODEL" \
        --data_json "$NOISY_DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --max_k 3 \
        --epochs 1 \
        --batch_size_per_device 2 \
        --gradient_accumulation_steps 4 \
        --lr 2e-4 \
        --lora_r 16 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --use_attention_pool \
        --bf16 \
        --load_in_4bit > "$TRAIN_LOG" 2>&1 &

    TRAIN_PID=$!
    log "Training PID: $TRAIN_PID"
    log "Monitor with: tail -f $TRAIN_LOG"
    log "Waiting for training to complete..."

    wait $TRAIN_PID
    log "Training complete. Checkpoint saved to: $CHECKPOINT_DIR"
}

# --- Step 3: Evaluation -------------------------------------------------------
run_evaluation() {
    banner "Step 3: Evaluation on Clean and Noisy Test Sets"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log "ERROR: Checkpoint not found: $CHECKPOINT_DIR"
        log "Run training first: bash $0 train"
        exit 1
    fi

    # 3a. Clean Test
    log "Evaluating on CLEAN test set..."
    uv run python "$PROJECT_ROOT/experiments/midlm/eval_midlm_bidirectional.py" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_json "$MIXED_DATASET" \
        --split test \
        --max_k 3 \
        --experiments_dir "$EVAL_CLEAN_DIR"

    # 3b. Noisy Test
    log "Evaluating on NOISY test set..."
    uv run python "$PROJECT_ROOT/experiments/midlm/eval_midlm_bidirectional.py" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_json "$NOISY_DATASET" \
        --split test \
        --max_k 3 \
        --experiments_dir "$EVAL_NOISY_DIR"

    # 3c. Summary
    banner "RESULTS SUMMARY"
    uv run python - <<'PY'
import json
import glob
import os

for label, root in [("CLEAN", "experiments/qwen3b_v2_clean"),
                    ("NOISY", "experiments/qwen3b_v2_noisy")]:
    runs = sorted(glob.glob(os.path.join(root, "*", "metrics.json")))
    if not runs:
        print(f"  {label}: no results found")
        continue
    latest = runs[-1]
    with open(latest) as f:
        m = json.load(f)
    print(f"  {label:5s} test ({m['num_examples']:>4} examples):")
    print(f"    ExactMatch : {m['exact_match_accuracy']*100:.2f}%")
    print(f"    K-Acc      : {m['k_accuracy']*100:.2f}%")
    print(f"    Micro-F1   : {m['micro_f1']*100:.2f}%")
    print(f"    Macro-F1   : {m['macro_f1']*100:.2f}%")
    print()
PY
}

# --- Main Dispatcher ----------------------------------------------------------
case "${1:-all}" in
    setup)
        setup_data
        ;;
    train)
        setup_data
        run_training
        ;;
    eval)
        run_evaluation
        ;;
    all)
        setup_data
        run_training
        run_evaluation
        ;;
    *)
        echo "Usage: $0 {all|setup|train|eval}"
        exit 1
        ;;
esac

log "Done."
