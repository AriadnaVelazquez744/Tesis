#!/usr/bin/env bash
# =============================================================================
# Evaluate any v2-trained model on BOTH the clean (mixed) and noisy test sets.
# =============================================================================
# Usage:
#   bash run_v2_eval.sh <model_id>
#
# model_id ∈ {qwen3b, qwen7b, mistral7b, phi3}
#
# Outputs:
#   experiments/<model_id>_v2_clean/<timestamp>/metrics.json
#   experiments/<model_id>_v2_noisy/<timestamp>/metrics.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

MODEL_ID="${1:-qwen3b}"

declare -A MODEL_DIRS=(
    [qwen3b]="Qwen2.5-3B-Instruct_midlm_bidirectional"
    [qwen7b]="Qwen2.5-7B-Instruct_midlm_bidirectional"
    [mistral7b]="Mistral-7B-Instruct-v0.3_midlm_bidirectional"
    [phi3]="Phi-3-mini-4k-instruct_midlm_bidirectional"
)

if [[ -z "${MODEL_DIRS[$MODEL_ID]:-}" ]]; then
    echo "Unknown model_id: $MODEL_ID"
    echo "Valid: ${!MODEL_DIRS[*]}"
    exit 1
fi

CHECKPOINT="$PROJECT_ROOT/training/midlm/adapters/trained_models_bidirectional_v2/${MODEL_DIRS[$MODEL_ID]}"
CLEAN_DATA="$PROJECT_ROOT/training/midlm/data/WeaveClinc150_mixed.json"
NOISY_DATA="$PROJECT_ROOT/training/midlm/data/WeaveClinc150_mixed_noisy.json"
CLEAN_EXP="$PROJECT_ROOT/experiments/midlm/runs/${MODEL_ID}_v2_clean"
NOISY_EXP="$PROJECT_ROOT/experiments/midlm/runs/${MODEL_ID}_v2_noisy"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Train first:  bash run_v2_train.sh $MODEL_ID"
    exit 1
fi

echo "=== CLEAN test set ==="
uv run python "$PROJECT_ROOT/experiments/midlm/eval_midlm_bidirectional.py" \
    --checkpoint_dir "$CHECKPOINT" \
    --data_json "$CLEAN_DATA" \
    --split test \
    --max_k 3 \
    --experiments_dir "$CLEAN_EXP"

echo
echo "=== NOISY test set ==="
uv run python "$PROJECT_ROOT/experiments/midlm/eval_midlm_bidirectional.py" \
    --checkpoint_dir "$CHECKPOINT" \
    --data_json "$NOISY_DATA" \
    --split test \
    --max_k 3 \
    --experiments_dir "$NOISY_EXP"

echo
echo "============================================================"
echo "  RESULTS — $MODEL_ID  (v2: Smart Pooling + HNM)"
echo "============================================================"
uv run python - <<PY
import json, glob, os
for label, root in [("CLEAN", "$CLEAN_EXP"), ("NOISY", "$NOISY_EXP")]:
    runs = sorted(glob.glob(os.path.join(root, "*", "metrics.json")))
    if not runs:
        print(f"  {label}: no results found")
        continue
    with open(runs[-1]) as f:
        m = json.load(f)
    print(f"  {label:5s} test ({m.get('num_examples', '?'):>4} examples):")
    print(f"    Exact Match : {m['exact_match_accuracy']*100:6.2f}%")
    print(f"    K-Accuracy  : {m['k_accuracy']*100:6.2f}%")
    print(f"    Micro-F1    : {m['micro_f1']*100:6.2f}%")
    print(f"    Macro-F1    : {m['macro_f1']*100:6.2f}%")
    print()
PY
