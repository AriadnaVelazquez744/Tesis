#!/bin/bash
# Retrain all 4 models with full dataset + validation split
# Usage: ./retrain_all.sh [3b|7b|all]

set -e

REPO="$(cd "$(dirname "$0")"/../../.. && pwd)"
TRAIN_DATA="$REPO/src/Vagueness_Judge/data/interactions/interaction_data_train.jsonl"

run_3b() {
    local MODEL_NAME="$1"
    echo "=== Training 3B model: $MODEL_NAME ==="
    uv run python "$REPO/src/Vagueness_Judge/training/sft_3b.py" \
        --model_dir_name "$MODEL_NAME" \
        --train_data_path "$TRAIN_DATA" \
        --load_in_8bit --max_seq_length 1024 \
        --epochs 3 --lr 1e-5 \
        --batch_size_per_device 2 --gradient_accumulation_steps 4 \
        --validation_split 0.15 \
        --logging_step 50 --save_steps 500 --save_total_limit 2 \
        --seed 0
    echo "=== Done: $MODEL_NAME ==="
}

run_7b() {
    local MODEL_NAME="$1"
    echo "=== Training 7B model: $MODEL_NAME ==="
    uv run python "$REPO/src/Vagueness_Judge/training/sft_7b.py" \
        --model_dir_name "$MODEL_NAME" \
        --train_data_path "$TRAIN_DATA" \
        --load_in_4bit --max_seq_length 1024 \
        --epochs 3 --lr 1e-5 \
        --batch_size_per_device 1 --gradient_accumulation_steps 4 \
        --validation_split 0.15 \
        --logging_step 50 --save_steps 500 --save_total_limit 2 \
        --seed 0
    echo "=== Done: $MODEL_NAME ==="
}

MODEL_SIZE="${1:-all}"

case "$MODEL_SIZE" in
    3b)
        run_3b "Qwen2.5-3B-Instruct"
        run_3b "Phi-3-mini-4k-instruct"
        ;;
    7b)
        run_7b "Qwen2.5-7B-Instruct"
        run_7b "Mistral-7B-Instruct-v0.3"
        ;;
    all)
        run_3b "Qwen2.5-3B-Instruct"
        run_3b "Phi-3-mini-4k-instruct"
        run_7b "Qwen2.5-7B-Instruct"
        run_7b "Mistral-7B-Instruct-v0.3"
        ;;
    *)
        echo "Usage: $0 [3b|7b|all]"
        exit 1
        ;;
esac

echo "=== All training runs complete! ==="
