#!/bin/bash
# Train and merge LoRA model for LM Studio
# Usage: ./train_and_merge.sh [3b|7b] [max_train_samples]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_SIZE=${1:-3b}
MAX_SAMPLES=${2:-24}

if [ "$MODEL_SIZE" = "3b" ]; then
    MODEL_NAME="Qwen2.5-3B-Instruct"
    SCRIPT="src/Vagueness_Judge/training/sft_3b.py"
    ARGS="--load_in_8bit"
elif [ "$MODEL_SIZE" = "7b" ]; then
    MODEL_NAME="Qwen2.5-7B-Instruct"
    SCRIPT="src/Vagueness_Judge/training/sft_7b.py"
    ARGS="--load_in_4bit"
else
    echo "Usage: $0 [3b|7b] [max_train_samples]"
    exit 1
fi

echo "=== Training $MODEL_NAME ==="
uv run python "$SCRIPT" \
    --model_dir_name "$MODEL_NAME" \
    --train_data_path "src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
    $ARGS \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 512 \
    --max_train_samples "$MAX_SAMPLES" \
    --resume

echo ""
echo "=== Merging LoRA adapter with base model ==="
uv run python src/Vagueness_Judge/training/merge_lora.py \
    --base_model_path "src/base_models/$MODEL_NAME" \
    --lora_adapter_path "src/Vagueness_Judge/training_models/${MODEL_NAME}-Vagueness_Judge" \
    --output_path "src/Vagueness_Judge/trained_models_full/${MODEL_NAME}-Vagueness_Judge" \
    --bf16

echo ""
echo "=== Done! Model ready for LM Studio ==="
echo "Location: src/Vagueness_Judge/trained_models_full/${MODEL_NAME}-Vagueness_Judge"
