#!/usr/bin/env bash
# Retrain all 4 MIDLM models with bidirectional (non-causal) attention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/trained_models_bidirectional"
UV_RUN="uv run"  # uses project's venv
mkdir -p "$OUTPUT_DIR"

_is_done() {
  local model_name="$1"
  local dir="${OUTPUT_DIR}/${model_name}_midlm_bidirectional"
  # midlm_heads.pt is written only on successful completion
  [[ -f "${dir}/midlm_heads.pt" ]]
}

_run_model() {
  local model_name="$1"
  local out_dir="${OUTPUT_DIR}/${model_name}_midlm_bidirectional"
  shift

  if _is_done "$model_name"; then
    echo "=== Training skipped for $model_name (already trained) ==="
  else
    # Auto-resume from latest checkpoint if any exist
    resume_args=()
    latest_ckpt=
    if [ -d "$out_dir" ]; then
      latest_ckpt=$(find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    fi
    if [ -n "$latest_ckpt" ]; then
      echo "=== Resuming $model_name from $latest_ckpt ==="
      resume_args=(--resume_from_checkpoint "$latest_ckpt")
    fi

    echo "=== Training $model_name (bidirectional) ==="
    ${UV_RUN} python "${SCRIPT_DIR}/train_midlm_bidirectional.py" "${resume_args[@]}" "$@"
  fi

  rm -rf trained_models_bidirectional/*/checkpoint-*

  # Evaluate regardless — may have been trained in a prior run
  if [ -f "${out_dir}/midlm_heads.pt" ]; then
    echo "=== Evaluating $model_name (bidirectional) ==="
    ${UV_RUN} python "${PROJECT_ROOT}/experiments/midlm/eval_midlm_bidirectional.py" \
      --checkpoint "$out_dir" \
      --split test \
      --save_predictions
  else
    echo "=== WARNING: $model_name has no midlm_heads.pt — skipping eval ==="
  fi
}

_run_model Qwen2.5-3B-Instruct \
  --model_dir_name Qwen2.5-3B-Instruct \
  --max_seq_length 384 \
  --batch_size_per_device 4 \
  --gradient_accumulation_steps 2 \
  --epochs 1 \
  --lr 2e-4 \
  --seed 0 \
  --output_dir "$OUTPUT_DIR"

_run_model Qwen2.5-7B-Instruct \
  --model_dir_name Qwen2.5-7B-Instruct \
  --max_seq_length 384 \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --epochs 1 \
  --lr 2e-4 \
  --seed 0 \
  --output_dir "$OUTPUT_DIR"

_run_model Mistral-7B-Instruct-v0.3 \
  --model_dir_name Mistral-7B-Instruct-v0.3 \
  --max_seq_length 384 \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --epochs 1 \
  --lr 2e-4 \
  --seed 0 \
  --output_dir "$OUTPUT_DIR"

_run_model Phi-3-mini-4k-instruct \
  --model_dir_name Phi-3-mini-4k-instruct \
  --max_seq_length 384 \
  --batch_size_per_device 4 \
  --gradient_accumulation_steps 2 \
  --epochs 1 \
  --lr 2e-4 \
  --seed 0 \
  --output_dir "$OUTPUT_DIR"

echo "=== All models trained and evaluated ==="
