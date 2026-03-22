#!/usr/bin/env bash
set -euo pipefail

# Ajusta estos valores para tu experimento.
# Para entrenar en una sola GPU (12GB), deja un solo índice.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# Para una sola GPU:
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12345}"

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
TRAINING_DIR="${REPO_ROOT}/src/Vagueness_Judge/training"
cd "${TRAINING_DIR}"

# Modelo a entrenar (nombre de carpeta dentro de src/base_models/)
# Ejemplos:
#   MODEL_DIR_NAME="Mistral-7B-Instruct-v0.3"
#   MODEL_DIR_NAME="Llama-3.2-3B-Instruct"
# Cambia esta variable antes de ejecutar.
MODEL_DIR_NAME="${MODEL_DIR_NAME:-Mistral-7B-Instruct-v0.3}"

# Config opcional por modelo (si no existe, se usarán defaults en sft.py)
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/src/Vagueness_Judge/training/model_configs/${MODEL_DIR_NAME}.json}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${REPO_ROOT}/src/Vagueness_Judge/data/interactions/interaction_data_train.jsonl}"
DATA_SETTING="${DATA_SETTING:-MTMD}"

# Para 12GB en 4-bit, reducir por defecto:
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"

# Valores conservadores para 12GB en 4-bit:
BS_PER_DEVICE="${BS_PER_DEVICE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LOGGING_STEP="${LOGGING_STEP:-10}"
SAVE_STEPS="${SAVE_STEPS:-400}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/src/Vagueness_Judge/training_models}"

mkdir -p "${REPO_ROOT}/logs"

ARGS=(
  "${REPO_ROOT}/src/Vagueness_Judge/training/sft.py"
  --model_dir_name "${MODEL_DIR_NAME}"
  --model_config_path "${MODEL_CONFIG_PATH}"
  --train_data_path "${TRAIN_DATA_PATH}"
  --data_setting "${DATA_SETTING}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --batch_size_per_device "${BS_PER_DEVICE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --logging_step "${LOGGING_STEP}"
  --save_steps "${SAVE_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --target_modules "${TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj}"
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --gradient_checkpointing
)

# Recomendación para 12GB: QLoRA en 4-bit.
ARGS+=( --load_in_4bit )

# Si tu GPU soporta bf16, activa bf16. Si no, usa fp16.
if [[ "${USE_BF16:-1}" == "1" ]]; then
  ARGS+=( --bf16 )
else
  ARGS+=( --fp16 )
fi

echo "Final command:"
echo "torchrun ${DISTRIBUTED_ARGS} ${ARGS[*]}"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
  torchrun ${DISTRIBUTED_ARGS} "${ARGS[@]}" 2>&1 | tee "${REPO_ROOT}/logs/sft_${MODEL_DIR_NAME}.log"
else
  python "${ARGS[@]}" 2>&1 | tee "${REPO_ROOT}/logs/sft_${MODEL_DIR_NAME}.log"
fi
