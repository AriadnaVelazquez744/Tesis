# Merge Commands for Vagueness Judge Models

This file contains the commands to merge the LoRA adapters with their base models to create single files ready for LM Studio.

## Prerequisites

- Base models are stored in: `src/base_models/`
- Trained LoRA adapters are in: `src/Vagueness_Judge/training_models/`
- Merged models will be saved to: `src/Vagueness_Judge/trained_models_full/`

## Merge Commands

### 1. Qwen2.5-3B-Instruct (3B model, 8-bit training)

```bash
cd /home/gia/AriadnaVR/JdV_Training && uv run python src/Vagueness_Judge/training/merge_lora.py \
  --base_model_path "src/base_models/Qwen2.5-3B-Instruct" \
  --lora_adapter_path "src/Vagueness_Judge/training_models/Qwen2.5-3B-Instruct-Vagueness_Judge" \
  --output_path "src/Vagueness_Judge/trained_models_full/Qwen2.5-3B-Instruct-Vagueness_Judge" \
  --bf16
```

**Training command used:**
```bash
uv run python src/Vagueness_Judge/training/sft_3b.py \
  --model_dir_name "Qwen2.5-3B-Instruct" \
  --train_data_path "src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
  --load_in_8bit \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_seq_length 512 \
  --max_train_samples 24 \
  --resume
```

---

### 2. Qwen2.5-7B-Instruct (7B model, 4-bit training)

```bash
cd /home/gia/AriadnaVR/JdV_Training && uv run python src/Vagueness_Judge/training/merge_lora.py \
  --base_model_path "src/base_models/Qwen2.5-7B-Instruct" \
  --lora_adapter_path "src/Vagueness_Judge/training_models/Qwen2.5-7B-Instruct-Vagueness_Judge" \
  --output_path "src/Vagueness_Judge/trained_models_full/Qwen2.5-7B-Instruct-Vagueness_Judge" \
  --bf16
```

**Training command used:**
```bash
uv run python src/Vagueness_Judge/training/sft_7b.py \
  --model_dir_name "Qwen2.5-7B-Instruct" \
  --train_data_path "src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
  --load_in_4bit \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_seq_length 512 \
  --max_train_samples 24 \
  --resume
```

---

### 3. Phi-3-mini-4k-instruct (3B model, 4-bit training)

**Note:** Phi-3 requires `HF_HOME=.cache` due to permission issues with the global Hugging Face cache.

```bash
cd /home/gia/AriadnaVR/JdV_Training && HF_HOME=.cache uv run python src/Vagueness_Judge/training/merge_lora.py \
  --base_model_path "src/base_models/Phi-3-mini-4k-instruct" \
  --lora_adapter_path "src/Vagueness_Judge/training_models/Phi-3-mini-4k-instruct-Vagueness_Judge" \
  --output_path "src/Vagueness_Judge/trained_models_full/Phi-3-mini-4k-instruct-Vagueness_Judge" \
  --bf16
```

**Training command used:**
```bash
cd /home/gia/AriadnaVR/JdV_Training && HF_HOME=.cache uv run python src/Vagueness_Judge/training/sft_3b.py \
  --model_dir_name "Phi-3-mini-4k-instruct" \
  --train_data_path "src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
  --load_in_4bit \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_seq_length 512 \
  --max_train_samples 24 \
  --resume
```

---

### 4. Mistral-7B-Instruct-v0.3 (7B model, 4-bit training)

```bash
cd /home/gia/AriadnaVR/JdV_Training && uv run python src/Vagueness_Judge/training/merge_lora.py \
  --base_model_path "src/base_models/Mistral-7B-Instruct-v0.3" \
  --lora_adapter_path "src/Vagueness_Judge/training_models/Mistral-7B-Instruct-v0.3-Vagueness_Judge" \
  --output_path "src/Vagueness_Judge/trained_models_full/Mistral-7B-Instruct-v0.3-Vagueness_Judge" \
  --bf16
```

**Training command used:**
```bash
uv run python src/Vagueness_Judge/training/sft_7b.py \
  --model_dir_name "Mistral-7B-Instruct-v0.3" \
  --train_data_path "src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
  --load_in_4bit \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_seq_length 512 \
  --max_train_samples 24 \
  --resume
```

---

## Adjusting Directories

If your directory structure is different, adjust these variables:

| Variable | Default Path | Description |
|----------|---------------|-------------|
| `BASE_MODELS_DIR` | `src/base_models/` | Where base models are stored |
| `ADAPTER_DIR` | `src/Vagueness_Judge/training_models/` | Where LoRA adapters are saved after training |
| `OUTPUT_DIR` | `src/Vagueness_Judge/trained_models_full/` | Where merged models will be saved |

Example with custom directories:
```bash
export BASE_MODELS_DIR="custom/base_models/"
export ADAPTER_DIR="custom/adapters/"
export OUTPUT_DIR="custom/merged_models/"

uv run python src/Vagueness_Judge/training/merge_lora.py \
  --base_model_path "${BASE_MODELS_DIR}Qwen2.5-3B-Instruct" \
  --lora_adapter_path "${ADAPTER_DIR}Qwen2.5-3B-Instruct-Vagueness_Judge" \
  --output_path "${OUTPUT_DIR}Qwen2.5-3B-Instruct-Vagueness_Judge" \
  --bf16
```

## Notes

- All merged models will have the `-Vagueness_Judge` suffix as per the training output.
- Use `--bf16` for bfloat16 precision (recommended) or `--fp16` for float16.
- For Phi-3, always prefix with `HF_HOME=.cache` to avoid permission issues.
- The merge script handles both 3B and 7B models automatically.
