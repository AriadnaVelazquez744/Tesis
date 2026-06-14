# MIDLM Training

Training scripts for the MIDLM multi-intent detection model.

## Training commands

```bash
# Basic Unsloth training (3B)
python train_midlm_unsloth.py \
  --model_dir_name Qwen2.5-3B-Instruct \
  --max_seq_length 512 --epochs 1 --lr 2e-4

# Bidirectional v2 (Smart Pooling + HNM)
python train_midlm_bidirectional.py \
  --model_path ../base_models/Qwen2.5-3B-Instruct \
  --data_json data/WeaveClinc150_mixed_noisy.json \
  --output_dir adapters/trained_models_bidirectional_v2 \
  --max_k 3 --epochs 1 --lr 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --use_attention_pool --bf16 --load_in_4bit

# Evaluation
python eval_midlm.py \
  --checkpoint_dir adapters/Qwen2.5-3B-Instruct_midlm \
  --split test --max_k 3 --bf16 --load_in_4bit

python eval_midlm_bidirectional.py \
  --checkpoint_dir adapters/trained_models_bidirectional_v2/Qwen2.5-3B-Instruct_midlm_bidirectional \
  --data_json data/WeaveClinc150_mixed_noisy.json \
  --split test --max_k 3

# Batch scripts (see scripts/ directory)
bash scripts/run_v2_train.sh qwen3b
bash scripts/run_overnight.sh
```

See also: `README_TRAINING.md` (detailed docs), `MIDLM_NOISE_ROBUSTNESS_REPORT.md` (results).
