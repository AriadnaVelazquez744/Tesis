# MIDLM training (Unsloth, local models, 12GB VRAM)

This folder contains a runnable implementation of the **MIDLM** training recipe from *Huang et al., 2025* (“MIDLM: Multi-Intent Detection with Bidirectional Large Language Models”) adapted to this thesis repository and the dataset:

- `training/midlm/data/WeaveClinc150_rewritten.json`

## What is implemented (paper mapping)

From *Huang et al., 2025* §3.2–3.5 (see `toothless/midlm_training/pdfs/` if available):

- **Intent-number detection**: pooled hidden states \( \to \) classifier \( \to \) predicted count \(K\).
- **Multi-intent selection**: intent logits \( \to \) `TopK` using predicted \(K\).
- **Joint training loss**:
  - multi-label intent loss = BCE-with-logits
  - intent-count loss = Cross-Entropy over classes \(1..C\) (here \(C=\) `--max_k`, default 3)

Important limitation:

- The paper’s key architectural change is replacing the backbone’s causal mask with **global (non-causal) attention** during post-training. **Unsloth does not reliably support non-causal attention**, so this implementation keeps the backbone causal, but implements the *same heads, decoding, and joint loss* so you can run local training on 12GB VRAM.

## Dataset expectations

`WeaveClinc150_rewritten.json` must have top-level keys:

- `train`: list of rows
- `validation`: list of rows
- `test`: list of rows

Each row needs at least:

- `text`: string utterance
- `labels`: list of intent-name strings (2–3 intents in this dataset)

If you change the JSON, you can regenerate the TSVs used by TEXTOIR-style loaders via:

```bash
python training/midlm/export_textoir_tsv_from_weave_json.py
```

## Models: use your local `base_models/`

This repo expects the base models to be stored locally as Hugging Face model folders, e.g.:

- `training/base_models/Qwen2.5-3B-Instruct/`
- `training/base_models/Qwen2.5-7B-Instruct/`
- `training/base_models/Mistral-7B-Instruct-v0.3/`
- `training/base_models/Phi-3-mini-4k-instruct/`

## Training command (recommended first run)

All VRAM-saving defaults are on by default (4-bit, bf16, gradient checkpointing).  
Start with a smaller model to validate everything fits on 12GB:

```bash
python training/midlm/train_midlm_unsloth.py \
  --model_dir_name Qwen2.5-3B-Instruct \
  --max_seq_length 512 \
  --epochs 1 \
  --lr 2e-4
```

For a 7B model with tighter memory:

```bash
python training/midlm/train_midlm_unsloth.py \
  --model_dir_name Mistral-7B-Instruct-v0.3 \
  --max_seq_length 256 \
  --gradient_accumulation_steps 16
```

Disable defaults if needed (e.g. to load in 16-bit):

```bash
python training/midlm/train_midlm_unsloth.py \
  --no-load_in_4bit --no-bf16 --no-gradient-checkpointing
```

Artifacts are written to:

- `training/midlm/adapters/<BaseName>_midlm/`
  - `intent_vocab.json`: intent label list used for training
  - `train_config.json`: points back to the base model path + training settings
  - `midlm_heads.pt`: trained MIDLM task heads (intent classification + intent-number classifier)
  - LoRA adapter weights + tokenizer files (for later inference)

## Evaluation

```bash
python training/midlm/eval_midlm.py \
  --checkpoint_dir training/midlm/adapters/Qwen2.5-3B-Instruct_midlm \
  --split test \
  --max_k 3 \
  --max_seq_length 384 \
  --load_in_4bit \
  --bf16 \
  --save_predictions
```

Metrics:

- Exact match accuracy (predicted intent set equals gold set)
- \(K\) accuracy (predicted number of intents equals gold)
- Micro-F1 and Macro-F1 over intent labels

Evaluation artifacts are always saved under:

- `training/midlm/experiments/<checkpoint>__<split>__<timestamp>/`
  - `metrics.json`
  - `eval_config.json`
  - `predictions.json` (only when `--save_predictions` is set)

## How this will later connect to TEXTOIR (interface only for now)

Your analysis layer spec in `propose/Analisys_layer.md` needs MIDLM to output:

- intent scores over the intent vocabulary
- predicted intent count \(K\)
- selected intent set via TopK-by-K decoding

This training bundle ensures those outputs exist and are reproducible via:

- `midlm.model.MIDLMForMultiIntent` (intent logits + num logits)
- `midlm.decode.decode_topk_by_predicted_k` (paper-style selection)

LM Studio note:

- LM Studio is useful for *inference* and dataset rewriting, but Unsloth LoRA fine-tuning runs via Python (scripts above).
- `training/base_models/` is used as **read-only input** for training; checkpoints/adapters are always written to `training/midlm/adapters/`.

## Consistency with Huang et al. (2025) and original MIDLM procedure

- **Implemented faithfully**:
  - dual-task setup (intent multi-label + intent-number prediction)
  - joint objective \(L = \alpha L_{intent} + \beta L_{num}\)
  - TopK-by-predicted-K decoding
  - LoRA-based post-training of a decoder-only LLM backbone
- **Paper-like defaults in this repo**:
  - `--epochs 1`
  - `--lr 2e-4`
  - `--weight_decay 0.05`
  - `--lora_r 16`, `--lora_alpha 32`, `--lora_dropout 0.05`
  - optional tuning to match paper search space (`r in {16,32}`, `alpha in {32,64}`)
- **Known divergence (explicit)**:
  - Paper’s key contribution is non-causal/global attention during post-training.
  - Current Unsloth-based implementation keeps the backbone causal due to framework limitations.
