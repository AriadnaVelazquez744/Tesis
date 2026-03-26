#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from unsloth import FastLanguageModel

from midlm.data import MIDLMDataCollator, WeaveMultiIntentDataset, build_intent_vocab, load_weave_json
from midlm.model import MIDLMForMultiIntent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_midlm_unsloth")


def _ensure_tokenizer_setup(tokenizer) -> None:
    if tokenizer.eos_token is None:
        if tokenizer.pad_token is not None:
            tokenizer.eos_token = tokenizer.pad_token
        elif tokenizer.bos_token is not None:
            tokenizer.eos_token = tokenizer.bos_token
        elif tokenizer.unk_token is not None:
            tokenizer.eos_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no eos_token; cannot proceed.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"


def load_backbone_with_lora(args: argparse.Namespace):
    base_dir = Path(__file__).resolve().parents[1]  # src/
    base_models_dir = base_dir / "base_models"
    model_path = Path(args.model_path) if args.model_path else (base_models_dir / args.model_dir_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    logger.info("Loading backbone from: %s", str(model_path))
    backbone, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=bool(args.load_in_4bit),
    )
    _ensure_tokenizer_setup(tokenizer)

    target_modules: Any = args.target_modules
    if isinstance(target_modules, str):
        if target_modules.strip() != "all-linear":
            target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    backbone = FastLanguageModel.get_peft_model(
        backbone,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        backbone = FastLanguageModel.for_training(backbone)
    except Exception:
        pass
    backbone.config.use_cache = False
    return backbone, tokenizer, model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIDLM (Huang et al., 2025) with Unsloth + LoRA")

    # Model selection
    p.add_argument(
        "--model_dir_name",
        type=str,
        default="Qwen2.5-3B-Instruct",
        help="Subdirectory name inside src/base_models/.",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit model folder path (overrides --model_dir_name).",
    )

    # Data
    p.add_argument(
        "--data_json",
        type=str,
        default="src/MIDLM/data/WeaveClinc150_rewritten.json",
        help="WeaveClinc150 JSON path with train/validation/test lists.",
    )
    p.add_argument("--max_k", type=int, default=3, help="Maximum number of intents per utterance (C in the paper).")
    p.add_argument("--intent_vocab_from", type=str, default="train", choices=["train", "all"])

    # Sequence / batching
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size_per_device", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Optimization
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # Logging/checkpoints
    p.add_argument("--output_dir", type=str, default="src/MIDLM/trained_models")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=400)
    p.add_argument("--save_total_limit", type=int, default=2)

    # Unsloth / precision
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target module names, or 'all-linear'.",
    )

    # MIDLM loss weights
    p.add_argument("--alpha", type=float, default=1.0, help="Weight for intent multi-label loss.")
    p.add_argument("--beta", type=float, default=1.0, help="Weight for intent-number loss.")

    # Quick smoke run
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=500)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16) and not bf16

    backbone, tokenizer, model_path = load_backbone_with_lora(args)

    data = load_weave_json(args.data_json)
    train_rows = data["train"]
    val_rows = data["validation"]

    if args.intent_vocab_from == "all":
        vocab_rows = data["train"] + data["validation"] + data["test"]
    else:
        vocab_rows = data["train"]

    intents, intent_to_id = build_intent_vocab(vocab_rows)
    logger.info("Intent vocabulary size: %d", len(intents))

    if args.max_train_samples is not None:
        train_rows = train_rows[: int(args.max_train_samples)]
    train_ds = WeaveMultiIntentDataset(train_rows, intent_to_id=intent_to_id, max_k=args.max_k)

    eval_rows = val_rows[: int(args.max_eval_samples)]
    eval_ds = WeaveMultiIntentDataset(eval_rows, intent_to_id=intent_to_id, max_k=args.max_k)

    model = MIDLMForMultiIntent(
        backbone,
        num_intents=len(intents),
        max_k=int(args.max_k),
        alpha=float(args.alpha),
        beta=float(args.beta),
    )

    collator = MIDLMDataCollator(tokenizer, max_seq_length=args.max_seq_length)

    exp_name = f"{Path(model_path).name}_midlm"
    out_dir = Path(args.output_dir) / exp_name
    base_models_root = (Path(__file__).resolve().parents[1] / "base_models").resolve()
    if base_models_root in out_dir.resolve().parents or out_dir.resolve() == base_models_root:
        raise ValueError(
            f"Refusing to write outputs under base model store: {out_dir}. "
            "Use --output_dir under src/MIDLM/trained_models."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist training config + intent vocab for later inference/eval.
    (out_dir / "intent_vocab.json").write_text(json.dumps(intents, indent=2), encoding="utf-8")
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "data_json": str(args.data_json),
                "max_k": int(args.max_k),
                "max_seq_length": int(args.max_seq_length),
                "alpha": float(args.alpha),
                "beta": float(args.beta),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size_per_device),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        report_to=[],
        seed=int(args.seed),
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=int(args.save_steps),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting training. Output: %s", str(out_dir))
    trainer.train()

    # Save LoRA adapter + tokenizer in the same output folder.
    try:
        model.backbone.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
    except Exception as e:
        logger.warning("Could not save adapter/tokenizer via backbone: %s", str(e))
        trainer.save_model(str(out_dir))

    # Save MIDLM task heads explicitly (intent head + intent-number head).
    torch.save(
        {
            "intent_head": model.intent_head.state_dict(),
            "num_head": model.num_head.state_dict(),
            "num_intents": int(model.num_intents),
            "max_k": int(model.max_k),
            "alpha": float(model.alpha),
            "beta": float(model.beta),
        },
        out_dir / "midlm_heads.pt",
    )

    logger.info("Done. Saved to: %s", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

