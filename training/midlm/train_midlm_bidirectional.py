#!/usr/bin/env python3
"""Train MIDLM (Huang et al., 2025) with bidirectional (non-causal) attention.

Replaces Unsloth with standard HF + PEFT to support 4D attention masks.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "MIDLM"))

from midlm.data import MIDLMDataCollator, WeaveMultiIntentDataset, build_intent_vocab, load_weave_json
from midlm.model import MIDLMForMultiIntent


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODELS_DIR = _PROJECT_ROOT / "training" / "base_models"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_midlm_bidirectional")


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


def load_backbone(args: argparse.Namespace):
    model_path = Path(args.model_path) if args.model_path else (BASE_MODELS_DIR / args.model_dir_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    logger.info("Loading backbone from: %s", str(model_path))
    backbone = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=False,
    )

    # Break weight tying: Qwen2.5 (and most LLMs) tie lm_head.weight to
    # embed_tokens.weight to save memory, but safetensors refuses to save
    # shared tensors.  We detach & clone lm_head so the two become independent.
    if getattr(backbone.config, "tie_word_embeddings", False):
        backbone.lm_head.weight = torch.nn.Parameter(backbone.lm_head.weight.detach().clone())
        backbone.config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    _ensure_tokenizer_setup(tokenizer)

    target_modules: Any = args.target_modules
    if isinstance(target_modules, str):
        if target_modules.strip() == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.config.use_cache = False

    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    backbone.print_trainable_parameters()
    return backbone, tokenizer, model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MIDLM with bidirectional (non-causal) attention (Huang et al., 2025)"
    )

    p.add_argument("--model_dir_name", type=str, default="Qwen2.5-3B-Instruct")
    p.add_argument("--model_path", type=str, default=None)

    p.add_argument(
        "--data_json", type=str, default=str(SCRIPT_DIR / "data" / "WeaveClinc150_rewritten.json")
    )
    p.add_argument("--max_k", type=int, default=3)
    p.add_argument("--intent_vocab_from", type=str, default="train", choices=["train", "all"])

    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--batch_size_per_device", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "adapters" / "trained_models_bidirectional"))
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=400)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="all-linear")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument("--use_attention_pool", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=500)

    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16) and not bf16

    backbone, tokenizer, model_path = load_backbone(args)

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
        bidirectional=True,
        use_attention_pool=bool(args.use_attention_pool),
    )

    collator = MIDLMDataCollator(tokenizer, max_seq_length=args.max_seq_length)

    exp_name = f"{Path(model_path).name}_midlm_bidirectional"
    out_dir = Path(args.output_dir) / exp_name
    if BASE_MODELS_DIR.resolve() in out_dir.resolve().parents:
        raise ValueError(f"Refusing to write outputs under base model store: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
                "bidirectional": True,
                "use_attention_pool": bool(args.use_attention_pool),
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
        dataloader_pin_memory=False,
        eval_strategy="steps",
        eval_steps=int(args.save_steps),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting training. Output: %s", str(out_dir))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Delete all intermediate checkpoints to free disk space
    for ckpt_dir in out_dir.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)
            logger.info("Deleted checkpoint: %s", str(ckpt_dir))

    try:
        model.backbone.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
    except Exception as e:
        logger.warning("Could not save adapter/tokenizer via backbone: %s", str(e))
        trainer.save_model(str(out_dir))

    torch.save(
        {
            "intent_head": model.intent_head.state_dict(),
            "num_head": model.num_head.state_dict(),
            "num_intents": int(model.num_intents),
            "max_k": int(model.max_k),
            "alpha": float(model.alpha),
            "beta": float(model.beta),
            "bidirectional": True,
            "use_attention_pool": bool(model.use_attention_pool),
            "attention_query": (model.attention_query.detach().cpu() if model.use_attention_pool else None),
        },
        out_dir / "midlm_heads.pt",
    )

    logger.info("Done. Saved to: %s", str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
