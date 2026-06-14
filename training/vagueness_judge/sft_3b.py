#!/usr/bin/env python3
"""LoRA fine-tuning for 3B models optimized for 12GB VRAM.
Uses 8-bit (default) or 16-bit precision - no 4-bit needed for 3B model.
8-bit: ~3GB model + low memory overhead = fits easily in 12GB VRAM.
16-bit: ~6GB model + low memory overhead = fits in 12GB VRAM.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(str(Path(__file__).parent))

from dataset_wrapper import PromptIterableDataset, collator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TASK_DESCRIPTION = """You are an agent trying to understand the user's goal and summarize it. Please first ask users for more specific details with options, and finally summarize the user's intention.
--- Step 1: initial thought generation ---
1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.
2. List the important missing details and some according options if the task is vague.
--- Step 2: inquiry for more information if vague ---
1. If the task is vague, inquire about more details with options according to the list in [INITIAL THOUGHT].
2. Think about what information you have and what to inquire next in [INQUIRY THOUGHT].
3. Present your inquiry with options for the user to choose after [INQUIRY], and be friendly.
4. You could repeat Step 2 multiple times (but less than 5 times), or directly skip Step 2 if the user task is clear initially.
--- Step 3: summarize the user's intention ---
1. Make the summary once the information is enough. You do not need to inquire about every missing detail in [INITIAL THOUGHT].
2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be the same as rounds of chatting.
3. Give the final summary after [SUMMARY] with comprehensive details in one or two sentences."""


def format_missing_details(i: int, detail: Dict[str, Any]) -> str:
    return f"- {detail['description']}: {', '.join(detail['options'])}\n"


def format_one(
    idx: int,
    thought: str,
    action: Dict[str, Any],
    is_vague: bool,
    missing_details: List[Dict[str, Any]],
) -> str:
    response = ""
    if action["role"] == "user":
        response = f"{action['content']}"
    else:
        if action["type"] == "New":
            response = (
                f"[INQUIRY THOUGHT] {action['thought']}\n[INQUIRY] {action['content']}"
            )
        elif action["type"] == "summary":
            response = (
                f"[SUMMARY THOUGHT] {action['thought']}\n[SUMMARY] {action['content']}"
            )

    initial_thought_str = f"[INITIAL THOUGHT] {thought}"
    if idx == 0:
        if is_vague:
            details_str = " Some aspects of missing details and potential options are as follows:\n"
            for j, detail in enumerate(missing_details):
                details_str += format_missing_details(j, detail)
            response = initial_thought_str + details_str + response
        else:
            response = initial_thought_str + "\n" + response
    return response


def preprocess_data(data: Dict[str, Any], args: argparse.Namespace) -> Any:
    first_query = f"{TASK_DESCRIPTION}\n\nHere is the task:\n{data['task']}"
    sequences: List[str] = [first_query]

    if args.data_setting == "MTSD":
        sequences.extend([
            format_one(idx, data["thought"], action, data["vague"], data["missing_details"])
            for idx, action in enumerate(data["actions"])
        ])
        return {"data": sequences}

    if args.data_setting == "MTMD":
        dataset = []
        for idx, action in enumerate(data["actions"]):
            sequences.append(format_one(idx, data["thought"], action, data["vague"], data["missing_details"]))
            if action["role"] == "assistant":
                dataset.append({"data": sequences.copy()})
        return dataset

    raise ValueError(f"Not supported type {args.data_setting}")


def load_raw_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if args.data_setting == "MTSD":
                dataset.append(preprocess_data(data, args))
            elif args.data_setting == "MTMD":
                dataset.extend(preprocess_data(data, args))
            else:
                raise ValueError(f"Not supported type {args.data_setting}")

    random.shuffle(dataset)
    if args.max_train_samples is not None:
        dataset = dataset[:args.max_train_samples]
    return dataset


def load_model_and_tokenizer(args: argparse.Namespace):
    base_dir = Path(__file__).resolve().parents[2]
    base_models_dir =     base_dir / "training" / "base_models"
    model_path = base_models_dir / args.model_dir_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    logger.info("Loading model from: %s", str(model_path))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

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
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # 3B model: 4-bit (for Phi-3), 8-bit (default), or 16-bit
    if args.load_in_4bit:
        logger.info("Loading 3B model in 4-bit quantization (for Phi-3)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.load_in_8bit:
        logger.info("Loading 3B model in 8-bit quantization (~3GB VRAM)")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.bf16 or args.fp16:
        logger.info("Loading 3B model in 16-bit precision (~6GB VRAM)")
        quantization_config = None
    else:
        # Default: 8-bit for memory efficiency
        logger.info("Loading 3B model in 8-bit quantization (default for 3B)")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="cuda:0",
        attn_implementation='eager',  # Avoid flash_attn issues with Phi-3
    )

    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA config for 3B model
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="3B model LoRA training for 12GB VRAM")

    parser.add_argument("--model_dir_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--data_setting", type=str, default="MTMD", choices=["MTSD", "MTMD"])
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size_per_device", type=int, default=2,
                        help="3B model can use batch_size=2 in 12GB VRAM")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--validation_split", type=float, default=0.15,
                        help="Fraction of training data to hold out for validation")

    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Reduce to 256 if still OOM")

    # 3B model options: 8-bit (default), 4-bit, or 16-bit (bf16/fp16)
    parser.add_argument("--load_in_8bit", action="store_true", default=True,
                        help="8-bit quantization (default for 3B, ~3GB VRAM)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4-bit quantization (for Phi-3 compatibility)")
    parser.add_argument("--bf16", action="store_true",
                        help="16-bit brain float (higher precision, ~6GB VRAM)")
    parser.add_argument("--fp16", action="store_true",
                        help="16-bit float (alternative to bf16)")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj")

    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set default paths
    base_dir = Path(__file__).resolve().parents[2]
    if args.train_data_path is None:
        args.train_data_path = str(base_dir / "src" / "Vagueness_Judge" / "data" / "interactions" / "interaction_data_train.jsonl")
    if args.output_dir is None:
        args.output_dir = str(base_dir / "src" / "Vagueness_Judge" / "training_models")

    model, tokenizer = load_model_and_tokenizer(args)

    raw_dataset = load_raw_dataset(args)

    # Train/validation split
    if args.validation_split > 0:
        train_raw, val_raw = train_test_split(
            raw_dataset, test_size=args.validation_split, random_state=args.seed
        )
        logger.info("Split dataset: train=%d, validation=%d", len(train_raw), len(val_raw))
    else:
        train_raw, val_raw = raw_dataset, []

    train_dataset = PromptIterableDataset(
        train_raw,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        teacher_forcing=True,
        truncate_method="tail",
    )
    val_dataset = (
        PromptIterableDataset(
            val_raw,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            teacher_forcing=True,
            truncate_method="tail",
        )
        if val_raw
        else None
    )

    train_data_collator = lambda features: collator(tokenizer, features)

    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16) and not bf16

    exp_output = Path(args.output_dir) / f"{args.model_dir_name}-Vagueness_Judge"
    exp_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(exp_output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch" if val_dataset else "no",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_step,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_data_collator,
    )

    resume_from_checkpoint = None
    if args.resume:
        if exp_output.exists():
            ckpt_dirs = sorted(exp_output.glob("checkpoint-*"))
            if ckpt_dirs:
                resume_from_checkpoint = str(ckpt_dirs[-1])
                logger.info("Resuming training from checkpoint: %s", resume_from_checkpoint)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(str(exp_output))
    tokenizer.save_pretrained(str(exp_output))

    # Save train_config.json so evaluation knows which base model to use
    base_dir = Path(__file__).resolve().parents[2]
    base_models_dir =     base_dir / "training" / "base_models"
    model_path = str((base_models_dir / args.model_dir_name).resolve())
    train_config = {
        "model_path": model_path,
        "train_data_path": str(Path(args.train_data_path).resolve()),
        "validation_split": args.validation_split,
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    (exp_output / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    logger.info("Training finished. Saved to: %s", str(exp_output))


if __name__ == "__main__":
    main()
