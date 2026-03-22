import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import Trainer, TrainingArguments

from unsloth import FastLanguageModel

# Ensure imports work regardless of current working directory.
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
    """
    Produce one assistant/user "turn text" for the alternating dialogue format
    expected by `dataset_wrapper.PromptIterableDataset`.
    """
    response = ""
    if action["role"] == "user":
        response = f"{action['content']}"
    else:
        if action["type"] == "New":
            response = f"[INQUIRY THOUGHT] {action['thought']}\n[INQUIRY] {action['content']}"
        elif action["type"] == "summary":
            response = f"[SUMMARY THOUGHT] {action['thought']}\n[SUMMARY] {action['content']}"

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
        sequences.extend(
            [
                format_one(idx, data["thought"], action, data["vague"], data["missing_details"])
                for idx, action in enumerate(data["actions"])
            ]
        )
        return {"data": sequences}

    if args.data_setting == "MTMD":
        dataset = []
        for idx, action in enumerate(data["actions"]):
            sequences.append(
                format_one(idx, data["thought"], action, data["vague"], data["missing_details"])
            )
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
        dataset = dataset[: args.max_train_samples]
    return dataset


def _load_model_config_if_exists(model_config_path: Optional[str]) -> Dict[str, Any]:
    if not model_config_path:
        return {}
    config_path = Path(model_config_path)
    if not config_path.exists():
        logger.warning("Model config not found: %s (using defaults)", str(config_path))
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(args: argparse.Namespace) -> Any:
    base_dir = Path(__file__).resolve().parents[2]  # src/
    base_models_dir = base_dir / "base_models"
    model_path = base_models_dir / args.model_dir_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model_config = _load_model_config_if_exists(args.model_config_path)
    peft_cfg = model_config.get("peft", {})

    lora_r = int(peft_cfg.get("r", args.lora_r))
    lora_alpha = int(peft_cfg.get("lora_alpha", args.lora_alpha))
    lora_dropout = float(peft_cfg.get("lora_dropout", args.lora_dropout))
    target_modules = peft_cfg.get("target_modules", None) or args.target_modules
    if isinstance(target_modules, str):
        # Accept either a special marker (e.g. "all-linear") or a comma-separated list.
        if target_modules.strip() != "all-linear":
            target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    # dtype: prefer bf16 if requested and supported
    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    logger.info("Loading model from: %s", str(model_path))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=bool(args.load_in_4bit),
    )

    # Robust tokenizer setup for generic models
    if tokenizer.eos_token is None:
        # Prefer pad_token as a fallback; otherwise, try bos_token/unk_token.
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
        # Many decoder-only models use eos_token as a start surrogate.
        tokenizer.bos_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Unsloth usually wants this set for training
    try:
        model = FastLanguageModel.for_training(model)
    except Exception:
        pass

    model.config.use_cache = False
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument(
        "--model_dir_name",
        type=str,
        required=True,
        help="Subdirectory name inside src/base_models/ (downloaded by douwnload_models.py).",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="Optional JSON file with PEFT config for this model (e.g. target_modules).",
    )

    # Data
    base_dir = Path(__file__).resolve().parents[2]  # src/
    default_train = base_dir / "Vagueness_Judge" / "data" / "interactions" / "interaction_data_train.jsonl"
    parser.add_argument("--train_data_path", type=str, default=str(default_train))
    parser.add_argument("--data_setting", type=str, default="MTMD", choices=["MTSD", "MTMD"])
    parser.add_argument("--max_train_samples", type=int, default=None)

    # Training hyperparams
    parser.add_argument("--output_dir", type=str, default=str(base_dir / "Vagueness_Judge" / "training_models"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size_per_device", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Reduce VRAM via gradient checkpointing")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    # Sequence/model params
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Unsloth/QLoRA params
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # LoRA defaults (overridable via model_config_path)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj",
        help="Comma-separated target module names for LoRA (LLaMA/Mistral-like by default).",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_modules_list = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    args.target_modules = target_modules_list

    model, tokenizer = load_model_and_tokenizer(args)

    raw_dataset = load_raw_dataset(args)
    train_dataset = PromptIterableDataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        teacher_forcing=True,
        truncate_method="tail",
    )

    train_data_collator = partial(collator, tokenizer)

    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16) and not bf16

    exp_output = Path(args.output_dir) / args.model_dir_name
    exp_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(exp_output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=train_data_collator,
    )

    trainer.train()

    # Save adapters + tokenizer
    model.save_pretrained(str(exp_output))
    tokenizer.save_pretrained(str(exp_output))
    logger.info("Training finished. Saved to: %s", str(exp_output))


if __name__ == "__main__":
    main()
