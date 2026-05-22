#!/usr/bin/env python3
"""Generate conversation records for all models on the IN3 test set."""
import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import MODELS, IN3_TEST_PATH, LABELLER_PATH, OUTPUTS_DIR, TASK_DESCRIPTION

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("inference")

MAX_ROUNDS = 5
MAX_GEN_LENGTH = 512


def ensure_tokenizer(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token or tokenizer.bos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token


def load_model(model_cfg: Dict[str, Any]):
    base_path = model_cfg["base_model_path"]
    model_type = model_cfg.get("type", "3b")
    adapter_path = model_cfg.get("adapter_path")

    logger.info("Loading base model from: %s", base_path)

    if model_type == "7b":
        logger.info("Using 4-bit quantization for 7B model")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        dtype = torch.bfloat16
    else:
        logger.info("Using 8-bit quantization for 3B model")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.float16

    use_trust = "phi" not in Path(base_path).name.lower()
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=use_trust)
    ensure_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=use_trust,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info("Loading LoRA adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            base_model_name_or_path=base_path,
        )
        logger.info("Adapter loaded successfully")
    else:
        logger.info("No adapter — running as untrained baseline")

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new: int = MAX_GEN_LENGTH) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_part = full[len(prompt):].strip()
    return new_part


def parse_initial_thought(text: str) -> Dict[str, Any]:
    result = {"vague": True, "thought": "", "has_summary": False}
    text_upper = text.upper()

    if "[SUMMARY]" in text_upper or "SUMMARY]" in text:
        result["has_summary"] = True
        result["vague"] = False

    if "[INITIAL THOUGHT]" in text:
        parts = text.split("[INITIAL THOUGHT]")
        after_init = parts[1].split("[INQUIRY THOUGHT]")[0].split("[INQUIRY]")[0].split("[SUMMARY THOUGHT]")[0].split("[SUMMARY]")[0].strip()
        result["thought"] = after_init

    if "clear" in result["thought"].lower() and "vague" not in result["thought"].lower().split("clear")[0]:
        result["vague"] = False
    elif "vague" in result["thought"].lower():
        result["vague"] = True

    return result


def parse_inquiry(text: str) -> Optional[Dict[str, Any]]:
    if "[INQUIRY]" not in text and "[INQUIRY THOUGHT]" not in text:
        return None

    inquiry_text = text
    if "[INQUIRY]" in text:
        inquiry_text = text.split("[INQUIRY]")[-1].strip()
    elif "[INQUIRY THOUGHT]" in text:
        parts = text.split("[INQUIRY THOUGHT]")
        if len(parts) > 1:
            after_thought = parts[1]
            if "[INQUIRY]" in after_thought:
                inquiry_text = after_thought.split("[INQUIRY]")[-1].strip()
            else:
                inquiry_text = after_thought.strip()

    inquiry_text = re.split(r'\[(?:SUMMARY|SUMMARY THOUGHT)\]', inquiry_text)[0].strip()
    inquiry_text = inquiry_text.strip()

    if not inquiry_text:
        return None

    options = []
    question = inquiry_text

    sep_patterns = [
        r'\?',
        r':',
    ]

    question_part = inquiry_text
    for sep in sep_patterns:
        parts = re.split(sep, inquiry_text, maxsplit=1)
        if len(parts) > 1 and len(parts[1].strip()) > 5:
            question_part = parts[0] + sep
            options_text = parts[1]
            raw_options = re.split(r'[,;]', options_text)
            options = [o.strip().strip('?.').strip() for o in raw_options if o.strip()]
            break

    options = [o for o in options if len(o) > 1][:10]

    return {"query": question_part.strip(), "options": options}


def parse_summary(text: str) -> Optional[str]:
    if "[SUMMARY]" in text:
        parts = text.split("[SUMMARY]")
        if len(parts) > 1:
            summary = parts[-1].strip()
            summary = re.split(r'\[(?:INITIAL THOUGHT|INQUIRY THOUGHT|INQUIRY)\]', summary)[0].strip()
            return summary
    return None


def simulate_user_response(
    inquiry: Dict[str, Any],
    ground_truth_info: List[Dict[str, Any]],
) -> str:
    query_lower = inquiry["query"].lower()
    options = inquiry["options"]

    for info in ground_truth_info:
        desc = info["description"].lower()
        if any(word in query_lower for word in desc.split()):
            if options:
                chosen = options[0]
            else:
                chosen = desc
            templates = [
                "Let's go with {opt} — that fits best.",
                "I'll go with {opt}.",
                "Oh, {opt} works for me.",
                "{opt} sounds good.",
            ]
            return random.choice(templates).format(opt=chosen)

    return "I don't need to specify that, let's move on."


def run_task(
    model,
    tokenizer,
    task: str,
    ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = f"{TASK_DESCRIPTION}\n\nHere is the task:\n{task}"
    full_prompt = prompt

    actions = []
    vague = True
    summary_text = None
    query_options_list = []

    for turn in range(MAX_ROUNDS):
        agent_prefix = "\nAgent: " if turn > 0 or "Agent:" in full_prompt else "Agent: "
        full_prompt += agent_prefix

        response = generate(model, tokenizer, full_prompt)

        if turn == 0:
            init_info = parse_initial_thought(response)
            vague = init_info["vague"]

            if init_info.get("thought"):
                actions.append({
                    "role": "assistant",
                    "thought": init_info["thought"],
                    "content": response,
                    "type": "initial_thought",
                })

        has_summary = "[SUMMARY]" in response
        inquiry = None if has_summary else parse_inquiry(response)

        if has_summary:
            summary_text = parse_summary(response)
            thought_text = ""
            if "[SUMMARY THOUGHT]" in response:
                thought_text = response.split("[SUMMARY THOUGHT]")[1].split("[SUMMARY]")[0].strip()

            actions.append({
                "role": "assistant",
                "thought": thought_text,
                "content": summary_text or response,
                "type": "summary",
            })
            break

        if inquiry:
            thought_text = ""
            if "[INQUIRY THOUGHT]" in response:
                thought_text = response.split("[INQUIRY THOUGHT]")[1].split("[INQUIRY]")[0].strip()

            actions.append({
                "role": "assistant",
                "thought": thought_text,
                "content": inquiry["query"],
                "type": "New",
            })

            query_options_list.append([inquiry])

            ground_truth_info = []
            if ground_truth:
                ground_truth_info.extend(ground_truth.get("user_approve", []))
                ground_truth_info.extend(ground_truth.get("user_rectify", []))
                ground_truth_info.extend(ground_truth.get("user_add", []))

            user_reply = simulate_user_response(inquiry, ground_truth_info)
            full_prompt += response + "\n"
            full_prompt += user_reply

            actions.append({
                "role": "user",
                "thought": None,
                "content": user_reply,
                "type": "response",
            })
        else:
            # Model didn't produce a proper inquiry or summary — force summary
            actions.append({
                "role": "assistant",
                "thought": "Summarizing based on gathered information.",
                "content": "The user's goal is: " + task,
                "type": "summary",
            })
            break

    if summary_text is None:
        summary_text = "No summary generated."

    return {
        "vague": vague,
        "actions": actions,
        "query_options_list": query_options_list,
        "summary": summary_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate conversation records")
    parser.add_argument("--model", type=str, default="all",
                        help="Model key from config, or 'all' for all models")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test tasks (for debugging)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if output already exists")
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    # Load test tasks
    test_tasks = []
    with open(IN3_TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            test_tasks.append(json.loads(line))

    if args.limit:
        test_tasks = test_tasks[:args.limit]

    logger.info("Loaded %d test tasks", len(test_tasks))

    # Load ground truth labels
    labeller = {}
    with open(LABELLER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            labeller[entry["task"]] = entry

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.warning("Unknown model: %s, skipping", model_key)
            continue

        safe_key = model_key.replace("/", "_").replace(" ", "_")
        out_path = OUTPUTS_DIR / f"{safe_key}_interactions.jsonl"
        if out_path.exists() and not args.force:
            existing_lines = sum(1 for _ in open(out_path))
            if existing_lines == len(test_tasks):
                logger.info("Skipping %s — output already exists (%d tasks)", model_key, existing_lines)
                continue
            else:
                logger.warning("%s has incomplete output (%d/%d), re-running", model_key, existing_lines, len(test_tasks))

        logger.info("=" * 60)
        logger.info("Running inference for model: %s", model_key)
        logger.info("=" * 60)

        model_cfg = MODELS[model_key]
        try:
            model, tokenizer = load_model(model_cfg)
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_key, e)
            continue

        records = []
        for i, task_entry in enumerate(test_tasks):
            task = task_entry["task"]
            gt = labeller.get(task, {})

            logger.info("  [%d/%d] %s", i + 1, len(test_tasks), task[:60])
            try:
                result = run_task(model, tokenizer, task, gt)
                result["category"] = task_entry.get("category", "")
                result["task"] = task
                records.append(result)
            except Exception as e:
                logger.error("  Error on task %d: %s", i, e)
                records.append({
                    "vague": False,
                    "task": task,
                    "actions": [],
                    "query_options_list": [],
                    "summary": "",
                    "error": str(e),
                })

        # Save
        safe_key = model_key.replace("/", "_").replace(" ", "_")
        out_path = OUTPUTS_DIR / f"{safe_key}_interactions.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Saved %d records to %s", len(records), out_path)

        # Clean up to free VRAM for next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All done!")


if __name__ == "__main__":
    main()
