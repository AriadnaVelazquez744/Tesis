#!/usr/bin/env python3
"""
Batch Vagueness Judge runner — runs the trained JDV model locally (HF+LoRA)
over evaluation queries and saves structured JSON results.

Designed for Colab GPU or local machine with CUDA. Does not require LM Studio
or the full CAO pipeline.

Usage:
    uv run python experiments/cao/jdv_runner.py \\
        --adapter-path /path/to/Qwen2.5-3B-Instruct-Vagueness_Judge

    # Or with a merged full model:
    uv run python experiments/cao/jdv_runner.py \\
        --merged-model-path /path/to/merged-model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cao.llm_shared import load_queries, sanitize_filename
from src.Vagueness_Judge.runtime.model_api import (
    TASK_DESCRIPTION,
    _build_messages,
    _parse_model_response,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("jdv_runner")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "cao" / "data" / "evaluation_sample_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "cao" / "data" / "jdv_results"
MAX_NEW_TOKENS = 512


def ensure_tokenizer(tokenizer) -> None:
    if tokenizer.eos_token is None:
        tokenizer.eos_token = (
            tokenizer.pad_token or tokenizer.bos_token or tokenizer.unk_token
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token


def load_jdv_model(
    *,
    base_model: str,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if merged_model_path:
        model_path = merged_model_path
        logger.info("Loading merged JDV model from: %s", model_path)
        use_trust = "phi" not in Path(model_path).name.lower()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=use_trust
        )
        ensure_tokenizer(tokenizer)
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=use_trust,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        if not adapter_path or not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}. "
                "Provide --adapter-path or --merged-model-path."
            )
        logger.info("Loading base model: %s", base_model)
        logger.info("Loading LoRA adapter: %s", adapter_path)
        use_trust = "phi" not in Path(base_model).name.lower()
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=use_trust
        )
        ensure_tokenizer(tokenizer)
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=use_trust,
            quantization_config=quant_config,
            device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            base_model_name_or_path=base_model,
        )
        logger.info("Adapter loaded successfully")

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, query: str) -> str:
    messages = _build_messages(
        {"mode": "initial", "query": query, "clarifications": [], "turns": []}
    )
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"{TASK_DESCRIPTION}\n\n"
        f"Here is the task:\n{query}\n\n"
        "Agent: "
    )


def generate_response(model, tokenizer, query: str) -> str:
    import torch

    prompt = build_prompt(tokenizer, query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()


def evaluate_query(model, tokenizer, query: str) -> Dict[str, Any]:
    raw = generate_response(model, tokenizer, query)
    parsed = _parse_model_response(raw, query, [])
    parsed["raw_response"] = raw
    return parsed


def get_completed_record_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not output_dir.exists():
        return completed
    for path in output_dir.glob("jdv_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            record_id = data.get("meta", {}).get("record_id", "")
            if record_id:
                completed.add(record_id)
        except (json.JSONDecodeError, OSError):
            continue
    return completed


def save_jdv_result(
    output_dir: Path,
    record: Dict[str, Any],
    index: int,
    jdv: Dict[str, Any],
    timestamp: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    record_id = record.get("id", f"q_{index:04d}")
    query = record.get("query", "").strip()
    slug = sanitize_filename(query, max_len=60)
    filename = f"jdv_{timestamp}_{index:04d}_{record_id}_{slug}.json"
    filepath = output_dir / filename

    payload = {
        "meta": {
            "record_id": record_id,
            "index": index,
            "query": query,
            "topic": record.get("topic", "unknown"),
            "oos": record.get("oos", False),
            "k": record.get("k", 1),
            "vague": record.get("vague", False),
            "domain_cluster": record.get("domain_cluster", ""),
            "timestamp": timestamp,
        },
        "jdv": {
            "status": jdv.get("status", "unknown"),
            "completed_query": jdv.get("completed_query", query),
            "summary": jdv.get("summary", ""),
            "summary_thought": jdv.get("summary_thought", ""),
            "raw_response": jdv.get("raw_response", jdv.get("_raw", "")),
        },
    }
    if jdv.get("status") == "needs_clarification":
        payload["jdv"]["question"] = jdv.get("question", "")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath


def write_manifest(output_dir: Path, manifest: Dict[str, str]) -> Path:
    path = output_dir / "jdv_results_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def load_jdv_results(jdv_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Load precomputed JDV results keyed by record_id and by query text."""
    by_id: Dict[str, Dict[str, Any]] = {}
    by_query: Dict[str, str] = {}

    for path in sorted(jdv_dir.glob("jdv_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {})
        record_id = meta.get("record_id", "")
        query = meta.get("query", "").strip()
        jdv = data.get("jdv", {})
        if not record_id:
            continue
        entry = {
            "status": jdv.get("status", "resolved"),
            "completed_query": jdv.get("completed_query", query),
            "summary": jdv.get("summary", ""),
            "summary_thought": jdv.get("summary_thought", ""),
            "raw_response": jdv.get("raw_response", ""),
            "question": jdv.get("question", ""),
            "source_file": path.name,
            "query": query,
        }
        by_id[record_id] = entry
        if query:
            by_query[query] = record_id

    return by_id, by_query


def run_batch(
    *,
    input_path: Path,
    output_dir: Path,
    base_model: str,
    adapter_path: Optional[str],
    merged_model_path: Optional[str],
    limit: Optional[int],
    skip_existing: bool,
) -> Dict[str, str]:
    records = load_queries(input_path)
    if limit:
        records = records[:limit]

    completed_ids = get_completed_record_ids(output_dir) if skip_existing else set()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest: Dict[str, str] = {}

    if output_dir.exists():
        existing_manifest = output_dir / "jdv_results_manifest.json"
        if existing_manifest.exists():
            with open(existing_manifest, encoding="utf-8") as f:
                manifest.update(json.load(f))

    total = len(records)
    logger.info("Loading JDV model...")
    model, tokenizer = load_jdv_model(
        base_model=base_model,
        adapter_path=adapter_path,
        merged_model_path=merged_model_path,
    )
    logger.info("Processing %d queries → %s", total, output_dir)

    completed = 0
    skipped = 0
    errors = 0

    for idx, record in enumerate(records):
        record_id = record.get("id", f"q_{idx:04d}")
        query = record.get("query", "").strip()

        print(f"[{idx + 1:04d}/{total:04d}] {record_id}")
        print(f"  Query: {query[:120]}...")

        if record_id in completed_ids:
            print("  [SKIP] Already processed")
            skipped += 1
            print()
            continue

        try:
            jdv = evaluate_query(model, tokenizer, query)
            filepath = save_jdv_result(output_dir, record, idx, jdv, timestamp)
            manifest[record_id] = filepath.name
            status = jdv.get("status", "unknown")
            print(f"  [OK] status={status}")
            print(f"  [SAVED] {filepath.name}")
            completed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors += 1

        print()

    write_manifest(output_dir, manifest)
    print("=" * 60)
    print(
        f"[JDV] Complete: {total} total, {completed} completed, "
        f"{skipped} skipped, {errors} errors"
    )
    print(f"[JDV] Results in: {output_dir}")
    print("=" * 60)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch JDV runner (local HF inference)")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to evaluation queries JSON")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Directory for JDV result JSON files")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="HuggingFace base model ID or path")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (*-Vagueness_Judge)")
    parser.add_argument("--merged-model-path", type=str, default=None,
                        help="Path to merged full model (alternative to adapter)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N queries")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip record_ids already present in output dir")
    args = parser.parse_args()

    if not args.adapter_path and not args.merged_model_path:
        parser.error("Provide --adapter-path or --merged-model-path")

    run_batch(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
