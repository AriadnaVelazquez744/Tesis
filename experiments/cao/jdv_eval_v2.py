#!/usr/bin/env python3
"""
Interactive JDV evaluation for v2 — sends queries from evaluation_sample_v2.json
to a JDV model (via API or local), handles multi-turn clarification rounds,
and saves results with full v2 metadata.

Two modes:
  1. API mode (default): connect to a remote JDV server (e.g. running on Colab)
  2. Local mode: load model + adapter directly on this machine (requires GPU)

Usage:
  # API mode — Colab server
  export VAGUE_ENDPOINT_URL="https://xxxx.ngrok.io"
  uv run python experiments/cao/jdv_eval_v2.py

  # API mode — explicit URL
  uv run python experiments/cao/jdv_eval_v2.py --api-url https://xxxx.ngrok.io

  # Local mode (GPU required)
  uv run python experiments/cao/jdv_eval_v2.py \
      --adapter-path training/vagueness_judge/adapters/Qwen2.5-3B-Instruct-Vagueness_Judge

  # Process only first 5 queries
  uv run python experiments/cao/jdv_eval_v2.py --limit 5

  # Resume existing session (skips completed record_ids)
  uv run python experiments/cao/jdv_eval_v2.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("jdv_eval_v2")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "cao" / "data" / "evaluation_sample_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "cao" / "data" / "jdv_results_v2"
MAX_NEW_TOKENS = 512


# ── helpers ────────────────────────────────────────────────────────


def sanitize_filename(text: str, max_len: int = 60) -> str:
    import re
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:max_len].strip("_")


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def get_completed_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not output_dir.exists():
        return completed
    for path in output_dir.glob("jdv_v2_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            rid = data.get("meta", {}).get("record_id", "")
            if rid:
                completed.add(rid)
        except (json.JSONDecodeError, OSError):
            continue
    return completed


TASK_DESCRIPTION = (
    "You are an agent trying to understand the user's goal and summarize it. "
    "Please first ask users for more specific details with options, and finally summarize the user's intention.\n"
    "--- Step 1: initial thought generation ---\n"
    "1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.\n"
    "2. List the important missing details and some according options if the task is vague.\n"
    "--- Step 2: inquiry for more information if vague ---\n"
    "1. If the task is vague, inquire about more details with options according to the list in [INITIAL THOUGHT].\n"
    "2. Think about what information you have and what to inquire next in [INQUIRY THOUGHT].\n"
    "3. Present your inquiry with options for the user to choose after [INQUIRY], and be friendly.\n"
    "4. You could repeat Step 2 multiple times (but less than 5 times), or directly skip Step 2 if the user task is clear initially.\n"
    "--- Step 3: summarize the user's intention ---\n"
    "1. Make the summary once the information is enough. You do not need to inquire about every missing detail in [INITIAL THOUGHT].\n"
    "2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be the same as rounds of chatting.\n"
    "3. Give the final summary after [SUMMARY] with comprehensive details in one or two sentences."
)


# ── API mode ──────────────────────────────────────────────────────


def _build_messages(query: str, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build OpenAI-compatible messages for the JDV model."""
    if not turns:
        return [
            {"role": "system", "content": TASK_DESCRIPTION},
            {"role": "user", "content": f"Here is the task:\n{query}"},
        ]
    messages = [{"role": "system", "content": TASK_DESCRIPTION}]
    for turn in turns:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def call_jdv_api(
    api_url: str,
    query: str,
    turns: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Call the JDV model via OpenAI-compatible API and return parsed result."""
    import requests

    messages = _build_messages(query, turns)
    endpoint = api_url.rstrip("/")
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    model_text = data["choices"][0]["message"]["content"]
    return _parse_jdv_response(model_text, query)


def _parse_jdv_response(text: str, query: str) -> Dict[str, Any]:
    """Parse JDV model output extracting status, question, summary, etc."""
    has_summary = "[SUMMARY]" in text
    has_inquiry = "[INQUIRY]" in text

    if has_summary:
        summary = text.split("[SUMMARY]")[-1].strip() if "[SUMMARY]" in text else ""
        for tag in ["[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[INQUIRY", "[SUMMARY THOUGHT"]:
            if tag in summary:
                summary = summary.split(tag)[0].strip()

        summary_thought = ""
        if "[SUMMARY THOUGHT]" in text:
            st = text.split("[SUMMARY THOUGHT]")[-1].strip()
            for tag in ["[SUMMARY]", "[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[INQUIRY"]:
                if tag in st:
                    st = st.split(tag)[0].strip()
            specs = [
                l.strip().lstrip("-* ").strip()
                for l in st.split("\n")
                if l.strip().startswith("- ") or l.strip().startswith("* ")
            ]
            summary_thought = "\n".join(specs)

        completed = summary.strip() if summary.strip() else query.strip()
        return {
            "status": "resolved",
            "completed_query": completed,
            "summary": summary,
            "summary_thought": summary_thought,
            "raw_response": text,
        }

    if has_inquiry:
        question = text.split("[INQUIRY]")[-1].strip()
        for tag in ["[SUMMARY", "[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[SUMMARY THOUGHT"]:
            if tag in question:
                question = question.split(tag)[0].strip()
        if not question:
            question = "Could you provide more details about your request?"
        return {
            "status": "needs_clarification",
            "question": question,
            "raw_response": text,
        }

    question = text.strip()
    for tag in ["[INITIAL THOUGHT]", "[INQUIRY THOUGHT]", "[INQUIRY]", "[SUMMARY THOUGHT]", "[SUMMARY]"]:
        if tag in question:
            question = question.replace(tag, "").strip()
    question = question.strip().strip(":\n ")
    return {
        "status": "needs_clarification",
        "question": question or "Could you clarify your request?",
        "raw_response": text,
    }


# ── Local mode ────────────────────────────────────────────────────


def _ensure_tokenizer(tokenizer) -> None:
    if tokenizer.eos_token is None:
        tokenizer.eos_token = (
            tokenizer.pad_token or tokenizer.bos_token or tokenizer.unk_token
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token


def load_local_model(*, base_model: str, adapter_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_trust = "phi" not in Path(base_model).name.lower()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=use_trust)
    _ensure_tokenizer(tokenizer)

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
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def call_local_model(
    model,
    tokenizer,
    query: str,
    turns: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    import torch

    messages = _build_messages(query, turns)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full.startswith(prompt):
        raw = full[len(prompt):].strip()
    else:
        raw = full.strip()
    return _parse_jdv_response(raw, query)


# ── Orchestration ─────────────────────────────────────────────────


def run_interactive_evaluation(
    records: List[Dict[str, Any]],
    output_dir: Path,
    *,
    api_url: Optional[str],
    adapter_path: Optional[str],
    base_model: str,
    skip_existing: bool,
    resume: bool,
    non_interactive: bool,
) -> int:
    """Run the interactive JDV evaluation loop."""
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if resume or skip_existing:
        completed_ids = get_completed_ids(output_dir)
        if completed_ids:
            print(f"[SETUP] Found {len(completed_ids)} existing results, will skip")

    # Load model (only in local mode)
    model = tokenizer = None
    if api_url:
        print(f"[SETUP] API mode — endpoint: {api_url}")
    else:
        if not adapter_path:
            adapter_path = str(
                PROJECT_ROOT
                / "training"
                / "vagueness_judge"
                / "adapters"
                / "Qwen2.5-3B-Instruct-Vagueness_Judge"
            )
        print(f"[SETUP] Local mode — adapter: {adapter_path}")
        model, tokenizer = load_local_model(base_model=base_model, adapter_path=adapter_path)
        print("[SETUP] Model loaded successfully")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(records)
    completed = 0
    skipped = 0
    errors = 0
    needs_clarification_count = 0

    print(f"\n{'='*60}")
    print(f"JDV Interactive Evaluation — v2 ({total} queries)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    for idx, record in enumerate(records):
        record_id: str = record.get("id", f"q_{idx:04d}")
        query: str = record.get("query", "").strip()
        is_vague: bool = record.get("vague", False)
        is_oos: bool = record.get("oos", False)
        k: int = record.get("k", 1)
        topic: str = record.get("topic", "unknown")
        domain: str = record.get("domain_cluster", "")

        # ── skip logic ──
        if record_id in completed_ids:
            print(f"[{idx+1:04d}/{total:04d}] {record_id} [SKIP] already exists")
            skipped += 1
            continue

        print(f"[{idx+1:04d}/{total:04d}] {record_id} | topic={topic} domain={domain} "
              f"vague={is_vague} oos={is_oos} k={k}")
        print(f"  Query: {query[:120]}...")

        # ── conversation loop ──
        turns: List[Dict[str, str]] = []
        decision: Optional[Dict[str, Any]] = None
        round_num = 0
        max_rounds = 5
        clarification_history: List[str] = []

        while round_num < max_rounds:
            round_num += 1
            try:
                if api_url:
                    decision = call_jdv_api(api_url, query, turns)
                else:
                    decision = call_local_model(model, tokenizer, query, turns)
            except Exception as e:
                print(f"  [ERROR] API call failed: {e}")
                errors += 1
                decision = None
                break

            if decision["status"] == "resolved":
                if round_num > 1:
                    print(f"  [CLARIFIED] → resolved after {round_num-1} round(s)")
                else:
                    print(f"  [OK] resolved (no clarification needed)")
                break

            if decision["status"] == "needs_clarification":
                needs_clarification_count += 1
                question = decision.get("question", "Could you provide more details?")
                print(f"\n  ┌─ JDV asks (round {round_num}) ──────────────────────────")
                print(f"  │ {question}")
                print(f"  └────────────────────────────────────────────────")

                if non_interactive:
                    user_answer = "[auto-skip]"
                    print(f"  [--non-interactive] auto-reply: '{user_answer}'")
                else:
                    try:
                        user_answer = input("  Your answer: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n  [INTERRUPTED]")
                        errors += 1
                        decision = None
                        break

                if not user_answer:
                    print("  [SKIP] empty answer, marking as needs_clarification")
                    clarification_history.append(question)
                    break

                clarification_history.append(question)
                turns.append({"role": "assistant", "content": decision.get("raw_response", question)})
                turns.append({"role": "user", "content": user_answer})
                continue

            # Unknown status
            print(f"  [WARN] Unknown status: {decision.get('status')}")
            break

        if decision is None or decision.get("status") == "needs_clarification":
            print(f"  [INCOMPLETE] still needs clarification after {round_num} rounds")
            skipped += 1
            print()
            continue

        # ── save result ──
        slug = sanitize_filename(query, max_len=60)
        filename = f"jdv_v2_{timestamp}_{idx:04d}_{record_id}_{slug}.json"
        filepath = output_dir / filename

        payload = {
            "meta": {
                "record_id": record_id,
                "index": idx,
                "query": query,
                "topic": topic,
                "domain_cluster": domain,
                "oos": is_oos,
                "k": k,
                "vague": is_vague,
                "clarification_rounds": round_num - 1 if round_num > 1 else 0,
                "timestamp": timestamp,
                "source": "evaluation_sample_v2.json",
            },
            "jdv": {
                "status": decision.get("status", "unknown"),
                "completed_query": decision.get("completed_query", query),
                "summary": decision.get("summary", ""),
                "summary_thought": decision.get("summary_thought", ""),
                "raw_response": decision.get("raw_response", ""),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"  [SAVED] {filename}")
        completed += 1
        print()

    # ── summary ──
    print("=" * 60)
    print(f"Complete: {total} total, {completed} OK, {skipped} skipped, {errors} errors")
    print(f"Queries needing clarification: {needs_clarification_count}")
    print(f"Results in: {output_dir}")
    print("=" * 60)
    return errors


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Interactive JDV evaluation on v2 queries")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to evaluation_sample_v2.json")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for JDV result JSONs")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="HuggingFace base model ID")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (local mode)")
    parser.add_argument("--api-url", type=str, default=None,
                        help="JDV API endpoint URL (Colab server)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N queries")
    parser.add_argument("--resume", action="store_true",
                        help="Skip record_ids already in output dir")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip clarification questions (auto-reply)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip queries already processed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    records = load_records(input_path)
    if args.limit:
        records = records[:args.limit]

    # Determine API URL from args or env
    api_url = args.api_url or os.environ.get("VAGUE_ENDPOINT_URL") or os.environ.get("LLMSTUDIO_BASE_URL")

    exit_code = run_interactive_evaluation(
        records=records,
        output_dir=Path(args.output_dir),
        api_url=api_url,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        skip_existing=args.skip_existing,
        resume=args.resume,
        non_interactive=args.non_interactive,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
