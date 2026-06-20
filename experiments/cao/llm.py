#!/usr/bin/env python3
"""
Raw query processor — runs raw queries through Fireworks Qwen 3.7 Plus
and saves responses to experiments/cao/data/llm_results/.

Usage:
    uv run python experiments/cao/llm.py [--limit N] [--data PATH] [--skip-existing]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cao.llm_shared import (  # noqa: E402
    FIREWORKS_MODEL_ID,
    RAW_SYSTEM_PROMPT,
    call_fireworks,
    get_completed_record_ids,
    load_queries,
    sanitize_filename,
    save_result,
)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "cao" / "data" / "llm_results"
SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = SCRIPT_DIR / "data" / "evaluation_sample_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw query Fireworks LLM processor")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N queries")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to queries JSON file (overrides default)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (overrides default)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip record_ids already present in output dir (default: on)")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing",
                        help="Re-process queries even if a result file already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    data_file = Path(args.data) if args.data else _DEFAULT_DATA

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("[LLM] ERROR: FIREWORKS_API_KEY not found in environment!")
        sys.exit(1)

    queries = load_queries(data_file)
    if args.limit:
        queries = queries[: args.limit]

    completed_ids = (
        get_completed_record_ids(output_dir, "llm")
        if args.skip_existing
        else set()
    )

    total = len(queries)
    completed = 0
    skipped = 0
    errors = 0

    print(f"\n[LLM] Processing {total} queries using {FIREWORKS_MODEL_ID}")
    print(f"[LLM] Output directory: {output_dir}\n")

    for idx, record in enumerate(queries):
        query = record.get("query", "").strip()
        record_id = record.get("id", f"q_{idx:04d}")
        topic = record.get("topic", "unknown")

        print(f"[{idx + 1:04d}/{total:04d}] {record_id} | topic={topic}")
        print(f"  Query: {query[:120]}...")

        if record_id in completed_ids:
            print("  [SKIP] Already processed")
            skipped += 1
            print()
            continue

        slug = sanitize_filename(query, max_len=60)
        messages = [
            {"role": "system", "content": RAW_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        try:
            start_time = time.time()
            response_text = call_fireworks(messages, api_key, FIREWORKS_MODEL_ID)
            elapsed = time.time() - start_time
            print(f"  [OK] Response received in {elapsed:.2f}s")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            payload = {
                "meta": {
                    "query": query,
                    "record_id": record_id,
                    "index": idx,
                    "topic": topic,
                    "domain_cluster": record.get("domain_cluster", ""),
                    "oos": record.get("oos", False),
                    "k": record.get("k", 1),
                    "vague": record.get("vague", False),
                    "model": FIREWORKS_MODEL_ID,
                    "timestamp": timestamp,
                    "elapsed_seconds": elapsed,
                    "source_data": str(data_file),
                },
                "response": response_text,
            }
            filepath = save_result(
                output_dir, "llm", record_id, idx, slug, payload
            )
            completed_ids.add(record_id)
            print(f"  [SAVED] {filepath.name}")
            completed += 1

        except Exception as e:
            print(f"  [ERROR] Failed to process query: {e}")
            errors += 1

        print()

    print("=" * 60)
    print(
        f"[LLM] Complete: {total} total, {completed} completed, "
        f"{skipped} skipped, {errors} errors"
    )
    print(f"[LLM] Results in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
