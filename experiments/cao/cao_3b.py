#!/usr/bin/env python3
"""
CAO+JDV processor — runs CAO records through a local 3B model (LM Studio)
and saves responses to experiments/cao/data/cao_3b_results/.

Usage:
    uv run python experiments/cao/cao_3b.py [--limit N] [--cao-dir PATH]

Requires RESPONSE_3B_MODEL_ID and VAGUE_ENDPOINT_URL in .env.
Load the base Qwen 2.5 3B Instruct model in LM Studio (not the JDV adapter).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cao.llm_shared import (  # noqa: E402
    build_cao_prompt,
    call_openai_compatible,
    extract_jdv_info,
    get_completed_record_ids,
    load_cao_records,
    sanitize_filename,
    save_result,
)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DEFAULT_CAO_DIR = PROJECT_ROOT / "experiments" / "cao" / "data" / "CAO_results"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "cao" / "data" / "cao_3b_results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAO+JDV local 3B LLM processor"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N CAO records")
    parser.add_argument("--cao-dir", type=str, default=None,
                        help="Directory with CAO JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip record_ids already present in output dir")
    args = parser.parse_args()

    cao_dir = Path(args.cao_dir) if args.cao_dir else DEFAULT_CAO_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    endpoint = os.environ.get("VAGUE_ENDPOINT_URL", "").strip()
    model_id = os.environ.get("RESPONSE_3B_MODEL_ID", "").strip()

    if not endpoint:
        print("[CAO_3B] ERROR: VAGUE_ENDPOINT_URL not found in environment!")
        sys.exit(1)
    if not model_id:
        print("[CAO_3B] ERROR: RESPONSE_3B_MODEL_ID not found in environment!")
        sys.exit(1)

    if not cao_dir.exists():
        print(f"[CAO_3B] ERROR: CAO directory not found: {cao_dir}")
        print("[CAO_3B] Run batch_cao.py --with-jdv first.")
        sys.exit(1)

    records = load_cao_records(cao_dir)
    if args.limit:
        records = records[: args.limit]

    completed_ids = (
        get_completed_record_ids(output_dir, "cao_3b")
        if args.skip_existing
        else set()
    )

    total = len(records)
    completed = 0
    skipped = 0
    errors = 0

    print(f"\n[CAO_3B] Processing {total} CAO records using {model_id}")
    print(f"[CAO_3B] Endpoint: {endpoint}")
    print(f"[CAO_3B] CAO dir: {cao_dir}")
    print(f"[CAO_3B] Output dir: {output_dir}\n")

    for record in records:
        record_id = record.get("record_id", "")
        index = record.get("index", 0)
        query = record.get("query", "")

        print(f"[{index + 1:04d}/{total:04d}] {record_id}")
        print(f"  Query: {query[:120]}...")

        if record_id in completed_ids:
            print("  [SKIP] Already processed")
            skipped += 1
            print()
            continue

        slug = sanitize_filename(query, max_len=60)
        system_prompt, user_prompt = build_cao_prompt(record)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            start_time = time.time()
            response_text = call_openai_compatible(
                messages, endpoint, model_id, max_tokens=512, timeout=60
            )
            elapsed = time.time() - start_time
            print(f"  [OK] Response received in {elapsed:.2f}s")

            payload = {
                "meta": {
                    "record_id": record_id,
                    "index": index,
                    "original_query": query,
                    "jdv": extract_jdv_info(record),
                    "model": model_id,
                    "endpoint": endpoint,
                    "cao_source_file": record.get("source_file", ""),
                    "elapsed_seconds": elapsed,
                },
                "prompt": {"system": system_prompt, "user": user_prompt},
                "response": response_text,
            }
            filepath = save_result(
                output_dir, "cao_3b", record_id, index, slug, payload
            )
            print(f"  [SAVED] {filepath.name}")
            completed += 1

        except Exception as e:
            print(f"  [ERROR] Failed to process record: {e}")
            errors += 1

        print()

    print("=" * 60)
    print(
        f"[CAO_3B] Complete: {total} total, {completed} completed, "
        f"{skipped} skipped, {errors} errors"
    )
    print(f"[CAO_3B] Results in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
