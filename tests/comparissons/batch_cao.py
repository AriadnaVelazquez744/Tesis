#!/usr/bin/env python3
"""
Batch CAO generator — runs queries through the full pipeline (without vagueness)
and saves CAOs to test/data/CAO_results/.

Usage:
    docker compose exec app python test/scripts/batch_cao.py [--queries N] [--limit M]

Skips any query that requires human intervention (vagueness clarification).
Does not modify any existing files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "tests" / "comparissons" / "data" / "CAO_results"
SCRIPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = (
    SCRIPT_DIR / "data" / "evaluation_sample_v1.json"
)
TEST_DATA_FILE = _DEFAULT_DATA


def _make_resolved_vagueness(
    user_text: str, state: Any
) -> Dict[str, Any]:
    """Bypass vagueness: always return resolved with the raw text."""
    from src.Vagueness_Judge.runtime.pipeline import _reset_state

    return {
        "status": "resolved",
        "completed_query": user_text.strip(),
        "summary": "",
        "summary_thought": "",
        "raw_response": "",
        "updated_state": _reset_state(state) if state else None,
    }


def _patch_vagueness() -> None:
    """Patch all references to handle_vagueness_turn so vagueness is fully bypassed."""
    import src.Vagueness_Judge.runtime.pipeline as vp
    import src.Vagueness_Judge.runtime as vr
    vp.handle_vagueness_turn = _make_resolved_vagueness
    vr.handle_vagueness_turn = _make_resolved_vagueness


def _load_queries(filepath: Path) -> list[Dict[str, Any]]:
    print(f"[BATCH] Loading queries from {filepath}")
    with open(filepath) as f:
        data = json.load(f)
    return data["records"]


def _save_cao(
    cao: Dict[str, Any],
    query: str,
    record_id: str,
    status: str,
    index: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.lingo.pipeline import _sanitize_filename
    slug = _sanitize_filename(query, max_len=60)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cao_{timestamp}_{index:04d}_{record_id}_{slug}.json"
    filepath = OUTPUT_DIR / filename
    meta = {
        "_batch_info": {
            "query": query,
            "record_id": record_id,
            "index": index,
            "timestamp": timestamp,
        }
    }
    output = {"meta": meta, "cao": cao}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {filepath.name}")


def _wait_for_amr(max_retries: int = 30, interval: int = 5) -> bool:
    import requests
    print("[BATCH] Waiting for AMR service at http://127.0.0.1:8001/health ...")
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get("http://127.0.0.1:8001/health", timeout=3)
            if r.status_code < 500:
                print(f"[BATCH] AMR ready (attempt {attempt})")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"  AMR not ready, retrying in {interval}s ({attempt}/{max_retries})")
        time.sleep(interval)
    print("[BATCH] AMR not ready after all retries — continuing anyway")
    return False


def _wait_for_external_services() -> None:
    import requests
    endpoints = {
        "MIDLM": os.environ.get("MIDLM_ENDPOINT_URL", ""),
        "TEXTOIR": os.environ.get("TEXTOIR_ENDPOINT_URL", ""),
    }
    for name, url in endpoints.items():
        if not url:
            print(f"[BATCH] {name} endpoint not configured — skipping")
            continue
        print(f"[BATCH] Checking {name} at {url} ...")
        for attempt in range(1, 13):
            try:
                r = requests.get(f"{url}/health", timeout=5)
                if r.status_code < 500:
                    print(f"  {name} ready (attempt {attempt})")
                    break
            except requests.exceptions.RequestException:
                pass
            print(f"  {name} not ready, retrying in 5s ({attempt}/12)")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CAO generator")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N queries")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to queries JSON file (overrides default)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for CAOs (overrides default)")
    args = parser.parse_args()

    global TEST_DATA_FILE, OUTPUT_DIR
    if args.data:
        TEST_DATA_FILE = Path(args.data)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    queries = _load_queries(TEST_DATA_FILE)
    if args.limit:
        queries = queries[: args.limit]

    total = len(queries)
    resolved = 0
    skipped = 0
    errors = 0

    print(f"\n[BATCH] Loading {total} queries from {TEST_DATA_FILE.name}")
    print(f"[BATCH] Output dir: {OUTPUT_DIR}")
    print(f"[BATCH] Vagueness: BYPASSED (all queries treated as resolved)\n")

    # Wait for services
    _wait_for_amr()
    _wait_for_external_services()
    print()

    # Quick TEXTOIR smoke test
    import requests
    _tx_url = os.environ.get("TEXTOIR_ENDPOINT_URL", "")
    if _tx_url:
        try:
            _r = requests.post(f"{_tx_url}/predict", json={"text": "set an alarm"}, timeout=5)
            print(f"[BATCH] TEXTOIR smoke test: {_r.json()}")
        except Exception as e:
            print(f"[BATCH] TEXTOIR smoke test failed: {e}")
    print()

    # Patch vagueness BEFORE importing pipeline (which imports handle_vagueness_turn)
    _patch_vagueness()

    # Import pipeline after patching, then also patch its local reference
    import src.lingo.pipeline as _lp
    _lp.handle_vagueness_turn = _make_resolved_vagueness
    run_main_pipeline = _lp.run_main_pipeline
    from src.Vagueness_Judge.runtime.pipeline import default_clarification_state

    # Capture CAOs by intercepting _save_cao_to_storage
    _captured_caos: list[dict] = []
    _original_save = _lp._save_cao_to_storage
    def _dual_save(cao, query, config):
        _captured_caos.append(dict(cao))
        _original_save(cao, query, config)
    _lp._save_cao_to_storage = _dual_save

    for idx, record in enumerate(queries):
        query: str = record.get("query", "").strip()
        record_id: str = record.get("id", f"q_{idx:04d}")
        is_vague: bool = record.get("vague", False)
        is_oos: bool = record.get("oos", False)
        expected_k: int = record.get("k", 1)
        topic: str = record.get("topic", "unknown")

        print(
            f"[{idx + 1:04d}/{total:04d}] "
            f"{record_id} | topic={topic} "
            f"vague={is_vague} oos={is_oos} k={expected_k}"
        )
        print(f"  Query: {query[:120]}...")

        config: Dict[str, Any] = {
            "session_id": f"batch_{datetime.now().strftime('%Y%m%d')}",
            "textoir_dataset": record.get("domain_cluster", "oos"),
            "textoir_known_cls_ratio": 0.75,
            "textoir_threshold": 0.5,
            "textoir_seed": 0,
        }
        cs = default_clarification_state()

        try:
            result = run_main_pipeline(
                user_text=query,
                history=[{"role": "user", "content": query}],
                config=config,
                clarification_state=cs,
            )

            pipeline_status = result.get("meta", {}).get("pipeline_phase", "")

            if result.get("clarification_state", {}).get("active") or \
               pipeline_status == "vagueness_clarification":
                print(f"  [SKIP] Needs human intervention (status={pipeline_status})")
                skipped += 1
                continue

            cao = _captured_caos[-1] if _captured_caos else result

            _save_cao(cao, query, record_id, pipeline_status, idx)
            resolved += 1

            oos_status = cao.get("intent", {}).get("oos_ind_status", "N/A")
            confidence = cao.get("intent", {}).get("confidence", "N/A")
            intents = cao.get("intent", {}).get("selected_intents", [])
            print(f"  OOS={oos_status} conf={confidence} intents={intents}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            errors += 1

        print()

    print("=" * 60)
    print(f"[BATCH] Complete: {total} total, {resolved} resolved, "
          f"{skipped} skipped, {errors} errors")
    print(f"[BATCH] Results in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
