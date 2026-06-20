#!/usr/bin/env python3
"""
Batch CAO generator — runs queries through the full pipeline and saves CAOs
to experiments/cao/data/CAO_results/.

Usage:
    uv run python experiments/cao/batch_cao.py [--limit M] [--with-jdv]
    uv run python experiments/cao/batch_cao.py --jdv-dir experiments/cao/data/jdv_results

By default vagueness (JDV) is bypassed. Use --with-jdv for live JDV, or
--jdv-dir to inject precomputed JDV JSONs from jdv_runner.py (Colab).
Skips any query that requires human intervention (vagueness clarification).
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

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "cao" / "data" / "CAO_results"
SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = (
    SCRIPT_DIR / "data" / "evaluation_sample_v1.json"
)
TEST_DATA_FILE = _DEFAULT_DATA
_JDV_BY_RECORD_ID: Dict[str, Dict[str, Any]] = {}


def _load_precomputed_jdv(jdv_dir: Path) -> Dict[str, Dict[str, Any]]:
    from experiments.cao.jdv_runner import load_jdv_results

    by_id, _ = load_jdv_results(jdv_dir)
    print(f"[BATCH] Loaded {len(by_id)} precomputed JDV records from {jdv_dir}")
    return by_id


def _make_precomputed_vagueness(
    user_text: str, state: Any
) -> Dict[str, Any]:
    """Return JDV resolution from precomputed jdv_results/ files."""
    from src.Vagueness_Judge.runtime.pipeline import _reset_state

    record_id = ""
    if isinstance(state, dict):
        record_id = str(state.get("_batch_record_id", "")).strip()

    jdv = _JDV_BY_RECORD_ID.get(record_id, {})
    if not jdv:
        for entry in _JDV_BY_RECORD_ID.values():
            if entry.get("query", "").strip() == user_text.strip():
                jdv = entry
                break

    if jdv.get("status") == "needs_clarification":
        question = jdv.get("question") or "Could you provide more details?"
        return {
            "status": "needs_user_input",
            "assistant_message": question,
            "raw_response": jdv.get("raw_response", ""),
            "updated_state": state,
        }

    return {
        "status": "resolved",
        "completed_query": jdv.get("completed_query", user_text.strip()),
        "summary": jdv.get("summary", ""),
        "summary_thought": jdv.get("summary_thought", ""),
        "raw_response": jdv.get("raw_response", ""),
        "updated_state": _reset_state(state) if state else None,
    }


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
    jdv: Dict[str, Any] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.lingo.pipeline import _sanitize_filename
    slug = _sanitize_filename(query, max_len=60)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cao_{timestamp}_{index:04d}_{record_id}_{slug}.json"
    filepath = OUTPUT_DIR / filename
    meta: Dict[str, Any] = {
        "_batch_info": {
            "query": query,
            "record_id": record_id,
            "index": index,
            "timestamp": timestamp,
            "pipeline_status": status,
        }
    }
    if jdv:
        meta["jdv"] = jdv
        cao_meta = cao.setdefault("meta", {})
        if isinstance(cao_meta, dict):
            for key in ("completed_query", "summary", "summary_thought"):
                if jdv.get(key):
                    cao_meta[key] = jdv[key]
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
    parser.add_argument("--with-jdv", action="store_true",
                        help="Run real Vagueness Judge instead of bypassing it")
    parser.add_argument("--jdv-dir", type=str, default=None,
                        help="Use precomputed JDV JSONs from this directory")
    args = parser.parse_args()

    if args.with_jdv and args.jdv_dir:
        parser.error("Use either --with-jdv or --jdv-dir, not both")

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

    with_jdv = args.with_jdv
    jdv_dir = Path(args.jdv_dir) if args.jdv_dir else None
    use_precomputed_jdv = jdv_dir is not None

    if use_precomputed_jdv:
        global _JDV_BY_RECORD_ID
        _JDV_BY_RECORD_ID = _load_precomputed_jdv(jdv_dir)

    if use_precomputed_jdv:
        vagueness_mode = f"PRECOMPUTED ({jdv_dir})"
    elif with_jdv:
        vagueness_mode = "JDV ENABLED"
    else:
        vagueness_mode = "BYPASSED"

    print(f"\n[BATCH] Loading {total} queries from {TEST_DATA_FILE.name}")
    print(f"[BATCH] Output dir: {OUTPUT_DIR}")
    print(f"[BATCH] Vagueness: {vagueness_mode}\n")

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

    if use_precomputed_jdv:
        import src.Vagueness_Judge.runtime.pipeline as vp
        import src.Vagueness_Judge.runtime as vr
        vp.handle_vagueness_turn = _make_precomputed_vagueness
        vr.handle_vagueness_turn = _make_precomputed_vagueness
    elif not with_jdv:
        # Patch vagueness BEFORE importing pipeline (which imports handle_vagueness_turn)
        _patch_vagueness()

    # Import pipeline (patch local reference when not using live JDV)
    import src.lingo.pipeline as _lp
    if use_precomputed_jdv:
        _lp.handle_vagueness_turn = _make_precomputed_vagueness
    elif not with_jdv:
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
        if use_precomputed_jdv:
            cs["_batch_record_id"] = record_id  # type: ignore[typeddict-unknown-key]

        if use_precomputed_jdv and record_id not in _JDV_BY_RECORD_ID:
            print(f"  [SKIP] No precomputed JDV for {record_id}")
            skipped += 1
            print()
            continue

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

            result_meta = result.get("meta", {})
            if use_precomputed_jdv:
                pre = _JDV_BY_RECORD_ID.get(record_id, {})
                jdv_info = {
                    "status": pre.get("status", "resolved"),
                    "completed_query": pre.get("completed_query", query),
                    "summary": pre.get("summary", ""),
                    "summary_thought": pre.get("summary_thought", ""),
                    "raw_response": pre.get("raw_response", ""),
                    "source_file": pre.get("source_file", ""),
                }
            else:
                jdv_info = {
                    "status": "resolved",
                    "completed_query": result_meta.get(
                        "completed_query",
                        cao.get("meta", {}).get("completed_query", query),
                    ),
                    "summary": result_meta.get(
                        "summary",
                        cao.get("meta", {}).get("summary", ""),
                    ),
                    "summary_thought": result_meta.get("summary_thought", ""),
                }
            _save_cao(cao, query, record_id, pipeline_status, idx, jdv=jdv_info)
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
