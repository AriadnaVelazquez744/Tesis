#!/usr/bin/env python3
"""
Deduplicate expanded queries and identify gaps.

Scans expanded_queries.jsonl for:
  1. Duplicate record IDs (same id field within the same seed_source)
  2. Duplicate query text (exact match within the same seed_source)

Duplicates are removed (first occurrence is kept). Seeds with < 10 records
after dedup are reported so expand_queries.py can be re-run to fill gaps.

Usage:
    python tests/dedup_expanded.py              # check + report only
    python tests/dedup_expanded.py --fix        # remove duplicates in-place
    python tests/dedup_expanded.py --fix --fill # remove dups + re-run expansion

Environment:
    LLM_API_BASE, LLM_MODEL, LLM_API_KEY — passed through to expand_queries.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANDED_PATH = HERE / "fixtures" / "expanded" / "expanded_queries.jsonl"
TARGET_PER_SEED = 10


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        print(f"No records found at {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN line {i}: invalid JSON — {e}")
    return records


def save_records(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def group_by_seed(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        meta = rec.get("meta")
        sid = meta.get("seed_source", "__unknown__") if isinstance(meta, dict) else "__unknown__"
        groups[sid].append(rec)
    return dict(groups)


def find_duplicates(records: list[dict]) -> set[int]:
    """
    Return set of indices (into `records`) that are duplicates and should be removed.
    Duplicates are detected by (seed_source, id) and (seed_source, query) collisions.
    """
    seen_ids: dict[tuple[str, str], int] = {}
    seen_queries: dict[tuple[str, str], int] = {}
    remove_indices: set[int] = set()

    for i, rec in enumerate(records):
        sid = rec.get("meta", {}).get("seed_source", "__unknown__")
        rid = rec.get("id", "")
        query = rec.get("query", "")

        id_key = (sid, rid)
        if rid and id_key in seen_ids:
            print(f"  DUP id    [{i}] '{rid}' collides with [{seen_ids[id_key]}] — removing")
            remove_indices.add(i)
            continue
        if rid:
            seen_ids[id_key] = i

        query_key = (sid, query)
        if query and query_key in seen_queries:
            print(f"  DUP query [{i}] '{query[:60]}...' collides with [{seen_queries[query_key]}] — removing")
            remove_indices.add(i)
            continue
        if query:
            seen_queries[query_key] = i

    return remove_indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate expanded queries")
    parser.add_argument("--fix", action="store_true", help="Remove duplicates in-place")
    parser.add_argument("--fill", action="store_true", help="Also re-run expand_queries.py to fill gaps")
    args = parser.parse_args()

    records = load_records(EXPANDED_PATH)
    if not records:
        return 0

    print(f"Total records loaded: {len(records)}")
    print()

    dup_indices = find_duplicates(records)

    if not dup_indices:
        print("No duplicates found.")
    elif not args.fix:
        print(f"\n{len(dup_indices)} duplicate(s) found. Run with --fix to remove them.")
    else:
        kept = [r for i, r in enumerate(records) if i not in dup_indices]
        print(f"\nRemoved {len(records) - len(kept)} duplicate(s).")
        save_records(kept, EXPANDED_PATH)
        records = kept

    print()

    groups = group_by_seed(records)
    print("=== Per-seed record counts ===")
    gaps = []
    total_records = 0
    for sid in sorted(groups):
        count = len(groups[sid])
        total_records += count
        status = "OK" if count >= TARGET_PER_SEED else f"MISSING {TARGET_PER_SEED - count}"
        print(f"  {sid:30s} {count:3d}  {status}")
        if count < TARGET_PER_SEED:
            gaps.append(sid)
    print(f"  {'TOTAL':30s} {total_records:3d}")
    print()

    if not gaps:
        print(f"All seeds have {TARGET_PER_SEED} records. No gaps to fill.")
        return 0

    print(f"{len(gaps)} seed(s) with < {TARGET_PER_SEED} records:")
    for sid in gaps:
        print(f"  - {sid}")

    if args.fill:
        print("\nRe-running expand_queries.py to fill gaps ...")
        result = subprocess.run(
            [sys.executable, str(HERE / "expand_queries.py")],
            cwd=HERE,
        )
        print(f"exit code: {result.returncode}")
        return result.returncode

    print(f"\nRun `python tests/expand_queries.py` to regenerate missing records,"
          f" or `python tests/dedup_expanded.py --fix --fill` to do both.")
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
