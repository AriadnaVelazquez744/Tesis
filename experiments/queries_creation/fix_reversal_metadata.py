#!/usr/bin/env python3
"""
Fix reversal pair metadata in expanded_queries.jsonl.

Deterministic post-processor — no LLM calls.
Corrects 3 known issues:
  1. reversal_pair_id and equivalent_to typos/mismatches in reversal B records
  2. Missing is_reversed_version on OOS reversal B records
  3. Missing equivalent_to on OOS reversal B records

Usage:
    python experiments/queries_creation/fix_reversal_metadata.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANDED_PATH = HERE / "fixtures" / "expanded" / "expanded_queries.jsonl"


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    if not EXPANDED_PATH.exists():
        print(f"Not found: {EXPANDED_PATH}")
        return 1

    records = load_records(EXPANDED_PATH)
    original_count = len(records)

    # Group by seed
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sid = rec.get("meta", {}).get("seed_source", "__unknown__")
        by_seed[sid].append(rec)

    # Track fixes for reporting
    fix_equiv = 0
    fix_pair_id_b = 0
    fix_pair_id_a = 0
    fix_is_rev = 0
    changes: list[str] = []

    for sid, recs in sorted(by_seed.items()):
        # Find reversal A (variant_index == 8) and B (variant_index == 9)
        rev_a = None
        rev_b = None
        for r in recs:
            vi = r.get("meta", {}).get("variant_index")
            if vi == 8:
                rev_a = r
            elif vi == 9:
                rev_b = r

        if not rev_a or not rev_b:
            continue

        # Fix reversal A: ensure reversal_pair_id equals its own id
        a_id = rev_a["id"]
        a_meta = rev_a.setdefault("meta", {})
        if a_meta.get("reversal_pair_id") != a_id:
            old = a_meta.get("reversal_pair_id")
            a_meta["reversal_pair_id"] = a_id
            fix_pair_id_a += 1
            changes.append(f"{sid} v8: reversal_pair_id '{old}' → '{a_id}'")

        # Fix reversal A: ensure is_reversed_version is False
        if a_meta.get("is_reversed_version") is not False:
            a_meta["is_reversed_version"] = False
            fix_is_rev += 1
            changes.append(f"{sid} v8: is_reversed_version → false")

        # Fix reversal B: ensure is_reversed_version is True
        b_meta = rev_b.setdefault("meta", {})
        if b_meta.get("is_reversed_version") is not True:
            b_meta["is_reversed_version"] = True
            fix_is_rev += 1
            changes.append(f"{sid} v9: is_reversed_version → true")

        # Fix reversal B: ensure reversal_pair_id equals A's id
        if b_meta.get("reversal_pair_id") != a_id:
            old = b_meta.get("reversal_pair_id")
            b_meta["reversal_pair_id"] = a_id
            fix_pair_id_b += 1
            changes.append(f"{sid} v9: reversal_pair_id '{old}' → '{a_id}'")

        # Fix reversal B: ensure equivalent_to equals A's id
        if b_meta.get("equivalent_to") != a_id:
            old = b_meta.get("equivalent_to")
            b_meta["equivalent_to"] = a_id
            fix_equiv += 1
            changes.append(f"{sid} v9: equivalent_to '{old}' → '{a_id}'")

    # Save
    save_records(records, EXPANDED_PATH)

    # Report
    print(f"Records processed: {original_count}")
    print(f"  reversal A pair_id fixes:   {fix_pair_id_a}")
    print(f"  reversal B pair_id fixes:   {fix_pair_id_b}")
    print(f"  reversal B equivalent_to fixes: {fix_equiv}")
    print(f"  is_reversed_version fixes:  {fix_is_rev}")
    print(f"  Total changes:              {fix_pair_id_a + fix_pair_id_b + fix_equiv + fix_is_rev}")

    if changes:
        print("\nDetail:")
        for c in changes:
            print(f"  {c}")

    final_count = sum(1 for _ in open(EXPANDED_PATH, encoding="utf-8") if _.strip())
    print(f"\nFinal record count: {final_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
