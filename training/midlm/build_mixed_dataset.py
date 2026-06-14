#!/usr/bin/env python3
"""
Build balanced mixed dataset from WeaveClinc150_rewritten.json (multi-intent)
and WeaveClinc150_single_intent.json (single-intent).

Strategy:
  - Subsample all groups (k=1, k=2, k=3) to match the smallest count (k=2)
  - Output: WeaveClinc150_mixed.json

Target distribution per split: ~33% k=1, ~33% k=2, ~33% k=3
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MULTI_PATH = SCRIPT_DIR / "data" / "WeaveClinc150_rewritten.json"
SINGLE_PATH = SCRIPT_DIR / "data" / "WeaveClinc150_single_intent.json"
OUTPUT = SCRIPT_DIR / "data" / "WeaveClinc150_mixed.json"

SEED = 42


def main():
    with MULTI_PATH.open("r", encoding="utf-8") as f:
        multi = json.load(f)
    with SINGLE_PATH.open("r", encoding="utf-8") as f:
        single = json.load(f)

    rng = random.Random(SEED)
    result = {}

    for split in ("train", "validation", "test"):
        multi_rows = multi.get(split, [])
        single_rows = single.get(split, [])

        k1 = len(single_rows)
        k2 = [r for r in multi_rows if r.get("metadata", {}).get("blend_size", 0) == 2]
        k3 = [r for r in multi_rows if r.get("metadata", {}).get("blend_size", 0) == 3]

        print(f"\n  {split}:")
        print(f"    k=1 (single): {k1}")
        print(f"    k=2 (multi):  {len(k2)}")
        print(f"    k=3 (multi):  {len(k3)}")

        # Subsample all groups to match k=2 count (the bottleneck)
        target = len(k2)
        k1_sample = rng.sample(single_rows, min(target, len(single_rows)))
        k3_sample = rng.sample(k3, min(target, len(k3)))

        merged = k1_sample + k2 + k3_sample
        rng.shuffle(merged)
        result[split] = merged

        k1_actual = sum(1 for r in merged if r.get("metadata", {}).get("blend_size", 0) == 1)
        k2_actual = sum(1 for r in merged if r.get("metadata", {}).get("blend_size", 0) == 2)
        k3_actual = sum(1 for r in merged if r.get("metadata", {}).get("blend_size", 0) == 3)

        print(f"    merged: {len(merged)} total")
        print(f"      k=1: {k1_actual} ({100*k1_actual/len(merged):.1f}%)")
        print(f"      k=2: {k2_actual} ({100*k2_actual/len(merged):.1f}%)")
        print(f"      k=3: {k3_actual} ({100*k3_actual/len(merged):.1f}%)")

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
