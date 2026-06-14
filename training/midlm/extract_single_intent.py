#!/usr/bin/env python3
"""
Extract single-intent examples from WeaveClinc150_rewritten.json.

Each multi-intent entry has source_texts[] and source_intents[].
This pairs each source_text with its source_intent to create
single-intent (k=1) training examples.

Output: WeaveClinc150_single_intent.json (same format as original)
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "data" / "WeaveClinc150_rewritten.json"
OUTPUT = SCRIPT_DIR / "data" / "WeaveClinc150_single_intent.json"


def extract(split_name: str, rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        texts = r.get("source_texts", [])
        intents = r.get("source_intents", [])
        if not texts or not intents or len(texts) != len(intents):
            continue
        for text, intent in zip(texts, intents):
            key = (intent, text.lower().strip())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "text": text,
                "labels": [intent],
                "source_intents": [intent],
                "source_texts": [text],
                "metadata": {
                    "split": split_name,
                    "blend_size": 1,
                    "was_rewritten": False,
                },
            })
    return out


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    total = 0
    for split in ("train", "validation", "test"):
        rows = data.get(split, [])
        single = extract(split, rows)
        result[split] = single
        total += len(single)
        print(f"  {split}: {len(single)} single-intent examples (from {len(rows)} multi-intent)")

    print(f"  TOTAL: {total}")

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
