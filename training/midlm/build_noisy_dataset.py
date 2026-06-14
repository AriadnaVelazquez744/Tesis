#!/usr/bin/env python3
"""
Build noise-robust dataset (v2) using a Non-Intent Noise Pool.

Changes from v1:
  - Uses the dedicated `noise_pool.json` (non-intent bearing statements).
  - Politeness markers ("can you", "please", "I want") are NOT filtered out.
  - Hard Negatives: Uses truly non-intent sentences that may *look* like requests
    but do not map to any of the 150 CLINC150 intents.

Output: WeaveClinc150_mixed_noisy.json
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MIXED_PATH = SCRIPT_DIR / "data" / "WeaveClinc150_mixed.json"
NOISE_POOL_PATH = SCRIPT_DIR / "data" / "noise_pool.json"
OUTPUT_PATH = SCRIPT_DIR / "data" / "WeaveClinc150_mixed_noisy.json"

SEED = 42


def load_noise_pool() -> list[str]:
    with NOISE_POOL_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["non_intent_statements"]


def inject_noise(text: str, noise_pool: list[str], rng: random.Random) -> str:
    import re

    num_noise = rng.randint(1, 3)
    noise_sentences = rng.sample(noise_pool, num_noise)

    parts = re.split(r'(, | and | but |\.)', text)
    parts = [p for p in parts if p]

    for ns in noise_sentences:
        pos = rng.randint(0, len(parts))
        parts.insert(pos, f" {ns} ")

    joined = "".join(parts).strip()
    joined = joined.replace("..", ".").replace(" .", ".").replace("  ", " ")
    return joined


def main():
    with MIXED_PATH.open("r", encoding="utf-8") as f:
        mixed = json.load(f)

    rng = random.Random(SEED)
    noise_pool = load_noise_pool()
    print(f"Noise pool size: {len(noise_pool)} unique non-intent statements")

    result = {}

    for split in ("train", "validation", "test"):
        rows = mixed.get(split, [])
        noisy_rows = []

        for r in rows:
            noisy_rows.append(r)

            should_noise = False
            if split == "train":
                if rng.random() < 0.5:
                    should_noise = True
            else:
                should_noise = True

            if should_noise:
                noisy_text = inject_noise(r["text"], noise_pool, rng)
                noisy_twin = {
                    "text": noisy_text,
                    "labels": r["labels"],
                    "source_intents": r.get("source_intents", r["labels"]),
                    "source_texts": r.get("source_texts", []),
                    "metadata": {
                        **r.get("metadata", {}),
                        "is_noisy": True,
                        "blend_size": r.get("metadata", {}).get("blend_size", 1),
                        "noise_source": "non_intent_pool",
                    },
                }
                noisy_rows.append(noisy_twin)

        rng.shuffle(noisy_rows)
        result[split] = noisy_rows
        noisy_count = sum(1 for r in noisy_rows if r.get("metadata", {}).get("is_noisy"))
        print(f"  {split}: {len(rows)} clean -> {len(noisy_rows)} mixed ({noisy_count} noisy)")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved noisy dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
