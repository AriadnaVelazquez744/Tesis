"""Merge multiple JSONL interaction files (shuffle, dedupe optional)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_jsonl_files(
    inputs: List[Path],
    output: Path,
    *,
    shuffle_seed: int = 42,
) -> int:
    """
    Concatenate all JSONL files, shuffle rows with `shuffle_seed`, write `output`.
    Returns number of rows written.
    """
    merged: List[dict] = []
    for p in inputs:
        merged.extend(read_jsonl(p))
    rng = random.Random(shuffle_seed)
    rng.shuffle(merged)
    write_jsonl(output, merged)
    return len(merged)
