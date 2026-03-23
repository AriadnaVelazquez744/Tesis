"""Load seeds, synthesize dialogues, stratified train/test split, write JSONL + manifest."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dialogue import build_full_record
from .schemas import validate_interaction_record


def _line_rng(base_seed: int, task: str) -> random.Random:
    h = hashlib.sha256(f"{base_seed}:{task}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def load_seed_jsonl_files(seeds_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not seeds_dir.is_dir():
        raise FileNotFoundError(f"Seeds directory not found: {seeds_dir}")
    for path in sorted(seeds_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return rows


def _needs_llm_annotation(row: Dict[str, Any]) -> bool:
    """True if row is task-only style (missing annotations)."""
    return "thought" not in row or "vague" not in row or "missing_details" not in row


def expand_seeds(
    rows: List[Dict[str, Any]],
    *,
    annotate: bool,
    base_seed: int,
) -> List[Dict[str, Any]]:
    from . import annotate as annotate_mod

    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if _needs_llm_annotation(r):
            if not annotate:
                raise ValueError(
                    "Seed row missing thought/vague/missing_details; run with --annotate "
                    "or provide full IN3-style fields."
                )
            r = annotate_mod.annotate_seed_row(r)
        out.append(r)
    return out


def build_records(
    rows: List[Dict[str, Any]],
    base_seed: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        task = str(row["task"])
        rng = _line_rng(base_seed, task)
        rec = build_full_record(row, rng=rng)
        records.append(rec)
    return records


def stratified_split(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
    group_key: str = "strand",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split each strand group independently so every strand contributes to test when possible."""
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key, "default"))
        buckets[key].append(row)

    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for key in sorted(buckets.keys()):
        items = buckets[key][:]
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        n_test = int(round(n * test_ratio))
        if 0 < test_ratio < 1 and n >= 2 and n_test == 0:
            n_test = 1
        if n_test > n:
            n_test = n
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_records(records: List[Dict[str, Any]]) -> None:
    for i, rec in enumerate(records):
        ok, errs = validate_interaction_record(rec, line_hint=f"line {i}")
        if not ok:
            raise ValueError("Validation failed: " + "; ".join(errs))


def run_pipeline(
    seeds_dir: Path,
    out_dir: Path,
    *,
    test_ratio: float = 0.2,
    base_seed: int = 42,
    annotate: bool = False,
    group_key: str = "strand",
) -> Dict[str, Any]:
    """
    Full pipeline: load seeds → optional LLM annotation → synthesize actions → split → write.

    Returns manifest dict.
    """
    raw = load_seed_jsonl_files(seeds_dir)
    if not raw:
        raise ValueError(f"No seed rows found under {seeds_dir}")

    expanded = expand_seeds(raw, annotate=annotate, base_seed=base_seed)
    records = build_records(expanded, base_seed=base_seed)
    validate_records(records)

    train_rows, test_rows = stratified_split(records, test_ratio=test_ratio, seed=base_seed, group_key=group_key)

    out_dir = out_dir.resolve()
    train_path = out_dir / "interaction_data_train.jsonl"
    test_path = out_dir / "interaction_data_test.jsonl"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    counts: Dict[str, Any] = {
        "total": len(records),
        "train": len(train_rows),
        "test": len(test_rows),
        "by_strand": {},
    }
    for row in records:
        s = str(row.get("strand", row.get("category", "unknown")))
        counts["by_strand"][s] = counts["by_strand"].get(s, 0) + 1

    manifest = {
        "seeds_dir": str(seeds_dir.resolve()),
        "out_dir": str(out_dir),
        "test_ratio": test_ratio,
        "base_seed": base_seed,
        "annotate": annotate,
        "counts": counts,
        "outputs": {
            "train": str(train_path),
            "test": str(test_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
