#!/usr/bin/env python3
"""
Evaluate the Qwen2.5-3B-Instruct v2 model on both clean and noisy test sets.

This script wraps `eval_midlm_bidirectional.py` to reproduce the v2 results:
  Clean Test : 67.2% Exact Match
  Noisy Test : 69.3% Exact Match (the headline result)

Usage:
    uv run python eval_qwen3b_best.py
    uv run python eval_qwen3b_best.py --noisy-only
    uv run python eval_qwen3b_best.py --clean-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = _PROJECT_ROOT / "training" / "midlm" / "adapters" / "trained_models_bidirectional_v2" / "Qwen2.5-3B-Instruct_midlm_bidirectional"
MIXED_DATASET = _PROJECT_ROOT / "training" / "midlm" / "data" / "WeaveClinc150_mixed.json"
NOISY_DATASET = _PROJECT_ROOT / "training" / "midlm" / "data" / "WeaveClinc150_mixed_noisy.json"
CLEAN_EXP_DIR = _PROJECT_ROOT / "experiments" / "midlm" / "runs" / "qwen3b_v2_clean"
NOISY_EXP_DIR = _PROJECT_ROOT / "experiments" / "midlm" / "runs" / "qwen3b_v2_noisy"

MAX_K = 3


def run_eval(data_json: Path, exp_dir: Path, label: str) -> int:
    cmd = [
        sys.executable, "eval_midlm_bidirectional.py",
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--data_json", str(data_json),
        "--split", "test",
        "--max_k", str(MAX_K),
        "--experiments_dir", str(exp_dir),
    ]
    print(f"\n>>> Evaluating on {label} test set ({data_json.name})")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    return subprocess.call(cmd, cwd=_PROJECT_ROOT)


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY — Qwen2.5-3B v2 (Smart Pooling + HNM)")
    print("=" * 60)
    for label, root in [("CLEAN", CLEAN_EXP_DIR), ("NOISY", NOISY_EXP_DIR)]:
        metrics_files = sorted(root.glob("*/metrics.json"))
        if not metrics_files:
            print(f"  {label:5s}: no results found in {root}")
            continue
        with metrics_files[-1].open() as f:
            m = json.load(f)
        print(f"\n  {label} test ({m.get('num_examples', '?'):>4} examples):")
        print(f"    Exact Match : {m['exact_match_accuracy']*100:6.2f}%")
        print(f"    K-Accuracy  : {m['k_accuracy']*100:6.2f}%")
        print(f"    Micro-F1    : {m['micro_f1']*100:6.2f}%")
        print(f"    Macro-F1    : {m['macro_f1']*100:6.2f}%")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-only", action="store_true")
    p.add_argument("--noisy-only", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not CHECKPOINT_DIR.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_DIR}")
        print("Run training first: uv run python train_qwen3b_best.py")
        return 1

    do_clean = not args.noisy_only
    do_noisy = not args.clean_only
    rc = 0

    if do_clean:
        rc |= run_eval(MIXED_DATASET, CLEAN_EXP_DIR, "CLEAN")
    if do_noisy:
        rc |= run_eval(NOISY_DATASET, NOISY_EXP_DIR, "NOISY")
    if not args.no_summary:
        print_summary()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
