#!/usr/bin/env python3
"""
Train Qwen2.5-3B-Instruct with the BEST noise-robust configuration (v2).

This is a one-shot training script that wraps `train_midlm_bidirectional.py`
with the exact configuration that produced the best results:
  - Smart Pooling (Learnable Attention)
  - Hard Negative Mining (Non-Intent Pool, 19,698 statements)
  - Dropout (0.1) and Label Smoothing (0.1)
  - LoRA on all linear layers (r=16, alpha=32)
  - 4-bit NF4 quantization + bfloat16

Expected results on the Noisy Test Set:
  Exact Match : ~69.3%
  K-Accuracy  : ~96.9%
  Micro-F1    : ~84.3%

Usage:
    uv run python train_qwen3b_best.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = SCRIPT_DIR.parents[1]
BASE_MODEL = _PROJECT_ROOT / "training" / "base_models" / "Qwen2.5-3B-Instruct"
NOISY_DATASET = SCRIPT_DIR / "data" / "WeaveClinc150_mixed_noisy.json"
OUTPUT_DIR = SCRIPT_DIR / "adapters" / "trained_models_bidirectional_v2"

# Best-result hyperparameters (do not change unless you have a good reason)
MAX_K = 3
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
MAX_SEQ_LENGTH = 384


def build_command(extra_args: list[str] | None = None) -> list[str]:
    cmd = [
        sys.executable, "train_midlm_bidirectional.py",
        "--model_path", str(BASE_MODEL),
        "--data_json", str(NOISY_DATASET),
        "--output_dir", str(OUTPUT_DIR),
        "--max_k", str(MAX_K),
        "--epochs", str(EPOCHS),
        "--batch_size_per_device", str(BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRAD_ACCUM),
        "--lr", str(LR),
        "--lora_r", str(LORA_R),
        "--lora_alpha", str(LORA_ALPHA),
        "--target_modules", "all-linear",
        "--use_attention_pool",          # Smart Pooling
        "--bf16",
        "--load_in_4bit",
        "--max_seq_length", str(MAX_SEQ_LENGTH),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the command without executing it",
    )
    p.add_argument(
        "--background", action="store_true",
        help="Run training in background (nohup) and return immediately",
    )
    p.add_argument(
        "--log-file", type=str, default="training_qwen3b_best.log",
        help="Log file path (used with --background)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_command()
    print("Training command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    if args.dry_run:
        return 0

    if args.background:
        log_path = SCRIPT_DIR / args.log_file
        print(f"Launching in background. Log: {log_path}")
        with log_path.open("w") as logf:
            subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
        return 0

    return subprocess.call(cmd, cwd=SCRIPT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
