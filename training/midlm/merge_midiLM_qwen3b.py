#!/usr/bin/env python3
"""One-time merge of LoRA adapter into base Qwen2.5-3B for MIDLM server.

Produces a standalone model directory that the server can load directly
without PeftModel, shaving ~2s off startup.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = SCRIPT_DIR.parents[1]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("merge_midiLM")


def main() -> int:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base Qwen2.5-3B")
    p.add_argument(
        "--base_model",
        type=str,
        default=str(_PROJECT_ROOT / "training" / "base_models" / "Qwen2.5-3B-Instruct"),
    )
    p.add_argument(
        "--adapter",
        type=str,
        default=str(SCRIPT_DIR / "adapters" / "trained_models_bidirectional" / "Qwen2.5-3B-Instruct_midlm_bidirectional"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "adapters" / "trained_models_merged" / "Qwen2.5-3B-Instruct_midlm_merged"),
    )
    args = p.parse_args()

    base_path = Path(args.base_model)
    adapter_path = Path(args.adapter)
    out_path = Path(args.output)

    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # Load base model in bf16 (no quantization for merge)
    logger.info("Loading base model: %s", base_path)
    backbone = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )

    # Load LoRA adapter
    logger.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(backbone, str(adapter_path))

    # Merge weights into base and unload adapter
    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model to: %s", out_path)
    model.save_pretrained(str(out_path))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=False)
    tokenizer.save_pretrained(str(out_path))

    # Copy MIDLM artifacts alongside
    for fname in ("intent_vocab.json", "midlm_heads.pt", "train_config.json"):
        src = adapter_path / fname
        if src.exists():
            shutil.copy2(src, out_path / fname)
            logger.info("Copied %s", fname)
        else:
            logger.warning("Missing %s in adapter dir", fname)

    # Report size
    total_bytes = sum(f.stat().st_size for f in out_path.iterdir() if f.is_file())
    logger.info("Done. Output: %s (%.1f GB)", out_path, total_bytes / 1e9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
