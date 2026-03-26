#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import DataLoader

from unsloth import FastLanguageModel

from midlm.data import MIDLMDataCollator, WeaveMultiIntentDataset, load_weave_json
from midlm.decode import decode_topk_by_predicted_k
from midlm.metrics import compute_metrics
from midlm.model import MIDLMForMultiIntent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_midlm")


def _ensure_tokenizer_setup(tokenizer) -> None:
    if tokenizer.eos_token is None:
        if tokenizer.pad_token is not None:
            tokenizer.eos_token = tokenizer.pad_token
        elif tokenizer.bos_token is not None:
            tokenizer.eos_token = tokenizer.bos_token
        elif tokenizer.unk_token is not None:
            tokenizer.eos_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no eos_token; cannot proceed.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MIDLM checkpoint on WeaveClinc150 splits")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory produced by train_midlm_unsloth.py (contains intent_vocab.json + adapter files).",
    )
    p.add_argument("--data_json", type=str, default="src/MIDLM/data/WeaveClinc150_rewritten.json")
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    p.add_argument("--max_k", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--experiments_dir",
        type=str,
        default="src/MIDLM/experiments",
        help="Directory where eval artifacts (metrics/config/predictions) are stored.",
    )
    p.add_argument(
        "--save_predictions",
        action="store_true",
        help="If set, save per-example predicted/gold intent ids for later comparisons.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt}")

    intents = json.loads((ckpt / "intent_vocab.json").read_text(encoding="utf-8"))
    if not isinstance(intents, list) or not intents:
        raise ValueError("Invalid intent_vocab.json")
    intent_to_id = {s: i for i, s in enumerate(intents)}

    # Reload base model path if available
    cfg_path = ckpt / "train_config.json"
    base_model_path = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_model_path = cfg.get("model_path")
    if base_model_path is None:
        raise ValueError("train_config.json missing model_path; cannot reload base model.")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    logger.info("Loading base model: %s", base_model_path)
    backbone, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_model_path),
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=bool(args.load_in_4bit),
    )
    _ensure_tokenizer_setup(tokenizer)

    # Load LoRA adapter from checkpoint dir (Unsloth/HF-compatible)
    try:
        from peft import PeftModel

        backbone = PeftModel.from_pretrained(backbone, str(ckpt))
    except Exception as e:
        logger.warning("Could not load adapter via PeftModel; continuing without adapter. Error: %s", str(e))

    backbone.config.use_cache = False
    model = MIDLMForMultiIntent(backbone, num_intents=len(intents), max_k=int(args.max_k))

    # Load trained MIDLM task heads (required for correct eval).
    heads_path = ckpt / "midlm_heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(
            f"Missing {heads_path}. This checkpoint has no trained MIDLM heads; "
            "please retrain with the current train_midlm_unsloth.py."
        )
    heads = torch.load(heads_path, map_location="cpu")
    model.intent_head.load_state_dict(heads["intent_head"])
    model.num_head.load_state_dict(heads["num_head"])
    model.eval()

    data = load_weave_json(args.data_json)
    rows = data[args.split]
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    ds = WeaveMultiIntentDataset(rows, intent_to_id=intent_to_id, max_k=args.max_k)
    collator = MIDLMDataCollator(tokenizer, max_seq_length=args.max_seq_length)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)

    pred_sets: List[Set[int]] = []
    gold_sets: List[Set[int]] = []
    pred_k: List[int] = []
    gold_k: List[int] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_multi = batch["labels_multi_hot"]
            y_num = batch["labels_num"]

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            decoded = decode_topk_by_predicted_k(
                intent_logits=out["intent_logits"].cpu(),
                num_logits=out["num_logits"].cpu(),
            )

            for i in range(input_ids.shape[0]):
                pred_ids = set(decoded[i].intent_ids)
                gold_ids = set(torch.nonzero(y_multi[i]).squeeze(-1).tolist())
                pred_sets.append(pred_ids)
                gold_sets.append(gold_ids)
                pred_k.append(int(decoded[i].k))
                gold_k.append(int(y_num[i].item()) + 1)

    metrics = compute_metrics(
        pred_sets=pred_sets,
        gold_sets=gold_sets,
        pred_k=pred_k,
        gold_k=gold_k,
        num_intents=len(intents),
    )
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{ckpt.name}__{args.split}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = experiments_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics_payload = {
        "checkpoint_dir": str(ckpt),
        "split": args.split,
        "num_examples": len(pred_sets),
        "exact_match_accuracy": metrics.exact_match_accuracy,
        "k_accuracy": metrics.k_accuracy,
        "micro_f1": metrics.micro_f1,
        "macro_f1": metrics.macro_f1,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (run_dir / "eval_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.save_predictions:
        rows_payload = []
        for idx in range(len(pred_sets)):
            rows_payload.append(
                {
                    "idx": idx,
                    "pred_intent_ids": sorted(list(pred_sets[idx])),
                    "gold_intent_ids": sorted(list(gold_sets[idx])),
                    "pred_k": int(pred_k[idx]),
                    "gold_k": int(gold_k[idx]),
                }
            )
        (run_dir / "predictions.json").write_text(json.dumps(rows_payload, indent=2), encoding="utf-8")

    logger.info("Split: %s  N=%d", args.split, len(pred_sets))
    logger.info("ExactMatchAcc: %.4f", metrics.exact_match_accuracy)
    logger.info("K-Acc:         %.4f", metrics.k_accuracy)
    logger.info("Micro-F1:      %.4f", metrics.micro_f1)
    logger.info("Macro-F1:      %.4f", metrics.macro_f1)
    logger.info("Saved eval artifacts to: %s", str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

