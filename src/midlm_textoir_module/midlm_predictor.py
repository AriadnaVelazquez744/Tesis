from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from peft import PeftModel
from unsloth import FastLanguageModel

from src.MIDLM.midlm.decode import decode_topk_by_predicted_k
from src.MIDLM.midlm.model import MIDLMForMultiIntent


@dataclass(frozen=True)
class MIDLMRuntime:
    model: MIDLMForMultiIntent
    tokenizer: Any
    intent_vocab: List[str]
    max_seq_length: int
    device: torch.device


def _ensure_tokenizer_setup(tokenizer: Any) -> None:
    if getattr(tokenizer, "eos_token", None) is None:
        if getattr(tokenizer, "pad_token", None) is not None:
            tokenizer.eos_token = tokenizer.pad_token
        elif getattr(tokenizer, "bos_token", None) is not None:
            tokenizer.eos_token = tokenizer.bos_token
        else:
            raise ValueError("Tokenizer has no eos_token; cannot proceed.")

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"


@lru_cache(maxsize=4)
def get_midlm_runtime(
    *,
    checkpoint_dir: str,
    device_type: str,
    load_in_4bit: bool,
    bf16: bool,
) -> MIDLMRuntime:
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"MIDLM checkpoint_dir not found: {ckpt}")

    intent_vocab_path = ckpt / "intent_vocab.json"
    heads_path = ckpt / "midlm_heads.pt"
    cfg_path = ckpt / "train_config.json"
    for p in (intent_vocab_path, heads_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required MIDLM artifact: {p}")

    intent_vocab = json.loads(intent_vocab_path.read_text(encoding="utf-8"))
    if not isinstance(intent_vocab, list) or not intent_vocab:
        raise ValueError(f"Invalid intent_vocab.json at {intent_vocab_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_model_path = cfg.get("model_path")
    if not base_model_path:
        raise ValueError(f"Missing 'model_path' in {cfg_path}")

    max_seq_length = int(cfg.get("max_seq_length", 512))
    device = torch.device(device_type)
    dtype = torch.bfloat16 if bf16 else torch.float16

    backbone, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_model_path),
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=bool(load_in_4bit),
    )
    _ensure_tokenizer_setup(tokenizer)

    backbone.config.use_cache = False
    backbone = PeftModel.from_pretrained(backbone, str(ckpt))

    model = MIDLMForMultiIntent(
        backbone,
        num_intents=len(intent_vocab),
        max_k=int(cfg.get("max_k", 3)),
        alpha=float(cfg.get("alpha", 1.0)),
        beta=float(cfg.get("beta", 1.0)),
    )

    heads = torch.load(heads_path, map_location="cpu")
    if "intent_head" not in heads or "num_head" not in heads:
        raise ValueError(f"Invalid midlm_heads.pt format at {heads_path}")
    model.intent_head.load_state_dict(heads["intent_head"])
    model.num_head.load_state_dict(heads["num_head"])
    model.to(device)
    model.eval()

    return MIDLMRuntime(
        model=model,
        tokenizer=tokenizer,
        intent_vocab=[str(x) for x in intent_vocab],
        max_seq_length=max_seq_length,
        device=device,
    )


def predict_topk_intents(
    *,
    text: str,
    checkpoint_dir: str,
    device_type: Optional[str] = None,
    load_in_4bit: bool = False,
    bf16: bool = False,
) -> Tuple[List[str], int, Optional[List[float]]]:
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    rt = get_midlm_runtime(
        checkpoint_dir=checkpoint_dir,
        device_type=str(device_type),
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )

    enc = rt.tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=int(rt.max_seq_length),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(rt.device)
    attention_mask = enc["attention_mask"].to(rt.device)

    with torch.inference_mode():
        out = rt.model(input_ids=input_ids, attention_mask=attention_mask)

    intent_logits = out["intent_logits"].detach().cpu()
    num_logits = out["num_logits"].detach().cpu()

    decoded = decode_topk_by_predicted_k(
        intent_logits=intent_logits,
        num_logits=num_logits,
    )[0]

    k = int(decoded.k)
    intent_ids = [int(i) for i in decoded.intent_ids]
    intent_labels = [rt.intent_vocab[i] for i in intent_ids if 0 <= i < len(rt.intent_vocab)]

    raw_intent_logits_list = intent_logits.squeeze(0).tolist()
    return intent_labels, k, raw_intent_logits_list

