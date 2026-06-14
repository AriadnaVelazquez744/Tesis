#!/usr/bin/env python3
"""MIDLM Bidirectional Inference Server.

Runs an OpenAI-compatible HTTP server that classifies multi-intent text
using the MIDLM architecture (Huang et al., 2025) with bidirectional
attention.  Designed to run alongside LM Studio on the same GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/midlm_textoir_module → src → root
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "MIDLM"))

from midlm.model import MIDLMForMultiIntent
from midlm.decode import decode_topk_by_predicted_k

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Logging — stdout + file
# ---------------------------------------------------------------------------
LOG_DIR = _PROJECT_ROOT / "experiments" / "midlm" / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "midlm_server.log"),
    ],
)
logger = logging.getLogger("midlm_server")

# ---------------------------------------------------------------------------
# Global model state (populated at startup)
# ---------------------------------------------------------------------------
class ModelState:
    model: MIDLMForMultiIntent
    tokenizer: AutoTokenizer
    id_to_intent: dict[int, str]
    num_intents: int
    max_k: int
    device: torch.device
    max_seq_length: int


state = ModelState()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_tokenizer(tokenizer) -> None:
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


def load_midiLM(checkpoint_dir: str, load_in_4bit: bool, max_seq_length: int) -> None:
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # --- intent vocabulary ---
    intents: list[str] = json.loads((ckpt / "intent_vocab.json").read_text(encoding="utf-8"))
    id_to_intent: dict[int, str] = {i: s for i, s in enumerate(intents)}

    # --- training config → base model path ---
    cfg = json.loads((ckpt / "train_config.json").read_text(encoding="utf-8"))
    base_model_path = cfg.get("model_path")
    if base_model_path is None:
        raise ValueError("train_config.json missing model_path")
    max_k = int(cfg.get("max_k", 3))
    use_attention_pool = bool(cfg.get("use_attention_pool", False))

    # --- load backbone ---
    dtype = torch.bfloat16
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    logger.info("Loading backbone: %s (4bit=%s)", base_model_path, load_in_4bit)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=False,
    )

    # Break weight tying (safetensors compatibility)
    if getattr(backbone.config, "tie_word_embeddings", False):
        backbone.lm_head.weight = torch.nn.Parameter(
            backbone.lm_head.weight.detach().clone()
        )
        backbone.config.tie_word_embeddings = False

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)
    _ensure_tokenizer(tokenizer)

    # --- LoRA adapter ---
    adapter_path = ckpt / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info("Loading LoRA adapter: %s", ckpt)
        backbone = PeftModel.from_pretrained(backbone, str(ckpt))

    backbone.config.use_cache = False

    # --- MIDLM wrapper + heads ---
    model = MIDLMForMultiIntent(
        backbone,
        num_intents=len(intents),
        max_k=max_k,
        bidirectional=True,
        use_attention_pool=use_attention_pool,
    )

    heads_path = ckpt / "midlm_heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"Missing {heads_path}")
    heads = torch.load(heads_path, map_location="cpu", weights_only=False)
    model.intent_head.load_state_dict(heads["intent_head"])
    model.num_head.load_state_dict(heads["num_head"])
    if use_attention_pool and "attention_query" in heads:
        with torch.no_grad():
            model.attention_query.copy_(heads["attention_query"].to(model.attention_query.device))
        logger.info("Loaded attention_query (shape=%s)", tuple(heads["attention_query"].shape))
    model.intent_head.to(dtype=dtype)
    model.num_head.to(dtype=dtype)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- store in global state ---
    state.model = model
    state.tokenizer = tokenizer
    state.id_to_intent = id_to_intent
    state.num_intents = len(intents)
    state.max_k = max_k
    state.device = device
    state.max_seq_length = max_seq_length

    logger.info(
        "Model ready: %d intents, max_k=%d, device=%s, attention_pool=%s",
        state.num_intents, state.max_k, state.device, use_attention_pool,
    )

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_intents(text: str) -> dict:
    enc = state.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=state.max_seq_length,
        padding=True,
    )
    input_ids = enc["input_ids"].to(state.device)
    attention_mask = enc["attention_mask"].to(state.device)

    with torch.no_grad():
        out = state.model(input_ids=input_ids, attention_mask=attention_mask)

    decoded = decode_topk_by_predicted_k(
        intent_logits=out["intent_logits"].cpu(),
        num_logits=out["num_logits"].cpu(),
    )
    result = decoded[0]
    intent_names = [state.id_to_intent[i] for i in result.intent_ids]
    return {
        "k": result.k,
        "intent_ids": result.intent_ids,
        "intents": intent_names,
    }

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "midlm-qwen3b-bidirectional"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0


class IntentResult(BaseModel):
    k: int
    intent_ids: List[int]
    intents: List[str]


class Choice(BaseModel):
    index: int
    message: ChatMessage
    intent_result: IntentResult
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MIDLM Bidirectional Server")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    user_msg = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if user_msg is None:
        raise ValueError("No user message found in messages list")

    text = user_msg.content

    t0 = time.monotonic()
    result = predict_intents(text)
    latency = time.monotonic() - t0

    logger.info(
        "text=%r k=%d intents=%s latency=%.3fs",
        text, result["k"], result["intents"], latency,
    )

    content = json.dumps({
        "k": result["k"],
        "intent_ids": result["intent_ids"],
        "intents": result["intents"],
    })

    prompt_tokens = len(state.tokenizer.encode(text, truncation=True, max_length=state.max_seq_length))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                intent_result=IntentResult(**result),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        ),
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "midlm-qwen3b-bidirectional",
                "object": "model",
                "created": 0,
                "owned_by": "midlm",
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Qwen2.5-3B-Instruct_midlm_bidirectional",
        "device": str(state.device),
        "num_intents": state.num_intents,
        "max_k": state.max_k,
        "bidirectional": True,
    }

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="MIDLM Bidirectional Inference Server")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            _PROJECT_ROOT
            / "training"
            / "midlm"
            / "adapters"
            / "trained_models_bidirectional_v2"
            / "Qwen2.5-3B-Instruct_midlm_bidirectional"
        ),
        help="MIDLM checkpoint directory (adapter + heads)",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=1235)
    p.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load backbone in 4-bit NF4 (default). Use --no-load_in_4bit for bf16.",
    )
    p.add_argument("--max_seq_length", type=int, default=384)
    args = p.parse_args()

    logger.info("Loading MIDLM from %s ...", args.checkpoint)
    t0 = time.monotonic()
    load_midiLM(args.checkpoint, args.load_in_4bit, args.max_seq_length)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
