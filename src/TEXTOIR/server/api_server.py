from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, BertTokenizer

_SERVER_DIR = Path(__file__).resolve().parent
_TEXTOIR_OID_DIR = _SERVER_DIR.parent / "open_intent_detection"
if str(_TEXTOIR_OID_DIR) not in sys.path:
    sys.path.insert(0, str(_TEXTOIR_OID_DIR))

from backbones.bert import BERT  # type: ignore  # noqa: E402
from dataloaders import max_seq_lengths  # type: ignore  # noqa: E402


app = FastAPI(title="TEXTOIR Intent Classifier")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    status: str
    confidence: float
    method_used: str


class HealthResponse(BaseModel):
    status: str
    model: str
    dataset: str
    known_cls_ratio: float
    threshold: float


def _load_config_from_env() -> Dict[str, Any]:
    return {
        "model_dir": os.environ.get("TEXTOIR_MODEL_DIR", ""),
        "dataset": os.environ.get("TEXTOIR_DATASET", "oos"),
        "known_cls_ratio": float(os.environ.get("TEXTOIR_KNOWN_CLS_RATIO", "0.75")),
        "seed": int(os.environ.get("TEXTOIR_SEED", "0")),
        "threshold": float(os.environ.get("TEXTOIR_THRESHOLD", "0.5")),
        "bert_model_name": os.environ.get("TEXTOIR_BERT_MODEL", "bert-base-uncased"),
        "method": os.environ.get("TEXTOIR_METHOD", "msp"),
        "device_type": os.environ.get("TEXTOIR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    }


_model_runtime = None


def _load_model() -> None:
    global _model_runtime

    cfg = _load_config_from_env()
    model_dir = cfg["model_dir"]
    device = torch.device(cfg["device_type"])

    num_labels_known = int(round(150 * cfg["known_cls_ratio"]))
    if cfg["dataset"] not in max_seq_lengths:
        raise KeyError(f"Unknown dataset: {cfg['dataset']}")
    max_seq_length = int(max_seq_lengths[cfg["dataset"]])

    bert_cfg = AutoConfig.from_pretrained(cfg["bert_model_name"])
    args = SimpleNamespace(
        num_labels=num_labels_known,
        activation="tanh",
        device=device,
    )
    model = BERT(bert_cfg, args)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(cfg["bert_model_name"], do_lower_case=True)

    weights_path = Path(model_dir) / "pytorch_model.bin"
    has_weights = weights_path.exists()
    if has_weights:
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    _model_runtime = {
        "model": model,
        "tokenizer": tokenizer,
        "num_labels_known": num_labels_known,
        "max_seq_length": max_seq_length,
        "threshold": cfg["threshold"],
        "device": device,
        "has_weights": has_weights,
        "method": cfg["method"],
        "model_dir": cfg["model_dir"],
        "dataset": cfg["dataset"],
        "known_cls_ratio": cfg["known_cls_ratio"],
    }


@app.on_event("startup")
async def startup() -> None:
    _load_model()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _model_runtime is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=str(_model_runtime["model_dir"]),
        dataset=_model_runtime["dataset"],
        known_cls_ratio=_model_runtime["known_cls_ratio"],
        threshold=_model_runtime["threshold"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _model_runtime is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rt = _model_runtime
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    tokens = rt["tokenizer"].tokenize(text)
    max_tokens = rt["max_seq_length"] - 2
    tokens = tokens[: max(0, max_tokens)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = rt["tokenizer"].convert_tokens_to_ids(tokens)

    pad_len = rt["max_seq_length"] - len(input_ids)
    if pad_len < 0:
        input_ids = input_ids[: rt["max_seq_length"]]
        pad_len = 0

    input_ids = input_ids + [0] * pad_len
    input_mask = [1] * len(tokens) + [0] * pad_len
    segment_ids = [0] * rt["max_seq_length"]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=rt["device"])
    input_mask_t = torch.tensor([input_mask], dtype=torch.long, device=rt["device"])
    segment_ids_t = torch.tensor([segment_ids], dtype=torch.long, device=rt["device"])

    with torch.inference_mode():
        _, logits = rt["model"](input_ids_t, segment_ids_t, input_mask_t)
        probs = F.softmax(logits, dim=1)
        max_prob = float(probs.max(dim=1).values.item())

    status = "OOS" if max_prob < rt["threshold"] else "IND"

    return PredictResponse(
        status=status,
        confidence=max_prob,
        method_used=rt["method"],
    )


def main() -> None:
    port = int(os.environ.get("TEXTOIR_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
