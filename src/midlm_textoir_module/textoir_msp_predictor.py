from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, BertTokenizer


@dataclass(frozen=True)
class TextoirMSPRuntime:
    model: torch.nn.Module
    tokenizer: Any
    num_labels_known: int
    max_seq_length: int
    threshold: float
    has_weights: bool
    device: torch.device


def _textoir_open_intent_detection_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "src" / "TEXTOIR" / "open_intent_detection"


def _ensure_textoir_on_path() -> None:
    root = _textoir_open_intent_detection_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _build_known_label_count(
    *,
    dataset: str,
    known_cls_ratio: float,
    seed: int,
) -> int:
    """
    TEXTOIR's known-class selection size:
      n_known_cls = round(len(all_label_list) * known_cls_ratio)
    """

    _ensure_textoir_on_path()
    from dataloaders import benchmark_labels  # type: ignore

    all_labels = benchmark_labels[dataset]
    _ = np.random.RandomState(seed)
    return int(round(len(all_labels) * float(known_cls_ratio)))


@lru_cache(maxsize=2)
def get_textoir_msp_runtime(
    *,
    dataset: str,
    known_cls_ratio: float,
    seed: int,
    model_output_dir: str,
    bert_model_name: str,
    threshold: float,
    device_type: str,
) -> TextoirMSPRuntime:
    """
    MSP = Max Softmax Probability thresholding:
      IND if max softmax prob >= threshold else OOS.
    """

    _ensure_textoir_on_path()
    from backbones.bert import BERT  # type: ignore
    from dataloaders import max_seq_lengths  # type: ignore

    device = torch.device(device_type)

    num_labels_known = _build_known_label_count(
        dataset=dataset,
        known_cls_ratio=known_cls_ratio,
        seed=seed,
    )
    if dataset not in max_seq_lengths:
        raise KeyError(f"Unknown TEXTOIR dataset: {dataset}")
    max_seq_length = int(max_seq_lengths[dataset])

    cfg = AutoConfig.from_pretrained(bert_model_name)
    args = SimpleNamespace(
        num_labels=num_labels_known,
        activation="tanh",  # matches TEXTOIR MSP default config
        device=device,
    )
    model = BERT(cfg, args)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    weights_path = Path(model_output_dir) / "pytorch_model.bin"
    has_weights = weights_path.exists()
    if has_weights:
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return TextoirMSPRuntime(
        model=model,
        tokenizer=tokenizer,
        num_labels_known=num_labels_known,
        max_seq_length=max_seq_length,
        threshold=float(threshold),
        has_weights=bool(has_weights),
        device=device,
    )


def predict_msp_ind_oos(
    *,
    text: str,
    model_output_dir: str,
    dataset: str = "oos",
    known_cls_ratio: float = 0.75,
    seed: int = 0,
    threshold: float = 0.5,
    bert_model_name: str = "bert-base-uncased",
    device_type: Optional[str] = None,
) -> Tuple[str, Optional[float], str]:
    """
    Returns:
      - status: "IND" | "OOS" | "UNKNOWN"
      - confidence: max softmax prob (if available)
      - method_used: "msp"
    """

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    rt = get_textoir_msp_runtime(
        dataset=dataset,
        known_cls_ratio=known_cls_ratio,
        seed=seed,
        model_output_dir=model_output_dir,
        bert_model_name=bert_model_name,
        threshold=threshold,
        device_type=str(device_type),
    )

    if not rt.has_weights:
        return "UNKNOWN", None, "missing_model_weights"

    tokens = rt.tokenizer.tokenize(text)
    max_tokens = rt.max_seq_length - 2
    tokens = tokens[: max(0, max_tokens)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = rt.tokenizer.convert_tokens_to_ids(tokens)

    pad_len = rt.max_seq_length - len(input_ids)
    if pad_len < 0:
        input_ids = input_ids[: rt.max_seq_length]
        pad_len = 0

    input_ids = input_ids + [0] * pad_len
    input_mask = [1] * len(tokens) + [0] * pad_len
    segment_ids = [0] * rt.max_seq_length

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=rt.device)
    input_mask_t = torch.tensor([input_mask], dtype=torch.long, device=rt.device)
    segment_ids_t = torch.tensor([segment_ids], dtype=torch.long, device=rt.device)

    with torch.inference_mode():
        _, logits = rt.model(input_ids_t, segment_ids_t, input_mask_t)
        probs = F.softmax(logits, dim=1)
        max_prob = float(probs.max(dim=1).values.item())

    status = "OOS" if max_prob < rt.threshold else "IND"
    return status, max_prob, "msp"

