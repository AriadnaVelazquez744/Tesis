from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def load_weave_json(path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("train", "validation", "test"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"Expected top-level list '{key}' in {p}")
    return data


def build_intent_vocab(
    rows: Sequence[Dict[str, Any]],
    *,
    min_freq: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    for r in rows:
        labels = r.get("labels", [])
        if not isinstance(labels, list):
            raise TypeError("Row 'labels' must be a list")
        for lab in labels:
            lab_s = str(lab)
            counts[lab_s] = counts.get(lab_s, 0) + 1
    intents = sorted([k for k, v in counts.items() if v >= min_freq])
    stoi = {s: i for i, s in enumerate(intents)}
    return intents, stoi


def labels_to_multi_hot(
    labels: Sequence[str],
    *,
    intent_to_id: Dict[str, int],
) -> torch.Tensor:
    y = torch.zeros(len(intent_to_id), dtype=torch.float32)
    for lab in labels:
        if lab in intent_to_id:
            y[intent_to_id[lab]] = 1.0
    return y


class WeaveMultiIntentDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        intent_to_id: Dict[str, int],
        max_k: int,
    ) -> None:
        self.rows = list(rows)
        self.intent_to_id = intent_to_id
        self.max_k = int(max_k)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        text = r.get("text", "")
        labels = r.get("labels", [])
        if not isinstance(labels, list):
            raise TypeError("Row 'labels' must be a list")
        labels_s = [str(x) for x in labels]
        y = labels_to_multi_hot(labels_s, intent_to_id=self.intent_to_id)
        k = len({lab for lab in labels_s if lab in self.intent_to_id})
        k = max(1, min(self.max_k, int(k)))
        return {
            "text": str(text),
            "labels_multi_hot": y,
            "labels_num": torch.tensor(k - 1, dtype=torch.long),  # 0..max_k-1
        }


@dataclass
class MIDLMTrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels_multi_hot: torch.Tensor
    labels_num: torch.Tensor


class MIDLMDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)

        if self.tokenizer.pad_token is None:
            # Match repo convention: use eos as pad for decoder-only models.
            if self.tokenizer.eos_token is None:
                raise ValueError("Tokenizer has no pad_token or eos_token; cannot collate")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For decoder-only models, left padding is generally safer for packed batches.
        self.tokenizer.padding_side = "left"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        labels_multi_hot = torch.stack([f["labels_multi_hot"] for f in features], dim=0)
        labels_num = torch.stack([f["labels_num"] for f in features], dim=0)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels_multi_hot": labels_multi_hot,
            "labels_num": labels_num,
        }

