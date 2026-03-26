from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class MIDLMDecoded:
    k: int
    intent_ids: List[int]


def decode_topk_by_predicted_k(
    *,
    intent_logits: torch.Tensor,  # (B, M)
    num_logits: torch.Tensor,  # (B, C) where C=max_k
) -> List[MIDLMDecoded]:
    if intent_logits.ndim != 2 or num_logits.ndim != 2:
        raise ValueError("Expected 2D tensors for intent_logits and num_logits")
    if intent_logits.shape[0] != num_logits.shape[0]:
        raise ValueError("Batch size mismatch between intent_logits and num_logits")

    # Predict K as argmax over classes (0..max_k-1) then map to 1..max_k.
    pred_k = torch.argmax(num_logits, dim=-1) + 1

    decoded: List[MIDLMDecoded] = []
    for b in range(intent_logits.shape[0]):
        k = int(pred_k[b].item())
        k = max(1, min(k, int(intent_logits.shape[1])))
        topk = torch.topk(intent_logits[b], k=k, dim=-1).indices.tolist()
        decoded.append(MIDLMDecoded(k=k, intent_ids=[int(x) for x in topk]))
    return decoded

