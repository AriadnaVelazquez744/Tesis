from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: (B, T, H)
    attention_mask: (B, T) with 1 for real tokens, 0 for pad
    returns: (B, H)
    """
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


class MIDLMForMultiIntent(nn.Module):
    """
    Implements Huang et al. (2025) MIDLM heads:
    - intent multi-label classification (BCE with logits)
    - intent-number classification over 1..max_k (CE)

    Note: the paper also changes the attention mask to global (non-causal) during post-training.
    Unsloth does not reliably support non-causal masks; this repo keeps the backbone causal by default.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        *,
        num_intents: int,
        max_k: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_intents = int(num_intents)
        self.max_k = int(max_k)
        self.alpha = float(alpha)
        self.beta = float(beta)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config")

        self.intent_head = nn.Linear(hidden_size, self.num_intents)
        self.num_head = nn.Linear(hidden_size, self.max_k)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_multi_hot: Optional[torch.Tensor] = None,
        labels_num: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]
        pooled = masked_mean_pool(last_hidden, attention_mask)

        intent_logits = self.intent_head(pooled)
        num_logits = self.num_head(pooled)

        loss = None
        if labels_multi_hot is not None and labels_num is not None:
            if labels_multi_hot.shape != intent_logits.shape:
                raise ValueError(
                    f"labels_multi_hot shape {tuple(labels_multi_hot.shape)} != intent_logits {tuple(intent_logits.shape)}"
                )
            # Multi-label intent loss (Eq. 8)
            loss_intent = F.binary_cross_entropy_with_logits(intent_logits, labels_multi_hot)
            # Intent number loss (Eq. 9) where labels_num is 0..max_k-1
            loss_num = F.cross_entropy(num_logits, labels_num)
            loss = self.alpha * loss_intent + self.beta * loss_num

        # Hugging Face Trainer expects a dict-like output with "loss".
        out_dict: Dict[str, torch.Tensor] = {
            "intent_logits": intent_logits,
            "num_logits": num_logits,
        }
        if loss is not None:
            out_dict["loss"] = loss
        return out_dict

