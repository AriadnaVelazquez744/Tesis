from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class MIDLMMetricResult:
    exact_match_accuracy: float
    k_accuracy: float
    micro_f1: float
    macro_f1: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_metrics(
    *,
    pred_sets: Sequence[Set[int]],
    gold_sets: Sequence[Set[int]],
    pred_k: Sequence[int],
    gold_k: Sequence[int],
    num_intents: int,
) -> MIDLMMetricResult:
    if not (len(pred_sets) == len(gold_sets) == len(pred_k) == len(gold_k)):
        raise ValueError("Mismatched lengths in metric inputs")

    n = len(pred_sets)
    exact = sum(1 for p, g in zip(pred_sets, gold_sets) if p == g)
    k_acc = sum(1 for pk, gk in zip(pred_k, gold_k) if int(pk) == int(gk))

    # Micro stats across all labels
    tp = 0
    fp = 0
    fn = 0
    for p, g in zip(pred_sets, gold_sets):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = _f1(micro_p, micro_r)

    # Macro F1: compute per-intent F1 then average
    per_f1: List[float] = []
    for lab in range(num_intents):
        tp_i = 0
        fp_i = 0
        fn_i = 0
        for p, g in zip(pred_sets, gold_sets):
            p_has = lab in p
            g_has = lab in g
            if p_has and g_has:
                tp_i += 1
            elif p_has and not g_has:
                fp_i += 1
            elif (not p_has) and g_has:
                fn_i += 1
        p_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        r_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        per_f1.append(_f1(p_i, r_i))
    macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0

    return MIDLMMetricResult(
        exact_match_accuracy=exact / n if n else 0.0,
        k_accuracy=k_acc / n if n else 0.0,
        micro_f1=float(micro_f1),
        macro_f1=float(macro_f1),
    )

