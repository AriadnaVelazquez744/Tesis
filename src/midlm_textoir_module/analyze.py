from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, List

from src.buffer_structure.cao_formatters import format_cao_as_markdown
from src.buffer_structure.cao_types import CognitiveAnalysisObject, IntentOOSStatus


class AnalyzeResult(TypedDict):
    cao: CognitiveAnalysisObject
    content: str


def _safe_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def analyze_midlm_textoir(
    *,
    original_query: str = "",
    completed_query: str,
    summary: str,
    textoir_msp_model_dir: Optional[str],
    # TEXTOIR/MSP knobs
    textoir_dataset: str = "oos",
    textoir_known_cls_ratio: float = 0.75,
    textoir_threshold: float = 0.5,
    textoir_seed: int = 0,
    bert_model_name: str = "bert-base-uncased",
    device_type: Optional[str] = None,
) -> AnalyzeResult:
    """
    Builds a CAO (Cognitive Analysis Object) for the interface.

    MIDLM reads endpoint config from env (MIDLM_ENDPOINT_URL, MIDLM_MODEL_ID).
    MIDLM uses original_query (raw user text) to avoid misleading the intent
    classifier with the verbose JDV summary.
    TEXTOIR loads via HTTP when TEXTOIR_ENDPOINT_URL is set, or falls back to
    loading locally from textoir_msp_model_dir on disk.
    """

    midlm_input = original_query.strip() if original_query else completed_query.strip()
    textoir_input = f"{completed_query.strip()}\n\n[Summary]\n{summary.strip()}"

    # 1) MIDLM multi-intent selection (HTTP via env-configurable endpoint)
    selected_intents: List[str] = []
    k: int = 0
    midlm_ok = False

    print(f"[ANALYZE] MIDLM input: {midlm_input[:200]} (original_query={bool(original_query)})")

    if os.environ.get("MIDLM_ENDPOINT_URL"):
        try:
            from .midlm_http_predictor import predict_topk_intents

            selected_intents, k, _raw = predict_topk_intents(
                text=midlm_input,
            )
            midlm_ok = True
        except Exception:
            midlm_ok = False

    # 2) TEXTOIR IND/OOS decision
    oos_ind_status: IntentOOSStatus = "UNKNOWN"
    oos_confidence: Optional[float] = None
    textoir_method_used: Optional[str] = None

    textoir_endpoint = os.environ.get("TEXTOIR_ENDPOINT_URL", "")

    if textoir_endpoint:
        try:
            from .textoir_http_predictor import predict_ind_oos

            status, conf, method_used = predict_ind_oos(
                text=textoir_input,
            )
            oos_ind_status = status  # type: ignore[assignment]
            oos_confidence = conf
            textoir_method_used = method_used
            print(f"[ANALYZE] TEXTOIR via HTTP: {oos_ind_status} (conf={oos_confidence})")
        except Exception as e:
            print(f"[ANALYZE] TEXTOIR HTTP failed, trying local fallback: {e}")
            oos_ind_status = "UNKNOWN"

    if oos_ind_status == "UNKNOWN" and _safe_exists(textoir_msp_model_dir):
        try:
            from .textoir_msp_predictor import predict_msp_ind_oos

            status, conf, method_used = predict_msp_ind_oos(
                text=textoir_input,
                model_output_dir=str(textoir_msp_model_dir),
                dataset=textoir_dataset,
                known_cls_ratio=textoir_known_cls_ratio,
                threshold=textoir_threshold,
                seed=textoir_seed,
                bert_model_name=bert_model_name,
                device_type=device_type,
            )
            oos_ind_status = status  # type: ignore[assignment]
            oos_confidence = conf
            textoir_method_used = method_used
            print(f"[ANALYZE] TEXTOIR local: {oos_ind_status} (conf={oos_confidence})")
        except Exception as e:
            print(f"[ANALYZE] TEXTOIR local also failed: {e}")
            oos_ind_status = "UNKNOWN"

    # 3) CAO assembly
    if oos_ind_status == "OOS":
        selected_intents = []
        k = 0

    cao: CognitiveAnalysisObject = {
        "intent": {
            "oos_ind_status": oos_ind_status,
            "k": int(k),
            "selected_intents": selected_intents,
            "confidence": oos_confidence if oos_confidence is not None else None,  # type: ignore[arg-type]
        },
        "nesting": {
            "nesting_graph": [],
        },
        "meta_reasoning": {
            "association_triggers": [],
        },
        "state": {
            "triples": [],
        },
        "meta": {
            "buffer_input_used": True,
            "midlm_ok": midlm_ok,
            "textoir_method_used": textoir_method_used,
            "original_query": original_query,
            "completed_query": completed_query,
            "summary": summary,
        },
    }

    # Clean None values for nicer display (optional)
    if "confidence" in cao["intent"] and cao["intent"]["confidence"] is None:
        del cao["intent"]["confidence"]  # type: ignore[arg-type]

    return {
        "cao": cao,
        "content": format_cao_as_markdown(cao),
    }

