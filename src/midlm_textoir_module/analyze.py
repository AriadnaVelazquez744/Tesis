from __future__ import annotations

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
    completed_query: str,
    summary: str,
    midlm_checkpoint_dir: Optional[str],
    textoir_msp_model_dir: Optional[str],
    # TEXTOIR/MSP knobs
    textoir_dataset: str = "oos",
    textoir_known_cls_ratio: float = 0.75,
    textoir_threshold: float = 0.5,
    textoir_seed: int = 0,
    bert_model_name: str = "bert-base-uncased",
    device_type: Optional[str] = None,
    # MIDLM knobs
    midlm_load_in_4bit: bool = False,
    midlm_bf16: bool = False,
) -> AnalyzeResult:
    """
    Builds a CAO (Cognitive Analysis Object) for the interface.
    """

    buffer_input = f"{completed_query.strip()}\n\n[Summary]\n{summary.strip()}"

    # 1) MIDLM multi-intent selection
    selected_intents: List[str] = []
    k: int = 0
    midlm_ok = False

    if _safe_exists(midlm_checkpoint_dir):
        try:
            from .midlm_predictor import predict_topk_intents

            selected_intents, k, _intent_logits = predict_topk_intents(
                text=buffer_input,
                checkpoint_dir=str(midlm_checkpoint_dir),
                device_type=device_type,
                load_in_4bit=midlm_load_in_4bit,
                bf16=midlm_bf16,
            )
            midlm_ok = True
        except Exception:
            midlm_ok = False

    # 2) TEXTOIR IND/OOS decision (MSP)
    oos_ind_status: IntentOOSStatus = "UNKNOWN"
    oos_confidence: Optional[float] = None
    textoir_method_used: Optional[str] = None

    if _safe_exists(textoir_msp_model_dir):
        try:
            from .textoir_msp_predictor import predict_msp_ind_oos

            status, conf, method_used = predict_msp_ind_oos(
                text=buffer_input,
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
        except Exception:
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
            "completed_query": completed_query,
            "summary": summary,
        },
    }

    # Clean None values for nicer display (optional)
    if "confidence" in cao["intent"] and cao["intent"]["confidence"] is None:
        # TypedDict allows NotRequired, but we keep key stable by deleting it
        del cao["intent"]["confidence"]  # type: ignore[arg-type]

    return {
        "cao": cao,
        "content": format_cao_as_markdown(cao),
    }

