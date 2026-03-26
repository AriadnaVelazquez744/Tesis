from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from src.Vagueness_Judge.runtime import (
    default_clarification_state,
    handle_vagueness_turn,
)

from src.midlm_textoir_module.analyze import analyze_midlm_textoir
from .engine import process_query


class Message(TypedDict):
    role: str
    content: str


def run_main_pipeline(
    user_text: str,
    history: List[Message] | None = None,
    config: Dict[str, Any] | None = None,
    clarification_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Main orchestration pipeline.

    Flow:
    1. Route user text through vagueness module.
    2. If clarification is required, return assistant question and keep state.
    3. If resolved, call engine with completed query and return final response.
    """
    if history is None:
        history = []
    if config is None:
        config = {}
    if clarification_state is None:
        clarification_state = default_clarification_state()

    vagueness_result = handle_vagueness_turn(user_text, clarification_state)
    updated_state = vagueness_result["updated_state"]

    if vagueness_result["status"] == "needs_user_input":
        return {
            "content": vagueness_result["assistant_message"],
            "meta": {
                "pipeline_phase": "vagueness_clarification",
                "tone": vagueness_result.get("tone"),
                "engine": "vagueness_controller",
            },
            "clarification_state": updated_state,
        }

    completed_query = vagueness_result["completed_query"]
    summary = vagueness_result.get("summary", "")
    tone = vagueness_result.get("tone")

    # Analysis configuration is passed through `config` so checkpoints/backbones
    # can be swapped without changing pipeline code.
    midlm_checkpoint_dir = config.get("midlm_checkpoint_dir")
    textoir_msp_model_dir = config.get("textoir_msp_model_dir")

    textoir_dataset = config.get("textoir_dataset", "oos")
    textoir_known_cls_ratio = float(config.get("textoir_known_cls_ratio", 0.75))
    textoir_threshold = float(config.get("textoir_threshold", 0.5))
    textoir_seed = int(config.get("textoir_seed", 0))
    bert_model_name = str(config.get("textoir_bert_model_name", "bert-base-uncased"))

    device_type = config.get("analysis_device_type")
    midlm_load_in_4bit = bool(config.get("midlm_load_in_4bit", False))
    midlm_bf16 = bool(config.get("midlm_bf16", False))

    analysis_result = analyze_midlm_textoir(
        completed_query=completed_query,
        summary=summary,
        midlm_checkpoint_dir=midlm_checkpoint_dir,
        textoir_msp_model_dir=textoir_msp_model_dir,
        textoir_dataset=str(textoir_dataset),
        textoir_known_cls_ratio=textoir_known_cls_ratio,
        textoir_threshold=textoir_threshold,
        textoir_seed=textoir_seed,
        bert_model_name=bert_model_name,
        device_type=device_type,
        midlm_load_in_4bit=midlm_load_in_4bit,
        midlm_bf16=midlm_bf16,
    )

    merged_meta: Dict[str, Any] = {}
    cao_meta = analysis_result["cao"].get("meta", {})
    if isinstance(cao_meta, dict):
        merged_meta.update(cao_meta)

    # Keep the engine stub meta fields (when available) for easier debugging.
    try:
        engine_response = process_query(
            query=completed_query,
            history=history,
            config=config,
        )
        merged_meta.update(dict(engine_response.get("meta", {})))
    except Exception:
        pass

    merged_meta.update(
        {
            "pipeline_phase": "resolved_and_analyzed_structured",
            "completed_query": completed_query,
            "summary": summary,
            "tone": tone,
            "analysis_engine": "MIDLM+TEXTOIR",
        }
    )

    return {
        "content": analysis_result["content"],
        "meta": merged_meta,
        "clarification_state": updated_state,
    }
