from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from src.Vagueness_Judge.runtime import (
    default_clarification_state,
    handle_vagueness_turn,
)

from src.midlm_textoir_module.analyze import analyze_midlm_textoir
from src.buffer_structure.cao_formatters import format_cao_as_markdown
from src.AMR.amr_processor import process_amr_into_cao
from .engine import process_query
from src.Semantic_Grounding.KeyBERT_processor import extract_and_ground
from src.Logic_Transformer.transformer import transform_logic


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    s = text.lower().strip()
    s = re.sub(r'[^a-z0-9\s_-]', '', s)
    s = re.sub(r'[\s_-]+', '_', s)
    return s[:max_len].strip('_')


def _save_cao_to_storage(cao: dict, completed_query: str, config: dict) -> None:
    try:
        storage_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../storage/cao")
        )
        os.makedirs(storage_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = config.get("session_id", "default")
        query_slug = _sanitize_filename(completed_query) or "resolved_query"

        filename = f"cao_{timestamp}_sess_{session_id}_{query_slug}.json"
        filepath = os.path.join(storage_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cao, f, indent=2, ensure_ascii=False)
        print(f"[STORAGE] CAO saved to: {filepath}")
    except Exception as e:
        print(f"[STORAGE] Error saving CAO: {e}")


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
    3. If resolved, run full analysis pipeline → Engine Lingo → natural response.
    """
    if history is None:
        history = []
    if config is None:
        config = {}
    if clarification_state is None:
        clarification_state = default_clarification_state()  # type: ignore

    vagueness_result = handle_vagueness_turn(user_text, clarification_state)  # type: ignore
    updated_state = vagueness_result["updated_state"]

    if vagueness_result["status"] == "needs_user_input":
        return {
            "content": vagueness_result["assistant_message"],
            "meta": {
                "pipeline_phase": "vagueness_clarification",
                "vagueness_raw": vagueness_result.get("raw_response", ""),
                "engine": "vagueness_controller",
            },
            "clarification_state": updated_state,
        }

    completed_query = vagueness_result["completed_query"]
    summary = vagueness_result.get("summary", "")

    # Extract the original user query from history
    original_query = ""
    for msg in history:
        if msg.get("role") == "user":
            original_query = msg["content"]
            break

    textoir_msp_model_dir = config.get("textoir_msp_model_dir")

    textoir_dataset = config.get("textoir_dataset", "oos")
    textoir_known_cls_ratio = float(config.get("textoir_known_cls_ratio", 0.75))
    textoir_threshold = float(config.get("textoir_threshold", 0.5))
    textoir_seed = int(config.get("textoir_seed", 0))
    bert_model_name = str(config.get("textoir_bert_model_name", "bert-base-uncased"))

    device_type = config.get("analysis_device_type")

    analysis_result = analyze_midlm_textoir(
        original_query=original_query,
        completed_query=completed_query,
        summary=summary,
        textoir_msp_model_dir=textoir_msp_model_dir,
        textoir_dataset=str(textoir_dataset),
        textoir_known_cls_ratio=textoir_known_cls_ratio,
        textoir_threshold=textoir_threshold,
        textoir_seed=textoir_seed,
        bert_model_name=bert_model_name,
        device_type=device_type,
    )

    # Integrate AMR processing into CAO
    buffer_input = completed_query.strip()

    try:
        analysis_result["cao"] = process_amr_into_cao(  # type: ignore
            analysis_result["cao"],  # type: ignore
            buffer_input,  # type: ignore
        )
        amr_ok = analysis_result["cao"].get("meta", {}).get("amr_ok", False)
        amr_error = analysis_result["cao"].get("meta", {}).get("amr_error", "none")
        print(f"[PIPELINE] AMR processing result: amr_ok={amr_ok}, error={amr_error}")
        if not amr_ok:
            print(
                f"[PIPELINE] AMR error detail: {analysis_result['cao'].get('meta', {}).get('amr_error_detail', 'unknown')}"
            )
    except Exception as e:
        print(f"[PIPELINE] AMR integration failed: {e}")
        analysis_result["cao"].setdefault("meta", {})["amr_ok"] = False
        analysis_result["cao"]["meta"]["amr_error"] = str(e)

    # Integrate KeyBERT grounding step
    try:
        analysis_result["cao"] = extract_and_ground( # type: ignore
            analysis_result["cao"],  # type: ignore
            buffer_input
        )
    except Exception as e:
        print(f"[PIPELINE] KeyBERT integration failed: {e}")

    # Logic Transformation (triple refinement & Tarski typing)
    try:
        analysis_result["cao"] = transform_logic(analysis_result["cao"], threshold=0.5)
    except Exception as e:
        print(f"[PIPELINE] Logic Transformer execution failed: {e}")
        analysis_result["cao"].setdefault("meta", {})["logic_transformer_ok"] = False

    # ── Engine Lingo: consume CAO and generate natural-language response ──
    try:
        engine_response = process_query(
            query=completed_query,
            history=history,
            config=config,
            cao=analysis_result["cao"],  # now passing the full CAO
        )
    except Exception as e:
        print(f"[PIPELINE] Engine Lingo failed, falling back to stub: {e}")
        engine_response = {
            "content": format_cao_as_markdown(analysis_result["cao"]),
            "meta": {"engine": "fallback_stub"},
        }

    # ── assemble final metadata ─────────────────────────────────────────
    merged_meta: Dict[str, Any] = {}

    # Pipeline-stage meta
    merged_meta.update(
        {
            "pipeline_phase": "resolved_and_analyzed_structured",
            "completed_query": completed_query,
            "summary": summary,
            "summary_thought": vagueness_result.get("summary_thought", ""),
            "vagueness_raw": vagueness_result.get("raw_response", ""),
            "analysis_engine": "MIDLM+TEXTOIR",
        }
    )

    # CAO meta (AMR, KeyBERT, Logic Transformer info)
    cao_meta = analysis_result["cao"].get("meta", {})
    if isinstance(cao_meta, dict):
        merged_meta.update(cao_meta)

    # Engine meta
    engine_meta = engine_response.get("meta", {})
    if isinstance(engine_meta, dict):
        merged_meta.update(engine_meta)

    # Persist final CAO to disk
    _save_cao_to_storage(analysis_result["cao"], completed_query, config)

    return {
        "content": engine_response["content"],
        "meta": merged_meta,
        "clarification_state": updated_state,
    }
