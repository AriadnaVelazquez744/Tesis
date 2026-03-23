from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from src.Vagueness_Judge.runtime import (
    default_clarification_state,
    handle_vagueness_turn,
)

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

    engine_response = process_query(
        query=completed_query,
        history=history,
        config=config,
    )
    merged_meta = dict(engine_response.get("meta", {}))
    merged_meta.update(
        {
            "pipeline_phase": "resolved_and_answered",
            "completed_query": completed_query,
            "summary": summary,
            "tone": tone,
        }
    )

    return {
        "content": engine_response.get("content", ""),
        "meta": merged_meta,
        "clarification_state": updated_state,
    }
