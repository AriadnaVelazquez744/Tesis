from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from .model_api import call_vagueness_model


class VaguenessDecision(TypedDict, total=False):
    status: str
    question: str
    completed_query: str
    summary: str
    tone: str | None


def _safe_parse_response(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "status": "needs_clarification",
            "question": (
                "Could you rephrase your request with more concrete details?"
            ),
            "tone": None,
        }
    if not isinstance(data, dict):
        return {
            "status": "needs_clarification",
            "question": (
                "Could you provide a more specific version of your request?"
            ),
            "tone": None,
        }
    return data


def _normalize_decision(data: Dict[str, Any]) -> VaguenessDecision:
    status = str(data.get("status", "needs_clarification"))
    if status == "resolved":
        return VaguenessDecision(
            status="resolved",
            completed_query=str(data.get("completed_query", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            tone=data.get("tone"),
        )
    return VaguenessDecision(
        status="needs_clarification",
        question=str(
            data.get(
                "question",
                "Could you clarify your objective with a concrete expected result?",
            )
        ).strip(),
        tone=data.get("tone"),
    )


def evaluate_initial_query(query: str) -> VaguenessDecision:
    prompt = json.dumps(
        {"mode": "initial", "query": query, "clarifications": []},
        ensure_ascii=True,
    )
    raw = call_vagueness_model(prompt)
    return _normalize_decision(_safe_parse_response(raw))


def evaluate_refined_query(
    initial_query: str,
    clarifications: List[str],
) -> VaguenessDecision:
    prompt = json.dumps(
        {
            "mode": "refine",
            "query": initial_query,
            "clarifications": clarifications,
        },
        ensure_ascii=True,
    )
    raw = call_vagueness_model(prompt)
    return _normalize_decision(_safe_parse_response(raw))
