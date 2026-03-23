from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_VAGUE_HINTS = (
    "something",
    "stuff",
    "things",
    "help me",
    "improve",
    "better",
    "fix this",
    "do this",
)


def _extract_tone(text: str) -> Optional[str]:
    low = text.lower()
    if "formal" in low:
        return "formal"
    if "casual" in low:
        return "casual"
    if "friendly" in low:
        return "friendly"
    if "academic" in low:
        return "academic"
    return None


def _query_looks_vague(query: str) -> bool:
    words = [w for w in query.strip().split() if w]
    if len(words) < 6:
        return True
    low = query.lower()
    return any(hint in low for hint in _VAGUE_HINTS)


def _build_completed_query(initial_query: str, clarifications: List[str]) -> str:
    cleaned = [c.strip() for c in clarifications if c and c.strip()]
    if not cleaned:
        return initial_query.strip()
    return (
        f"{initial_query.strip()}\n\n"
        f"Clarifications from user: {'; '.join(cleaned)}."
    )


def call_vagueness_model(prompt: str) -> str:
    """
    Placeholder model-call function.

    The function currently expects `prompt` to be a JSON string with:
    - mode: "initial" | "refine"
    - query: initial user query
    - clarifications: optional list of user clarification texts

    Returns a JSON string representing the model answer. This signature remains
    stable so the placeholder can be replaced by a real model/API call later.
    """
    try:
        payload: Dict[str, Any] = json.loads(prompt)
    except json.JSONDecodeError:
        payload = {"mode": "initial", "query": prompt, "clarifications": []}

    mode = str(payload.get("mode", "initial"))
    query = str(payload.get("query", "")).strip()
    clarifications = payload.get("clarifications") or []
    if not isinstance(clarifications, list):
        clarifications = [str(clarifications)]
    clarifications = [str(item) for item in clarifications]

    tone = _extract_tone(" ".join([query] + clarifications))

    if mode == "initial" and _query_looks_vague(query):
        return json.dumps(
            {
                "status": "needs_clarification",
                "question": (
                    "Your request seems broad. Could you clarify the expected "
                    "output format, scope, and any constraints?"
                ),
                "tone": tone,
            }
        )

    if mode == "refine" and len([c for c in clarifications if c.strip()]) < 1:
        return json.dumps(
            {
                "status": "needs_clarification",
                "question": (
                    "Please add one concrete requirement (for example desired "
                    "output format, audience, or level of detail)."
                ),
                "tone": tone,
            }
        )

    completed_query = _build_completed_query(query, clarifications)
    summary = (
        "The request is now specific enough to continue with the main pipeline."
    )
    return json.dumps(
        {
            "status": "resolved",
            "completed_query": completed_query,
            "summary": summary,
            "tone": tone,
        }
    )
