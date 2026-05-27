from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from .service import evaluate_initial_query, evaluate_refined_query


class ClarificationTurn(TypedDict):
    role: str
    content: str


class ClarificationState(TypedDict):
    active: bool
    initial_query: str
    turns: List[ClarificationTurn]
    pending_question: str
    resolved: bool


def default_clarification_state() -> ClarificationState:
    return ClarificationState(
        active=False,
        initial_query="",
        turns=[],
        pending_question="",
        resolved=False,
    )


def _reset_state(state: ClarificationState) -> ClarificationState:
    state["active"] = False
    state["initial_query"] = ""
    state["turns"] = []
    state["pending_question"] = ""
    state["resolved"] = False
    return state


def handle_vagueness_turn(
    user_text: str,
    state: ClarificationState | None,
) -> Dict[str, Any]:
    """
    Handle one user turn through the vagueness-first controller.

    Returns dict with:
    - status: "needs_user_input" | "resolved"
    - assistant_message: text to display in Streamlit when clarification is needed
    - completed_query / summary / raw_response when resolved
    - updated_state
    """
    state = state or default_clarification_state()

    if not state["active"]:
        initial_decision = evaluate_initial_query(user_text)
        if initial_decision.get("status") == "resolved":
            return {
                "status": "resolved",
                "completed_query": initial_decision.get("completed_query", user_text),
                "summary": initial_decision.get("summary", ""),
                "summary_thought": initial_decision.get("summary_thought", ""),
                "raw_response": initial_decision.get("raw_response", ""),
                "updated_state": _reset_state(state),
            }

        state["active"] = True
        state["resolved"] = False
        state["initial_query"] = user_text.strip()
        state["turns"] = [
            {"role": "user", "content": user_text.strip()},
        ]
        question = initial_decision.get(
            "question",
            "Could you provide more details?",
        )
        state["pending_question"] = question
        state["turns"].append({
            "role": "assistant",
            "content": initial_decision.get("raw_response", question),
        })
        return {
            "status": "needs_user_input",
            "assistant_message": question,
            "raw_response": initial_decision.get("raw_response", ""),
            "updated_state": state,
        }

    # Clarification loop: current user turn is interpreted as clarification.
    state["turns"].append({"role": "user", "content": user_text.strip()})
    clarification_texts = [
        t["content"]
        for t in state["turns"]
        if t["role"] == "user" and t["content"] != state["initial_query"]
    ]

    refined = evaluate_refined_query(state["initial_query"], clarification_texts, state["turns"])
    if refined.get("status") == "resolved":
        state["resolved"] = True
        return {
            "status": "resolved",
            "completed_query": refined.get("completed_query", state["initial_query"]),
            "summary": refined.get("summary", ""),
            "summary_thought": refined.get("summary_thought", ""),
            "raw_response": refined.get("raw_response", ""),
            "updated_state": _reset_state(state),
        }

    question = refined.get(
        "question",
        "Could you clarify the expected format and constraints?",
    )
    state["pending_question"] = question
    state["turns"].append({
        "role": "assistant",
        "content": refined.get("raw_response", question),
    })
    return {
        "status": "needs_user_input",
        "assistant_message": question,
        "raw_response": refined.get("raw_response", ""),
        "updated_state": state,
    }
