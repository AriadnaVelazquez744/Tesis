"""Synthesize multi-turn actions from IN3-style annotations (deterministic)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .schemas import sort_missing_details, validate_interaction_record

USER_TEMPLATES = [
    "I'll go with {choice}.",
    "Let's go with {choice} — that fits best.",
    "I'd prefer {choice}, thanks!",
    "Oh, {choice} works for me.",
    "{choice} sounds perfect.",
]


def _pick_option_index(rng: random.Random, n_options: int) -> int:
    if n_options <= 0:
        return 0
    return rng.randrange(0, n_options)


def _user_reply_for_choice(choice: str, rng: random.Random) -> str:
    tmpl = rng.choice(USER_TEMPLATES)
    return tmpl.format(choice=choice)


def _inquiry_content(detail: Dict[str, Any]) -> str:
    q = detail.get("inquiry")
    if isinstance(q, str) and q.strip():
        return q.strip()
    desc = detail.get("description", "this aspect")
    opts = detail.get("options") or []
    if opts:
        return f"Could you clarify {desc}? Options: {', '.join(opts)}."
    return f"Could you clarify {desc}?"


def _inquiry_thought(detail: Dict[str, Any], round_idx: int) -> str:
    desc = detail.get("description", "detail")
    return f"Need to clarify {desc} before proceeding (round {round_idx + 1})."


def build_actions_from_annotations(
    task: str,
    thought: str,
    vague: bool,
    missing_details: List[Dict[str, Any]],
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Build `actions` list matching existing interaction JSONL conventions.
    For vague tasks, one inquiry per missing detail (sorted by importance desc).
    """
    rng = rng or random.Random(0)
    actions: List[Dict[str, Any]] = []

    if not vague:
        summary_thought = (
            f"The initial task is clear enough to summarize without further questions. "
            f"Here are the user preferences and constraints:\n- Task as stated: {task[:200]}"
        )
        summary_content = (
            f"The user's goal is: {task.strip()} "
            "(no additional constraints were required)."
        )
        actions.append(
            {
                "role": "assistant",
                "content": summary_content,
                "thought": summary_thought,
                "type": "summary",
            }
        )
        return actions

    ordered = sort_missing_details(list(missing_details))
    if not ordered:
        raise ValueError("vague=true requires non-empty missing_details")

    resolved: List[Tuple[str, str]] = []

    for i, detail in enumerate(ordered):
        opts = detail.get("options") or []
        idx = _pick_option_index(rng, len(opts))
        chosen = opts[idx] if opts else "the first option"

        actions.append(
            {
                "role": "assistant",
                "content": _inquiry_content(detail),
                "thought": _inquiry_thought(detail, i),
                "type": "New",
            }
        )
        actions.append(
            {
                "role": "user",
                "content": _user_reply_for_choice(chosen, rng),
                "thought": None,
                "type": "response",
            }
        )
        resolved.append((detail.get("description", f"detail_{i}"), chosen))

    # Summary
    bullet_lines = "\n".join(f"- {desc}: {val}" for desc, val in resolved)
    summary_thought = (
        "The user has provided sufficient information over the course of the conversation. "
        "Here are the user preferences and constraints:\n" + bullet_lines
    )
    summary_bits = "; ".join(f"{d}: {v}" for d, v in resolved)
    summary_content = (
        f"Refined goal for the assistant: {task.strip()} "
        f"(with these clarifications: {summary_bits})."
    )

    actions.append(
        {
            "role": "assistant",
            "content": summary_content,
            "thought": summary_thought,
            "type": "summary",
        }
    )

    return actions


def build_full_record(
    base: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Given a seed dict with task, vague, thought, missing_details, category/strand optional,
    attach synthesized actions and validate.
    """
    rng = rng or random.Random(0)
    task = base["task"]
    thought = base["thought"]
    vague = bool(base["vague"])
    md = base.get("missing_details") or []

    actions = build_actions_from_annotations(task, thought, vague, md, rng=rng)

    row = {
        "category": base.get("category", "Augmented"),
        "task": task,
        "vague": vague,
        "thought": thought,
        "missing_details": md if vague else [],
        "actions": actions,
    }
    for key in ("strand", "source", "seed_id"):
        if key in base:
            row[key] = base[key]

    ok, errs = validate_interaction_record(row)
    if not ok:
        raise ValueError("Built invalid record: " + "; ".join(errs))

    return row
