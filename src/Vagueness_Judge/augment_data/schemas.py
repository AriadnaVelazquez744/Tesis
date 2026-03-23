"""Validate interaction records compatible with `training/sft.py` loaders."""

from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_TOP = ("task", "vague", "thought", "missing_details", "actions")
REQUIRED_ACTION_ASSISTANT = ("role", "content", "thought", "type")
REQUIRED_ACTION_USER = ("role", "content", "type")
REQUIRED_DETAIL = ("description", "importance", "inquiry", "options")


def _importance_key(detail: Dict[str, Any]) -> int:
    raw = detail.get("importance", "0")
    try:
        return int(str(raw).strip())
    except ValueError:
        return 0


def validate_interaction_record(row: Dict[str, Any], line_hint: str = "") -> Tuple[bool, List[str]]:
    """
    Return (ok, errors). Compatible with JSONL lines used by `load_raw_dataset`.
    Extra keys (e.g. strand, category, source) are allowed.
    """
    errors: List[str] = []
    prefix = f"{line_hint}: " if line_hint else ""

    for k in REQUIRED_TOP:
        if k not in row:
            errors.append(f"{prefix}missing key '{k}'")

    if errors:
        return False, errors

    if not isinstance(row["task"], str) or not row["task"].strip():
        errors.append(f"{prefix}task must be non-empty string")

    vague = row["vague"]
    if not isinstance(vague, bool):
        errors.append(f"{prefix}vague must be bool")

    if not isinstance(row["thought"], str):
        errors.append(f"{prefix}thought must be string")

    md = row["missing_details"]
    if not isinstance(md, list):
        errors.append(f"{prefix}missing_details must be list")
    else:
        for i, detail in enumerate(md):
            if not isinstance(detail, dict):
                errors.append(f"{prefix}missing_details[{i}] must be object")
                continue
            for dk in REQUIRED_DETAIL:
                if dk not in detail:
                    errors.append(f"{prefix}missing_details[{i}] missing '{dk}'")
            opts = detail.get("options")
            if opts is not None and (not isinstance(opts, list) or len(opts) == 0):
                errors.append(f"{prefix}missing_details[{i}].options must be non-empty list")

    if isinstance(vague, bool):
        if vague and isinstance(md, list) and len(md) == 0:
            errors.append(f"{prefix}vague=true requires non-empty missing_details")
        if not vague and isinstance(md, list) and len(md) > 0:
            errors.append(f"{prefix}vague=false requires empty missing_details")

    actions = row["actions"]
    if not isinstance(actions, list) or len(actions) == 0:
        errors.append(f"{prefix}actions must be non-empty list")
        return False, errors

    for i, act in enumerate(actions):
        if not isinstance(act, dict):
            errors.append(f"{prefix}actions[{i}] must be object")
            continue
        role = act.get("role")
        if role == "assistant":
            for ak in REQUIRED_ACTION_ASSISTANT:
                if ak not in act:
                    errors.append(f"{prefix}actions[{i}] missing '{ak}'")
            t = act.get("type")
            if t not in ("New", "summary"):
                errors.append(f"{prefix}actions[{i}] assistant type must be New or summary")
        elif role == "user":
            for ak in REQUIRED_ACTION_USER:
                if ak not in act:
                    errors.append(f"{prefix}actions[{i}] missing '{ak}'")
            if act.get("type") != "response":
                errors.append(f"{prefix}actions[{i}] user type must be response")
        else:
            errors.append(f"{prefix}actions[{i}] invalid role {role!r}")

    # Alternation assistant / user
    for i, act in enumerate(actions):
        if not isinstance(act, dict):
            continue
        want = "assistant" if i % 2 == 0 else "user"
        if act.get("role") != want:
            errors.append(f"{prefix}actions[{i}] expected role {want}, got {act.get('role')!r}")

    last = actions[-1]
    if isinstance(last, dict):
        if last.get("role") != "assistant" or last.get("type") != "summary":
            errors.append(f"{prefix}last action must be assistant summary")

    return len(errors) == 0, errors


def normalize_missing_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with importance as string (matches existing IN3 style)."""
    out = dict(detail)
    imp = out.get("importance", "2")
    out["importance"] = str(imp).strip()
    return out


def sort_missing_details(missing_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by importance descending (3 before 1), stable within same rank."""
    return sorted(
        (normalize_missing_detail(d) for d in missing_details),
        key=_importance_key,
        reverse=True,
)


def strip_metadata_for_training(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keys expected by training (extra keys are harmless; optional helper)."""
    base = {
        "category": row.get("category", "Augmented"),
        "task": row["task"],
        "vague": row["vague"],
        "thought": row["thought"],
        "missing_details": row["missing_details"],
        "actions": row["actions"],
    }
    return base
