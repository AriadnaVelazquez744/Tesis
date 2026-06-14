"""Optional LLM annotation for task-only seed rows (OpenAI-compatible Chat Completions API)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

JSON_SCHEMA_HINT = """You must respond with a single JSON object only, with keys:
- "vague": boolean
- "thought": string explaining why the task is vague or clear
- "missing_details": array of objects, each with keys "description", "importance" (string "1"|"2"|"3"), "inquiry", "options" (array of 3 short strings). Use [] if vague is false.
"""


def annotate_seed_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill vague, thought, missing_details for a row that at minimum contains "task".
    Preserves strand, category, source, seed_id when present.
    """
    task = row.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("annotate_seed_row requires a non-empty string 'task'")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for annotation mode. "
            "Set it in the environment or use fully specified seed JSONL without --annotate."
        )

    base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    base_url = base_url.rstrip("/")
    model = os.environ.get("AUGMENT_LLM_MODEL", "gpt-4o-mini")

    system = (
        "You annotate user tasks for a downstream agent that must judge vagueness and missing details. "
        + JSON_SCHEMA_HINT
    )
    user_msg = f"Task:\n{task.strip()}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }
    # Prefer JSON mode when supported
    payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:2000]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response: {data!r}") from e

    parsed = json.loads(content)
    vague = bool(parsed["vague"])
    thought = str(parsed["thought"])
    missing_details: List[Dict[str, Any]] = parsed.get("missing_details") or []

    if vague and not missing_details:
        raise RuntimeError("Model returned vague=true with empty missing_details")
    if not vague and missing_details:
        missing_details = []

    out: Dict[str, Any] = {
        "task": task.strip(),
        "vague": vague,
        "thought": thought,
        "missing_details": missing_details,
    }
    for key in ("category", "strand", "source", "seed_id"):
        if key in row:
            out[key] = row[key]
    if "category" not in out:
        out["category"] = "Annotated"
    return out
