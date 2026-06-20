"""Shared utilities for CAO evaluation LLM processors."""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL_ID = "accounts/fireworks/models/qwen3p7-plus"

CAO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the Vagueness Judge resolution and the "
    "Cognitive Analysis Object (CAO) below to answer the user naturally and completely. "
    "If the CAO marks the query as OOS (out-of-scope), acknowledge the limitation "
    "instead of inventing capabilities."
)

RAW_SYSTEM_PROMPT = "You are a helpful assistant."


def sanitize_filename(text: str, max_len: int = 60) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:max_len].strip("_")


def load_queries(filepath: Path) -> list[Dict[str, Any]]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def load_cao_records(cao_dir: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for path in sorted(cao_dir.glob("cao_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        batch_info = data.get("meta", {}).get("_batch_info", {})
        records.append(
            {
                "source_file": path.name,
                "source_path": path,
                "record_id": batch_info.get("record_id", ""),
                "index": batch_info.get("index", 0),
                "query": batch_info.get("query", ""),
                "meta": data.get("meta", {}),
                "cao": data.get("cao", {}),
            }
        )
    records.sort(key=lambda r: r.get("index", 0))
    return records


def extract_jdv_info(record: Dict[str, Any]) -> Dict[str, str]:
    meta = record.get("meta", {})
    cao_meta = record.get("cao", {}).get("meta", {})
    jdv = meta.get("jdv", {})
    return {
        "completed_query": (
            jdv.get("completed_query")
            or cao_meta.get("completed_query")
            or record.get("query", "")
        ),
        "summary": jdv.get("summary") or cao_meta.get("summary", ""),
        "summary_thought": (
            jdv.get("summary_thought") or cao_meta.get("summary_thought", "")
        ),
        "status": jdv.get("status", "resolved"),
    }


def compact_cao_for_prompt(cao: Dict[str, Any]) -> Dict[str, Any]:
    compact = copy.deepcopy(cao)
    meta = compact.get("meta", {})
    if isinstance(meta, dict):
        for key in ("amr_ast", "amr_graph", "vagueness_raw"):
            meta.pop(key, None)
    state = compact.get("state", {})
    if isinstance(state, dict):
        state.pop("unrefined_triples", None)
    return compact


def build_cao_prompt(record: Dict[str, Any]) -> Tuple[str, str]:
    jdv = extract_jdv_info(record)
    cao_meta = record.get("cao", {}).get("meta", {})
    original_query = (
        cao_meta.get("original_query")
        or record.get("query", "")
    )

    parts = [
        f"## Original query\n{original_query}",
        f"## Resolved query (JDV)\n{jdv['completed_query']}",
        "## JDV summary",
        jdv["summary"] or "(none)",
    ]
    if jdv["summary_thought"]:
        parts.append(jdv["summary_thought"])

    cao_json = json.dumps(
        compact_cao_for_prompt(record["cao"]),
        ensure_ascii=False,
        indent=2,
    )
    parts.append(f"## Cognitive Analysis Object\n{cao_json}")

    return CAO_SYSTEM_PROMPT, "\n\n".join(parts)


def call_fireworks(
    messages: List[Dict[str, str]],
    api_key: str,
    model_id: str = FIREWORKS_MODEL_ID,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 45,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    max_retries = 5
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{FIREWORKS_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            if response.status_code == 429:
                print(
                    f"  [WARNING] Rate limited (429). Retrying in {delay}s... "
                    f"({attempt}/{max_retries})"
                )
                time.sleep(delay)
                delay *= 2
            else:
                print(
                    f"  [ERROR] API returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
                time.sleep(delay)
                delay *= 2
        except requests.exceptions.RequestException as e:
            print(
                f"  [ERROR] Network error: {e}. Retrying in {delay}s... "
                f"({attempt}/{max_retries})"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Failed to get response from Fireworks API after all retries.")


def call_openai_compatible(
    messages: List[Dict[str, str]],
    base_url: str,
    model_id: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    endpoint = base_url.rstrip("/")
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_id:
        payload["model"] = model_id

    max_retries = 5
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{endpoint}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            print(
                f"  [ERROR] API returned {response.status_code}: "
                f"{response.text[:200]}"
            )
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.RequestException as e:
            print(
                f"  [ERROR] Network error: {e}. Retrying in {delay}s... "
                f"({attempt}/{max_retries})"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(
        f"Failed to get response from {endpoint} after all retries."
    )


def get_completed_record_ids(output_dir: Path, prefix: str) -> set[str]:
    completed: set[str] = set()
    if not output_dir.exists():
        return completed
    for path in output_dir.glob(f"{prefix}_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            record_id = data.get("meta", {}).get("record_id", "")
            if record_id:
                completed.add(record_id)
        except (json.JSONDecodeError, OSError):
            continue
    return completed


def save_result(
    output_dir: Path,
    prefix: str,
    record_id: str,
    index: int,
    slug: str,
    payload: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{index:04d}_{record_id}_{slug}.json"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath
