from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests


def _get_endpoint_url() -> str:
    url = os.environ.get("MIDLM_ENDPOINT_URL")
    if not url:
        return ""
    return url.rstrip("/")


def _get_model_id() -> str:
    return os.environ.get("MIDLM_MODEL_ID", "midlm-qwen3b-bidirectional")


def predict_topk_intents(
    *,
    text: str,
) -> Tuple[List[str], int, Optional[Dict[str, Any]]]:
    """
    Call the MIDLM intent classifier via HTTP.

    Server contract (see midlm_access_report.md):
      POST /v1/chat/completions
      { "model": "...", "messages": [{"role": "user", "content": "..."}],
        "max_tokens": 128, "temperature": 0.0 }

    Response carries intents in choices[0].intent_result (primary)
    or choices[0].message.content as JSON (fallback).

    Returns:
      (intent_labels, k, raw_response_dict_or_None)
    """
    endpoint = _get_endpoint_url()
    if not endpoint:
        raise RuntimeError("MIDLM_ENDPOINT_URL not set")

    model_id = _get_model_id()
    body: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 128,
        "temperature": 0.0,
    }

    print(f"[MIDLM_HTTP] POST {endpoint}/v1/chat/completions (model={model_id})")
    print(f"[MIDLM_HTTP] Input: {text[:200]}")

    try:
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=15,
        )
        print(f"[MIDLM_HTTP] Response status: {resp.status_code}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"MIDLM server unreachable at {endpoint}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"MIDLM server timed out at {endpoint}") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"MIDLM request failed: {e}") from e

    if resp.status_code >= 400:
        raise RuntimeError(f"MIDLM server returned {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"[MIDLM_HTTP] Failed to parse response JSON: {e}")
        return [], 0, None

    try:
        choice = data["choices"][0]
    except (KeyError, IndexError) as e:
        print(f"[MIDLM_HTTP] Unexpected response structure: {e}")
        return [], 0, None

    # Primary: intent_result field (custom server extension)
    if "intent_result" in choice:
        r = choice["intent_result"]
        k = int(r.get("k", 0))
        intents = list(r.get("intents", []))
        print(f"[MIDLM_HTTP] Result: k={k}, intents={intents}")
        return intents, k, r

    # Fallback: message.content as JSON string
    try:
        content = json.loads(choice["message"]["content"])
        k = int(content.get("k", 0))
        intents = list(content.get("intents", []))
        print(f"[MIDLM_HTTP] Result (from message.content): k={k}, intents={intents}")
        return intents, k, content
    except (KeyError, json.JSONDecodeError, TypeError) as e:
        print(f"[MIDLM_HTTP] Failed to parse message.content: {e}")
        print(f"[MIDLM_HTTP] Raw message: {choice.get('message', {}).get('content', '')[:300]}")
        return [], 0, choice
