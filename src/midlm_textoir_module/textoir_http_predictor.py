from __future__ import annotations

import os
from typing import Optional, Tuple

import requests


def _get_endpoint_url() -> str:
    url = os.environ.get("TEXTOIR_ENDPOINT_URL")
    if not url:
        return ""
    return url.rstrip("/")


def predict_ind_oos(
    *,
    text: str,
    timeout: int = 10,
) -> Tuple[str, Optional[float], str]:
    """
    Call the TEXTOIR intent classifier via HTTP.

    Server contract:
      POST /predict
      { "text": "..." }

    Response:
      { "status": "IND"|"OOS", "confidence": 0.97, "method_used": "msp" }

    Returns:
      (status, confidence, method_used)
        status: "IND" | "OOS" | "UNKNOWN"
        confidence: max softmax probability (or None on error)
        method_used: method name or error description
    """
    endpoint = _get_endpoint_url()
    if not endpoint:
        return "UNKNOWN", None, "TEXTOIR_ENDPOINT_URL not set"

    print(f"[TEXTOIR_HTTP] POST {endpoint}/predict")
    print(f"[TEXTOIR_HTTP] Input: {text[:200]}")

    try:
        resp = requests.post(
            f"{endpoint}/predict",
            headers={"Content-Type": "application/json"},
            json={"text": text},
            timeout=timeout,
        )
        print(f"[TEXTOIR_HTTP] Response status: {resp.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"[TEXTOIR_HTTP] Server unreachable at {endpoint}: {e}")
        return "UNKNOWN", None, f"connection_error:{e}"
    except requests.exceptions.Timeout as e:
        print(f"[TEXTOIR_HTTP] Server timed out: {e}")
        return "UNKNOWN", None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"[TEXTOIR_HTTP] Request failed: {e}")
        return "UNKNOWN", None, f"request_error:{e}"

    if resp.status_code >= 400:
        print(f"[TEXTOIR_HTTP] Server returned {resp.status_code}: {resp.text[:300]}")
        return "UNKNOWN", None, f"http_{resp.status_code}"

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[TEXTOIR_HTTP] Invalid JSON response: {e}")
        return "UNKNOWN", None, "invalid_json"

    status = str(data.get("status", "UNKNOWN"))
    confidence = data.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    method_used = str(data.get("method_used", "unknown"))

    print(f"[TEXTOIR_HTTP] Result: status={status}, confidence={confidence}, method={method_used}")
    return status, confidence, method_used
