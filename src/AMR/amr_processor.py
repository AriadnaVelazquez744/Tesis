# Tesis/src/AMR/amr_processor.py
import requests
from typing import Dict, Any

AMR_API_URL = "http://127.0.0.1:8001/parse"
AMR_HEALTH_URL = "http://127.0.0.1:8001/health"


def process_amr_into_cao(cao: Dict[str, Any], text: str) -> Dict[str, Any]:
    cao.setdefault("meta", {})
    print(f"[AMR] Processing text: {text[:80]}...")

    print(f"[AMR] Checking health at {AMR_HEALTH_URL}")
    try:
        health_resp = requests.get(AMR_HEALTH_URL, timeout=5)
        print(f"[AMR] Health status: {health_resp.status_code} - {health_resp.json()}")
    except Exception as e:
        print(f"[AMR] Health check FAILED: {type(e).__name__}: {e}")

    print(f"[AMR] Sending parse request to {AMR_API_URL}")
    try:
        response = requests.post(AMR_API_URL, json={"text": text}, timeout=60)
        print(f"[AMR] Parse response status: {response.status_code}")
        response.raise_for_status()

        amr_result = response.json()
        print(f"[AMR] Parse successful! Keys: {list(amr_result.keys())}")

        # 1. Store the initial, unrefined triplets for the Logic Transformer
        cao["state"]["unrefined_triples"] = amr_result.get("unrefined_triples", [])
        
        # 2. Store the raw Abstract Syntax Tree (AST) for KeyBERT fidelity checks
        cao["meta"]["amr_ast"] = amr_result.get("ast", {})
        
        # 3. Store the display string
        cao["meta"]["amr_ok"] = True
        cao["meta"]["amr_graph"] = amr_result.get("amr_graph", "")

        print(f"[AMR] Extracted {len(cao['state']['unrefined_triples'])} unrefined triples.")
        print(f"[AMR] Unrefined triples sample: {cao['state']['unrefined_triples']}")  # Show triples
        print("[AMR] AMR integration COMPLETED successfully")

        print(f"[AMR] AMR graph: {cao['meta']['amr_graph'][:200]}...")
        print("[AMR] AMR integration COMPLETED successfully")

    except requests.exceptions.ConnectionError as e:
        print(f"[AMR] ERROR Connection Failed: {type(e).__name__}: {e}")
        cao["meta"]["amr_ok"] = False
        cao["meta"]["amr_error"] = "connection_failed"
        cao["meta"]["amr_error_detail"] = str(e)
    except requests.exceptions.Timeout as e:
        print(f"[AMR] ERROR Timeout: {type(e).__name__}: {e}")
        cao["meta"]["amr_ok"] = False
        cao["meta"]["amr_error"] = "timeout"
    except requests.exceptions.HTTPError as e:
        print(f"[AMR] ERROR HTTP Error: {type(e).__name__}: {e}")
        cao["meta"]["amr_ok"] = False
        cao["meta"]["amr_error"] = "http_error"
        cao["meta"]["amr_error_detail"] = str(e)
    except requests.exceptions.RequestException as e:
        print(f"[AMR] ERROR Request Failed: {type(e).__name__}: {e}")
        cao["meta"]["amr_ok"] = False
        cao["meta"]["amr_error"] = "request_failed"
        cao["meta"]["amr_error_detail"] = str(e)
    except Exception as e:
        print(f"[AMR] ERROR Unexpected: {type(e).__name__}: {e}")
        cao["meta"]["amr_ok"] = False
        cao["meta"]["amr_error"] = "unexpected"
        cao["meta"]["amr_error_detail"] = str(e)

    return cao
