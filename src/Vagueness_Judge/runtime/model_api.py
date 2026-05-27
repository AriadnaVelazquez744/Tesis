from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests


TASK_DESCRIPTION = (
    "You are an agent trying to understand the user's goal and summarize it. "
    "Please first ask users for more specific details with options, and finally summarize the user's intention.\n"
    "--- Step 1: initial thought generation ---\n"
    "1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.\n"
    "2. List the important missing details and some according options if the task is vague.\n"
    "--- Step 2: inquiry for more information if vague ---\n"
    "1. If the task is vague, inquire about more details with options according to the list in [INITIAL THOUGHT].\n"
    "2. Think about what information you have and what to inquire next in [INQUIRY THOUGHT].\n"
    "3. Present your inquiry with options for the user to choose after [INQUIRY], and be friendly.\n"
    "4. You could repeat Step 2 multiple times (but less than 5 times), or directly skip Step 2 if the user task is clear initially.\n"
    "--- Step 3: summarize the user's intention ---\n"
    "1. Make the summary once the information is enough. You do not need to inquire about every missing detail in [INITIAL THOUGHT].\n"
    "2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be the same as rounds of chatting.\n"
    "3. Give the final summary after [SUMMARY] with comprehensive details in one or two sentences."
)


def _get_endpoint_url() -> str:
    url = os.environ.get("VAGUE_ENDPOINT_URL") or os.environ.get("LLMSTUDIO_BASE_URL")
    if not url:
        print("[VAGUENESS] No VAGUE_ENDPOINT_URL or LLMSTUDIO_BASE_URL set in environment")
        return ""
    return url.rstrip("/")


def _build_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    mode = payload.get("mode", "initial")
    query = str(payload.get("query", "")).strip()
    turns = payload.get("turns") or []

    if mode == "initial" or not turns:
        return [
            {"role": "system", "content": TASK_DESCRIPTION},
            {"role": "user", "content": f"Here is the task:\n{query}"},
        ]

    messages = [
        {"role": "system", "content": TASK_DESCRIPTION},
    ]
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append({"role": role, "content": content})
    return messages


def _parse_model_response(text: str, query: str, clarifications: List[str]) -> Dict[str, Any]:
    has_summary = "[SUMMARY]" in text
    has_inquiry = "[INQUIRY]" in text
    has_initial_thought = "[INITIAL THOUGHT]" in text

    if has_summary:
        summary = ""
        if "[SUMMARY]" in text:
            summary = text.split("[SUMMARY]")[-1].strip()
        for tag in ["[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[INQUIRY", "[SUMMARY THOUGHT"]:
            if tag in summary:
                summary = summary.split(tag)[0].strip()

        summary_thought = ""
        if "[SUMMARY THOUGHT]" in text:
            st_part = text.split("[SUMMARY THOUGHT]")[-1].strip()
            for tag in ["[SUMMARY]", "[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[INQUIRY"]:
                if tag in st_part:
                    st_part = st_part.split(tag)[0].strip()
            specs_lines = [
                l.strip().lstrip("-* ").strip()
                for l in st_part.split("\n")
                if l.strip().startswith("- ") or l.strip().startswith("* ")
            ]
            summary_thought = "\n".join(specs_lines)

        cleaned = [c.strip() for c in clarifications if c and c.strip()]
        if summary.strip():
            completed_query = summary.strip()
        elif cleaned:
            completed_query = f"{query.strip()}\n\nClarifications from user: {'; '.join(cleaned)}."
        else:
            completed_query = query.strip()

        return {
            "status": "resolved",
            "completed_query": completed_query,
            "summary": summary,
            "summary_thought": summary_thought,
            "_raw": text,
        }

    if has_inquiry:
        question = text.split("[INQUIRY]")[-1].strip()
        for tag in ["[SUMMARY", "[INITIAL THOUGHT", "[INQUIRY THOUGHT", "[SUMMARY THOUGHT"]:
            if tag in question:
                question = question.split(tag)[0].strip()
        if not question:
            question = "Could you provide more details about your request?"
        return {
            "status": "needs_clarification",
            "question": question,
            "_raw": text,
        }

    if has_initial_thought:
        thought = text.split("[INITIAL THOUGHT]")[-1].strip()
        thought_clean = thought.split("[INQUIRY THOUGHT]")[0].split("[INQUIRY]")[0].split("[SUMMARY THOUGHT]")[0].split("[SUMMARY]")[0].strip()
        if "clear" in thought_clean.lower() and "vague" not in thought_clean.lower().split("clear")[0]:
            return {
                "status": "resolved",
                "completed_query": query.strip(),
                "summary": "The request is clear.",
                "_raw": text,
            }

    # No recognizable tags — use the model's raw response as the question
    question = text.strip()
    for tag in ["[INITIAL THOUGHT]", "[INQUIRY THOUGHT]", "[INQUIRY]", "[SUMMARY THOUGHT]", "[SUMMARY]"]:
        if tag in question:
            question = question.replace(tag, "").strip()
    question = question.strip().strip(":\n ")
    if not question:
        question = "Could you clarify your request with more specific details?"
    return {
        "status": "needs_clarification",
        "question": question,
        "_raw": text,
    }


def call_vagueness_model(prompt: str) -> str:
    try:
        payload: Dict[str, Any] = json.loads(prompt)
    except json.JSONDecodeError:
        payload = {"mode": "initial", "query": prompt, "clarifications": [], "turns": []}

    mode = str(payload.get("mode", "initial"))
    query = str(payload.get("query", "")).strip()
    clarifications = payload.get("clarifications") or []
    if not isinstance(clarifications, list):
        clarifications = [str(clarifications)]
    clarifications = [str(item) for item in clarifications]

    endpoint = _get_endpoint_url()
    if not endpoint:
        print("[VAGUENESS] No endpoint configured — returning fallback")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": "ERROR: No endpoint configured",
        })

    messages = _build_messages(payload)

    model_id = os.environ.get("VAGUE_MODEL_ID", "")
    if model_id:
        print(f"[VAGUENESS] POST {endpoint}/v1/chat/completions (mode={mode}, model={model_id})")
    else:
        print(f"[VAGUENESS] POST {endpoint}/v1/chat/completions (mode={mode})")
    print(f"[VAGUENESS] Messages: {json.dumps(messages, ensure_ascii=False)[:500]}...")

    body: Dict[str, Any] = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    if model_id:
        body["model"] = model_id

    try:
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=30,
        )
        print(f"[VAGUENESS] API response status: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[VAGUENESS] Connection refused to {endpoint}")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": f"ERROR: Connection refused to {endpoint}",
        })
    except requests.exceptions.Timeout:
        print("[VAGUENESS] Request timed out")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": "ERROR: Request timed out",
        })
    except requests.exceptions.RequestException as e:
        print(f"[VAGUENESS] Request failed: {e}")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": f"ERROR: {e}",
        })

    if resp.status_code >= 400:
        print(f"[VAGUENESS] API returned {resp.status_code}: {resp.text[:500]}")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": f"ERROR: API returned {resp.status_code}",
        })

    try:
        data = resp.json()
        model_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[VAGUENESS] Failed to parse API response: {e}")
        return json.dumps({
            "status": "needs_clarification",
            "question": "Could you clarify your request with more specific details?",
            "_raw": f"ERROR: Failed to parse response: {e}",
        })

    print(f"[VAGUENESS] Model output: {model_text[:500]}")

    result = _parse_model_response(model_text, query, clarifications)
    return json.dumps(result, ensure_ascii=False)
