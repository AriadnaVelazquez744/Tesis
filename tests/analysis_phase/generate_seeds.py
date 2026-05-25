#!/usr/bin/env python3
"""
Generate seed queries for the Analysis Phase Evaluation Dataset.

Generates ONE seed per LLM call (15 calls per topic, 75 total) so each
response is small and JSON-parsing is reliable.

Features:
  - Each seed is saved immediately upon successful validation.
  - Duplicate seed_id prevention: IDs are generated in the script.
  - Auto-retry: if a seed fails schema validation, the LLM is called again
    with a correction prompt (up to 3 attempts).
  - Checkpoint/resume by reading existing seed_ids from the output file.

Usage:
    python tests/generate_seeds.py

Environment:
    LLM_API_BASE    (default: http://10.6.125.216:8080/v1)
    LLM_MODEL       (default: qwen2.5-32b-instruct)
    LLM_API_KEY     (optional, default: "not-needed")
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
PROMPTS = FIXTURES / "prompts"
SEEDS = FIXTURES / "seeds"

LLM_BASE = os.environ.get("LLM_API_BASE", "http://10.6.125.216:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-32b-instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

VALID_RELATION_TYPES = {"ASYMMETRIC", "INVERSE", "SYMMETRIC", "ORDERING", "FUNCTIONAL"}
VALID_ANCHOR_TYPES = {"entity", "concept"}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```[\s\S]*$", "", text)
        text = text.strip()
    brace = text.find("{")
    if brace == -1:
        raise json.JSONDecodeError("No opening brace found", text, 0)
    text = text[brace:]
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def seed_errors(seed: dict, clinc150: set[str]) -> list[str]:
    errs = []
    sid = seed.get("seed_id", "???")

    if not isinstance(sid, str) or not sid.strip():
        errs.append("seed_id missing or empty")

    topic = seed.get("topic")
    if topic not in {"natural_logic", "mathematical_calculation", "information_search", "human_psychology", "mixed"}:
        errs.append(f"invalid topic '{topic}'")

    domain = seed.get("domain_cluster")
    if domain not in {"banking_finance", "travel", "home_auto", "food_dining", "healthcare", "communication", "info_utilities", "entertainment"}:
        errs.append(f"invalid domain_cluster '{domain}'")

    k = seed.get("k")
    if not isinstance(k, int) or k < 1 or k > 3:
        errs.append(f"k={k} must be int 1-3")

    oos = seed.get("oos")
    if not isinstance(oos, bool):
        errs.append("oos must be bool")

    query = seed.get("query", "")
    if not isinstance(query, str) or not query.strip():
        errs.append("query is empty")

    intents = seed.get("intents", [])
    if not isinstance(intents, list):
        errs.append("intents must be list")
    else:
        if oos:
            if len(intents) != 0:
                errs.append(f"oos=true but intents={intents} (should be [])")
        else:
            if len(intents) != k:
                errs.append(f"len(intents)={len(intents)} != k={k}")
            for intent in intents:
                if intent not in clinc150:
                    errs.append(f"intent '{intent}' not in CLINC150")

    anchors = seed.get("expected_grounding_anchors", [])
    if not isinstance(anchors, list) or not anchors:
        errs.append("expected_grounding_anchors must be non-empty list")
    else:
        for i, a in enumerate(anchors):
            if not isinstance(a, dict):
                errs.append(f"anchors[{i}] not a dict")
            else:
                if not isinstance(a.get("term"), str) or not a["term"].strip():
                    errs.append(f"anchors[{i}] missing/invalid 'term'")
                if a.get("type") not in VALID_ANCHOR_TYPES:
                    errs.append(f"anchors[{i}].type='{a.get('type')}' must be entity or concept")

    triples = seed.get("expected_unrefined_triples", [])
    if not isinstance(triples, list) or not triples:
        errs.append("expected_unrefined_triples must be non-empty list")
    else:
        for i, t in enumerate(triples):
            if not isinstance(t, dict):
                errs.append(f"triples[{i}] not a dict")
                continue
            triple_arr = t.get("triple")
            if not isinstance(triple_arr, list) or len(triple_arr) != 3:
                errs.append(f"triples[{i}].triple must be [S,R,O] list")
            else:
                for j, elem in enumerate(triple_arr):
                    if not isinstance(elem, str):
                        errs.append(f"triples[{i}].triple[{j}] must be string, got {type(elem).__name__}")
            rtype = t.get("relation_type")
            if rtype not in VALID_RELATION_TYPES:
                errs.append(f"triples[{i}].relation_type='{rtype}' invalid")

    rtypes = seed.get("relation_types", {})
    if not isinstance(rtypes, dict):
        errs.append("relation_types must be dict")
    else:
        triple_relations = set()
        for t in triples:
            if isinstance(t, dict) and isinstance(t.get("triple"), list) and len(t["triple"]) == 3:
                triple_relations.add(t["triple"][1])
        for rel in triple_relations:
            if rel not in rtypes:
                errs.append(f"relation '{rel}' in triples but missing from relation_types")
        for rel, val in rtypes.items():
            if val not in VALID_RELATION_TYPES:
                errs.append(f"relation_types['{rel}']='{val}' invalid")

    rt = seed.get("reversal_type")
    if rt not in {"inverse_voice", "inverse_relation", "clause_order", "symmetric_direction", None}:
        errs.append(f"invalid reversal_type '{rt}'")

    return errs


def call_llm(system_prompt: str, temperature: float = 0.3) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system_prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{LLM_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}: {resp.text[:2000]}"
                if attempt < MAX_RETRIES:
                    print(f"\n      API error, retrying ({attempt}/{MAX_RETRIES}) ...")
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                raise RuntimeError(last_error)

            return resp.json()["choices"][0]["message"]["content"]

        except (json.JSONDecodeError, KeyError) as e:
            last_error = f"API response parse failure: {e}"
            if attempt < MAX_RETRIES:
                print(f"\n      {last_error}, retrying ({attempt}/{MAX_RETRIES}) ...")
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise RuntimeError(last_error) from e

    raise RuntimeError(last_error)


def build_seed_prompt(
    topic: dict,
    domain: dict,
    seed_id: str,
    intents: list[str],
    k: int,
    is_oos: bool,
    need_reversal: bool,
    correction_hint: str = "",
) -> str:
    tid = topic["id"]
    did = domain["id"]

    all_intents = "\n".join(f"- {i}" for i in intents)

    enum_k = f"k={k}; if oos then k=0"
    intent_rule = "Set expected_intents=[] (empty)" if is_oos else f"expected_intents must have exactly {k} intents from the list above"

    prompt = (
        f"You generate exactly ONE seed record. Return ONLY a JSON object — no markdown, no code fences.\n\n"
        f"## Parameters\n"
        f"- seed_id: {seed_id} (use this EXACT value)\n"
        f"- topic: {tid}\n"
        f"- domain_cluster: {did}\n"
        f"- k: {k}\n"
        f"- oos: {str(is_oos).lower()}\n"
        f"- need_reversal: {str(need_reversal).lower()}\n\n"
        f"## Topic\n{topic['description']}\n\n"
        f"## Domain\n{domain['description']}\n\n"
        f"## CLINC150 intents (use ONLY these)\n{all_intents}\n\n"
        f"## Valid relation types (use EXACT strings, case-sensitive)\n"
        f"- ASYMMETRIC — fixed direction, most verbs (causes, owns, kicks, drives)\n"
        f"- INVERSE — has distinct inverse (parent↔child, wrote↔was_written_by)\n"
        f"- SYMMETRIC — both directions valid (equals, is_sibling_of, is_married_to)\n"
        f"- ORDERING — defines order (greater_than, before, after, older_than)\n"
        f"- FUNCTIONAL — one-to-one (is_capital_of, has_ssn)\n\n"
        f"## Required JSON structure (use EXACT field names)\n"
        f"{{\n"
        f'  "seed_id": "{seed_id}",\n'
        f'  "topic": "{tid}",\n'
        f'  "domain_cluster": "{did}",\n'
        f'  "query": "natural user utterance (6-30 words)",\n'
        f'  "k": {k},\n'
        f'  "intents": [/* {k} CLINC150 intents */],\n'
        f'  "oos": {str(is_oos).lower()},\n'
        f'  "expected_grounding_anchors": [\n'
        f'    {{"term": "entity_or_phrase", "type": "entity"}},\n'
        f'    {{"term": "concept_name", "type": "concept"}}\n'
        f'  ],\n'
        f'  "expected_unrefined_triples": [\n'
        f'    {{"triple": ["Subject", "relationVerb", "Object"], "relation_type": "ASYMMETRIC"}}\n'
        f'  ],\n'
        f'  "relation_types": {{"relationVerb": "ASYMMETRIC"}},\n'
        f'  "reversal_type": null\n'
        f"}}\n\n"
        f"## Rules\n"
        f"1. All triple elements must be strings (no numbers, no nulls).\n"
        f"2. anchor.type must be exactly 'entity' or 'concept'.\n"
        f"3. Every relation in unrefined_triples[].triple[1] must have a key in relation_types.\n"
        f"4. relation_types values must be one of: ASYMMETRIC, INVERSE, SYMMETRIC, ORDERING, FUNCTIONAL.\n"
        f"5. Use raw forms in triples — preserve 'I', 'my', original verb tenses.\n"
        f"6. {intent_rule}.\n"
    )

    if need_reversal:
        prompt += (
            f"7. This MUST have a non-null reversal_type: inverse_voice, inverse_relation, clause_order, or symmetric_direction.\n"
        )

    if correction_hint:
        prompt += f"\n## Previous attempt had errors — FIX THESE:\n{correction_hint}\n"

    return prompt


def _auto_fix_seed(seed: dict, k_target: int, is_oos: bool, clinc150: set[str]) -> dict:
    s = dict(seed)
    if is_oos:
        s["intents"] = []
    else:
        s["intents"] = [i for i in s.get("intents", []) if i in clinc150][:k_target]
    return s


def _force_fix_seed(seed: dict, tid: str, domain_id: str, k_target: int, is_oos: bool, clinc150: set[str]) -> dict:
    s = dict(seed)
    s["seed_id"] = seed.get("seed_id", "?")
    s["topic"] = tid
    s["domain_cluster"] = domain_id
    s["k"] = k_target
    s["oos"] = is_oos
    if is_oos:
        s["intents"] = []
    else:
        s["intents"] = [i for i in s.get("intents", []) if i in clinc150][:k_target]
    anchors = s.get("expected_grounding_anchors", [])
    if isinstance(anchors, list):
        for a in anchors:
            if isinstance(a, dict) and a.get("type") not in {"entity", "concept"}:
                a["type"] = "entity"
    triples = s.get("expected_unrefined_triples", [])
    if isinstance(triples, list):
        for t in triples:
            if isinstance(t, dict):
                arr = t.get("triple")
                if isinstance(arr, list):
                    for j in range(len(arr)):
                        if not isinstance(arr[j], str):
                            arr[j] = str(arr[j])
                rtype = t.get("relation_type")
                if isinstance(rtype, str) and rtype.upper() in VALID_RELATION_TYPES:
                    t["relation_type"] = rtype.upper()
    return s


def append_seed(tid: str, seed: dict) -> None:
    path = SEEDS / f"{tid}_seeds.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(seed, ensure_ascii=False) + "\n")


def load_existing_seed_ids(tid: str) -> set[str]:
    path = SEEDS / f"{tid}_seeds.jsonl"
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("seed_id", "")
                if sid:
                    seen.add(sid)
            except json.JSONDecodeError:
                pass
    return seen


def main() -> int:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    SEEDS.mkdir(parents=True, exist_ok=True)

    print("Loading definitions ...")
    topics_data = load_yaml(PROMPTS / "topics.yaml")
    domains_data = load_yaml(PROMPTS / "domains.yaml")
    intents = load_json(PROMPTS / "clinc150_intents.json")
    clinc150 = set(intents)

    topics = topics_data["topics"]
    domains = domains_data["domains"]

    topic_abbrev_map = {t["id"]: t["abbrev"] for t in topics}
    domain_abbrev_map = {d["id"]: d["abbrev"] for d in domains}
    domain_by_id = {d["id"]: d for d in domains}

    print(f"  Topics        : {[t['id'] for t in topics]}")
    print(f"  Domains       : {[d['id'] for d in domains]}")
    print(f"  CLINC150      : {len(intents)} intents")
    print(f"  LLM endpoint  : {LLM_BASE}/chat/completions")
    print(f"  LLM model     : {LLM_MODEL}")
    print()

    total_new = 0
    total_skipped = 0
    total_failed = 0

    for topic in topics:
        tid = topic["id"]
        tabbrev = topic_abbrev_map[tid]
        existing_ids = load_existing_seed_ids(tid)

        # Counter per domain for unique seed_id numbering
        domain_counter: dict[str, int] = {}

        plan = [
            {"k": 1, "oos": False, "reversal": True, "domain_id": "banking_finance"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "travel"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "home_auto"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "food_dining"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "healthcare"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "communication"},
            {"k": 1, "oos": False, "reversal": False, "domain_id": "info_utilities"},
            {"k": 1, "oos": False, "reversal": True, "domain_id": "entertainment"},
            {"k": 2, "oos": False, "reversal": False, "domain_id": "banking_finance"},
            {"k": 2, "oos": False, "reversal": False, "domain_id": "travel"},
            {"k": 2, "oos": False, "reversal": False, "domain_id": "home_auto"},
            {"k": 2, "oos": False, "reversal": False, "domain_id": "food_dining"},
            {"k": 2, "oos": False, "reversal": False, "domain_id": "healthcare"},
            {"k": 3, "oos": False, "reversal": False, "domain_id": "info_utilities"},
            {"k": 3, "oos": True, "reversal": False, "domain_id": "communication"},
        ]

        print(f"[{tid}] {len(plan)} seeds planned")
        sys.stdout.flush()

        for idx, spec in enumerate(plan):
            domain_id = spec["domain_id"]
            k = spec["k"]
            is_oos = spec["oos"]
            need_reversal = spec["reversal"]

            domain_abbrev = domain_abbrev_map[domain_id]
            domain_counter[domain_id] = domain_counter.get(domain_id, 0) + 1
            num = domain_counter[domain_id]
            seed_id = f"{tabbrev}-{domain_abbrev}-{num:03d}"

            if seed_id in existing_ids:
                print(f"  [{idx+1}/{len(plan)}] {seed_id} already exists — SKIPPED")
                total_skipped += 1
                continue

            domain_obj = domain_by_id[domain_id]

            print(f"  [{idx+1}/{len(plan)}] {seed_id} (k={k}, oos={is_oos}) ...", end=" ")
            sys.stdout.flush()

            # Up to 3 attempts per seed with correction
            seed_obj = None
            for attempt in range(1, 4):
                correction = ""
                if attempt > 1 and seed_obj is not None:
                    errs = seed_errors(seed_obj, clinc150)
                    correction = "Fix these errors:\n" + "\n".join(f"- {e}" for e in errs[:5])

                prompt = build_seed_prompt(
                    topic, domain_obj, seed_id, intents, k, is_oos, need_reversal,
                    correction_hint=correction,
                )

                try:
                    raw = call_llm(prompt, temperature=0.3)
                    candidate = _extract_json(raw)
                except Exception as e:
                    if attempt < 3:
                        print(f"\n      LLM error, retry ({attempt}/3) ...")
                        time.sleep(RETRY_DELAY_SEC)
                        continue
                    print(f"FAILED after 3 LLM attempts: {e}")
                    total_failed += 1
                    seed_obj = None
                    break

                errs = seed_errors(candidate, clinc150)
                if not errs:
                    seed_obj = candidate
                    break

                # Attempt auto-fix for known model quirks before retry
                candidate = _auto_fix_seed(candidate, k, is_oos, clinc150)
                fixed_errs = seed_errors(candidate, clinc150)
                if not fixed_errs:
                    seed_obj = candidate
                    print("*", end=" ")
                    sys.stdout.flush()
                    break

                if attempt < 3:
                    print(f"\n      schema errors, retry ({attempt}/3) ...")
                    seed_obj = candidate  # keep last attempt for potential final fix
                else:
                    # Last resort: force-fix remaining issues
                    candidate = _force_fix_seed(candidate, tid, domain_id, k, is_oos, clinc150)
                    final_errs = seed_errors(candidate, clinc150)
                    if not final_errs:
                        seed_obj = candidate
                        print("!", end=" ")
                        sys.stdout.flush()
                    else:
                        print(f"\n      FAILED: {'; '.join(final_errs[:3])}")
                        total_failed += 1
                        seed_obj = None
                        break

            if seed_obj is None:
                continue

            # Override fields the script manages
            seed_obj["seed_id"] = seed_id
            seed_obj["topic"] = tid
            seed_obj["domain_cluster"] = domain_id
            seed_obj["k"] = k
            seed_obj["oos"] = is_oos

            append_seed(tid, seed_obj)
            print(f"→ {seed_id}")
            total_new += 1

        print()

    print("=== Summary ===")
    for p in sorted(SEEDS.glob("*_seeds.jsonl")):
        count = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
        print(f"  {p.name}: {count}")
    print(f"  NEW: {total_new}")
    if total_skipped:
        print(f"  SKIPPED (existing): {total_skipped}")
    if total_failed:
        print(f"  FAILED: {total_failed}")
    print()

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
