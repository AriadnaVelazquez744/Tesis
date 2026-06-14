#!/usr/bin/env python3
"""
Fix cross-seed duplicate queries by regenerating one record per pair.

Targets 3 truly accidental cross-seed duplicate pairs where different seeds
produced the same query text. Keeps the first occurrence, regenerates the second.

Usage:
    python tests/fix_cross_dupes.py              # fix all 3 pairs (3 LLM calls)
    python tests/fix_cross_dupes.py --seed MATH-COMM-001 --variant 7  # single fix
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE
FIXTURES = ANALYSIS / "fixtures"
PROMPTS = FIXTURES / "prompts"
EXPANDED = FIXTURES / "expanded"
EXPANDED_PATH = EXPANDED / "expanded_queries.jsonl"

LLM_BASE = os.environ.get("LLM_API_BASE", "http://10.6.125.216:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-32b-instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")

MAX_K = 5
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

VALID_TOPICS = {"natural_logic", "mathematical_calculation", "information_search", "human_psychology", "mixed"}
VALID_DOMAINS = {"banking_finance", "travel", "home_auto", "food_dining", "healthcare", "communication", "info_utilities", "entertainment"}
VALID_RELATION_TYPES = {"ASYMMETRIC", "INVERSE", "SYMMETRIC", "ORDERING", "FUNCTIONAL"}
VALID_INFERRED_BY = {
    "amr_arg0_arg1", "amr_passive", "preposition_directional",
    "lexical_mutuality", "lexical_equality", "lexical_possession",
    "lexical_kinship_symmetric", "lexical_kinship_asymmetric",
    "lexical_causal", "lexical_location", "lexical_comparative",
    "unique_identification", "fallback",
}
TOPIC_ABBR = {
    "natural_logic": "NLG", "mathematical_calculation": "MATH",
    "information_search": "INFO", "human_psychology": "PSYCH", "mixed": "MIX",
}
DOMAIN_ABBR = {
    "banking_finance": "BANK", "travel": "TRAV", "home_auto": "HOME",
    "food_dining": "FOOD", "healthcare": "HLTH", "communication": "COMM",
    "info_utilities": "UTIL", "entertainment": "ENTR",
}


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_records(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```[\s\S]*$", "", text)
        text = text.strip()
    brace = text.find("{")
    if brace == -1:
        raise json.JSONDecodeError("No opening brace", text, 0)
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj

def format_intent_list(intents: list[str]) -> str:
    return "\n".join(f"- {i}" for i in intents)

def call_llm(system_prompt: str, temperature: float) -> str:
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
                timeout=300,
            )
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}: {resp.text[:2000]}"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                raise RuntimeError(last_error)
            return resp.json()["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            last_error = f"API parse failure: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise RuntimeError(last_error) from e
    raise RuntimeError(last_error)

def record_errors(record: dict) -> list[str]:
    errs = []
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        errs.append("id missing")
    if not isinstance(record.get("query"), str) or not record["query"].strip():
        errs.append("query empty/missing")
    if not isinstance(record.get("summary"), str) or not record["summary"].strip():
        errs.append("summary empty/missing")
    topic = record.get("topic")
    if topic not in VALID_TOPICS:
        errs.append(f"invalid topic '{topic}'")
    domain = record.get("domain_cluster")
    if domain not in VALID_DOMAINS:
        errs.append(f"invalid domain_cluster '{domain}'")
    if record.get("vague") is not False:
        errs.append("vague must be false")
    k = record.get("k")
    if not isinstance(k, int) or k < 1 or k > MAX_K:
        errs.append(f"k={k} must be int 1-{MAX_K}")
    oos = record.get("oos")
    if not isinstance(oos, bool):
        errs.append("oos must be bool")
    intents = record.get("expected_intents", [])
    if not isinstance(intents, list):
        errs.append("expected_intents must be list")
    elif oos:
        if intents:
            errs.append("oos=true but intents non-empty")
    else:
        if len(intents) != k:
            errs.append(f"len(expected_intents)={len(intents)} != k={k}")
    anchors = record.get("expected_grounding_anchors", [])
    if not isinstance(anchors, list) or not anchors:
        errs.append("expected_grounding_anchors empty")
    for triple_key in ("expected_unrefined_triples", "expected_refined_triples"):
        triples = record.get(triple_key, [])
        if not isinstance(triples, list) or not triples:
            errs.append(f"{triple_key} empty")
        else:
            for t in triples:
                arr = t.get("triple")
                if not isinstance(arr, list) or len(arr) != 3:
                    errs.append(f"{triple_key}.triple not [S,R,O]")
                rtype = t.get("relation_type")
                if rtype not in VALID_RELATION_TYPES:
                    errs.append(f"{triple_key}.relation_type='{rtype}' invalid")
    rtypes = record.get("relation_types", {})
    if not isinstance(rtypes, dict):
        errs.append("relation_types must be dict")
    nesting = record.get("expected_nesting_graph", [])
    if not isinstance(nesting, list):
        errs.append("expected_nesting_graph must be list")
    elif k > 1 and not nesting:
        errs.append(f"k={k} > 1 but nesting_graph empty")
    elif k == 1 and nesting:
        errs.append("k=1 but nesting_graph non-empty")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        errs.append("meta missing/not dict")
    return errs

def auto_fix_record(record: dict) -> list[str]:
    if record.get("oos"):
        record["expected_intents"] = []
    k = record.get("k", 1)
    if k > MAX_K:
        record["k"] = MAX_K
        k = MAX_K
    intents = record.get("expected_intents", [])
    if not record.get("oos"):
        if len(intents) > k:
            record["expected_intents"] = intents[:k]
        elif len(intents) < k and intents:
            record["k"] = len(intents)
            k = len(intents)
    nesting = record.get("expected_nesting_graph", [])
    if k > 1 and not nesting:
        intents_list = record.get("expected_intents", [])
        if len(intents_list) >= 2:
            record["expected_nesting_graph"] = [
                {"parent": intents_list[0], "child": intents_list[1], "relation": "parallel"}
            ]
        else:
            record["expected_nesting_graph"] = [
                {"parent": "primary", "child": "secondary", "relation": "parallel"}
            ]
    for triple_key in ("expected_unrefined_triples", "expected_refined_triples"):
        for t in record.get(triple_key, []):
            if t.get("relation_type") == "ASYMMETRICAL":
                t["relation_type"] = "ASYMMETRIC"
    return record_errors(record)


def build_same_k_spec(record: dict, seed: dict, existing_queries: list[str]) -> str:
    """Build a spec that keeps the same k and intents but different wording."""
    k = record.get("k", 1)
    intents = record.get("expected_intents", [])
    oos = record.get("oos", False)
    oos_label = record.get("expected_oos_label")

    if oos:
        spec = (
            f"Generate alternative wording for this OOS query.\n"
            f"Type: OOS variant — same OOS concept, different wording.\n"
            f"- Rewrite the query so it has no matching CLINC150 intent\n"
            f"- Set oos=true and expected_intents=[]\n"
            f"- Set expected_oos_label to \"{oos_label or 'out_of_scope'}\"\n"
            f"- The query should be plausible and natural, just out-of-scope\n"
        )
    else:
        spec = (
            f"Generate alternative wording for this query.\n"
            f"Type: SAME K — same intents and k={k}, different wording.\n"
            f"- Rewrite the query with DIFFERENT vocabulary and sentence structure\n"
            f"- Preserve the exact set of intents: {json.dumps(intents)}\n"
            f"- Preserve k={k}\n"
            f"- Vary: formality, verb choice, question vs. statement\n"
            f"- Do NOT change the underlying meaning or the required triples\n"
            f"- oos: false\n"
        )

    if existing_queries:
        spec += (
            f"\n"
            f"## ANTI-REPEAT: avoid these already-existing phrasings in this seed\n"
            f"Your output MUST be DIFFERENT from all of these:\n"
        )
        for i, q in enumerate(existing_queries):
            spec += f"  Variant {i + 1}: \"{q[:120]}\"\n"
        spec += (
            f"\n"
            f"- Do NOT use the same opening phrase as any variant above\n"
            f"- Change: question vs statement, formality level, sentence structure\n"
        )

    return spec


def generate_replacement(seed: dict, target_record: dict, prompt_template: str,
                          intents_list: list[str], assigned_id: str,
                          seed_records: list[dict]) -> dict | None:
    """Generate a replacement for a duplicate record with different wording."""
    seed_id = seed.get("seed_id", "")
    variant_idx = target_record.get("meta", {}).get("variant_index", 0)
    topic = seed.get("topic", "")
    domain = seed.get("domain_cluster", "")

    # Collect existing queries from this seed for anti-repeat
    existing_queries = [r.get("query", "") for r in seed_records
                        if r.get("id") != target_record.get("id") and r.get("query")]

    spec = build_same_k_spec(target_record, seed, existing_queries)
    system_prompt = (
        prompt_template
        .replace("{seed_json}", json.dumps(seed, indent=2))
        .replace("{intent_list_formatted}", format_intent_list(intents_list))
        .replace("{specification}", spec)
        .replace("{assigned_id}", assigned_id)
    )

    for attempt in range(1, 4):
        try:
            raw = call_llm(system_prompt, temperature=0.8)
            result = _extract_json(raw)
            if not isinstance(result.get("id"), str):
                result["id"] = assigned_id
            if not result.get("query"):
                result["query"] = seed.get("query", "")
            result.setdefault("meta", {})
            result["meta"]["seed_source"] = seed_id
            result["meta"]["variant_index"] = variant_idx
            if not result.get("oos"):
                if not result.get("expected_intents"):
                    result["expected_intents"] = seed.get("intents", [])
                if result.get("k", 0) > MAX_K:
                    result["k"] = MAX_K
                    result["expected_intents"] = result["expected_intents"][:MAX_K]
        except Exception as e:
            if attempt < 3:
                continue
            print(f"      LLM error: {e}")
            return None

        errs = record_errors(result)
        if not errs:
            result["topic"] = topic
            result["domain_cluster"] = domain
            return result

        errs = auto_fix_record(result)
        if not errs:
            result["topic"] = topic
            result["domain_cluster"] = domain
            return result

        if attempt < 3:
            continue
        else:
            print(f"      FAILED: {'; '.join(errs[:3])}")
            return None

    return None


# The 3 truly accidental cross-seed duplicate pairs (different base queries)
# Format: (seed_to_regenerate, variant_index_to_regenerate)
ACCIDENTAL_PAIRS = [
    ("PSYCH-COMM-001", 7),   # "public speaking tips" → also in MIX-TRAV-002 v7
    ("INFO-UTIL-001", 7),    # "meaning of life" → also in INFO-UTIL-002 v7
    ("MATH-COMM-001", 7),    # "quantum entanglement" → also in MIX-UTIL-002 v7
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Specific seed (e.g. MATH-COMM-001)")
    parser.add_argument("--variant", type=int, help="Variant index")
    args = parser.parse_args()

    print("Loading records, seeds, prompts ...")
    records = load_records(EXPANDED_PATH)

    seeds: dict[str, dict] = {}
    for fpath in sorted((FIXTURES / "seeds").glob("*_seeds.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    seeds[s["seed_id"]] = s

    prompt_template = (PROMPTS / "expand_single.md").read_text(encoding="utf-8")
    intents_list = load_json(PROMPTS / "clinc150_intents.json")

    # Determine targets
    if args.seed and args.variant is not None:
        targets = [(args.seed, args.variant)]
    else:
        targets = ACCIDENTAL_PAIRS

    # Build lookup
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_seed[r.get("meta", {}).get("seed_source", "")].append(r)

    # Compute next available ID
    max_num = 0
    for r in records:
        rid = r.get("id", "")
        parts = rid.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            n = int(parts[1])
            if n > max_num:
                max_num = n
    next_id = max_num + 1

    total_generated = 0
    total_failed = 0
    new_records: list[dict] = []

    for seed_id, variant_idx in targets:
        seed = seeds.get(seed_id)
        if not seed:
            print(f"  SKIP: seed {seed_id} not found")
            total_failed += 1
            continue

        recs = by_seed.get(seed_id, [])
        target = next((r for r in recs if r.get("meta", {}).get("variant_index") == variant_idx), None)
        if not target:
            print(f"  SKIP: {seed_id} v{variant_idx} not found")
            total_failed += 1
            continue

        topic = seed.get("topic", "")
        domain = seed.get("domain_cluster", "")
        topic_abbr = TOPIC_ABBR.get(topic, "GEN")
        domain_abbr = DOMAIN_ABBR.get(domain, "GEN")
        assigned_id = f"{topic_abbr}-{domain_abbr}-{next_id}"
        next_id += 1

        print(f"[{seed_id}] Replacing v{variant_idx} (duplicate query) → {assigned_id}")

        new_rec = generate_replacement(
            seed, target, prompt_template, intents_list,
            assigned_id, recs,
        )

        if new_rec is None:
            print(f"  ✗ {seed_id} v{variant_idx}: failed")
            total_failed += 1
            continue

        # Remove the old record
        records = [r for r in records if r.get("id") != target["id"]]
        new_records.append(new_rec)
        total_generated += 1
        print(f"  ✓ {seed_id} v{variant_idx}: {assigned_id}")

    for rec in new_records:
        records.append(rec)

    save_records(records, EXPANDED_PATH)

    print(f"\n=== Summary ===")
    print(f"  Generated: {total_generated}")
    print(f"  Failed:    {total_failed}")

    final_count = sum(1 for _ in open(EXPANDED_PATH, encoding="utf-8") if _.strip())
    print(f"  Total:     {final_count}")

    # Report remaining cross-seed duplicates
    seen = {}
    dupes = 0
    for r in records:
        q = r.get("query", "")
        sid = r.get("meta", {}).get("seed_source", "")
        if q in seen:
            prev_sid = seen[q]
            if prev_sid != sid:
                dupes += 1
        else:
            seen[q] = sid
    if dupes:
        print(f"  Cross-seed duplicates remaining: {dupes}")
    else:
        print(f"  Cross-seed duplicates remaining: 0 ✓")

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
