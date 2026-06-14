#!/usr/bin/env python3
"""
Targeted regeneration: fill gaps left after dedup with anti-repeat + high temp.

Reads expanded_queries.jsonl, finds seeds with < 10 records, determines
which variant slots are missing, and regenerates only those variants with:
  - temperature 0.8 for same-k variants (forces diversity)
  - anti-repeat instructions listing previously generated queries
  - candidate intent suggestions for k+1 variants

Usage:
    python tests/regenerate_variants.py

Dependencies:
    python tests/dedup_expanded.py --fix   (run this first)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
PROMPTS = FIXTURES / "prompts"
SEEDS = FIXTURES / "seeds"
EXPANDED = FIXTURES / "expanded"

LLM_BASE = os.environ.get("LLM_API_BASE", "http://10.6.125.216:8080/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-32b-instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5
MAX_K = 5
TARGET_PER_SEED = 10

OOS_LABEL_EXAMPLES = [
    "hypothetical_time_travel", "third_party_app_question",
    "non_english_mix", "game_trivia", "philosophical_question",
    "personal_opinion", "future_prediction", "creative_writing",
    "unrelated_topic", "ambiguous_request",
]

TOPIC_ABBR = {
    "natural_logic": "NLG", "mathematical_calculation": "MATH",
    "information_search": "INFO", "human_psychology": "PSYCH", "mixed": "MIX",
}

DOMAIN_ABBR = {
    "banking_finance": "BANK", "travel": "TRAV", "home_auto": "HOME",
    "food_dining": "FOOD", "healthcare": "HLTH", "communication": "COMM",
    "info_utilities": "UTIL", "entertainment": "ENTR",
}

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


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_prompt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_seeds() -> dict[str, dict]:
    seeds = {}
    for fpath in sorted(SEEDS.glob("*_seeds.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    seeds[s["seed_id"]] = s
    return seeds


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
        if record.get("expected_oos_label") is None:
            errs.append("oos=true but expected_oos_label is null")
    else:
        if len(intents) != k:
            errs.append(f"len(expected_intents)={len(intents)} != k={k}")
        if record.get("expected_oos_label") is not None:
            errs.append("oos=false but expected_oos_label is non-null")
    anchors = record.get("expected_grounding_anchors", [])
    if not isinstance(anchors, list) or not anchors:
        errs.append("expected_grounding_anchors empty")
    else:
        for a in anchors:
            if a.get("type") not in ("entity", "concept"):
                errs.append(f"anchor type='{a.get('type')}' invalid")
    for triple_key in ("expected_unrefined_triples", "expected_refined_triples"):
        triples = record.get(triple_key, [])
        if not isinstance(triples, list) or not triples:
            errs.append(f"{triple_key} empty")
        else:
            for t in triples:
                arr = t.get("triple")
                if not isinstance(arr, list) or len(arr) != 3:
                    errs.append(f"{triple_key}.triple not [S,R,O]")
                else:
                    for elem in arr:
                        if not isinstance(elem, str):
                            errs.append(f"{triple_key}.triple element not string")
                rtype = t.get("relation_type")
                if rtype not in VALID_RELATION_TYPES:
                    errs.append(f"{triple_key}.relation_type='{rtype}' invalid")
    rtypes = record.get("relation_types", {})
    if not isinstance(rtypes, dict):
        errs.append("relation_types must be dict")
    else:
        for rel, info in rtypes.items():
            if isinstance(info, dict):
                ft = info.get("formal_type")
                if ft not in VALID_RELATION_TYPES:
                    errs.append(f"relation_types['{rel}'].formal_type='{ft}' invalid")
                ib = info.get("inferred_by")
                if ib and ib not in VALID_INFERRED_BY:
                    errs.append(f"relation_types['{rel}'].inferred_by='{ib}' invalid")
    nesting = record.get("expected_nesting_graph", [])
    if not isinstance(nesting, list):
        errs.append("expected_nesting_graph must be list")
    elif k > 1 and not nesting:
        errs.append(f"k={k} > 1 but nesting_graph empty")
    elif k == 1 and nesting:
        errs.append("k=1 but nesting_graph non-empty")
    triggers = record.get("expected_association_triggers", [])
    if not isinstance(triggers, list):
        errs.append("expected_association_triggers must be list")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        errs.append("meta missing/not dict")
    else:
        ss = meta.get("seed_source")
        if not isinstance(ss, str) or not ss.strip():
            errs.append("meta.seed_source missing/empty")
        vi = meta.get("variant_index")
        if vi is not None and not isinstance(vi, int):
            errs.append("meta.variant_index must be int or null")
        rpid = meta.get("reversal_pair_id")
        if rpid is not None and (not isinstance(rpid, str) or not rpid.strip()):
            errs.append("meta.reversal_pair_id invalid")
        rt = meta.get("reversal_type")
        if rt is not None and rt not in ("inverse_voice", "inverse_relation", "clause_order", "symmetric_direction"):
            errs.append(f"meta.reversal_type='{rt}' invalid")
        irv = meta.get("is_reversed_version")
        if irv is not None and not isinstance(irv, bool):
            errs.append("meta.is_reversed_version must be bool or null")
        eq = meta.get("equivalent_to")
        if eq is not None and (not isinstance(eq, str) or not eq.strip()):
            errs.append("meta.equivalent_to invalid")
    return errs


def auto_fix_record(record: dict) -> list[str]:
    if record.get("oos"):
        record["expected_intents"] = []
        if record.get("expected_oos_label") is None:
            record["expected_oos_label"] = "out_of_scope"
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
            rtype = t.get("relation_type", "")
            if rtype == "ASYMMETRICAL":
                t["relation_type"] = "ASYMMETRIC"
    return record_errors(record)


def build_spec(
    variant_idx: int,
    seed: dict,
    existing_queries: list[str] | None = None,
    candidate_intents: list[str] | None = None,
    variant_ids: dict[int, str] | None = None,
) -> str:
    seed_k = seed.get("k", 1)
    seed_intents = seed.get("intents", [])
    seed_oos = seed.get("oos", False)

    if seed_oos:
        return (
            f"Generate variant {variant_idx + 1} of {TARGET_PER_SEED} for this seed.\n"
            f"Type: OOS variant — same OOS concept, different wording.\n"
            f"- Rewrite the query so it has no matching CLINC150 intent\n"
            f"- Set oos=true and expected_intents=[]\n"
            f"- Set expected_oos_label to a short snake_case label describing the OOS category\n"
            f"- The query should be plausible and natural, just out-of-scope\n"
            f"- Ensure the query does not accidentally match any CLINC150 intent\n"
            f"- Examples of OOS labels: {', '.join(OOS_LABEL_EXAMPLES)}"
        )

    if variant_idx < 5:
        spec = (
            f"Generate variant {variant_idx + 1} of {TARGET_PER_SEED} for this seed.\n"
            f"Type: SAME K — same intents and k={seed_k}, different wording.\n"
            f"- Rewrite the query with DIFFERENT vocabulary and sentence structure\n"
            f"- Preserve the exact set of intents and k level (k={seed_k})\n"
            f"- Vary: formality, verb choice, question vs. statement, sentence complexity\n"
            f"- Do NOT change the underlying meaning or the required triples\n"
            f"- expected_intents: {json.dumps(seed_intents)}\n"
            f"- oos: false\n"
        )
        if existing_queries:
            spec += (
                f"\n"
                f"## ANTI-REPEAT: avoid these already-generated phrasings\n"
                f"The following variants already exist for this seed. "
                f"Your output MUST be DIFFERENT from all of them:\n"
            )
            for i, q in enumerate(existing_queries):
                spec += f"  Variant {i + 1}: \"{q[:100]}\"\n"
            spec += (
                f"\n"
                f"- Do NOT use the same opening phrase as any variant above\n"
                f"- Change: question vs statement, formality level, sentence structure\n"
                f"- Be creative — use different verbs, different sentence patterns\n"
            )
        return spec

    if variant_idx < 7:
        new_k = seed_k + 1
        if new_k > MAX_K:
            spec = (
                f"Generate variant {variant_idx + 1} of {TARGET_PER_SEED} for this seed.\n"
                f"Type: INTENT SUBSTITUTION — replace one intent with a compatible one.\n"
                f"- Keep k={seed_k} (maximum {MAX_K} allowed)\n"
                f"- Substitute ONE intent from the original set with a different compatible CLINC150 intent\n"
                f"- The new intent set should contain {seed_k - 1} of the original intents + 1 new intent\n"
                f"- The query must naturally express all {seed_k} intents\n"
                f"- oos: false\n"
            )
        else:
            spec = (
                f"Generate variant {variant_idx + 1} of {TARGET_PER_SEED} for this seed.\n"
                f"Type: K+1 — add a compatible secondary intent.\n"
                f"- Keep the original intent(s) intact: {json.dumps(seed_intents)}\n"
                f"- Add ONE additional CLINC150 intent that is semantically compatible\n"
                f"- The resulting query must be plausible and natural\n"
                f"- Update k to {new_k} and include both the original and new intent in expected_intents\n"
                f"- oos: false\n"
            )
        if candidate_intents:
            spec += (
                f"\n"
                f"## Candidate intents to consider\n"
                f"These CLINC150 intents are compatible with the {seed.get('domain_cluster', '')} domain. "
                f"Pick the one that best fits:\n"
            )
            for ci in candidate_intents:
                spec += f"  - {ci}\n"
        return spec

    if variant_idx < 8:
        return (
            f"Generate variant {variant_idx + 1} of {TARGET_PER_SEED} for this seed (OOS variant).\n"
            f"Type: OOS — Out of Scope.\n"
            f"- Rewrite the query so it has no matching CLINC150 intent\n"
            f"- Set oos=true and expected_intents=[]\n"
            f"- Set expected_oos_label to a short snake_case label describing the OOS category\n"
            f"- The query should be plausible and natural, just out-of-scope\n"
            f"- Ensure the query does not accidentally match any CLINC150 intent\n"
            f"- Examples of OOS labels: {', '.join(OOS_LABEL_EXAMPLES)}"
        )

    is_a = variant_idx == 8
    vid_map = variant_ids or {}
    pair_id = vid_map.get(8, f"REV-{seed.get('seed_id', 'unknown')}")
    if is_a:
        return (
            f"Generate variant 9 of {TARGET_PER_SEED} for this seed (reversal pair A).\n"
            f"Type: REVERSAL PAIR — first of a mirror pair.\n"
            f"- is_reversed_version: false\n"
            f"- reversal_pair_id: \"{pair_id}\"\n"
            f"- equivalent_to: null (the B variant will reference this ID)\n"
            f"- reversal_type: choose the most natural mechanism for this query\n"
            f"  (inverse_voice, inverse_relation, clause_order, or symmetric_direction)\n"
            f"- k={seed_k}, same intents as seed: {json.dumps(seed_intents)}\n"
            f"- oos: false"
        )
    else:
        rev_a_id = vid_map.get(8, pair_id)
        return (
            f"Generate variant 10 of {TARGET_PER_SEED} for this seed (reversal pair B).\n"
            f"Type: REVERSAL PAIR — second of a mirror pair (reversed version).\n"
            f"- is_reversed_version: true\n"
            f"- reversal_pair_id: \"{pair_id}\"\n"
            f"- equivalent_to: \"{rev_a_id}\" (must match variant A's ID exactly)\n"
            f"- reversal_type must match variant A's value\n"
            f"- The triples after refinement must be isomorphic with variant A's\n"
            f"- k={seed_k}, same intents as seed: {json.dumps(seed_intents)}\n"
            f"- oos: false"
        )


def load_domain_candidates() -> dict[str, list[str]]:
    path = HERE / "data" / "domain_intent_map.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("  WARN: domain_intent_map.json not found — no candidate suggestions")
    return {}


def main() -> int:
    (FIXTURES / "data").mkdir(parents=True, exist_ok=True)
    EXPANDED.mkdir(parents=True, exist_ok=True)

    expanded_path = EXPANDED / "expanded_queries.jsonl"
    done_path = EXPANDED / "regenerated_done.json"

    if not expanded_path.exists():
        print(f"No expanded queries found at {expanded_path}")
        return 0

    print("Loading records, seeds, prompts ...")
    records: list[dict] = []
    with open(expanded_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    seeds = load_seeds()
    prompt_template = load_prompt(PROMPTS / "expand_single.md")
    intents_list = load_json(PROMPTS / "clinc150_intents.json")
    domain_candidates = load_domain_candidates()

    # Group records by seed
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        sid = rec.get("meta", {}).get("seed_source", "__unknown__")
        by_seed[sid].append(rec)

    # Find seeds with gaps
    done: set[str] = set()
    if done_path.exists():
        done = set(json.load(open(done_path, encoding="utf-8")))

    gaps = {sid: recs for sid, recs in by_seed.items()
            if len(recs) < TARGET_PER_SEED and sid not in done and sid in seeds}

    # Also check seeds not in the file at all (should not happen, but be safe)
    all_seed_ids = set(seeds.keys())
    in_file = set(by_seed.keys())
    missing_seeds = all_seed_ids - in_file
    for sid in missing_seeds:
        if sid not in done:
            gaps[sid] = []

    if not gaps:
        print("No gaps found — all seeds have 10 records.")
        return 0

    print(f"Found {len(gaps)} seed(s) with gaps:")
    for sid, recs in sorted(gaps.items()):
        existing = len(recs)
        needed = TARGET_PER_SEED - existing
        print(f"  {sid}: {existing}/{TARGET_PER_SEED} (need {needed} more)")

    # Determine max existing ID number
    max_num = 0
    for rec in records:
        rid = rec.get("id", "")
        parts = rid.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            num = int(parts[1])
            if num > max_num:
                max_num = num
    next_id_num = max_num + 1

    print(f"\nNext available ID number: {next_id_num}")
    print()

    total_generated = 0
    total_failed = 0

    for seed_id in sorted(gaps):
        seed = seeds[seed_id]
        existing_recs = gaps[seed_id]
        existing_count = len(existing_recs)
        needed = TARGET_PER_SEED - existing_count

        # Determine which variant slots are filled vs missing
        filled_slots = set()
        for rec in existing_recs:
            vi = rec.get("meta", {}).get("variant_index")
            if vi is not None:
                filled_slots.add(vi)

        # Determine missing variant indices
        missing_slots = []
        for v in range(TARGET_PER_SEED):
            if v not in filled_slots:
                missing_slots.append(v)

        # If we don't have variant_index info, infer from count
        if not missing_slots:
            missing_slots = list(range(existing_count, TARGET_PER_SEED))

        # Collect existing queries for anti-repeat
        existing_queries = [r.get("query", "") for r in existing_recs if r.get("query")]

        topic = seed.get("topic", "")
        domain = seed.get("domain_cluster", "")
        topic_abbr = TOPIC_ABBR.get(topic, "GEN")
        domain_abbr = DOMAIN_ABBR.get(domain, "GEN")

        # Get domain candidate intents for k+1 variants
        candidates = domain_candidates.get(domain, [])

        print(f"[{seed_id}] Filling {len(missing_slots)} slot(s): {missing_slots}")
        sys.stdout.flush()

        new_records: list[dict] = []
        seed_failed = 0
        existing_queries = [r.get("query", "") for r in existing_recs if r.get("query")]

        seed_oos = seed.get("oos", False)

        # Pre-compute IDs for all missing variants so reversal pairs can reference each other
        variant_ids: dict[int, str] = {}
        temp_for_v: dict[int, float] = {}
        for v in missing_slots:
            if v < 5 and not seed_oos:
                temp_for_v[v] = 0.8
            elif seed_oos or v >= 7:
                temp_for_v[v] = 0.4
            else:
                temp_for_v[v] = 0.6
            variant_ids[v] = f"{topic_abbr}-{domain_abbr}-{next_id_num + len(variant_ids)}"

        for v in missing_slots:
            temp = temp_for_v[v]
            assigned_id = variant_ids[v]
            next_id_num += 1

            spec = build_spec(v, seed, existing_queries, candidates, variant_ids)
            prompt = (
                prompt_template
                .replace("{seed_json}", json.dumps(seed, indent=2))
                .replace("{intent_list_formatted}", format_intent_list(intents_list))
                .replace("{specification}", spec)
                .replace("{assigned_id}", assigned_id)
            )

            record = None
            for attempt in range(1, 4):
                try:
                    raw = call_llm(prompt, temperature=temp)
                    result = _extract_json(raw)
                    if not isinstance(result.get("id"), str):
                        result["id"] = assigned_id
                    if not result.get("query"):
                        result["query"] = seed.get("query", "")
                    result.setdefault("meta", {})
                    result["meta"]["seed_source"] = seed_id
                    result["meta"]["variant_index"] = v
                    if not result.get("oos"):
                        if not result.get("expected_intents"):
                            result["expected_intents"] = seed.get("intents", [])
                        if result.get("k", 0) > MAX_K:
                            result["k"] = MAX_K
                            result["expected_intents"] = result["expected_intents"][:MAX_K]
                except Exception as e:
                    if attempt < 3:
                        continue
                    print(f"      variant v{v}: LLM error: {e}")
                    record = None
                    break

                errs = record_errors(result)
                if not errs:
                    result["topic"] = topic
                    result["domain_cluster"] = domain
                    record = result
                    break

                errs = auto_fix_record(result)
                if not errs:
                    result["topic"] = topic
                    result["domain_cluster"] = domain
                    record = result
                    break

                if attempt < 3:
                    continue
                else:
                    print(f"      variant v{v}: FAILED — {'; '.join(errs[:3])}")
                    record = None
                    break

            if record is None:
                total_failed += 1
                seed_failed += 1
                continue

            new_records.append(record)
            existing_queries.append(record.get("query", ""))
            print(f"      variant v{v} done — {record['id']} ({temp=})")
            sys.stdout.flush()

        # Save new records
        for rec in new_records:
            with open(expanded_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_generated += len(new_records)
        if seed_failed == 0:
            done.add(seed_id)
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump(sorted(done), f)

        print(f"  → {seed_id}: added {len(new_records)} record(s)")
        sys.stdout.flush()
        print()

    print("=== Summary ===")
    print(f"  Generated: {total_generated}")
    print(f"  Failed:    {total_failed}")

    if expanded_path.exists():
        final_count = sum(1 for _ in open(expanded_path, encoding="utf-8") if _.strip())
        print(f"  Total:     {final_count}")

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
