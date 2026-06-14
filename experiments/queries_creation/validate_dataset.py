"""
Comprehensive validation and auto-fix tool for the evaluation dataset.

Usage:
    uv run python3 experiments/queries_creation/validate_dataset.py              # validate only
    uv run python3 experiments/queries_creation/validate_dataset.py --fix        # validate + auto-fix
    uv run python3 experiments/queries_creation/validate_dataset.py --fix --output custom.jsonl

Checks performed:
  1. Schema completeness       all required JSON fields present and non‑empty
  2. ID uniqueness             no duplicate IDs across the dataset
  3. k vs intents              k == len(intents) for IND; k == 1 for OOS
  4. OOS consistency           oos=True → empty intents + label; oos=False → intents + null label
  5. CLINC150 vocabulary       all declared intents are in the approved list
  6. Triple structure          each triple is [S, R, O] with non‑empty strings
  7. Relation types            formal_type ∈ valid set; inferred_by ∈ valid set
  8. Nesting graph             present iff k > 1; parent/child intents exist on the record
  9. Reversal cross‑refs       equivalent_to / reversal_pair_id point to real records
  10. Reversal pair Jaccard    Jaccard index ≥ 0.8 between paired records' unrefined triples
  11. Semantic duplicates      near‑duplicate queries flagged via text overlap
  12. Anchor coverage          grounding‑anchor terms appear in the query text
"""

import json
import os
import re
import sys
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "fixtures", "expanded", "expanded_queries.jsonl")
FIX_PATH = os.path.join(HERE, "fixtures", "expanded", "expanded_queries_fixed.jsonl")

# The five topics
VALID_TOPICS = {
    "natural_logic", "mathematical_calculation",
    "information_search", "human_psychology", "mixed",
}

# The eight CLINC150 domain clusters
VALID_DOMAINS = {
    "banking_finance", "travel", "home_auto", "food_dining",
    "healthcare", "communication", "info_utilities", "entertainment",
}

# The five relational types from the Tarski algebra
VALID_RELATION_TYPES = {"SYMMETRIC", "INVERSE", "ASYMMETRIC", "ORDERING", "FUNCTIONAL"}

# Inference rules defined in relation_taxonomy.yaml
VALID_INFERRED_BY = {
    "amr_arg0_arg1", "amr_passive", "preposition_directional",
    "lexical_mutuality", "lexical_equality", "lexical_possession",
    "lexical_kinship_symmetric", "lexical_kinship_asymmetric",
    "lexical_causal", "lexical_location", "lexical_comparative", "fallback",
}

# CLINC150 vocabulary (loaded from prompts file)
CLINC150_PATH = os.path.join(HERE, "fixtures", "prompts", "clinc150_intents.json")

# Nesting graph relation values
VALID_NESTING_RELATIONS = {"parallel", "sequential", "conditional", "optional"}

# ---------------------------------------------------------------------------
# Load CLINC150 vocabulary
# ---------------------------------------------------------------------------

def load_clinc150():
    """Return a set of all valid CLINC150 intent names."""
    if os.path.exists(CLINC150_PATH):
        with open(CLINC150_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return set(data)
            if isinstance(data, dict) and "intents" in data:
                return set(data["intents"])
    return set()


# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------

def load_records(path=DATA_PATH):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(records, path):
    """Write records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Individual checks
#   Each check is a function check_*(records, vocab)
#   Returns a list of (record_id, issue_severity, message) tuples.
#   Severity: "ERROR" (must fix) or "WARN" (review recommended).
# ---------------------------------------------------------------------------

def check_schema(records, vocab):
    """1. Schema completeness — every required field present and of correct type."""
    issues = []
    for rec in records:
        rid = rec.get("id", "?")
        errs = []

        # scalar fields
        if not isinstance(rec.get("query"), str) or not rec["query"].strip():
            errs.append("query missing or empty")
        if not isinstance(rec.get("summary"), str) or not rec["summary"].strip():
            errs.append("summary missing or empty")
        if not isinstance(rec.get("topic"), str) or rec["topic"] not in VALID_TOPICS:
            errs.append(f"topic '{rec.get('topic')}' invalid or missing")
        if not isinstance(rec.get("domain_cluster"), str) or rec["domain_cluster"] not in VALID_DOMAINS:
            errs.append(f"domain_cluster '{rec.get('domain_cluster')}' invalid or missing")
        if rec.get("vague") is not False:
            errs.append("vague must be false")
        if not isinstance(rec.get("k"), int) or rec["k"] < 0:
            errs.append(f"k={rec.get('k')} invalid or missing")
        if not isinstance(rec.get("oos"), bool):
            errs.append("oos must be a boolean")

        # intents
        if not isinstance(rec.get("expected_intents"), list):
            errs.append("expected_intents must be a list")

        # oos label
        if rec.get("oos") and rec.get("expected_oos_label") is None:
            errs.append("oos=true but expected_oos_label is null")
        if not rec.get("oos") and rec.get("expected_oos_label") is not None:
            errs.append("oos=false but expected_oos_label is not null")

        # anchors
        anchors = rec.get("expected_grounding_anchors", [])
        if not isinstance(anchors, list):
            errs.append("expected_grounding_anchors must be a list")

        # triples
        for key in ("expected_unrefined_triples", "expected_refined_triples"):
            triples = rec.get(key, [])
            if not isinstance(triples, list):
                errs.append(f"{key} must be a list")

        # relation_types
        rtypes = rec.get("relation_types", {})
        if not isinstance(rtypes, dict):
            errs.append("relation_types must be a dict")

        # nesting
        if not isinstance(rec.get("expected_nesting_graph"), list):
            errs.append("expected_nesting_graph must be a list")

        # triggers
        if not isinstance(rec.get("expected_association_triggers"), list):
            errs.append("expected_association_triggers must be a list")

        # meta
        meta = rec.get("meta", {})
        if not isinstance(meta, dict):
            errs.append("meta must be a dict")

        for msg in errs:
            issues.append((rid, "ERROR", msg))
    return issues


def check_id_uniqueness(records, vocab):
    """2. No duplicate IDs across the whole dataset."""
    issues = []
    seen = defaultdict(list)
    for i, rec in enumerate(records):
        rid = rec.get("id")
        seen[rid].append(i)
    for rid, indices in seen.items():
        if len(indices) > 1:
            issues.append((rid, "ERROR", f"duplicate ID appears {len(indices)} times (lines {indices})"))
    return issues


def check_k_vs_intents(records, vocab):
    """3. k == len(intents) for IND; k == 1 for OOS."""
    issues = []
    for rec in records:
        rid = rec["id"]
        k = rec.get("k", 0)
        intents = rec.get("expected_intents", [])
        oos = rec.get("oos", False)

        if oos:
            if k != 1:
                issues.append((rid, "WARN", f"OOS record has k={k} (expected 1)"))
            if intents:
                issues.append((rid, "ERROR", f"OOS record has {len(intents)} intents (expected 0)"))
        else:
            if len(intents) != k:
                issues.append((
                    rid, "ERROR",
                    f"k={k} but len(expected_intents)={len(intents)}",
                ))
    return issues


def check_clinc150_vocab(records, vocab):
    """5. All declared intents must be in the CLINC150 vocabulary."""
    issues = []
    if not vocab:
        return issues  # skip if vocabulary not loaded
    for rec in records:
        rid = rec["id"]
        for intent in rec.get("expected_intents", []):
            if intent not in vocab:
                issues.append((rid, "ERROR", f"intent '{intent}' not in CLINC150 vocabulary"))
    return issues


def check_triple_structure(records, vocab):
    """6. Each triple is [S, R, O] with all non-empty strings."""
    issues = []
    for rec in records:
        rid = rec["id"]
        for key in ("expected_unrefined_triples", "expected_refined_triples"):
            for t in rec.get(key, []):
                arr = t.get("triple")
                if not isinstance(arr, list) or len(arr) != 3:
                    issues.append((rid, "ERROR", f"{key}: triple not [S,R,O]"))
                    continue
                for j, label in enumerate(["S", "R", "O"]):
                    if not isinstance(arr[j], str) or not arr[j].strip():
                        issues.append((rid, "ERROR", f"{key}: {label} is empty/missing"))
                rtype = t.get("relation_type")
                if rtype not in VALID_RELATION_TYPES:
                    issues.append((rid, "ERROR", f"{key}: relation_type '{rtype}' invalid"))
    return issues


def check_relation_types(records, vocab):
    """7. relation_types entries have valid formal_type and inferred_by."""
    issues = []
    for rec in records:
        rid = rec["id"]
        for rel, info in rec.get("relation_types", {}).items():
            if isinstance(info, dict):
                ft = info.get("formal_type")
                if ft and ft not in VALID_RELATION_TYPES:
                    issues.append((rid, "ERROR", f"relation_types['{rel}'].formal_type='{ft}' invalid"))
                ib = info.get("inferred_by")
                if ib and ib not in VALID_INFERRED_BY:
                    issues.append((rid, "WARN", f"relation_types['{rel}'].inferred_by='{ib}' unknown"))
    return issues


def check_nesting_graph(records, vocab):
    """8. Nesting graph: present iff k>1; parent/child intents exist on the record."""
    issues = []
    for rec in records:
        rid = rec["id"]
        k = rec.get("k", 1)
        nesting = rec.get("expected_nesting_graph", [])
        intents = set(rec.get("expected_intents", []))
        oos = rec.get("oos", False)

        if oos:
            continue  # OOS records have no intents, so skip

        if k > 1 and not nesting:
            issues.append((rid, "WARN", f"k={k} > 1 but nesting_graph is empty"))
        if k == 1 and nesting:
            issues.append((rid, "WARN", f"k=1 but nesting_graph has {len(nesting)} entries"))

        for ng in nesting:
            parent = ng.get("parent")
            child = ng.get("child")
            relation = ng.get("relation")
            if relation and relation not in VALID_NESTING_RELATIONS:
                issues.append((rid, "WARN", f"nesting_graph relation '{relation}' not standard"))
            if parent and parent not in intents:
                issues.append((rid, "WARN", f"nesting_graph parent '{parent}' not in expected_intents"))
            if child and child not in intents:
                issues.append((rid, "WARN", f"nesting_graph child '{child}' not in expected_intents"))
    return issues


def build_id_map(records):
    """Return {id: record} dict for cross-reference lookups."""
    id_map = {}
    for rec in records:
        rid = rec.get("id")
        if rid:
            id_map[rid] = rec
    return id_map


def check_reversal_crossrefs(records, vocab):
    """9. equivalent_to and reversal_pair_id point to real existing records."""
    id_map = build_id_map(records)
    issues = []

    # check that all reversal references are consistent
    pair_members = defaultdict(list)  # reversal_pair_id → [(record_id, is_reversed)]

    for rec in records:
        rid = rec["id"]
        meta = rec.get("meta", {})
        eq = meta.get("equivalent_to")
        rpid = meta.get("reversal_pair_id")
        is_rev = meta.get("is_reversed_version", False)

        if eq is not None:
            if eq not in id_map:
                issues.append((rid, "ERROR", f"equivalent_to='{eq}' does not exist in dataset"))
            elif eq == rid:
                issues.append((rid, "ERROR", f"equivalent_to points to self"))

        if rpid is not None:
            # collect for cross-check
            pair_members[rpid].append((rid, is_rev))

    # cross-check reversal pairs: each pair should have exactly (A, B) or (B, A)
    for rpid, members in pair_members.items():
        if len(members) == 1:
            mrid, is_rev = members[0]
            issues.append((mrid, "ERROR", f"reversal_pair_id='{rpid}' only has one member (unpaired)"))
        elif len(members) > 2:
            issues.append((rpid, "ERROR", f"reversal_pair_id='{rpid}' has {len(members)} members (expected 2)"))

        # check A↔B symmetry: if A says equivalent_to=B, B should say equivalent_to=A
        if len(members) == 2:
            a_id, a_rev = members[0]
            b_id, b_rev = members[1]
            a_meta = id_map.get(a_id, {}).get("meta", {})
            b_meta = id_map.get(b_id, {}).get("meta", {})
            if a_id and b_id:
                if a_meta.get("equivalent_to") != b_id:
                    issues.append((a_id, "ERROR", f"member of pair '{rpid}' has equivalent_to='{a_meta.get('equivalent_to')}' but sibling is '{b_id}'"))
                if b_meta.get("equivalent_to") != a_id:
                    issues.append((b_id, "ERROR", f"member of pair '{rpid}' has equivalent_to='{b_meta.get('equivalent_to')}' but sibling is '{a_id}'"))
                # exactly one should be the reversed version
                if a_rev == b_rev:
                    issues.append((a_id, "WARN", f"pair '{rpid}': both members have is_reversed_version={a_rev} (expected one true, one false)"))

    return issues


def jaccard(triples_a, triples_b):
    """Jaccard similarity between two lists of triples (based on the [S,R,O] string tuples)."""
    set_a = {tuple(t["triple"]) for t in triples_a if "triple" in t}
    set_b = {tuple(t["triple"]) for t in triples_b if "triple" in t}
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def check_reversal_jaccard(records, vocab):
    """10. Jaccard index ≥ 0.8 between paired records' unrefined triples."""
    id_map = build_id_map(records)
    issues = []
    checked_pairs = set()

    for rec in records:
        rid = rec["id"]
        meta = rec.get("meta", {})
        eq = meta.get("equivalent_to")
        rpid = meta.get("reversal_pair_id")

        if eq and rpid:
            pair_key = tuple(sorted([rid, eq]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            peer = id_map.get(eq)
            if peer is None:
                continue

            j = jaccard(
                rec.get("expected_unrefined_triples", []),
                peer.get("expected_unrefined_triples", []),
            )
            if j < 0.8:
                issues.append((
                    rid, "WARN",
                    f"reversal pair Jaccard={j:.2f} (rids: {rid} ↔ {eq}, pair_id={rpid})",
                ))

    return issues


def text_overlap_ratio(a, b):
    """Fraction of tokens in a that also appear in b (case-insensitive)."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def check_semantic_duplicates(records, vocab):
    """11. Flag near-duplicate queries across different seeds (overlap > 90%)."""
    issues = []
    # Only compare records from different seed sources
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            sid_a = a.get("meta", {}).get("seed_source", "")
            sid_b = b.get("meta", {}).get("seed_source", "")
            # skip records from the same seed (dedup is expected within a seed)
            if sid_a and sid_b and sid_a == sid_b:
                continue

            qa, qb = a.get("query", ""), b.get("query", "")
            if not qa or not qb:
                continue

            ratio = text_overlap_ratio(qa, qb)
            if ratio > 0.90 and qa.lower() != qb.lower():
                issues.append((
                    a["id"], "WARN",
                    f"query overlaps {ratio:.0%} with '{b['id']}' (seed: {sid_a} vs {sid_b})",
                ))
    return issues


def check_anchor_coverage(records, vocab):
    """12. Grounding‑anchor terms should appear in the query text."""
    issues = []
    for rec in records:
        rid = rec["id"]
        query_lower = rec.get("query", "").lower()
        for anchor in rec.get("expected_grounding_anchors", []):
            term = anchor.get("term", "").strip()
            if term and term.lower() not in query_lower:
                issues.append((
                    rid, "WARN",
                    f"anchor term '{term}' not found in query text",
                ))
    return issues


# ---------------------------------------------------------------------------
# Auto-fix functions
# ---------------------------------------------------------------------------

def auto_fix_all(records):
    """Apply every deterministic auto-fix to all records. Return list of fixes."""
    id_map = build_id_map(records)
    fixes = []

    for rec in records:
        rid = rec["id"]

        # Fix 1: OOS → ensure empty intents and non-null label
        if rec.get("oos"):
            if rec.get("expected_intents"):
                rec["expected_intents"] = []
                fixes.append((rid, "expected_intents set to [] (OOS)"))
            if rec.get("expected_oos_label") is None:
                rec["expected_oos_label"] = "out_of_scope"
                fixes.append((rid, "expected_oos_label set to 'out_of_scope' (OOS)"))

        # Fix 2: Opposite for IND
        if not rec.get("oos"):
            if rec.get("expected_oos_label") is not None:
                rec["expected_oos_label"] = None
                fixes.append((rid, "expected_oos_label set to null (IND)"))

        # Fix 3: Repair reversal cross-references using seed_id+vindex heuristic
        meta = rec.get("meta", {})
        if not isinstance(meta, dict):
            continue

        eq = meta.get("equivalent_to")
        rpid = meta.get("reversal_pair_id")
        is_rev = meta.get("is_reversed_version", False)
        sid = meta.get("seed_source", "")
        vi = meta.get("variant_index")

        # Heuristic: if OOS seed (variant_index >= 9 or seed contains OOS seed patterns)
        # the reversal logic is different
        if rec.get("oos"):
            continue

        # For reversal records (v8, v9), fix equivalent_to if missing or wrong
        if vi in (7, 8):  # 0-indexed: v8=index 7, v9=index 8
            # Find the sibling in the same seed
            peer = None
            sibling_vi = 8 if vi == 7 else 7
            for other in records:
                if other["id"] == rid:
                    continue
                other_meta = other.get("meta", {})
                if other_meta.get("seed_source") == sid and other_meta.get("variant_index") == sibling_vi:
                    peer = other["id"]
                    break

            if peer and eq != peer:
                rec["meta"]["equivalent_to"] = peer
                rec["meta"]["reversal_pair_id"] = sid.rstrip("0123456789")  # rough pair ID
                fixes.append((rid, f"equivalent_to auto-fixed to '{peer}'"))

        # Fix 4: k vs intents mismatch
        k = rec.get("k", 1)
        intents = rec.get("expected_intents", [])
        if not rec.get("oos") and len(intents) != k:
            if len(intents) < k:
                # Can't auto-fill intents, just fix k to match
                rec["k"] = len(intents)
                fixes.append((rid, f"k adjusted from {k} to {len(intents)} to match intents"))
            else:
                # Truncate intents to k
                rec["expected_intents"] = intents[:k]
                fixes.append((rid, f"expected_intents truncated from {len(intents)} to {k}"))

        # Fix 5: Nesting graph — remove entries referencing missing intents
        intents_set = set(rec.get("expected_intents", []))
        nesting = rec.get("expected_nesting_graph", [])
        cleaned = [ng for ng in nesting if ng.get("parent") in intents_set and ng.get("child") in intents_set]
        if len(cleaned) != len(nesting):
            removed = len(nesting) - len(cleaned)
            rec["expected_nesting_graph"] = cleaned
            fixes.append((rid, f"removed {removed} nesting_graph entries referencing unknown intents"))

        # Fix 6: Add nesting graph for k>1 if missing
        if k > 1 and not rec["expected_nesting_graph"] and not rec.get("oos"):
            # Create a default parallel nesting
            ints = rec.get("expected_intents", [])
            for parent, child in zip(ints[:-1], ints[1:]):
                rec["expected_nesting_graph"].append({
                    "parent": parent,
                    "child": child,
                    "relation": "parallel",
                })
            fixes.append((rid, f"added default parallel nesting_graph for k={k}"))

    return fixes


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(all_issues, fixes, fix_mode):
    """Pretty-print a summary of all issues and fixes."""
    # Aggregate by severity
    errors = [(r, m) for (r, s, m) in all_issues if s == "ERROR"]
    warnings = [(r, m) for (r, s, m) in all_issues if s == "WARN"]

    # Count by check (using message prefix grouping)
    check_counts = defaultdict(int)
    for _, _, msg in all_issues:
        key = msg.split(":")[0].split(" ")[0]
        check_counts[key] += 1

    print("=" * 72)
    print("  DATASET VALIDATION REPORT")
    print("=" * 72)

    print(f"\nTotal records checked: {len(load_records())}")
    print(f"Total issues found:    {len(all_issues)}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    print(f"\n─── Issues by category ───")
    for key, count in sorted(check_counts.items(), key=lambda x: -x[1]):
        severity = "ERR" if any(s == "ERROR" for _, s, _ in all_issues if key in _) else "WRN"
        print(f"  [{severity}] {key}: {count}")

    if errors:
        print(f"\n─── Errors (must fix) ───")
        for rid, msg in errors[:30]:
            print(f"  [{rid}] {msg}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more errors")

    if warnings:
        print(f"\n─── Warnings (review recommended) ───")
        for rid, msg in warnings[:30]:
            print(f"  [{rid}] {msg}")
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more warnings")

    if fixes:
        print(f"\n─── Auto-fixes applied ({len(fixes)}) ───")
        for rid, action in fixes[:25]:
            print(f"  [{rid}] {action}")
        if len(fixes) > 25:
            print(f"  ... and {len(fixes) - 25} more fixes")

    print("\n" + "=" * 72)

    # Health score
    total = len(load_records())
    error_free = total - len(set(r for (r, s, _) in all_issues if s == "ERROR"))
    warn_free = total - len(set(r for (r, s, _) in all_issues if s == "WARN"))
    health = (error_free / total * 0.6 + warn_free / total * 0.4) * 100
    print(f"  Dataset health score: {health:.0f}%  "
          f"(records with 0 errors: {error_free}/{total}, "
          f"0 warnings: {warn_free}/{total})")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate and auto-fix the evaluation dataset."
    )
    parser.add_argument("--fix", action="store_true",
                        help="Apply auto-fixes to fixable issues")
    parser.add_argument("--output", default=FIX_PATH,
                        help=f"Output path for fixed dataset (default: {FIX_PATH})")
    parser.add_argument("--data", default=DATA_PATH,
                        help=f"Input dataset path (default: {DATA_PATH})")
    args = parser.parse_args()

    # Load
    records = load_records(args.data)
    vocab = load_clinc150()
    print(f"Loaded {len(records)} records from {args.data}")
    if vocab:
        print(f"CLINC150 vocabulary: {len(vocab)} intents loaded")
    else:
        print("WARNING: CLINC150 vocabulary not found — skipping vocabulary check")

    # Run all checks
    checks = [
        ("Schema completeness", check_schema),
        ("ID uniqueness", check_id_uniqueness),
        ("k vs intents", check_k_vs_intents),
        ("CLINC150 vocabulary", check_clinc150_vocab),
        ("Triple structure", check_triple_structure),
        ("Relation types", check_relation_types),
        ("Nesting graph validity", check_nesting_graph),
        ("Reversal cross-references", check_reversal_crossrefs),
        ("Reversal Jaccard", check_reversal_jaccard),
        ("Semantic duplicates", check_semantic_duplicates),
        ("Anchor coverage", check_anchor_coverage),
    ]

    all_issues = []
    for name, check_fn in checks:
        issues = check_fn(records, vocab)
        all_issues.extend(issues)

    # Auto-fix (before fixing, so we can report what was fixed)
    fixes = []
    if args.fix:
        fixes = auto_fix_all(records)
        # Re-run checks after fixes
        all_issues_after = []
        for name, check_fn in checks:
            issues = check_fn(records, vocab)
            all_issues_after.extend(issues)
        all_issues = all_issues_after

    # Report
    print_report(all_issues, fixes, args.fix)

    # Save fixed dataset
    if args.fix:
        save_records(records, args.output)
        print(f"\nFixed dataset saved to: {args.output}")

    # Exit code
    has_errors = any(s == "ERROR" for _, s, _ in all_issues)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
