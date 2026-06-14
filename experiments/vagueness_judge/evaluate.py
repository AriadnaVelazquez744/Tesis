#!/usr/bin/env python3
"""Compute evaluation metrics for generated conversation records.

All 9 metrics are fully automated following the Tell Me More! paper methodology.
M3 is computed as user_details_in_summary / total_user_details (details actually discussed).
M5 is computed via heuristics (option length + relevance to target detail).
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from config import LABELLER_PATH, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

# Lazy-load sentence-transformers for semantic matching (M2, M3, M5)
# Falls back to simple keyword overlap if not available.
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Using sentence-transformers for semantic matching")
        return _embedder
    except Exception:
        logger.warning("sentence-transformers not available, using keyword matching fallback")
        return None


def semantic_similarity(a: str, b: str) -> float:
    emb = _get_embedder()
    if emb is not None:
        from sentence_transformers import util
        emb_a = emb.encode(a, convert_to_tensor=True)
        emb_b = emb.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b))
    else:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))


def best_match(query: str, descriptions: List[str], threshold: float = 0.35) -> int:
    if not descriptions:
        return -1
    scores = [semantic_similarity(query, d) for d in descriptions]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx if scores[best_idx] >= threshold else -1


DISMISS_PATTERNS = [
    "i don't need to specify",
    "let's move on",
    "i don't know",
    "not sure",
    "doesn't matter",
]


def _is_dismissive_user_response(text: str) -> bool:
    text_lower = text.lower().strip()
    for pat in DISMISS_PATTERNS:
        if pat in text_lower:
            return True
    return False


def compute_metrics(
    interactions: List[Dict[str, Any]],
    labeller: List[Dict[str, Any]],
) -> Dict[str, Any]:
    task_num = len(interactions)
    if task_num == 0:
        return {"error": "no tasks", "num_examples": 0}

    # Build lookup by task
    labeller_by_task = {entry["task"]: entry for entry in labeller}

    results = {}

    # --- M1: Vagueness Judgment Accuracy ---
    align_cnt = 0
    for rec in interactions:
        lab = labeller_by_task.get(rec["task"], {})
        if lab.get("user_vague") == rec.get("vague"):
            align_cnt += 1
    results["vagueness_judgment_accuracy"] = round(align_cnt / task_num, 4) if task_num else 0.0

    # --- M2: Missing Details Recovery Rate (per importance) ---
    mdrr = {"1": {"rate": 0.0, "cnt": 0}, "2": {"rate": 0.0, "cnt": 0}, "3": {"rate": 0.0, "cnt": 0}}
    total_recover = {"rate": 0.0, "cnt": 0}

    for rec in interactions:
        lab = labeller_by_task.get(rec["task"], {})
        if not rec.get("vague") or not lab.get("user_vague"):
            continue

        # Collect ground truth details
        human_info = []
        human_info.extend(lab.get("user_approve", []))
        human_info.extend(lab.get("user_rectify", []))
        human_info.extend(lab.get("user_add", []))
        descriptions = [info["description"] for info in human_info]

        if not descriptions:
            continue

        # Match model queries to ground truth details
        hit = {i: False for i in range(len(descriptions))}
        qol = rec.get("query_options_list", [])
        for turn_info in qol:
            for qo in turn_info:
                query = qo.get("query", "")
                if not query:
                    continue
                idx = best_match(query, descriptions)
                if idx >= 0:
                    hit[idx] = True

        # Per-importance stats
        task_imp = {}
        for i, info in enumerate(human_info):
            imp = str(info.get("importance", "2"))
            if imp not in task_imp:
                task_imp[imp] = {"hit": 0, "total": 0}
            task_imp[imp]["total"] += 1
            if hit[i]:
                task_imp[imp]["hit"] += 1

        for imp, vals in task_imp.items():
            if imp in mdrr:
                mdrr[imp]["rate"] += vals["hit"] / max(vals["total"], 1)
                mdrr[imp]["cnt"] += 1

        total_recover["rate"] += sum(1 for h in hit.values() if h) / max(len(hit), 1)
        total_recover["cnt"] += 1

    mdrr_out = {}
    for imp, vals in mdrr.items():
        mdrr_out[imp] = round(vals["rate"] / max(vals["cnt"], 1), 4)
    mdrr_out["total_recover_rate"] = round(
        total_recover["rate"] / max(total_recover["cnt"], 1), 4
    )
    results["missing_details_recover_rate"] = mdrr_out

    # --- M3: Summary Intention Coverage (paper-style, automated) ---
    # Paper metric: user_details_in_summary / total_user_details
    # total_user_details = ground truth details the user actually discussed
    # user_details_in_summary = of those, how many appear in the summary
    total_discussed = 0
    total_covered = 0
    for rec in interactions:
        lab = labeller_by_task.get(rec["task"], {})
        if not rec.get("vague") or not lab.get("user_vague"):
            continue

        summary = rec.get("summary", "")
        if not summary or summary == "No summary generated.":
            continue

        human_info = []
        human_info.extend(lab.get("user_approve", []))
        human_info.extend(lab.get("user_rectify", []))
        human_info.extend(lab.get("user_add", []))
        descriptions = [info["description"] for info in human_info]
        if not descriptions:
            continue

        actions = rec.get("actions", [])
        qol = rec.get("query_options_list", [])
        discussed_indices = set()
        qol_idx = 0
        for action in actions:
            if action.get("role") != "user":
                continue
            content = action.get("content", "")
            if _is_dismissive_user_response(content):
                qol_idx += 1
                continue
            # Match this user response to its preceding inquiry
            if qol_idx < len(qol):
                turn_info = qol[qol_idx]
                qol_idx += 1
                for qo in turn_info:
                    query = qo.get("query", "")
                    if not query:
                        continue
                    idx = best_match(query, descriptions)
                    if idx >= 0:
                        discussed_indices.add(idx)

        # Also advance qol_idx for assistant-only turns (no user response)
        # Already handled by the for loop — only user actions consume an inquiry

        if not discussed_indices:
            continue

        total_discussed += len(discussed_indices)
        for idx in discussed_indices:
            if semantic_similarity(summary, descriptions[idx]) > 0.35:
                total_covered += 1

    m3_rate = round(total_covered / max(total_discussed, 1), 4)
    results["summary_intention_coverage_rate"] = m3_rate

    # --- M4: Options Presenting Rate ---
    vague_cnt = 0
    options_presenting = 0.0
    for rec in interactions:
        if not rec.get("vague"):
            continue
        vague_cnt += 1
        num_details = 0
        num_with_options = 0
        for turn_info in rec.get("query_options_list", []):
            for qo in turn_info:
                num_details += 1
                if len(qo.get("options", [])) > 0:
                    num_with_options += 1
        if num_details > 0:
            options_presenting += num_with_options / num_details
    results["options_presenting_rate"] = round(
        options_presenting / max(vague_cnt, 1), 4
    )

    # --- M5: Options Reasonable Rate (paper-style, automated) ---
    # Paper metric: 1 - sum(inappropriate_options) / sum(total_options)
    # Estimates inappropriateness per option using heuristics:
    #   - overly long options (probable meta-text / hallucinations)
    #   - options that don't semantically relate to any ground truth detail
    m5_total_options = 0
    m5_bad_options = 0
    MAX_OPTION_CHARS = 80
    for rec in interactions:
        lab = labeller_by_task.get(rec["task"], {})
        if not rec.get("vague") or not lab.get("user_vague"):
            continue

        human_info = []
        human_info.extend(lab.get("user_approve", []))
        human_info.extend(lab.get("user_rectify", []))
        human_info.extend(lab.get("user_add", []))
        descriptions = [info["description"] for info in human_info]

        qol = rec.get("query_options_list", [])
        for turn_info in qol:
            for qo in turn_info:
                options = qo.get("options", [])
                query = qo.get("query", "")

                # Find which detail this inquiry targets
                target_idx = best_match(query, descriptions) if descriptions else -1

                for opt in options:
                    m5_total_options += 1
                    if len(opt) > MAX_OPTION_CHARS:
                        m5_bad_options += 1
                        continue
                    # If we know the target detail, check if option is relevant
                    if target_idx >= 0:
                        desc = descriptions[target_idx]
                        if semantic_similarity(opt, desc) < 0.2:
                            m5_bad_options += 1

    m5_rate = round(1 - (m5_bad_options / max(m5_total_options, 1)), 4) if m5_total_options > 0 else None
    results["options_reasonable_rate"] = m5_rate

    # --- M6: Average Provided Options ---
    vague_cnt = 0
    avg_options = 0.0
    for rec in interactions:
        if not rec.get("vague"):
            continue
        vague_cnt += 1
        num_details = 0
        num_options = 0
        for turn_info in rec.get("query_options_list", []):
            for qo in turn_info:
                num_details += 1
                num_options += len(qo.get("options", []))
        if num_details > 0:
            avg_options += num_options / num_details
    results["average_provided_options"] = round(
        avg_options / max(vague_cnt, 1), 4
    )

    # --- M7: Average Inquired Missing Details Per Round ---
    vague_cnt = 0
    avg_per_round = 0.0
    for rec in interactions:
        if not rec.get("vague"):
            continue
        vague_cnt += 1
        qol = rec.get("query_options_list", [])
        num_turns = max(len(qol), 1)
        num_queries = sum(len(ti) for ti in qol)
        avg_per_round += num_queries / num_turns
    results["average_inquired_missing_details_per_round"] = round(
        avg_per_round / max(vague_cnt, 1), 4
    )

    # --- M8: Average Conversation Rounds ---
    total_rounds = 0.0
    for rec in interactions:
        actions = rec.get("actions", [])
        num_assistant = sum(1 for a in actions if a.get("role") == "assistant")
        total_rounds += num_assistant
    results["average_conversation_rounds"] = round(total_rounds / task_num, 4)

    # --- M9: Average Inquired Missing Details ---
    vague_cnt = 0
    avg_total_queries = 0.0
    for rec in interactions:
        if not rec.get("vague"):
            continue
        vague_cnt += 1
        qol = rec.get("query_options_list", [])
        avg_total_queries += sum(len(ti) for ti in qol)
    results["average_inquired_missing_details"] = round(
        avg_total_queries / max(vague_cnt, 1), 4
    )

    results["num_examples"] = task_num
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--model", type=str, default="all",
                        help="Model key or 'all'")
    parser.add_argument("--input_dir", type=str, default=str(OUTPUTS_DIR),
                        help="Directory containing _interactions.jsonl files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input dir not found: %s", input_dir)
        return

    # Load labeller
    labeller = []
    with open(LABELLER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            labeller.append(json.loads(line))
    logger.info("Loaded %d labeller entries", len(labeller))

    if args.model == "all":
        pattern = "*_interactions.jsonl"
        model_keys = sorted(set(
            p.name.replace("_interactions.jsonl", "") for p in input_dir.glob(pattern)
        ))
    else:
        model_keys = [args.model]

    for model_key in model_keys:
        in_path = input_dir / f"{model_key}_interactions.jsonl"
        if not in_path.exists():
            logger.warning("No interactions file for %s at %s", model_key, in_path)
            continue

        interactions = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                interactions.append(json.loads(line))

        logger.info("Evaluating %s: %d records", model_key, len(interactions))

        metrics = compute_metrics(interactions, labeller)
        metrics["model"] = model_key

        out_path = OUTPUTS_DIR / f"{model_key}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved metrics to %s", out_path)

        # Print summary
        print(f"\n--- {model_key} ---")
        for k, v in metrics.items():
            if k in ("model", "num_examples"):
                continue
            if isinstance(v, dict):
                print(f"  {k}: {json.dumps(v)}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
