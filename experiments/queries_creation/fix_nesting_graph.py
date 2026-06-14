#!/usr/bin/env python3
"""
Post-processor: re-evaluate nesting_graph entries using a rule matrix.

Reads expanded_queries.jsonl, applies known intent-pair rules, and writes
the corrected version back in-place. Only modifies records where k > 1 and
the current relation is "parallel" (auto-fill default). LLM-generated
relations like "prerequisite", "subgoal", "alternative" are preserved.

Usage:
    python tests/fix_nesting_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANDED_PATH = HERE / "fixtures" / "expanded" / "expanded_queries.jsonl"

NESTING_RULES: dict[tuple[str, str], str] = {
    # ── prerequisite:  intent B should be resolved / checked before A ──
    ("book_hotel", "flight_status"): "prerequisite",
    ("book_hotel", "travel_alert"): "prerequisite",
    ("book_hotel", "travel_notification"): "prerequisite",
    ("book_flight", "travel_alert"): "prerequisite",
    ("book_flight", "flight_status"): "prerequisite",
    ("book_flight", "travel_notification"): "prerequisite",
    ("book_flight", "carry_on"): "prerequisite",
    ("pay_bill", "balance"): "prerequisite",
    ("pay_bill", "bill_due"): "prerequisite",
    ("pay_bill", "bill_balance"): "prerequisite",
    ("transfer", "balance"): "prerequisite",
    ("transfer", "transactions"): "prerequisite",
    ("schedule_meeting", "calendar"): "prerequisite",
    ("schedule_meeting", "calendar_update"): "prerequisite",
    ("schedule_maintenance", "calendar"): "prerequisite",
    ("order", "order_status"): "prerequisite",
    ("order", "spending_history"): "prerequisite",
    ("pto_request", "pto_balance"): "prerequisite",
    ("pto_request", "pto_used"): "prerequisite",
    ("insurance_change", "insurance"): "prerequisite",
    ("credit_limit_change", "credit_limit"): "prerequisite",
    ("credit_limit_change", "credit_score"): "prerequisite",
    ("improve_credit_score", "credit_score"): "prerequisite",
    ("restaurant_reservation", "restaurant_reviews"): "prerequisite",
    ("cancel_reservation", "restaurant_reservation"): "prerequisite",
    ("travel_notification", "book_flight"): "prerequisite",
    ("travel_notification", "book_hotel"): "prerequisite",

    # ── subgoal:  intent B is a step toward completing A ──
    ("book_hotel", "directions"): "subgoal",
    ("book_hotel", "travel_suggestion"): "subgoal",
    ("book_flight", "car_rental"): "subgoal",
    ("book_flight", "uber"): "subgoal",
    ("book_flight", "directions"): "subgoal",
    ("order", "shopping_list"): "subgoal",
    ("order", "ingredients_list"): "subgoal",
    ("schedule_maintenance", "last_maintenance"): "subgoal",
    ("schedule_maintenance", "oil_change_when"): "subgoal",
    ("restaurant_reservation", "restaurant_suggestion"): "subgoal",
    ("meal_suggestion", "nutrition_info"): "subgoal",
    ("meal_suggestion", "calories"): "subgoal",
    ("recipe", "ingredient_substitution"): "subgoal",
    ("recipe", "ingredients_list"): "subgoal",
    ("recipe", "cook_time"): "subgoal",
    ("reminder", "calendar_update"): "subgoal",
    ("play_music", "update_playlist"): "subgoal",
    ("todo_list", "todo_list_update"): "subgoal",
    ("shopping_list", "shopping_list_update"): "subgoal",

    # ── alternative:  either intent satisfies the same user need ──
    ("book_hotel", "car_rental"): "alternative",
    ("restaurant_reservation", "meal_suggestion"): "alternative",
    ("cook_time", "order"): "alternative",
    ("play_music", "next_song"): "alternative",
    ("directions", "current_location"): "alternative",
    ("timer", "alarm"): "alternative",
    ("todo_list", "shopping_list"): "alternative",
    ("schedule_meeting", "reminder"): "alternative",
}


def apply_nesting_rules(graph: list[dict]) -> list[dict]:
    out = []
    for edge in graph:
        parent = edge.get("parent", "")
        child = edge.get("child", "")
        current_rel = edge.get("relation", "parallel")

        rule_key = (parent, child)
        reverse_key = (child, parent)

        if current_rel == "parallel":
            if rule_key in NESTING_RULES:
                out.append({"parent": parent, "child": child, "relation": NESTING_RULES[rule_key]})
            elif reverse_key in NESTING_RULES:
                rel = NESTING_RULES[reverse_key]
                rel = {"prerequisite": "subgoal", "subgoal": "prerequisite",
                       "alternative": "alternative"}.get(rel, "parallel")
                out.append({"parent": parent, "child": child, "relation": rel})
            else:
                out.append(edge)
        else:
            out.append(edge)

    return out


def main() -> int:
    if not EXPANDED_PATH.exists():
        print(f"File not found: {EXPANDED_PATH}")
        return 1

    records: list[dict] = []
    with open(EXPANDED_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN line {i}: invalid JSON — {e}")

    changes = 0
    total_k_gt_1 = 0
    for rec in records:
        k = rec.get("k", 0)
        graph = rec.get("expected_nesting_graph", [])
        if k > 1 and graph:
            total_k_gt_1 += 1
            old = list(graph)
            new = apply_nesting_rules(graph)
            if old != new:
                rec["expected_nesting_graph"] = new
                changes += 1

    with open(EXPANDED_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Records with k>1 and nesting_graph: {total_k_gt_1}")
    print(f"Nesting graphs modified: {changes}")

    if changes:
        print(f"\nRules applied:")
        counts: dict[str, int] = {}
        for rec in records:
            for edge in rec.get("expected_nesting_graph", []):
                rel = edge.get("relation", "")
                if rel != "parallel":
                    counts[rel] = counts.get(rel, 0) + 1
        for rel, cnt in sorted(counts.items()):
            print(f"  {rel}: {cnt}")
    else:
        print("No changes needed — all nesting graphs already correct or empty.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
