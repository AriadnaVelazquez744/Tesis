from __future__ import annotations

from typing import Any, Dict, List


def infer_relational_type(predicate: str, amr_edges: List[Any]) -> str:
    pred_clean = predicate.lower()

    if "-of" in pred_clean:
        return "INVERSE"

    symmetric_indicators = [
        "equals", "same", "equivalent", "spouse", "brother",
        "sibling", "mutually",
    ]
    if any(ind in pred_clean for ind in symmetric_indicators):
        return "SYMMETRIC"

    if "arg" in pred_clean:
        return "ASYMMETRIC"

    return "ASYMMETRIC"


def transform_logic(cao: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    print("[LOGIC TRANSFORMER] Initiating graph refinement and Tarski typing...")

    state = cao.get("state", {})
    unrefined = state.get("unrefined_triples", [])
    anchors = state.get("grounding_anchors", [])
    amr_ast = cao.get("meta", {}).get("amr_ast", {})
    amr_edges = amr_ast.get("edges", [])

    print(f"[LOGIC TRANSFORMER] Received {len(unrefined)} unrefined triples, "
          f"{len(anchors)} grounding anchors (threshold={threshold})")

    knowledge_anchors = [
        anchor["term"].lower() for anchor in anchors if anchor.get("score", 0) >= threshold
    ]
    print(f"[LOGIC TRANSFORMER] Knowledge anchors after threshold: {knowledge_anchors}")

    refined_triples: List[Dict[str, Any]] = []
    noise_concepts = ["i", "want", "information", "thing", "amr-unknown", "possible-01"]

    noise_pruned = 0
    anchor_pruned = 0

    for i, triple in enumerate(unrefined):
        sub_node = triple.get("subject", {})
        obj_node = triple.get("object", {})
        predicate = triple.get("predicate", "rel")

        sub_val = str(sub_node.get("value", "")).lower()
        obj_val = str(obj_node.get("value", "")).lower()

        sub_hit = any(sub_val in a or a in sub_val for a in knowledge_anchors)
        obj_hit = any(obj_val in a or a in obj_val for a in knowledge_anchors)

        if (sub_val in noise_concepts) or (obj_val in noise_concepts):
            print(f"[LOGIC TRANSFORMER]  Prune triple #{i}: noise concept "
                  f"({sub_val} -> {predicate} -> {obj_val})")
            noise_pruned += 1
            continue

        if not (sub_hit or obj_hit):
            print(f"[LOGIC TRANSFORMER]  Prune triple #{i}: no anchor match "
                  f"({sub_val} -> {predicate} -> {obj_val}) "
                  f"[sub_hit={sub_hit} obj_hit={obj_hit}]")
            anchor_pruned += 1
            continue

        rel_type = infer_relational_type(predicate, amr_edges)
        print(f"[LOGIC TRANSFORMER]  Keep triple #{i}: "
              f"({sub_val} -> {predicate} -> {obj_val}) "
              f"type={rel_type}")

        final_sub = sub_node.get("value", "")
        final_pred = predicate
        final_obj = obj_node.get("value", "")

        if rel_type == "INVERSE":
            final_sub = obj_node.get("value", "")
            final_pred = f"{predicate.replace('-of', '')}_inverse"
            final_obj = sub_node.get("value", "")

            refined_triples.append({
                "subject": final_sub,
                "predicate": final_pred,
                "object": final_obj,
                "tarski_type": "INVERSE",
            })
            refined_triples.append({
                "subject": final_obj,
                "predicate": f"{final_pred}_mirror",
                "object": final_sub,
                "tarski_type": "INVERSE",
            })
            continue

        if rel_type == "SYMMETRIC":
            refined_triples.append({
                "subject": final_sub,
                "predicate": final_pred,
                "object": final_obj,
                "tarski_type": "SYMMETRIC",
            })
            continue

        refined_triples.append({
            "subject": final_sub,
            "predicate": final_pred,
            "object": final_obj,
            "tarski_type": "ASYMMETRIC",
        })

    state["triples"] = refined_triples
    cao.setdefault("meta", {})
    cao["meta"]["logic_transformer_ok"] = True
    cao["meta"]["logic_threshold_applied"] = threshold

    print(f"[LOGIC TRANSFORMER] Complete. "
          f"Pruned: {noise_pruned} noise + {anchor_pruned} no-anchor = {noise_pruned + anchor_pruned} total. "
          f"Emitted {len(refined_triples)} refined knowledge triples for VSA.")
    return cao
