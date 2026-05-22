#!/usr/bin/env python3
"""Compare our best model against the paper's reported results."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
OUR_DIR = BASE / "src" / "experiments" / "outputs"
PAPER_DIR = BASE / "Tell_Me_More-master" / "data" / "user_interaction_records" / "metrics"

PAPER_MODELS = [
    ("mistral-7b-instruct-v0.2-hf", "Mistral-7B v0.2"),
    ("Llama-2-7b-chat-hf", "LLaMA-2-7B-Chat"),
    ("gpt4", "GPT-4"),
    ("mistral-interact", "Mistral-Interact"),
]

METRICS = [
    ("vagueness_judgment_accuracy", "Vagueness Accuracy", True),
    ("missing_details_recover_rate", "Detail Recovery", True),
    ("summary_intention_coverage_rate", "Summary Coverage", True),
    ("options_presenting_rate", "Options Presenting", True),
    ("options_reasonable_rate", "Options Reasonable", True),
    ("average_provided_options", "Avg Options", False),
    ("average_inquired_missing_details_per_round", "Avg Inq / Round", False),
    ("average_conversation_rounds", "Avg Rounds", False),
    ("average_inquired_missing_details", "Avg Inq Details", False),
]


def load_data():
    # Paper merged
    with open(PAPER_DIR / "metric_mix_merged.json") as f:
        merged = json.load(f)
    paper = {}
    for mk in [m[0] for m in PAPER_MODELS]:
        paper[mk] = {}
        for metric_key, _, _ in METRICS:
            pk = metric_key.replace("judgment", "judgement")
            val = merged.get(pk, {}).get(mk)
            if val is None:
                val = merged.get(metric_key, {}).get(mk)
            if isinstance(val, dict):
                val = val.get("total_recover_rate", val)
            paper[mk][metric_key] = val

    # Individual paper files (for per-importance detail recovery)
    paper_indiv = {}
    for fname in PAPER_DIR.glob("metric_mix_*.json"):
        if fname.name == "metric_mix_merged.json":
            continue
        mk = fname.stem.replace("metric_mix_", "")
        if mk not in [m[0] for m in PAPER_MODELS]:
            continue
        with open(fname) as f:
            data = json.load(f)
        md = data.get("missing_details_recover_rate", {})
        if isinstance(md, dict) and "1" in md:
            paper_indiv[mk] = md

    # Our metrics
    our = {}
    for mf in sorted(OUR_DIR.glob("*_metrics.json")):
        key = mf.name.replace("_metrics.json", "")
        with open(mf) as f:
            data = json.load(f)
        our[key] = {}
        for metric_key, _, _ in METRICS:
            v = data.get(metric_key)
            if isinstance(v, dict):
                if "total_recover_rate" in v:
                    our[key][metric_key] = v
                elif "1" in v:
                    our[key][metric_key] = v
                else:
                    our[key][metric_key] = None
            elif isinstance(v, float):
                our[key][metric_key] = round(v, 4)
            else:
                our[key][metric_key] = v

    return paper, paper_indiv, our


def extract_val(v, metric_key):
    if isinstance(v, dict):
        if "total_recover_rate" in v:
            return v["total_recover_rate"]
        return None
    return v


def pick_best(our):
    best = {}
    for metric_key, _, _ in METRICS:
        candidates = []
        for k, v in our.items():
            raw = v.get(metric_key)
            if raw is None:
                continue
            val = extract_val(raw, metric_key)
            if val is not None and not isinstance(val, dict):
                candidates.append((k, val))
        if not candidates:
            continue
        best[metric_key] = max(candidates, key=lambda x: x[1])
    return best


def fmt(v, w):
    if v is None:
        return f"{'N/A':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.4f}"
    if isinstance(v, int):
        return f"{v:>{w}}"
    return f"{str(v):>{w}}"


OUTPUT_DIR = BASE / "src" / "experiments" / "outputs"


def main():
    paper, paper_indiv, our = load_data()
    best = pick_best(our)

    print("=" * 100)
    print("  Best of our 8 models vs Tell Me More! paper")
    print("=" * 100)

    # Header
    hdr = f"{'Metric':<22}"
    for _, label in PAPER_MODELS:
        hdr += f"  {label:>18}"
    hdr += f"  {'Our Best':>18}  {'Model':>18}"
    print(hdr)
    print("─" * len(hdr))

    # Rows
    for metric_key, label, higher_better in METRICS:
        row = f"{label:<22}"
        for mk, _ in PAPER_MODELS:
            pval = extract_val(paper.get(mk, {}).get(metric_key), metric_key)
            row += f"  {fmt(pval, 18)}"
        b = best.get(metric_key)
        if b:
            bmod, bval = b
            row += f"  {fmt(bval, 18)}  {bmod:>18}"
        else:
            row += f"  {'N/A':>18}  {'':>18}"
        print(row)

    print("─" * len(hdr))

    # Per-importance detail recovery
    print("\n── Detail Recovery by Importance ──")
    hdr2 = f"{'Importance':<22}"
    for _, label in PAPER_MODELS:
        hdr2 += f"  {label:>18}"
    hdr2 += f"  {'Our Best':>18}  {'Model':>18}"
    print(hdr2)
    print("─" * len(hdr2))
    for imp in ("1", "2", "3"):
        imp_label = {"1": "Importance 1", "2": "Importance 2", "3": "Importance 3"}[imp]
        row = f"{imp_label:<22}"
        for mk, _ in PAPER_MODELS:
            md = paper_indiv.get(mk, {})
            v = md.get(imp)
            row += f"  {fmt(v, 18)}"
        best_imp = 0.0
        best_imp_m = ""
        for mk, data in our.items():
            md = data.get("missing_details_recover_rate", {})
            if isinstance(md, dict) and imp in md:
                v = md[imp]
                if isinstance(v, (int, float)) and v > best_imp:
                    best_imp = v
                    best_imp_m = mk
        row += f"  {fmt(best_imp, 18)}  {best_imp_m:>18}"
        print(row)

    print("─" * len(hdr2))

    # Summary
    print("\n── Summary ──")
    for metric_key, label, higher_better in METRICS:
        b = best.get(metric_key)
        if not b:
            continue
        bmod, bval = b
        paper_vals = [(mk, paper.get(mk, {}).get(metric_key)) for mk, _ in PAPER_MODELS]
        paper_vals = [(mk, v) for mk, v in paper_vals if isinstance(v, (int, float))]
        if not paper_vals:
            continue
        best_paper = max(paper_vals, key=lambda x: x[1])
        beats = bval >= best_paper[1]
        print(f"  {'✅' if beats else ' '} {label:<22} our={bval:.4f} ({bmod}) vs paper={best_paper[1]:.4f} ({best_paper[0]})")

    print(f"\n  ✅ = our best matches or exceeds the best paper model")

    # Save to JSON
    result = {
        "models": {},
        "per_importance_detail_recovery": {},
        "best_per_metric": {},
        "summary": {},
    }
    for mk, label in PAPER_MODELS:
        result["models"][label] = {}
        for metric_key, _, _ in METRICS:
            pval = extract_val(paper.get(mk, {}).get(metric_key), metric_key)
            result["models"][label][metric_key] = pval
    for mk, label in PAPER_MODELS:
        result["per_importance_detail_recovery"][label] = dict(paper_indiv.get(mk, {}))

    for metric_key, label, higher_better in METRICS:
        b = best.get(metric_key)
        if b:
            bmod, bval = b
            result["best_per_metric"][metric_key] = {"value": bval, "model": bmod}
            paper_vals = [(mk, extract_val(paper.get(mk, {}).get(metric_key), metric_key))
                          for mk, _ in PAPER_MODELS]
            paper_vals = [(mk, v) for mk, v in paper_vals if isinstance(v, (int, float))]
            if paper_vals:
                best_paper = max(paper_vals, key=lambda x: x[1])
                beats_paper = bval >= best_paper[1]
                result["summary"][metric_key] = {
                    "our_value": bval,
                    "our_model": bmod,
                    "best_paper_value": best_paper[1],
                    "best_paper_model": best_paper[0],
                    "beats_paper": beats_paper,
                }

    # Best per-importance
    result["best_per_importance"] = {}
    for imp in ("1", "2", "3"):
        best_val = 0.0
        best_mod = ""
        for mk, data in our.items():
            md = data.get("missing_details_recover_rate", {})
            if isinstance(md, dict) and imp in md:
                v = md[imp]
                if isinstance(v, (int, float)) and v > best_val:
                    best_val = v
                    best_mod = mk
        result["best_per_importance"][imp] = {"value": best_val, "model": best_mod}

    out_path = OUTPUT_DIR / "comparison_best_vs_paper.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
