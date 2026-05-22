#!/usr/bin/env python3
"""Compare our results against the original Tell_Me_More paper's reported metrics."""
import json
from pathlib import Path

OUR_DIR = Path(__file__).resolve().parent / "outputs"
PAPER_DIR = Path(__file__).resolve().parent.parent.parent / "Tell_Me_More-master" / "data" / "user_interaction_records" / "metrics"

ORIGINAL_NAME = {
    "mistral-7b-instruct-v0.2-hf": "Mistral-7B-Instruct-v0.2 (paper)",
    "Llama-2-7b-chat-hf": "LLaMA-2-7B-Chat (paper)",
    "gpt4": "GPT-4 (paper)",
    "mistral-interact": "Mistral-Interact (paper)",
}

OUR_COLUMNS = [
    "Mistral-7B-Baseline",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-mini-Baseline",
    "Phi-3-mini-4k-instruct",
    "Qwen2.5-3B-Baseline",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Baseline",
    "Qwen2.5-7B-Instruct",
]

METRIC_LABELS = {
    "vagueness_judgment_accuracy": "Vagueness Judgment Accuracy",
    "missing_details_recover_rate": "Missing Details Recover Rate",
    "summary_intention_coverage_rate": "Summary Intention Coverage",
    "options_presenting_rate": "Options Presenting Rate",
    "options_reasonable_rate": "Options Reasonable Rate",
    "average_provided_options": "Avg Provided Options",
    "average_inquired_missing_details_per_round": "Avg Inquired / Round",
    "average_conversation_rounds": "Avg Conversation Rounds",
    "average_inquired_missing_details": "Avg Inquired Missing Details",
}


def load_paper_metrics(path):
    """Load a single paper metric file (e.g. metric_mix_gpt4.json)."""
    with open(path) as f:
        data = json.load(f)
    return data


def load_paper_merged():
    """Load the paper's merged comparison file."""
    merged_path = PAPER_DIR / "metric_mix_merged.json"
    if not merged_path.exists():
        return {}
    with open(merged_path) as f:
        return json.load(f)


def load_our_metrics():
    """Load our metrics files and return {model_key: {metric: value}}."""
    our = {}
    for mf in sorted(OUR_DIR.glob("*_metrics.json")):
        key = mf.name.replace("_metrics.json", "")
        with open(mf) as f:
            data = json.load(f)
        # Flatten nested metric dicts to a single value
        flat = {}
        for k, v in data.items():
            if k in ("model", "num_examples", "error"):
                continue
            if isinstance(v, dict):
                if "auto_estimate" in v:
                    flat[k] = v.get("human_annotation") or v.get("auto_estimate")
                elif "human_annotation" in v:
                    flat[k] = v.get("human_annotation")
                elif "total_recover_rate" in v:
                    flat[k] = v["total_recover_rate"]
                elif "rate" in v:
                    flat[k] = v["rate"]
                else:
                    flat[k] = None
            else:
                flat[k] = v
        our[key] = flat
    return our


def merge_paper_individuals():
    """Load individual paper metric files for models not in merged."""
    paper = {}
    for fname in PAPER_DIR.glob("metric_mix_*.json"):
        if fname.name == "metric_mix_merged.json":
            continue
        # Extract model key from filename: metric_mix_gpt4.json -> gpt4
        model_key = fname.stem.replace("metric_mix_", "")
        data = load_paper_metrics(fname)
        flat = {}
        for k, v in data.items():
            normalized_k = k.replace("judgement", "judgment")
            if isinstance(v, dict) and "total_recover_rate" in v:
                flat[normalized_k] = v["total_recover_rate"]
            elif isinstance(v, dict):
                flat[normalized_k] = next(iter(v.values()), v)
            else:
                flat[normalized_k] = v
        paper[model_key] = flat
    return paper


def fmt(val, width=10):
    if val is None:
        return f"{'NaN':>{width}}"
    if isinstance(val, float):
        return f"{val:>{width}.4f}"
    return f"{val:>{width}}"


def print_table(metric_names, paper_data, our_data, paper_models, our_models):
    headers = ["Metric"] + [ORIGINAL_NAME.get(m, m) for m in paper_models] + our_models
    col_width = max(len(h) for h in headers) + 2
    col_width = max(col_width, 14)

    sep = "─" * col_width
    header_row = f"{'Metric':<{col_width}}"
    for m in paper_models:
        header_row += f"  {ORIGINAL_NAME.get(m, m):>{col_width-2}}"
    for m in our_models:
        header_row += f"  {m:>{col_width-2}}"
    print(header_row)
    print("─" * len(header_row))

    for metric_key, label in METRIC_LABELS.items():
        row = f"{label:<{col_width}}"

        for m in paper_models:
            val = paper_data.get(m, {}).get(metric_key, None)
            row += f"  {fmt(val, col_width-2)}"

        for m in our_models:
            val = our_data.get(m, {}).get(metric_key, None)
            row += f"  {fmt(val, col_width-2)}"

        print(row)

    print("─" * len(header_row))
    print(f"  {'(paper)':>{col_width-2}}" + "  " * len(paper_models) + "  " + "  (our models)")


def print_detail_recovery(paper_data, our_data, paper_models, our_models):
    """Print per-importance detail recovery if available."""
    col_width = max(max(len(ORIGINAL_NAME.get(m, m)) for m in paper_models), max(len(m) for m in our_models)) + 2
    col_width = max(col_width, 14)

    # Load raw per-importance data from individual paper files
    paper_raw = {}
    for fname in PAPER_DIR.glob("metric_mix_*.json"):
        if fname.name == "metric_mix_merged.json":
            continue
        model_key = fname.stem.replace("metric_mix_", "")
        with open(fname) as f:
            data = json.load(f)
        md = data.get("missing_details_recover_rate", {})
        if isinstance(md, dict) and "1" in md:
            paper_raw[model_key] = md

    our_raw = {}
    for m in our_models:
        mf = OUR_DIR / f"{m}_metrics.json"
        if mf.exists():
            with open(mf) as f:
                data = json.load(f)
            md = data.get("missing_details_recover_rate", {})
            if isinstance(md, dict) and "1" in md:
                our_raw[m] = md

    print("\n── Missing Details Recovery by Importance ──")
    for imp in ("1", "2", "3", "total_recover_rate"):
        imp_label = {"1": "Importance 1", "2": "Importance 2", "3": "Importance 3", "total_recover_rate": "Total"}[imp]
        row = f"{imp_label:<{col_width}}"
        for m in paper_models:
            md = paper_raw.get(m, {})
            val = md.get(imp) if isinstance(md, dict) else None
            if val is None:
                val = paper_data.get(m, {}).get("missing_details_recover_rate", None)
            row += f"  {fmt(val, col_width-2)}"
        for m in our_models:
            md = our_raw.get(m, {})
            val = md.get(imp) if isinstance(md, dict) else None
            if val is None:
                val = our_data.get(m, {}).get("missing_details_recover_rate", None)
            row += f"  {fmt(val, col_width-2)}"
        print(row)


def main():
    # Load both sources
    paper_merged = load_paper_merged()
    paper_individuals = merge_paper_individuals()

    # Merge: merged file is dict-of-dicts, individuals are flat
    paper = {}
    for model_key, data in paper_merged.items():
        # Normalize metric names (paper uses British "judgement")
        normalized = {}
        for k, v in data.items():
            normalized[k.replace("judgement", "judgment")] = v
        paper[model_key] = normalized
    for model_key, data in paper_individuals.items():
        if model_key not in paper:
            paper[model_key] = data

    our = load_our_metrics()

    # Filter to only models that exist in both or are interesting
    paper_models = [m for m in ORIGINAL_NAME if m in paper]
    our_models = [m for m in OUR_COLUMNS if m in our]

    print("=" * 120)
    print("  Comparison: Our models vs Tell Me More! paper results")
    print("=" * 120)

    print_table(list(METRIC_LABELS.keys()), paper, our, paper_models, our_models)
    print_detail_recovery(paper, our, paper_models, our_models)

    # Save JSON
    out = {"our_models": our, "paper_models": paper}
    out_path = OUR_DIR / "comparison_vs_paper.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed JSON to {out_path}")

    # Highlight key comparisons
    print("\n── Key Highlights ──")
    if "mistral-7b-instruct-v0.2-hf" in paper and "Mistral-7B-Instruct-v0.3" in our:
        print(f"  Mistral (paper v0.2 vs our v0.3):")
        p = paper["mistral-7b-instruct-v0.2-hf"]
        o = our["Mistral-7B-Instruct-v0.3"]
        print(f"    Vagueness Accuracy:  {p.get('vagueness_judgment_accuracy', 'N/A')} (paper) vs {o.get('vagueness_judgment_accuracy', 'N/A')} (ours)")
        print(f"    Detail Recovery:     {p.get('missing_details_recover_rate', 'N/A')} (paper) vs {o.get('missing_details_recover_rate', 'N/A')} (ours)")
        print(f"    Summary Coverage:    {p.get('summary_intention_coverage_rate', 'N/A')} (paper) vs {o.get('summary_intention_coverage_rate', 'N/A')} (ours)")


if __name__ == "__main__":
    main()
