#!/usr/bin/env python3
"""Aggregate metrics across all models into a single comparison table."""
import json
from pathlib import Path

from config import OUTPUTS_DIR


def flatten_value(v):
    if v is None:
        return None
    if isinstance(v, dict):
        if "total_recover_rate" in v:
            return {k: round(float(v[k]), 4) if isinstance(v[k], float) else v[k]
                    for k in ("1", "2", "3", "total_recover_rate") if k in v}
        if "auto_estimate" in v:
            return flatten_value(v.get("human_annotation") or v.get("auto_estimate"))
        if "rate" in v and len(v) == 1:
            return flatten_value(v.get("rate"))
        return {k: flatten_value(val) for k, val in v.items()}
    if isinstance(v, float):
        return round(v, 4)
    return v


def main():
    metrics_files = sorted(OUTPUTS_DIR.glob("*_metrics.json"))
    if not metrics_files:
        print("No metrics files found in", OUTPUTS_DIR)
        return

    comparison = {}

    for mf in metrics_files:
        model_key = mf.name.replace("_metrics.json", "")
        data = json.loads(mf.read_text(encoding="utf-8"))
        for k, v in data.items():
            if k in ("model", "num_examples", "error"):
                continue
            if k not in comparison:
                comparison[k] = {}
            comparison[k][model_key] = flatten_value(v)

    out_path = OUTPUTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison to {out_path}")

    # Print table
    metrics_names = list(comparison.keys())
    model_names = list(comparison[next(iter(comparison))].keys()) if comparison else []

    print(f"\n{'Metric':<45} " + "  ".join(f"{m:<20}" for m in model_names))
    print("-" * (45 + 22 * len(model_names)))
    for metric in metrics_names:
        row = f"{metric:<45}"
        for model in model_names:
            val = comparison[metric].get(model, None)
            if val is None:
                row += f"  {'NaN':<20}"
            elif isinstance(val, dict):
                row += f"  {'(nested)':<20}"
            else:
                row += f"  {str(val):<20}"
        print(row)


if __name__ == "__main__":
    main()
