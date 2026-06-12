import json
import random
from collections import Counter

SOURCE_FILE = "tests/analysis_phase/fixtures/expanded/expanded_queries.jsonl"
OUTPUT_FILE = "tests/comparissons/data/evaluation_sample_v1.json"
SUMMARY_FILE = "tests/comparissons/data/evaluation_sample_v1_summary.json"

IND_TARGET = {
    "natural_logic":              {1: 7, 2: 7, 3: 6},
    "mathematical_calculation":   {1: 7, 2: 7, 3: 6},
    "information_search":         {1: 7, 2: 7, 3: 6},
    "human_psychology":           {1: 7, 2: 7, 3: 6},
    "mixed":                      {1: 7, 2: 7, 3: 6},
}

OOS_TARGET = {
    "natural_logic":              {1: 4, 2: 3, 3: 3},
    "mathematical_calculation":   {1: 4, 2: 3, 3: 3},
    "information_search":         {1: 4, 2: 3, 3: 3},
    "human_psychology":           {1: 4, 2: 3, 3: 3},
    "mixed":                      {1: 4, 2: 3, 3: 3},
}


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def check_stratum_availability(pool, targets, label):
    for topic in sorted(targets):
        for k, needed in sorted(targets[topic].items()):
            available = len([
                r for r in pool
                if r["topic"] == topic and r["k"] == k
            ])
            if available < needed:
                raise ValueError(
                    f"{label}: {topic} k={k} needs {needed} but only {available} available"
                )
    print(f"  [{label}] All strata have sufficient candidates.")


def sample_stratum(pool, topic, k, n, random_gen):
    candidates = [
        r for r in pool
        if r["topic"] == topic and r["k"] == k
    ]
    random_gen.shuffle(candidates)
    selected = candidates[:n]
    if len(selected) < n:
        raise RuntimeError(
            f"Cannot sample {n} records for {topic} k={k}: "
            f"only {len(candidates)} available"
        )
    return selected


def build_pools(records):
    ind_pool = []
    oos_pool = []
    for r in records:
        if r.get("oos", False):
            oos_pool.append(r)
        elif r["meta"]["variant_index"] not in (8, 9) and r["k"] in (1, 2, 3):
            ind_pool.append(r)
    return ind_pool, oos_pool


def compute_distribution(records):
    by_topic = Counter(r["topic"] for r in records)
    by_topic_k = Counter((r["topic"], r["k"]) for r in records)
    by_domain = Counter(r["domain_cluster"] for r in records)
    unique_seeds = len(set(r["meta"]["seed_source"] for r in records))
    return {
        "by_topic": dict(sorted(by_topic.items())),
        "by_topic_k": {str(k): c for (t, k), c in sorted(by_topic_k.items())},
        "by_domain": dict(sorted(by_domain.items())),
        "unique_seeds": unique_seeds,
    }


def main():
    print("Loading records...")
    records = load_records(SOURCE_FILE)
    print(f"  Total records loaded: {len(records)}")

    ind_pool, oos_pool = build_pools(records)
    print(f"  IND pool (non-reversal, k=1-3): {len(ind_pool)}")
    print(f"  OOS pool: {len(oos_pool)}")

    print("\nChecking stratum availability...")
    check_stratum_availability(ind_pool, IND_TARGET, "IND")
    check_stratum_availability(oos_pool, OOS_TARGET, "OOS")

    rng = random.Random(42)

    print("\nSampling IND...")
    ind_selected = []
    for topic in sorted(IND_TARGET):
        for k, n in sorted(IND_TARGET[topic].items()):
            picked = sample_stratum(ind_pool, topic, k, n, rng)
            ind_selected.extend(picked)
            print(f"    {topic} k={k}: selected {len(picked)}/{n}")

    print("\nSampling OOS...")
    oos_selected = []
    for topic in sorted(OOS_TARGET):
        for k, n in sorted(OOS_TARGET[topic].items()):
            picked = sample_stratum(oos_pool, topic, k, n, rng)
            oos_selected.extend(picked)
            print(f"    {topic} k={k}: selected {len(picked)}/{n}")

    all_selected = ind_selected + oos_selected
    print(f"\nTotal selected: {len(all_selected)} (IND: {len(ind_selected)}, OOS: {len(oos_selected)})")

    ind_dist = compute_distribution(ind_selected)
    oos_dist = compute_distribution(oos_selected)

    KEEP_FIELDS = {"id", "query", "topic", "domain_cluster", "vague", "k", "oos"}
    cleaned = [{k: r[k] for k in KEEP_FIELDS} for r in all_selected]

    output = {
        "meta": {
            "description": "Stratified sample for human evaluation of pipeline CAO vs raw query",
            "total": len(all_selected),
            "ind_count": len(ind_selected),
            "oos_count": len(oos_selected),
            "random_seed": 42,
            "distribution": {
                "ind": {
                    "target": IND_TARGET,
                    "actual": ind_dist,
                },
                "oos": {
                    "target": OOS_TARGET,
                    "actual": oos_dist,
                },
            },
            "created": "2026-06-12",
            "source_file": SOURCE_FILE,
        },
        "records": cleaned,
    }

    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    summary = {
        "meta": output["meta"],
        "ind_distribution": ind_dist,
        "oos_distribution": oos_dist,
        "sample_ids": [r["id"] for r in cleaned],
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {SUMMARY_FILE}")

    print("\n--- Sample Distribution Summary ---")
    for topic in sorted(IND_TARGET):
        counts = Counter(r["k"] for r in ind_selected if r["topic"] == topic)
        print(f"  IND {topic}: {dict(sorted(counts.items()))}")
    for topic in sorted(OOS_TARGET):
        counts = Counter(r["k"] for r in oos_selected if r["topic"] == topic)
        print(f"  OOS {topic}: {dict(sorted(counts.items()))}")
    print(f"  IND domains: {dict(sorted(Counter(r['domain_cluster'] for r in ind_selected).items()))}")
    print(f"  OOS domains: {dict(sorted(Counter(r['domain_cluster'] for r in oos_selected).items()))}")
    print(f"  Unique seeds in IND: {ind_dist['unique_seeds']}")
    print(f"  Unique seeds in OOS: {oos_dist['unique_seeds']}")
    print(f"  IND-OOS seed overlap: {len(set(r['meta']['seed_source'] for r in ind_selected) & set(r['meta']['seed_source'] for r in oos_selected))}")


if __name__ == "__main__":
    main()
