#!/usr/bin/env python3
"""
Full TEXTOIR benchmark runner for thesis experiments.

Runs all combinations of methods x datasets x known_cls_ratios x seeds,
collects results, and saves the best model.

Usage:
    python train_benchmark.py \\
        --output_dir /path/to/storage \\
        --data_dir /path/to/TEXTOIR/data \\
        --gpu_id 0

Environment:
    HF_HOME  : huggingface cache directory (avoid /root permission issues)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_BENCHMARK_GRID = {
    "methods": ["MSP", "OpenMax", "DOC", "DeepUnk", "LOF", "K+1-way", "SEG", "MDF", "ADB", "ARPL", "KNNCL"],
    "datasets": ["oos", "banking", "stackoverflow"],
    "known_cls_ratios": [0.25, 0.5, 0.75],
    "labeled_ratio": 1.0,
    "backbone": "bert",
    "seeds": [0, 1, 2, 3, 4],
}

_METHOD_CONFIG: Dict[str, Dict[str, str]] = {
    "MSP":     {"config": "MSP",     "loss": "CrossEntropyLoss",       "pretrain": False},
    "OpenMax": {"config": "OpenMax", "loss": "CrossEntropyLoss",       "pretrain": False},
    "DOC":     {"config": "DOC",     "loss": "Binary_CrossEntropyLoss","pretrain": False},
    "DeepUnk": {"config": "DeepUnk", "loss": "CosineFaceLoss",         "pretrain": False},
    "LOF":     {"config": "LOF",     "loss": "CrossEntropyLoss",       "pretrain": False},
    "K+1-way": {"config": "K+1-way", "loss": "CrossEntropyLoss",       "pretrain": False},
    "SEG":     {"config": "SEG",     "loss": "CrossEntropyLoss",       "pretrain": False},
    "MDF":     {"config": "MDF",     "loss": "CrossEntropyLoss",       "pretrain": True},
    "ADB":     {"config": "ADB",     "loss": "CrossEntropyLoss",       "pretrain": True},
    "ARPL":    {"config": "ARPL",    "loss": "CrossEntropyLoss",       "pretrain": True},
    "KNNCL":   {"config": "KNNCL",   "loss": "CrossEntropyLoss",       "pretrain": True},
}


@dataclass
class RunResult:
    method: str
    dataset: str
    known_cls_ratio: float
    seed: int
    acc: float = 0.0
    f1_overall: float = 0.0
    f1_known: float = 0.0
    f1_open: float = 0.0
    best_eval_score: float = 0.0
    model_path: str = ""
    status: str = "pending"


def _find_open_intent_detection_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "TEXTOIR" / "open_intent_detection"


def _find_data_dir(base_dir: Path) -> Path:
    candidate = base_dir.parents[1] / "base_implementations" / "TEXTOIR" / "data"
    return candidate.resolve()


def _build_run_dir(output_dir: str, method: str, dataset: str,
                   ratio: float, seed: int) -> Path:
    return (Path(output_dir) / "open_intent_detection"
            / f"{method}_{dataset}_{ratio}_1.0_bert_{seed}")


def _parse_log_for_results(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    results: Dict[str, float] = {}

    import re
    m = re.search(r"best_eval_score\s*=\s*([\d.]+)", text)
    if m:
        results["best_eval_score"] = float(m.group(1))

    m = re.search(r"Acc\s*=\s*([\d.]+)", text)
    if m:
        results["acc"] = float(m.group(1))

    m = re.search(r"F1\s*=\s*([\d.]+)", text)
    if m:
        results["f1_overall"] = float(m.group(1))

    m = re.search(r"F1-known\s*=\s*([\d.]+)", text)
    if m:
        results["f1_known"] = float(m.group(1))

    m = re.search(r"F1-open\s*=\s*([\d.]+)", text)
    if m:
        results["f1_open"] = float(m.group(1))

    return results if results else None


def run_single_experiment(
    oid_dir: Path,
    method: str,
    dataset: str,
    known_cls_ratio: float,
    seed: int,
    labeled_ratio: float,
    backbone: str,
    data_dir: str,
    output_dir: str,
    gpu_id: str,
) -> RunResult:
    cfg = _METHOD_CONFIG[method]
    result = RunResult(
        method=method,
        dataset=dataset,
        known_cls_ratio=known_cls_ratio,
        seed=seed,
    )

    run_out_dir = _build_run_dir(output_dir, method, dataset, known_cls_ratio, seed)
    model_path = run_out_dir / "models" / "pytorch_model.bin"

    cmd = [
        sys.executable, "run.py",
        "--dataset", dataset,
        "--method", method,
        "--known_cls_ratio", str(known_cls_ratio),
        "--labeled_ratio", str(labeled_ratio),
        "--seed", str(seed),
        "--backbone", backbone,
        "--config_file_name", cfg["config"],
        "--loss_fct", cfg["loss"],
        "--gpu_id", gpu_id,
        "--train",
        "--save_model",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--results_file_name", f"results_{method}.csv",
        "--save_results",
    ]
    if cfg["pretrain"]:
        cmd.append("--pretrain")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = oid_dir / "logs" / f"benchmark_{method}_{dataset}_{known_cls_ratio}_{seed}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[BENCH] {method} | {dataset} | ratio={known_cls_ratio} | seed={seed}")
    print(f"  Log: {log_path}")

    try:
        subprocess.run(
            cmd,
            cwd=str(oid_dir),
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            check=True,
            timeout=7200,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        result.status = "timeout"
        return result
    except subprocess.CalledProcessError as e:
        print(f"  FAILED (exit code {e.returncode})")
        result.status = f"failed:{e.returncode}"
        return result

    parsed = _parse_log_for_results(log_path)
    if parsed:
        result.acc = parsed.get("acc", 0.0)
        result.f1_overall = parsed.get("f1_overall", 0.0)
        result.f1_known = parsed.get("f1_known", 0.0)
        result.f1_open = parsed.get("f1_open", 0.0)
        result.best_eval_score = parsed.get("best_eval_score", 0.0)
        result.status = "ok"
    else:
        result.status = "no_results_parsed"

    if model_path.exists():
        result.model_path = str(model_path)

    print(f"  Acc={result.acc} F1={result.f1_overall} F1-known={result.f1_known} F1-open={result.f1_open}")
    return result


def save_results_csv(results: List[RunResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "known_cls_ratio", "seed",
            "acc", "f1_overall", "f1_known", "f1_open",
            "best_eval_score", "model_path", "status",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nResults saved to {path}")


def find_best_model(results: List[RunResult]) -> Optional[RunResult]:
    ok_results = [r for r in results if r.status == "ok" and r.f1_open > 0]
    if not ok_results:
        return None
    ok_results.sort(key=lambda r: r.f1_open, reverse=True)
    return ok_results[0]


def copy_best_model(best: RunResult, dest_dir: Path) -> None:
    src_model = Path(best.model_path)
    if not src_model.exists():
        print(f"WARNING: best model file not found at {src_model}")
        return

    src_config = src_model.parent / "config.json"
    dest_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(str(src_model), str(dest_dir / "pytorch_model.bin"))
    if src_config.exists():
        shutil.copy2(str(src_config), str(dest_dir / "config.json"))

    readme = dest_dir / "BEST_MODEL_INFO.txt"
    readme.write_text(
        f"Best model (by F1-open):\n"
        f"  method:        {best.method}\n"
        f"  dataset:       {best.dataset}\n"
        f"  ratio:         {best.known_cls_ratio}\n"
        f"  seed:          {best.seed}\n"
        f"  Acc:           {best.acc}\n"
        f"  F1-overall:    {best.f1_overall}\n"
        f"  F1-known:      {best.f1_known}\n"
        f"  F1-open:       {best.f1_open}\n"
    )
    print(f"Best model copied to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TEXTOIR full benchmark runner")
    parser.add_argument("--output_dir", default="./storage/textoir_training",
                        help="Root output directory for trained models and results")
    parser.add_argument("--data_dir", default="",
                        help="Path to TEXTOIR data directory (default: auto-detect)")
    parser.add_argument("--gpu_id", default="0", help="GPU device id")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be run without executing")
    parser.add_argument("--methods", nargs="+", default=[],
                        help="Override: specific methods to run")
    parser.add_argument("--datasets", nargs="+", default=[],
                        help="Override: specific datasets to run")
    parser.add_argument("--ratios", nargs="+", type=float, default=[],
                        help="Override: specific known_cls_ratios")
    parser.add_argument("--seeds", nargs="+", type=int, default=[],
                        help="Override: specific seeds")
    parser.add_argument("--best_model_dir", default="./storage/textoir_training/best_model",
                        help="Destination for the best model copy")
    args = parser.parse_args()

    oid_dir = _find_open_intent_detection_dir()
    print(f"TEXTOIR open_intent_detection dir: {oid_dir}")

    data_dir = str(Path(args.data_dir or _find_data_dir(oid_dir)).resolve())
    print(f"Data dir: {data_dir}")

    output_dir = str(Path(args.output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    methods = args.methods or _BENCHMARK_GRID["methods"]
    datasets = args.datasets or _BENCHMARK_GRID["datasets"]
    ratios = args.ratios or _BENCHMARK_GRID["known_cls_ratios"]
    seeds = args.seeds or _BENCHMARK_GRID["seeds"]
    labeled_ratio = _BENCHMARK_GRID["labeled_ratio"]
    backbone = _BENCHMARK_GRID["backbone"]

    total = len(methods) * len(datasets) * len(ratios) * len(seeds)
    print(f"Grid: {len(methods)} methods x {len(datasets)} datasets x "
          f"{len(ratios)} ratios x {len(seeds)} seeds = {total} runs")

    if args.dry_run:
        print("DRY RUN — no experiments executed")
        return

    all_results: List[RunResult] = []
    completed = 0

    for method in methods:
        for dataset in datasets:
            for ratio in ratios:
                for seed in seeds:
                    result = run_single_experiment(
                        oid_dir=oid_dir,
                        method=method,
                        dataset=dataset,
                        known_cls_ratio=ratio,
                        seed=seed,
                        labeled_ratio=labeled_ratio,
                        backbone=backbone,
                        data_dir=data_dir,
                        output_dir=output_dir,
                        gpu_id=args.gpu_id,
                    )
                    all_results.append(result)
                    completed += 1
                    print(f"  [{completed}/{total}] done\n")

    results_csv = Path(output_dir) / "benchmark_results.csv"
    save_results_csv(all_results, results_csv)

    best = find_best_model(all_results)
    if best:
        print(f"\nBest model (F1-open={best.f1_open}): {best.method} on "
              f"{best.dataset} ratio={best.known_cls_ratio} seed={best.seed}")
        best_dest = Path(args.best_model_dir)
        copy_best_model(best, best_dest)
    else:
        print("\nNo successful runs found to determine best model.")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
