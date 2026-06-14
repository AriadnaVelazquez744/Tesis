#!/usr/bin/env python3
"""
CLI for augmented interaction data.

Usage (from repository root)::

    python -m training.augment generate \\
        --seeds-dir training/augment/seeds \\
        --out-dir src/Vagueness_Judge/data/augmented

Training::

    TRAIN_DATA_PATH=src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl \\
    bash src/Vagueness_Judge/training/sft.sh

Or merge with the original IN3-style corpus::

    python src/Vagueness_Judge/augment_data/cli.py merge \\
        --inputs src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl \\
                src/Vagueness_Judge/data/interactions/interaction_data_train.jsonl \\
        --output src/Vagueness_Judge/data/augmented/interaction_data_train_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .merge import merge_jsonl_files
from .pipeline import run_pipeline


def _default_seeds_dir() -> Path:
    return Path(__file__).resolve().parent / "seeds"


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "augmented"


def cmd_generate(args: argparse.Namespace) -> int:
    manifest = run_pipeline(
        seeds_dir=Path(args.seeds_dir),
        out_dir=Path(args.out_dir),
        test_ratio=args.test_ratio,
        base_seed=args.seed,
        annotate=args.annotate,
        group_key=args.group_key,
    )
    print(json.dumps(manifest, indent=2))
    if args.merge_with:
        out_merged = Path(args.merge_output)
        n = merge_jsonl_files(
            [Path(args.out_dir) / "interaction_data_train.jsonl", Path(args.merge_with)],
            out_merged,
            shuffle_seed=args.seed,
        )
        print(f"Merged train files -> {out_merged} ({n} rows)", file=sys.stderr)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    n = merge_jsonl_files(
        [Path(p) for p in args.inputs],
        Path(args.output),
        shuffle_seed=args.seed,
    )
    print(f"Wrote {n} rows to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vagueness Judge augmented interaction data")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Synthesize interaction JSONL from seed files")
    g.add_argument("--seeds-dir", type=str, default=str(_default_seeds_dir()))
    g.add_argument("--out-dir", type=str, default=str(_default_out_dir()))
    g.add_argument("--test-ratio", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--annotate",
        action="store_true",
        help="Use OPENAI_API_KEY to annotate task-only seeds (requires partial rows)",
    )
    g.add_argument("--group-key", type=str, default="strand", help="Stratified split grouping key")
    g.add_argument(
        "--merge-with",
        type=str,
        default=None,
        help="Optional extra JSONL to merge with generated train (e.g. original interaction_data_train.jsonl)",
    )
    g.add_argument(
        "--merge-output",
        type=str,
        default=None,
        help="Output path for merged train (required if --merge-with is set)",
    )
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("merge", help="Merge JSONL files with shuffling")
    m.add_argument("--inputs", nargs="+", required=True, help="Input JSONL paths in order")
    m.add_argument("--output", "-o", required=True)
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    if args.command == "generate" and args.merge_with and not args.merge_output:
        parser.error("--merge-output is required when using --merge-with")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
