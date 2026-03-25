#!/usr/bin/env python3
"""
Export TEXTOIR-style TSV files from a WeaveClinc150 JSON (e.g. WeaveClinc150_rewritten.json).

Same format as generate_clinc150_multiintent.write_tsv:
  header: text<TAB>label
  label: sorted intent names joined by |

Split mapping matches the generator:
  train -> train.tsv
  validation -> dev.tsv
  test -> test.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export textoir TSV from WeaveClinc150 JSON")
    p.add_argument(
        "--input-json",
        type=Path,
        default=Path("src/MIDLM/data/WeaveClinc150_rewritten.json"),
        help="WeaveClinc150 or WeaveClinc150_rewritten JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/MIDLM/data/textoir_tsv"),
        help="Directory for train.tsv, dev.tsv, test.tsv",
    )
    return p.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "label"])
        for r in rows:
            labels = r.get("labels", [])
            if not isinstance(labels, list):
                raise TypeError(f"Row missing list 'labels': {repr(r)[:200]}")
            label = "|".join(sorted(str(x) for x in labels))
            writer.writerow([r.get("text", ""), label])


def main() -> int:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    train = data.get("train", [])
    val = data.get("validation", [])
    test = data.get("test", [])
    for name, obj in ("train", train), ("validation", val), ("test", test):
        if not isinstance(obj, list):
            raise RuntimeError(f"JSON key '{name}' must be a list.")

    out = args.output_dir
    write_tsv(out / "train.tsv", train)
    write_tsv(out / "dev.tsv", val)
    write_tsv(out / "test.tsv", test)
    print(f"Wrote {out}/train.tsv ({len(train)} rows)")
    print(f"Wrote {out}/dev.tsv ({len(val)} rows)")
    print(f"Wrote {out}/test.tsv ({len(test)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
