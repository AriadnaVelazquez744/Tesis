"""Tests for Vagueness_Judge augment_data (schema + dialogue; optional SFT preprocess)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import unittest
from pathlib import Path

# Repository root: tests/ -> parent
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Vagueness_Judge.augment_data.dialogue import build_actions_from_annotations, build_full_record
from Vagueness_Judge.augment_data.pipeline import stratified_split, validate_records
from Vagueness_Judge.augment_data.schemas import validate_interaction_record


class TestSchemas(unittest.TestCase):
    def test_valid_clear(self) -> None:
        row = {
            "task": "Find X.",
            "vague": False,
            "thought": "Clear.",
            "missing_details": [],
            "actions": [
                {
                    "role": "assistant",
                    "content": "Summary text.",
                    "thought": "Summary thought.",
                    "type": "summary",
                }
            ],
        }
        ok, errs = validate_interaction_record(row)
        self.assertTrue(ok, errs)

    def test_invalid_vague_empty_details(self) -> None:
        row = {
            "task": "Vague task.",
            "vague": True,
            "thought": "Vague.",
            "missing_details": [],
            "actions": [],
        }
        ok, errs = validate_interaction_record(row)
        self.assertFalse(ok)


class TestDialogue(unittest.TestCase):
    def test_clear_actions_single_summary(self) -> None:
        actions = build_actions_from_annotations(
            task="Compute 2+2.",
            thought="Clear.",
            vague=False,
            missing_details=[],
            rng=random.Random(0),
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "summary")

    def test_vague_roundtrip(self) -> None:
        md = [
            {
                "description": "Scope",
                "importance": "3",
                "inquiry": "Which scope?",
                "options": ["A", "B", "C"],
            }
        ]
        rec = build_full_record(
            {
                "task": "Do something.",
                "vague": True,
                "thought": "Vague.",
                "missing_details": md,
                "category": "Test",
            },
            rng=random.Random(123),
        )
        ok, errs = validate_interaction_record(rec)
        self.assertTrue(ok, errs)
        self.assertEqual(rec["actions"][-1]["type"], "summary")

    def test_deterministic_rng(self) -> None:
        r1 = build_full_record(
            {
                "task": "Same task.",
                "vague": True,
                "thought": "Vague.",
                "missing_details": [
                    {
                        "description": "x",
                        "importance": "2",
                        "inquiry": "Pick?",
                        "options": ["p", "q"],
                    }
                ],
            },
            rng=random.Random(99),
        )
        r2 = build_full_record(
            {
                "task": "Same task.",
                "vague": True,
                "thought": "Vague.",
                "missing_details": [
                    {
                        "description": "x",
                        "importance": "2",
                        "inquiry": "Pick?",
                        "options": ["p", "q"],
                    }
                ],
            },
            rng=random.Random(99),
        )
        self.assertEqual(r1["actions"], r2["actions"])


class TestStratifiedSplit(unittest.TestCase):
    def test_split_preserves_total(self) -> None:
        rows = [{"strand": s, "i": i} for s in ("a", "b") for i in range(5)]
        train, test = stratified_split(rows, test_ratio=0.2, seed=0, group_key="strand")
        self.assertEqual(len(train) + len(test), len(rows))


class TestSftCompatibility(unittest.TestCase):
    def test_preprocess_mtmd_runs(self) -> None:
        """Ensure generated lines work with sft.preprocess_data (MTMD)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available in this environment")
        import importlib

        sft = importlib.import_module("Vagueness_Judge.training.sft")

        seeds = REPO_ROOT / "src/Vagueness_Judge/augment_data/seeds/search.jsonl"
        line = seeds.read_text(encoding="utf-8").strip().split("\n")[0]
        seed_obj = json.loads(line)
        rec = build_full_record(seed_obj, rng=random.Random(0))

        ns = argparse.Namespace(data_setting="MTMD")
        out = sft.preprocess_data(rec, ns)
        self.assertIsInstance(out, list)
        self.assertTrue(len(out) > 0)
        for ex in out:
            self.assertIn("data", ex)


if __name__ == "__main__":
    unittest.main()
