"""Augmented interaction data generation for Vagueness Judge SFT.

Run::

    PYTHONPATH=src python -m Vagueness_Judge.augment_data generate \\
        --out-dir src/Vagueness_Judge/data/augmented

Point training at ``data/augmented/interaction_data_train.jsonl`` via ``TRAIN_DATA_PATH``.
"""

__all__ = [
    "schemas",
    "dialogue",
    "pipeline",
    "merge",
]
