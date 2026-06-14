#!/usr/bin/env python3
"""Fix broken base_model_name_or_path in existing adapter_config.json files.

The old training run saved paths pointing to /home/gia/AriadnaVR/JdV_Training/
instead of the actual location at /home/gia/AriadnaVR/thesis/JdV_Training/.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent  
TRAINING_MODELS_DIR = PROJECT_DIR / "src" / "Vagueness_Judge" / "jdv_adapters"
BASE_MODELS_DIR = PROJECT_DIR / "src" / "base_models"  # ~/thesis/base_models

OLD_BASE = "/home/ari/Collage/TESIS/Tesis/src/base_models"

def fix_one(adapter_dir: Path) -> bool:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return False

    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_path = config.get("base_model_name_or_path", "")

    if not old_path.startswith(OLD_BASE):
        print(f"  SKIP {adapter_dir.name}: path OK or unexpected ({old_path})")
        return False

    model_name = old_path.split("/")[-1]
    new_path = str(BASE_MODELS_DIR / model_name)

    if not (BASE_MODELS_DIR / model_name).exists():
        print(f"  WARN {adapter_dir.name}: new path {new_path} does not exist!")
        return False

    config["base_model_name_or_path"] = new_path
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"  FIXED {adapter_dir.name}: {old_path} -> {new_path}")
    return True


def main():
    if not TRAINING_MODELS_DIR.exists():
        print(f"Training models dir not found: {TRAINING_MODELS_DIR}")
        return

    found = 0
    for entry in sorted(TRAINING_MODELS_DIR.iterdir()):
        if entry.is_dir() and entry.name.endswith("Vagueness_Judge"):
            if fix_one(entry):
                found += 1

    print(f"\nFixed {found} adapter(s).")
    if found == 0:
        print("All adapter paths are already correct or up-to-date.")


if __name__ == "__main__":
    main()
