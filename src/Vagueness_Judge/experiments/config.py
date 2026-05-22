import json
import os
from collections import OrderedDict
from pathlib import Path

os.environ["HF_HOME"] = "/tmp/hf-cache"
os.environ["HF_MODULES_CACHE"] = "/tmp/hf-cache/modules"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
BASE_MODELS_DIR = PROJECT_DIR.parent / "base_models"
TRAINING_MODELS_DIR = PROJECT_DIR / "src" / "Vagueness_Judge" / "training_models"

IN3_TEST_PATH = PROJECT_DIR / "Tell_Me_More-master" / "data" / "IN3" / "test.jsonl"
LABELLER_PATH = PROJECT_DIR / "Tell_Me_More-master" / "data" / "data_labeling" / "test_data_report_mix.jsonl"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

TASK_DESCRIPTION = (
    "You are an agent trying to understand the user's goal and summarize it. "
    "Please first ask users for more specific details with options, and finally summarize the user's intention.\n"
    "--- Step 1: initial thought generation ---\n"
    "1. Generate [INITIAL THOUGHT] about if the task is vague or clear and why.\n"
    "2. List the important missing details and some according options if the task is vague.\n"
    "--- Step 2: inquiry for more information if vague ---\n"
    "1. If the task is vague, inquire about more details with options according to the list in [INITIAL THOUGHT].\n"
    "2. Think about what information you have and what to inquire next in [INQUIRY THOUGHT].\n"
    "3. Present your inquiry with options for the user to choose after [INQUIRY], and be friendly.\n"
    "4. You could repeat Step 2 multiple times (but less than 5 times), or directly skip Step 2 if the user task is clear initially.\n"
    "--- Step 3: summarize the user's intention ---\n"
    "1. Make the summary once the information is enough. You do not need to inquire about every missing detail in [INITIAL THOUGHT].\n"
    "2. List all the user's preferences and constraints in [SUMMARY THOUGHT]. The number of points should be the same as rounds of chatting.\n"
    "3. Give the final summary after [SUMMARY] with comprehensive details in one or two sentences."
)

def discover_models():
    models = OrderedDict()

    baselines = [
        ("Qwen2.5-3B-Baseline", "Qwen2.5-3B-Instruct", "3b"),
        ("Qwen2.5-7B-Baseline", "Qwen2.5-7B-Instruct", "7b"),
        ("Mistral-7B-Baseline", "Mistral-7B-Instruct-v0.3", "7b"),
        ("Phi-3-mini-Baseline", "Phi-3-mini-4k-instruct", "7b"),
    ]
    for key, model_name, model_type in baselines:
        models[key] = {
            "base_model_name": model_name,
            "base_model_path": str(BASE_MODELS_DIR / model_name),
            "adapter_path": None,
            "type": model_type,
        }

    adapter_dirs = sorted(TRAINING_MODELS_DIR.glob("*-Vagueness_Judge"))
    for adapter_dir in adapter_dirs:
        key = adapter_dir.name.replace("-Vagueness_Judge", "")

        # Read train_config.json to get base model path
        train_config_path = adapter_dir / "train_config.json"
        if train_config_path.exists():
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            base_model_path = train_config.get("model_path", "")
        else:
            # Fallback: derive from adapter_config.json
            adapter_config_path = adapter_dir / "adapter_config.json"
            if adapter_config_path.exists():
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                base_model_path = adapter_config.get("base_model_name_or_path", "")
            else:
                base_model_path = ""

        if not base_model_path or not Path(base_model_path).exists():
            base_model_path = str(BASE_MODELS_DIR / key)

        is_7b = "7B" in key or "Mistral-7B" in key or "Phi-3" in key
        models[key] = {
            "base_model_name": key,
            "base_model_path": base_model_path,
            "adapter_path": str(adapter_dir),
            "type": "7b" if is_7b else "3b",
        }

    return models

MODELS = discover_models()
