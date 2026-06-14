# Training

Model training scripts and outputs, separated from the engine (`src/`) and tests (`tests/`).

## Structure

```
training/
├── vagueness_judge/       JDV fine-tuning (LoRA SFT)
│   ├── sft.py, sft_3b.py, sft_7b.py    Training scripts
│   ├── merge_lora.py                    Merge LoRA into base model
│   ├── dataset_wrapper.py               Dataset utilities
│   ├── model_configs/                   Per-model hyperparameters
│   ├── adapters/                        Trained LoRA adapters (4 models)
│   └── scripts/                         Shell wrappers
│
├── midlm/                 MIDLM multi-intent detection training
│   ├── train_midlm_unsloth.py           Original Unsloth-based training
│   ├── train_midlm_bidirectional.py     Bidirectional attention training
│   ├── eval_midlm_bidirectional.py      Evaluation
│   ├── train_qwen3b_best.py / eval_qwen3b_best.py  Best-config wrappers
│   ├── build_mixed_dataset.py / build_noisy_dataset.py  Dataset builders
│   ├── midlm_server.py                  Inference server (for training)
│   ├── data/                            Datasets (JSON + TSV)
│   ├── scripts/                         Shell scripts for batch runs
│   ├── adapters/                        Trained LoRA adapters
│   ├── experiments/                     Evaluation results
│   └── logs/                            Training logs
│
├── textoir/               TEXTOIR benchmark training
│   ├── train_benchmark.py               Grid benchmark runner
│   ├── run_server.sh                    Server startup
│   └── adapters/                        Trained model checkpoints
│
├── augment/               Data augmentation scripts
│   └── ...                              (from src/Vagueness_Judge/augment_data/)
│
├── requirements-training.txt            Training-only Python dependencies
└── README.md                            This file
```

## Separation of concerns

- **`src/`** — Engine (runtime). Streamlit UI, pipeline, inference servers.
  Everything needed to run the container.
- **`tests/`** — Test suite. Analysis phase fixtures, CAO comparisons.
- **`training/`** — This directory. Model training/evaluation only.

### What stays in `src/` (engine, not duplicated):

| Module | Reason |
|--------|--------|
| `src/MIDLM/midlm/` | Core MIDLM package imported at runtime by `midlm_textoir_module/` |
| `src/TEXTOIR/open_intent_detection/` | Runtime framework for MSP/OpenMax inference |
| `src/TEXTOIR/server/api_server.py` | Inference API server |
| `src/Vagueness_Judge/runtime/` | Main pipeline used by `main.py` |
| `src/Vagueness_Judge/augment_data/` | Training data prep (also copied to `training/augment/`) |

## Running training

All scripts import the MIDLM core package from `src/MIDLM/midlm/` (not duplicated).
Make sure the project root is in your Python path or use `uv run` from the project root.

```bash
# From project root:
uv run python training/midlm/train_midlm_unsloth.py --help
uv run python training/midlm/train_midlm_bidirectional.py --help
uv run python training/vagueness_judge/sft.py --help

# Or cd into the training subdirectory:
cd training/midlm
uv run python train_midlm_bidirectional.py ...
```
