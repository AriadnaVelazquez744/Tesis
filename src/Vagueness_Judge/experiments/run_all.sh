#!/bin/bash
# Full evaluation pipeline: infer → evaluate → compare

set -e

REPO="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$REPO"

export HF_HOME=/tmp/hf-cache

echo "=========================================="
echo "  Vagueness Judge — Evaluation Pipeline"
echo "=========================================="

# Phase 0: Fix adapter paths (only needed for old adapters)
echo ""
echo "=== Phase 0: Fix adapter paths ==="
.venv/bin/python src/experiments/fix_adapter_paths.py

# Phase 1: Inference for all models
echo ""
echo "=== Phase 1: Inference ==="
.venv/bin/python src/experiments/inference.py --model all

# Phase 2: Auto-evaluate (all 9 metrics, 8 automated + M5 as NaN)
echo ""
echo "=== Phase 2: Auto-evaluation ==="
.venv/bin/python src/experiments/evaluate.py --model all

# Phase 3: Build comparison table
echo ""
echo "=== Phase 3: Comparison ==="
.venv/bin/python src/experiments/compare.py

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  See outputs/comparison.json for results."
echo "=========================================="
