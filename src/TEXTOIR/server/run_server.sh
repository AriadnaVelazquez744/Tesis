#!/usr/bin/env bash
# ============================================================================ #
# TEXTOIR API Server — startup script for toothless (server)
#
# Usage:
#   ./run_server.sh [--model-dir PATH] [--port PORT] [--gpu GPU_ID]
#
# Environment variables (override CLI args):
#   TEXTOIR_MODEL_DIR    path to the best model checkpoint
#   TEXTOIR_PORT         server port (default 8081)
#   TEXTOIR_DEVICE       cuda or cpu (default: cuda if available)
#   TEXTOIR_METHOD       method name for metadata (default "msp")
#   TEXTOIR_DATASET      dataset name (default "oos")
#   TEXTOIR_THRESHOLD    IND/OOS threshold (default 0.5)
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - trained model checkpoint in MODEL_DIR/pytorch_model.bin
# ============================================================================ #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────────────
MODEL_DIR="${TEXTOIR_MODEL_DIR:-${1:-}}"
PORT="${TEXTOIR_PORT:-${2:-8081}}"
GPU="${3:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --port)      PORT="$2";       shift 2 ;;
        --gpu)       GPU="$2";        shift 2 ;;
        *) shift ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: TEXTOIR_MODEL_DIR not set and --model-dir not provided."
    echo "Usage: $0 [--model-dir PATH] [--port PORT]"
    exit 1
fi

if [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "ERROR: model weights not found at $MODEL_DIR/pytorch_model.bin"
    echo "Run train_benchmark.py first, or point --model-dir to a valid checkpoint."
    exit 1
fi

# ── uv virtual environment ───────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv-server"

if [ ! -d "$VENV_DIR" ]; then
    echo "[TEXTOIR] Creating uv venv at $VENV_DIR ..."
    uv venv --python 3.12 "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[TEXTOIR] Installing dependencies ..."
uv pip install --python "$VENV_DIR/bin/python" \
    -r "$SCRIPT_DIR/requirements.txt"

# ── Set env vars for the server ──────────────────────────────────────────────
export TEXTOIR_MODEL_DIR
export TEXTOIR_PORT
export TEXTOIR_DEVICE="${TEXTOIR_DEVICE:-cuda}"
export TEXTOIR_METHOD="${TEXTOIR_METHOD:-msp}"
export TEXTOIR_DATASET="${TEXTOIR_DATASET:-oos}"
export TEXTOIR_THRESHOLD="${TEXTOIR_THRESHOLD:-0.5}"
export TEXTOIR_BERT_MODEL="${TEXTOIR_BERT_MODEL:-bert-base-uncased}"
export TEXTOIR_KNOWN_CLS_RATIO="${TEXTOIR_KNOWN_CLS_RATIO:-0.75}"
export TEXTOIR_SEED="${TEXTOIR_SEED:-0}"

# ── Start server ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " TEXTOIR API Server"
echo "========================================================"
echo "  Model dir : $MODEL_DIR"
echo "  Port      : $PORT"
echo "  Device    : $TEXTOIR_DEVICE"
echo "  Method    : $TEXTOIR_METHOD"
echo "  Dataset   : $TEXTOIR_DATASET"
echo "  Threshold : $TEXTOIR_THRESHOLD"
echo "========================================================"
echo ""

cd "$PROJECT_DIR"
python -m uvicorn src.TEXTOIR.server.api_server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1
