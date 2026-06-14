#!/usr/bin/env bash
# =============================================================================
# MIDLM Bidirectional Inference Server — v2 (Qwen2.5-3B + Smart Pooling + HNM)
# =============================================================================
# Loads the best-noise-robust Qwen2.5-3B-Instruct adapter and runs an
# OpenAI-compatible HTTP server on port 1235.  Designed to coexist with
# LM Studio (default port 1234) on a single 12GB GPU.
#
# Endpoints:
#   GET  /health
#   GET  /v1/models
#   POST /v1/chat/completions   (OpenAI-compatible, Open WebUI compatible)
#
# Logs:  logs/midlm_server.log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT_DIR="$PROJECT_DIR/training/midlm/adapters/trained_models_bidirectional_v2/Qwen2.5-3B-Instruct_midlm_bidirectional"
LOG_DIR="$PROJECT_DIR/experiments/midlm/logs"
HOST="${MIDLM_HOST:-0.0.0.0}"
PORT="${MIDLM_PORT:-1235}"
LOAD_IN_4BIT="${MIDLM_LOAD_IN_4BIT:-1}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_DIR"
    echo "Train first:  cd $PROJECT_DIR && uv run python training/midlm/train_qwen3b_best.py"
    exit 1
fi

PID_FILE="$LOG_DIR/midlm_server.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping previous server (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

EXTRA=""
if [ "$LOAD_IN_4BIT" = "1" ]; then
    EXTRA="--load_in_4bit"
else
    EXTRA="--no-load_in_4bit"
fi

echo "Launching MIDLM server (v2) on ${HOST}:${PORT}"
echo "  Checkpoint : $CHECKPOINT_DIR"
echo "  4-bit      : $LOAD_IN_4BIT"
echo "  Log        : $LOG_DIR/midlm_server.log"
echo

mkdir -p "$LOG_DIR"

nohup uv run python "$SCRIPT_DIR/midlm_inference_server.py" \
    --checkpoint "$CHECKPOINT_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA \
    --max_seq_length 384 \
    > "$LOG_DIR/midlm_server.log" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"
echo "Server PID: $PID"
echo "Tail logs with: tail -f $LOG_DIR/midlm_server.log"
echo "Stop with     : kill $PID"
