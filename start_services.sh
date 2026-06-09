#!/bin/bash
set -e

echo "=========================================="
echo "Starting AMR FastAPI daemon (Python 3.8 venv)..."
echo "=========================================="
.venv-amr/bin/python -m uvicorn src.AMR.amr_api:app --host 127.0.0.1 --port 8001 &
AMR_PID=$!

echo "Waiting for AMR service to be ready..."
echo "Checking: http://127.0.0.1:8001/health"
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "WARNING: AMR service failed to start after $MAX_RETRIES retries"
        echo "Continuing without AMR — pipeline will degrade gracefully"
        break
    fi
    echo "Retry $RETRY_COUNT/$MAX_RETRIES - waiting 30s..."
    sleep 30
done

echo "=========================================="
AMR_STATUS=$(curl -s http://127.0.0.1:8001/health | cat)
echo "AMR health status: $AMR_STATUS"
echo "=========================================="

sleep 5

echo "=========================================="
echo "Starting Streamlit UI (Python 3.12 venv)..."
echo "=========================================="
.venv/bin/streamlit run main.py --server.port=8501 --server.address=0.0.0.0
