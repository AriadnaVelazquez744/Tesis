#!/usr/bin/env bash
# cache_models.sh — Pre-download all models for Docker
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$PROJECT_DIR/storage/hf_cache"

echo "=== Step 1: bert-base-uncased (440 MB) ==="
.venv/bin/python -c "
from transformers import AutoConfig, BertTokenizer
AutoConfig.from_pretrained('bert-base-uncased')
BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print('bert-base-uncased OK')
"

echo "=== Step 2: all-MiniLM-L6-v2 (888 MB) ==="
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('all-MiniLM-L6-v2 OK')
"

echo "=== Step 3: AMR2-structbart-L (2.9 GB) ==="
export TORCH_HOME="$PROJECT_DIR/storage/torch_cache"
echo "TORCH_HOME=$TORCH_HOME"
.venv-amr/bin/python -c "
from transition_amr_parser.parse import AMRParser
AMRParser.from_pretrained('AMR2-structbart-L')
print('AMR2-structbart-L OK')
"

echo "=== Cache sizes ==="
du -sh storage/hf_cache/hub/*/
du -sh storage/torch_cache/