#!/usr/bin/env bash
set -euo pipefail

python src/base_models/douwnload_models.py

# Pre-cache the KeyBERT model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"