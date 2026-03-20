#!/usr/bin/env bash
set -euo pipefail

# Streamlit para la interfaz del proyecto.
streamlit run main.py \
  --server.address 0.0.0.0 \
  --server.port 8501

