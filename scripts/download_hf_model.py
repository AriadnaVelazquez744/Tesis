"""
Pre-download all-MiniLM-L6-v2 for KeyBERT into storage/hf_cache/ so that
the Docker container can mount it (avoiding SSL issues inside the container).

Usage:
    python scripts/download_hf_model.py

The model is saved to <repo_root>/storage/hf_cache/hub/
which is mounted at /root/.cache/huggingface inside the container.
"""
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cache_dir = os.path.join(repo_root, "storage", "hf_cache")
os.makedirs(cache_dir, exist_ok=True)

model_id = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Downloading {model_id} to {cache_dir} ...")
print(f"(This may take a few minutes on first run)")

try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"Download complete: {path}")
    sys.exit(0)
except ImportError:
    print("huggingface_hub not installed. Installing...")
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"]
    )
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"Download complete: {path}")
    sys.exit(0)
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
