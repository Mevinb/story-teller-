#!/bin/bash
# Story Teller — Launch Script
cd "$(dirname "$0")"

# Kill any existing instance on port 5000
kill $(lsof -t -i:5000) 2>/dev/null

echo "🖊️  Starting Story Teller..."
echo "   http://localhost:5000"
echo ""

# Make pip-installed CUDA runtime libraries visible to llama-cpp-python.
VENV_SITE="$PWD/venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

# Disable HuggingFace and TQDM progress bars to prevent Errno 5 EIO issues in threads/daemon process
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DISABLE_TQDM=1
export TQDM_DISABLE=1

exec ./venv/bin/gunicorn --worker-class gthread --threads 8 --workers 1 --bind 127.0.0.1:5000 app:app
