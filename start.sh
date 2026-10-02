#!/bin/bash
# Story Teller — Launch Script
cd "$(dirname "$0")"

# 1. Clean up any leftover Story Teller instance on port 5000
OLD_PIDS=$(lsof -t -iTCP:5000 -sTCP:LISTEN 2>/dev/null || true)
if [[ -n "$OLD_PIDS" ]]; then
    for pid in $OLD_PIDS; do
        if ps -p "$pid" -o cmd= 2>/dev/null | grep -qE "gunicorn|app:app"; then
            echo "⚠️  Found previous Story Teller process on port 5000 (PID: $pid). Stopping..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    # Check if any Story Teller processes are still lingering on port 5000
    REMAINING_PIDS=$(lsof -t -iTCP:5000 -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$REMAINING_PIDS" ]]; then
        for pid in $REMAINING_PIDS; do
            if ps -p "$pid" -o cmd= 2>/dev/null | grep -qE "gunicorn|app:app"; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        sleep 0.5
    fi
    # If still in use by an unrelated service, exit
    FINAL_PIDS=$(lsof -t -iTCP:5000 -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$FINAL_PIDS" ]]; then
        echo "Port 5000 is already in use by another process ($FINAL_PIDS); Story Teller was not started." >&2
        exit 1
    fi
fi

echo "🖊️  Starting Story Teller..."
echo "   http://localhost:5000"
echo ""

# Compile when source files change. The web server serves the built assets;
# Vite is needed only for development/builds, not while using the studio.
if [[ ! -f static/app/index.html ]] ||
   [[ -n $(find frontend/src frontend/index.html frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig.json -type f -newer static/app/index.html -print -quit 2>/dev/null) ]]; then
    bash scripts/build_frontend.sh || exit 1
fi

# Make pip-installed CUDA runtime libraries visible to llama-cpp-python.
VENV_SITE="$PWD/venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

# Disable HuggingFace and TQDM progress bars to prevent Errno 5 EIO issues in threads/daemon process
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DISABLE_TQDM=1
export TQDM_DISABLE=1
export STORY_AUTO_RESUME=${STORY_AUTO_RESUME:-true}

# Cleanup function to kill server and all child workers when script or terminal closes
cleanup() {
    trap - SIGINT SIGTERM SIGHUP EXIT
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "🛑 Shutting down Story Teller (PID: $SERVER_PID)..."
        # First send SIGTERM to children (workers) and master
        pkill -P "$SERVER_PID" -TERM 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        # If still alive, force kill
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            pkill -P "$SERVER_PID" -KILL 2>/dev/null || true
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    exit 0
}

# Trap signals: terminal close (SIGHUP), Ctrl+C (SIGINT), kill (SIGTERM), and normal exit
trap cleanup SIGINT SIGTERM SIGHUP EXIT

# Start Gunicorn in the background and monitor it
./venv/bin/python -m gunicorn -c gunicorn.conf.py app:app &
SERVER_PID=$!

# Wait for server or termination
wait "$SERVER_PID" 2>/dev/null || true
cleanup
