#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
if ! command -v npm >/dev/null 2>&1; then
    echo "Node.js 22.12+ and npm are required to build the React studio." >&2
    exit 1
fi
if [[ ! -d node_modules || package-lock.json -nt node_modules/.package-lock.json ]]; then
    npm ci --include=dev
fi
npm run build
