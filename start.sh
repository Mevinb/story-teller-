#!/bin/bash
# Story Teller — Launch Script
cd "$(dirname "$0")"

# Kill any existing instance on port 5000
kill $(lsof -t -i:5000) 2>/dev/null

echo "🖊️  Starting Story Teller..."
echo "   http://localhost:5000"
echo ""

exec ./venv/bin/python app.py
