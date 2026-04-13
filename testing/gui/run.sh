#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting backend..."
source testing/.venv/bin/activate
PYTHONPATH="$PROJECT_ROOT/testing" uvicorn gui.backend.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir testing/gui/backend &
BACKEND_PID=$!

echo "Starting frontend..."
cd testing/gui/frontend
npm install
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

echo ""
echo "==================================="
echo "  LLM Surgeon GUI"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://127.0.0.1:8000"
echo "==================================="
echo ""

wait
