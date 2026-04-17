#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
UVICORN_LOG_LEVEL="info"

for arg in "$@"; do
    case "$arg" in
        --debug)
            LOG_LEVEL="DEBUG"
            UVICORN_LOG_LEVEL="debug"
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "Starting backend (LOG_LEVEL=$LOG_LEVEL)..."
source testing/.venv/bin/activate
LOG_LEVEL="$LOG_LEVEL" PYTHONPATH="$PROJECT_ROOT/testing" uvicorn gui.backend.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir testing/gui/backend \
  --log-level "$UVICORN_LOG_LEVEL" &
BACKEND_PID=$!

trap "kill $BACKEND_PID 2>/dev/null" EXIT

# Wait for the backend to accept HTTP requests BEFORE starting vite.
# Rationale: if the browser's first fetch() hits ECONNREFUSED, Chrome's
# socket pool marks 127.0.0.1:8000 as failed and throttles reconnects
# for several seconds — the gap is below the JS runtime and can't be
# shortened from the page. Blocking vite startup on a successful HTTP
# probe avoids the failure in the first place.
#
# We probe /api/sessions (cheap) rather than just TCP listen: uvicorn
# accepts connections during lifespan startup but doesn't serve until
# the generator yields, so TCP readiness alone isn't enough.
echo -n "Waiting for backend to accept requests"
for i in $(seq 1 60); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo "Backend exited before becoming ready" >&2
    exit 1
  fi
  if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:8000/api/sessions; then
    echo " (ready after $((i * 200))ms)"
    break
  fi
  echo -n "."
  sleep 0.2
  if [ "$i" = "60" ]; then
    echo ""
    echo "Backend didn't respond within 12s; starting frontend anyway" >&2
  fi
done

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
