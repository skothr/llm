#!/usr/bin/env bash
set -euo pipefail

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

# Preflight: curl is needed for the readiness probe below. Fail fast rather
# than hanging in a probe loop that can never succeed.
command -v curl >/dev/null 2>&1 || { echo "run.sh: curl is required but not installed" >&2; exit 1; }

cd "$PROJECT_ROOT"

# Signal propagation: we run each child in its own process group via `setsid`,
# then TERM the whole group on EXIT. Without this, uvicorn --reload's worker
# subprocess can get orphaned on Ctrl+C because `kill $PID` only targets the
# reloader. `kill -TERM -$PGID` hits the group.
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill -TERM "-$FRONTEND_PID" 2>/dev/null || kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill -TERM "-$BACKEND_PID" 2>/dev/null || kill "$BACKEND_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "Starting backend (LOG_LEVEL=$LOG_LEVEL)..."
# shellcheck disable=SC1091
source testing/.venv/bin/activate
setsid env LOG_LEVEL="$LOG_LEVEL" PYTHONPATH="$PROJECT_ROOT/testing" uvicorn gui.backend.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir testing/gui/backend \
  --log-level "$UVICORN_LOG_LEVEL" &
BACKEND_PID=$!

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
READY=0
echo -n "Waiting for backend to accept requests"
for i in $(seq 1 60); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo "Backend exited before becoming ready" >&2
    exit 1
  fi
  if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:8000/api/sessions; then
    echo " (ready after ~$((i * 200))ms)"
    READY=1
    break
  fi
  echo -n "."
  sleep 0.2
done
if [[ "$READY" -ne 1 ]]; then
  echo ""
  echo "Backend didn't respond within 12s — refusing to start frontend." >&2
  echo "(Starting vite now would re-open the Chrome ECONNREFUSED throttle" >&2
  echo " this probe was added to avoid.)" >&2
  exit 1
fi

echo "Starting frontend..."
cd testing/gui/frontend
npm install
setsid npm run dev &
FRONTEND_PID=$!

echo ""
echo "==================================="
echo "  LLM Surgeon GUI"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://127.0.0.1:8000"
echo "==================================="
echo ""

wait
