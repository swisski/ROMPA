#!/usr/bin/env bash
# Start the ROMP metrics frontend: FastAPI backend serving a single-page
# Plotly UI at http://127.0.0.1:8000.
#
# Usage:
#     ./frontend/run.sh
#
# Environment overrides:
#     PY               path to a python interpreter (default: auto-detect)
#     ROMP_FRONTEND_HOST  bind address (default: 127.0.0.1)
#     ROMP_FRONTEND_PORT  bind port (default: 8000)
#     ROMP_DATA_ROOT   data directory (default: ROMPA/demo/data)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

HOST="${ROMP_FRONTEND_HOST:-127.0.0.1}"
PORT="${ROMP_FRONTEND_PORT:-8000}"

# auto-pick richer data root if present (from ./frontend/link_aice_data.sh)
if [ -z "${ROMP_DATA_ROOT:-}" ] && [ -d "$REPO_ROOT/.data_aice/obs" ]; then
    export ROMP_DATA_ROOT="$REPO_ROOT/.data_aice"
fi

# --- pick an interpreter that has uvicorn + momp available ---
pick_python() {
    if [ -n "${PY:-}" ] && "$PY" -c "import uvicorn, momp" >/dev/null 2>&1; then
        echo "$PY"; return 0
    fi
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/../monsoon-bench/.venv/bin/python" \
        "python3" "python"; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c "import uvicorn, momp" >/dev/null 2>&1; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

if ! PY_BIN="$(pick_python)"; then
    cat >&2 <<'EOF'
ROMP frontend: no Python interpreter found that has both `uvicorn` and
`momp` installed. Set one up once with:

    uv venv .venv
    . .venv/bin/activate
    uv pip install -e '.[frontend]'

Then re-run ./frontend/run.sh (the script will auto-detect .venv).

You can also point PY at a specific interpreter:
    PY=/path/to/python ./frontend/run.sh
EOF
    exit 1
fi

echo "ROMP frontend: using $PY_BIN"
echo "  host:  $HOST:$PORT"
echo "  data:  ${ROMP_DATA_ROOT:-$REPO_ROOT/demo/data}"
echo

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
exec "$PY_BIN" -m uvicorn frontend.api.app:app --host "$HOST" --port "$PORT" --reload
