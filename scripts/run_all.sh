#!/usr/bin/env bash
# Wrapper for scripts/run_all.py — run from repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  exec "$ROOT/venv/bin/python" "$ROOT/scripts/run_all.py" "$@"
fi
exec python3 "$ROOT/scripts/run_all.py" "$@"
