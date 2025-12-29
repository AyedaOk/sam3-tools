#!/bin/bash
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

VENV_PY="$APP_DIR/.venv/bin/python"
cd "$APP_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "Could not find venv python at: $VENV_PY"
  echo "Re-run the installer to recreate the venv."
  exit 1
fi

exec "$VENV_PY" main.py "$@"
