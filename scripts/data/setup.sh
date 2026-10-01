#!/usr/bin/env bash
# Create a venv with the data-pipeline dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$ROOT/.venv-data}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements-data.txt"
"$VENV/bin/pip" install -q -e "$ROOT" --no-deps
echo "ok: source $VENV/bin/activate ; export HF_HUB_ENABLE_HF_TRANSFER=1 ; export HF_TOKEN=..."
