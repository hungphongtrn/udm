#!/usr/bin/env bash
# Create a uv venv with the data-pipeline dependencies (uv is installed to ~/.local/bin if missing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$ROOT/.venv-data}"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -r "$ROOT/requirements-data.txt"
uv pip install -q --python "$VENV/bin/python" -e "$ROOT" --no-deps
echo "ok: source $VENV/bin/activate ; export HF_HUB_ENABLE_HF_TRANSFER=1 ; export HF_TOKEN=..."
