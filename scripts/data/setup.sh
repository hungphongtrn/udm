#!/usr/bin/env bash
# One-time data-machine setup: installs uv if missing, then syncs the uv project env (.venv) with the data group.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked --project "$ROOT" --no-default-groups --group data
echo "ok: run scripts/data/*.sh (they use 'uv run'); export HF_TOKEN=... for the push"
