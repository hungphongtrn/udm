# sourced by the other scripts
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then . "$REPO_ROOT/.venv/bin/activate"; fi
