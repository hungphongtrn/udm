# sourced by the other scripts
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$ROOT/.venv-data/bin/activate" ]; then source "$ROOT/.venv-data/bin/activate"; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"; export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
OUT="${OUT:-$ROOT/build/scrm_out}"
CACHE="${CACHE:-$ROOT/build/hf_cache}"
WORKERS="${WORKERS:-$(nproc)}"
