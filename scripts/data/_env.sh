# sourced by the other scripts: `py ...` runs Python in the uv project env with the data group only (no torch)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py() { uv run --locked --project "$ROOT" --no-default-groups --group data python "$@"; }
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
OUT="${OUT:-$ROOT/build/scrm_out}"
CACHE="${CACHE:-$ROOT/build/hf_cache}"
WORKERS="${WORKERS:-$(nproc)}"
