# sourced by the other scripts: `py ...` runs Python in the uv project env with the training groups
# (CU=128 default -> torch cu128; CU=130 -> torch cu130). scripts/train/setup.sh syncs the env; re-run it after a
# pull that changes uv.lock. `uv run --no-sync`: the causal-conv1d wheels' METADATA version (1.7.0) differs from their
# filename version (1.7.0+cu13torch...), so a syncing `uv run` reinstalls causal-conv1d on every call.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CU="${CU:-128}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"   # fast Hub transfers (huggingface_hub >= 1.0 uses hf_xet)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation with variable-length packs
uvr() { uv run --no-sync --project "$REPO_ROOT" --no-default-groups --group data --group train --group "cu$CU" "$@"; }
py() { uvr python "$@"; }
