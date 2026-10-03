# sourced by the other scripts: `py ...` runs Python in the uv project env with the training groups
# (CU=128 default -> torch cu128; CU=130 -> torch cu130). uv syncs the env on first use (scripts/train/setup.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CU="${CU:-128}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
uvr() { uv run --locked --project "$REPO_ROOT" --no-default-groups --group data --group train --group "cu$CU" "$@"; }
py() { uvr python "$@"; }
