#!/usr/bin/env bash
# scripts/train/lossgrid.sh <config.yaml> [key.sub=value ...]   (e.g. WORKERS=4 scripts/train/lossgrid.sh configs/lossgrid_qwen3_5_4b.yaml)
# Loss-function grid (all 15 subsets of ce/brier/bt/sigmoid x seeds) on cached frozen-backbone features. Resumable.
# Checkpoint = lowest validation loss; writes heads/<run>.pt and selected.json.
# With dindex.enabled (config or override) every head is also scored on the full Decision Index 0.2.1: the vLLM
# backbone first featurizes the suite once into dindex.feats_dir (skipped when complete; needs dindex_setup.sh), then
# training runs, then scoring. Env for that step as in dindex_frozen.sh: FEATS_CFG, SHARD_REQUESTS, DI_DIR,
# FEATS_OVERRIDES, FEATURES_PYTHON.
# WORKERS = parallel runs on the one GPU; each holds the train features (variant 0) + validation on the GPU.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: lossgrid.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
export DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
FEATS="$(py -m scrm.lossgrid --config "$CFG" --print-dindex-feats "$@" | tail -n 1)"
if [ -n "$FEATS" ]; then
    PY="${FEATURES_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python}"
    # shellcheck disable=SC2086
    "$PY" -m scrm.dindex_frozen featurize --config "${FEATS_CFG:-configs/features_qwen3_5_4b.yaml}" --out "$FEATS" \
        --shard-requests "${SHARD_REQUESTS:-1024}" ${FEATS_OVERRIDES:-}
fi
py -m scrm.lossgrid --config "$CFG" ${WORKERS:+--workers "$WORKERS"} "$@"
