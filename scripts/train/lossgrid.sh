#!/usr/bin/env bash
# scripts/train/lossgrid.sh <config.yaml> [key.sub=value ...]   (e.g. WORKERS=4 scripts/train/lossgrid.sh configs/lossgrid_qwen3_5_4b.yaml)
# Loss-function grid (all 15 subsets of ce/brier/bt/sigmoid x seeds) on cached frozen-backbone features. Resumable.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: lossgrid.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
py -m scrm.lossgrid --config "$CFG" ${WORKERS:+--workers "$WORKERS"} "$@"
