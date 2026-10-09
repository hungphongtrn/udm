#!/usr/bin/env bash
# scripts/train/lossgrid.sh <config.yaml> [key.sub=value ...]   (e.g. WORKERS=4 scripts/train/lossgrid.sh configs/lossgrid_qwen3_5_4b.yaml)
# Loss-function grid (all 15 subsets of ce/brier/bt/sigmoid x seeds) on cached frozen-backbone features. Resumable.
# Checkpoint = lowest validation loss; writes heads/<run>.pt and selected.json, then scores selected.json on the
# Decision Index with scripts/train/dindex_frozen.sh (its env applies; suite features are cached and reused, so a new
# grid only costs the scoring) into outputs/dindex_frozen/<basename of output_dir>_<scope>. DINDEX=0 skips scoring.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: lossgrid.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
py -m scrm.lossgrid --config "$CFG" ${WORKERS:+--workers "$WORKERS"} "$@"
[ "${DINDEX:-1}" = 0 ] && exit 0
OUT_DIR="$(py -c 'import sys; from scrm.lossgrid import load_grid_config
print(load_grid_config(sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")])["output_dir"])' "$CFG" "$@")"
"$REPO_ROOT/scripts/train/dindex_frozen.sh" "$OUT_DIR/selected.json"
