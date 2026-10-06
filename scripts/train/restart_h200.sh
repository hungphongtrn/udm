#!/usr/bin/env bash
# Fresh 2x H200 run of configs/scrm_qwen3_5_4b_h200.yaml (v2 recipe) -> outputs/scrm_qwen3_5_4b_h200_v2.
#   scripts/train/restart_h200.sh [key.sub=value ...]        e.g. train.max_steps=8000
#   RESUME=auto scripts/train/restart_h200.sh                 continue the v2 run from its last checkpoint
# Stop the old run first (it is refused while another scrm.train is alive). Logs: $OUT/train.log.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
CFG=configs/scrm_qwen3_5_4b_h200.yaml
OUT=outputs/scrm_qwen3_5_4b_h200_v2
if pgrep -f "scrm.train" >/dev/null; then
  echo "another scrm.train is running (pgrep -af scrm.train); stop it first" >&2; exit 1
fi
ARGS=()
if [ -n "${RESUME:-}" ]; then
  ARGS+=(--resume "$RESUME")
elif compgen -G "$OUT/step_*" >/dev/null; then
  echo "$OUT already has checkpoints; use RESUME=auto to continue or move it away for a fresh run" >&2; exit 1
fi
mkdir -p "$OUT"
NPROC=2 scripts/train/train.sh "$CFG" "${ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/train.log"
