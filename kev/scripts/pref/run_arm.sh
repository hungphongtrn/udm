#!/usr/bin/env bash
# Train one arm, then evaluate it. Usage: scripts/pref/run_arm.sh ARM SEED [GPU]   (env: FORCE=1, BATCH_FALLBACK=1, BCE_W, F_EPOCHS, TEST=1)
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
source scripts/pref/arms.sh
ARM=${1:?arm}; SEED=${2:?seed}; GPU=${3:-${CUDA_VISIBLE_DEVICES:-0}}
OUT=runs/pref/$ARM/seed$SEED
if [ -e "$OUT" ]; then
  if [ "${FORCE:-0}" = 1 ]; then rm -rf "$OUT"
  else echo "refusing: $OUT exists (head.pt: $([ -f $OUT/head.pt ] && echo yes || echo no)); FORCE=1 to delete and rerun" >&2; exit 1; fi
fi
[ "$ARM" = F ] && [ ! -f "runs/pref/C/seed$SEED/head.pt" ] && { echo "arm F needs runs/pref/C/seed$SEED (run C first)" >&2; exit 1; }
FLAGS=$(arm_flags "$ARM" "$SEED")
# kev.train refuses an existing --out, so the log lives beside it and is copied in afterwards
mkdir -p "$(dirname "$OUT")"; LOG="$OUT.train.log"
echo "[$(date -Is)] arm $ARM seed $SEED gpu $GPU -> $OUT"; echo "flags: $FLAGS" | tee "$LOG"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m kev.train $FLAGS --out "$OUT" 2>&1 | tee -a "$LOG"
[ -f "$OUT/head.pt" ] || { echo "training failed: no $OUT/head.pt (log: $LOG)" >&2; exit 1; }
cp "$LOG" "$OUT/train.log"
CUDA_VISIBLE_DEVICES=$GPU scripts/pref/eval_arm.sh "$OUT"
