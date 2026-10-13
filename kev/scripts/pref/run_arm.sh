#!/usr/bin/env bash
# Train one arm, then evaluate it (scripts/pref/eval_arm.sh, then the all-wrong minimal pairs: scripts/abstain/eval_allwrong.sh).
# Usage: scripts/pref/run_arm.sh ARM SEED [GPU]   (env: FORCE=1, BATCH_FALLBACK=1, ROW_BUDGET, BCE_W, F_EPOCHS, HALF_FRAC, TEST=1, ALLWRONG=0 skips the all-wrong eval)
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
source scripts/pref/arms.sh
ARM=${1:?arm}; SEED=${2:?seed}; GPU=${3:-0}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}   # physical indices = nvidia-smi order
# chat-format arms (C D E F H HP) run K option rows of varying length per question; the caching allocator fragments on them
# (arm C seed 0 ran out with 4 GiB reserved but unallocated). Allocator only: no effect on the numbers, so arms stay comparable.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# GPU is an index into the caller's CUDA_VISIBLE_DEVICES when that is set (e.g. two exported GPU UUIDs: 0 = the first),
# else a physical index; a UUID / non-numeric value is used as is
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ $GPU =~ ^[0-9]+$ ]]; then
  IFS=, read -ra _VIS <<< "$CUDA_VISIBLE_DEVICES"
  [ "$GPU" -lt "${#_VIS[@]}" ] || { echo "GPU index $GPU out of range for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 1; }
  GPU=${_VIS[$GPU]}
fi
OUT=runs/pref/$ARM/seed$SEED
if [ -e "$OUT" ]; then
  if [ "${FORCE:-0}" = 1 ]; then rm -rf "$OUT"
  elif [ -f "$OUT/head.pt" ]; then   # trained: evaluate only (eval_arm.sh skips splits already scored)
    echo "[$(date -Is)] arm $ARM seed $SEED already trained ($OUT/head.pt): evaluating only"
    CUDA_VISIBLE_DEVICES=$GPU scripts/pref/eval_arm.sh "$OUT"
    [ "${ALLWRONG:-1}" = 0 ] || CUDA_VISIBLE_DEVICES=$GPU scripts/abstain/eval_allwrong.sh "$OUT"; exit
  else echo "refusing: $OUT exists without head.pt (unfinished training; LoRA runs cannot resume); FORCE=1 or delete it to rerun" >&2; exit 1; fi
fi
# stage-2 arms warm-start from their stage 1 (arms.sh --init_from)
case $ARM in F) DEP=C ;; Fh) DEP=Ch ;; STh) DEP=Sh ;; *) DEP= ;; esac
[ -n "$DEP" ] && [ ! -f "runs/pref/$DEP/seed$SEED/head.pt" ] && { echo "arm $ARM needs runs/pref/$DEP/seed$SEED (run $DEP first)" >&2; exit 1; }
FLAGS=$(arm_flags "$ARM" "$SEED")
# kev.train refuses an existing --out, so the log lives beside it and is copied in afterwards
mkdir -p "$(dirname "$OUT")"; LOG="$OUT.train.log"
echo "[$(date -Is)] arm $ARM seed $SEED gpu $GPU -> $OUT"; echo "flags: $FLAGS" | tee "$LOG"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m kev.train $FLAGS --out "$OUT" 2>&1 | tee -a "$LOG"
[ -f "$OUT/head.pt" ] || { echo "training failed: no $OUT/head.pt (log: $LOG)" >&2; exit 1; }
cp "$LOG" "$OUT/train.log"
CUDA_VISIBLE_DEVICES=$GPU scripts/pref/eval_arm.sh "$OUT"
[ "${ALLWRONG:-1}" = 0 ] || CUDA_VISIBLE_DEVICES=$GPU scripts/abstain/eval_allwrong.sh "$OUT"
