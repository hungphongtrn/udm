#!/usr/bin/env bash
# The all-wrong abstention screen on three GPUs (docs/hypotheses/allwrong-abstention.md). One line, from kev/:
#   nohup scripts/abstain/queue.sh > runs/queue.log 2>&1 &
# env: GPUS (default 0,1,2: overrides an exported CUDA_VISIBLE_DEVICES), SEED (0), HALF_FRAC (0.5), plus run_arm.sh's.
# Builds the all-wrong data once (CPU), then three lanes in parallel, each arm trained + evaluated by run_arm.sh
# (eval_arm.sh, then eval_allwrong.sh):
#   GPU 0: all-wrong eval of the full A, C, F (D1, no training) -> Ah -> Ch -> HBh
#   GPU 1: Sh -> STh (stage 2 on Sh) -> Rh
#   GPU 2: R2h -> RYh -> Fh (waits for Ch's marker; stage 2 on Ch)
# Each arm leaves runs/queue/<arm>.done or .failed and its output in runs/queue/<arm>.log; a failed arm does not stop
# its lane. Rerunning is safe: run_arm.sh only evaluates arms already trained, and evaluation skips finished splits.
set -uo pipefail
cd "$(dirname "$0")/../.."            # kev/
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2}
SEED=${SEED:-0}; Q=runs/queue
mkdir -p "$Q"; rm -f "$Q"/*.done "$Q"/*.failed
scripts/abstain/eval_allwrong.sh --build || { echo "all-wrong data build failed" >&2; exit 1; }

arm() {   # arm ARM GPU_INDEX
  echo "[$(date -Is)] start $1 on gpu $2"
  if scripts/pref/run_arm.sh "$1" "$SEED" "$2" > "$Q/$1.log" 2>&1; then touch "$Q/$1.done"; echo "[$(date -Is)] done $1"
  else touch "$Q/$1.failed"; echo "[$(date -Is)] FAILED $1 (log: $Q/$1.log)"; fi
}
full() {  # D1: the full-data arms already trained and evaluated; only their all-wrong pass
  for a in A C F; do
    r=runs/pref/$a/seed$SEED
    if [ -f "$r/head.pt" ]; then
      echo "[$(date -Is)] all-wrong eval $r"
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} scripts/abstain/eval_allwrong.sh "$r" > "$Q/$a.allwrong.log" 2>&1 || echo "[$(date -Is)] FAILED all-wrong eval $r (log: $Q/$a.allwrong.log)"
    else echo "[$(date -Is)] skip $r: not trained"; fi
  done
}
wait_for() { until [ -e "$Q/$1.done" ] || [ -e "$Q/$1.failed" ]; do sleep 300; done; }

( full; arm Ah 0; arm Ch 0; arm HBh 0 ) &
( arm Sh 1; arm STh 1; arm Rh 1 ) &
( arm R2h 2; arm RYh 2; wait_for Ch; arm Fh 2 ) &
wait
echo "[$(date -Is)] queue finished: $(ls "$Q" | grep -c '\.done$') done, failed: $(cd "$Q" && ls *.failed 2>/dev/null | tr '\n' ' ')"
S=runs/pref; s=seed$SEED
echo "reports:"
echo ".venv/bin/python scripts/abstain/allwrong_report.py --arm A=$S/A/$s --arm C=$S/C/$s --arm F=$S/F/$s --out runs/allwrong/report-full"
echo ".venv/bin/python scripts/abstain/allwrong_report.py$(for a in Ah Ch Fh HBh Rh R2h Sh STh RYh; do printf ' --arm %s=%s/%s/%s' $a $S $a $s; done) --out runs/allwrong/report-half"
