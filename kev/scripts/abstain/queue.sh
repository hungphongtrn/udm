#!/usr/bin/env bash
# The all-wrong abstention screen on three GPUs (docs/hypotheses/in-progress/allwrong-abstention.md). One line, from
# the repo root:   nohup ./kev.sh scripts/abstain/queue.sh > kev/runs/queue.log 2>&1 &
# env: GPUS (default 0,1,2: overrides an exported CUDA_VISIBLE_DEVICES), SEED (0), HALF_FRAC (0.5), DI (1; 0 skips the
# Decision Index subsets), plus run_arm.sh's. Chat-format arms run at BATCH_FALLBACK=1 (1 record x accum 8, what C-F
# trained at; 2 x 4 risks running out on a 24 GB 3090), the Kev-format Ah at the default 4 x 2.
# Builds the all-wrong data once (CPU), then three lanes in parallel, each arm trained + evaluated by run_arm.sh
# (eval_arm.sh, then eval_allwrong.sh):
# P2h (docs/hypotheses/todo/pointwise-rcr-mlp.md) and its leave-one-out arms go first in each lane:
#   GPU 0: G0 fork parity (0.8B, minutes) -> Hh -> Dh -> all-wrong eval of the full A seeds 0-2, C, F (D1 / E0.2, no
#          training) -> Ah -> Ch -> PLh -> Decision Index subset of A seeds 0-2 (E0.1)
#   GPU 1: PRh -> P2Bh -> PUh -> PBh -> Sh -> STh (stage 2 on Sh) -> HBh
#   GPU 2: P2h -> R2Bh -> R2h -> Rh -> RYh -> Fh (waits for Ch's marker; stage 2 on Ch)
# Then G4 (K=1 vs K) on the pointwise arms. Each arm leaves runs/queue/<arm>.done or .failed and its output in
# runs/queue/<arm>.log; a failed arm does not stop its lane. Rerunning is safe: run_arm.sh only evaluates arms already
# trained, and evaluation skips finished splits.
set -uo pipefail
cd "$(dirname "$0")/../.."            # kev/
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}   # as run_arm.sh: steps and arms index the same physical GPUs
SEED=${SEED:-0}; Q=runs/queue
mkdir -p "$Q"; rm -f "$Q"/*.done "$Q"/*.failed
scripts/abstain/eval_allwrong.sh --build || { echo "all-wrong data build failed" >&2; exit 1; }
IFS=, read -ra VIS <<< "$CUDA_VISIBLE_DEVICES"

arm() {   # arm ARM GPU_INDEX
  local fb=1; [ "$1" = Ah ] && fb=0
  echo "[$(date -Is)] start $1 on gpu $2"
  if BATCH_FALLBACK=$fb scripts/pref/run_arm.sh "$1" "$SEED" "$2" > "$Q/$1.log" 2>&1; then touch "$Q/$1.done"; echo "[$(date -Is)] done $1"
  else touch "$Q/$1.failed"; echo "[$(date -Is)] FAILED $1 (log: $Q/$1.log)"; fi
}
step() {  # step NAME GPU_INDEX CMD...: a non-training job with its own log
  local name=$1 gpu=$2; shift 2
  echo "[$(date -Is)] $name"
  CUDA_VISIBLE_DEVICES=${VIS[$gpu]} "$@" > "$Q/$name.log" 2>&1 || echo "[$(date -Is)] FAILED $name (log: $Q/$name.log)"
}
full() {  # D1 / E0.2: the full-data arms already trained and evaluated; only their all-wrong pass
  for r in runs/pref/A/seed0 runs/pref/A/seed1 runs/pref/A/seed2 runs/pref/C/seed$SEED runs/pref/F/seed$SEED; do
    if [ -f "$r/head.pt" ]; then step "allwrong-$(echo "$r" | cut -d/ -f3,4 | tr / -)" 0 scripts/abstain/eval_allwrong.sh "$r"
    else echo "[$(date -Is)] skip $r: not trained"; fi
  done
}
dindex() {  # E0.1: Kev's softmax on the shared Decision Index subset, every A seed (D's +3.31 needs this baseline)
  [ "${DI:-1}" = 0 ] && return
  for s in 0 1 2; do
    [ -f runs/pref/A/seed$s/head.pt ] && step "dindex-A-seed$s" 0 scripts/pref/dindex.sh subset runs/pref/A/seed$s
  done
}
wait_for() { until [ -e "$Q/$1.done" ] || [ -e "$Q/$1.failed" ]; do sleep 300; done; }

( step G0-parity 0 .venv/bin/python scripts/abstain/fork_check.py parity --device cuda
  arm Hh 0; arm Dh 0; full; arm Ah 0; arm Ch 0; arm PLh 0; dindex ) &
( arm PRh 1; arm P2Bh 1; arm PUh 1; arm PBh 1; arm Sh 1; arm STh 1; arm HBh 1 ) &
( arm P2h 2; arm R2Bh 2; arm R2h 2; arm Rh 2; arm RYh 2; wait_for Ch; arm Fh 2 ) &
wait
S=runs/pref; s=seed$SEED
for a in Hh PBh PRh PUh P2h P2Bh PLh; do
  [ -f "$S/$a/$s/head.pt" ] && step "G4-$a" 0 .venv/bin/python scripts/abstain/fork_check.py k1 --run "$S/$a/$s" --device cuda
done
echo "[$(date -Is)] queue finished: $(ls "$Q" | grep -c '\.done$') done, failed: $(cd "$Q" && ls *.failed 2>/dev/null | tr '\n' ' ')"
echo "gates: $(cd "$Q" && grep -h -E 'G[04] (PASS|FAIL)' G0-parity.log G4-*.log 2>/dev/null | tr '\n' ' ')"
echo "reports (from the repo root):"
echo "./kev.sh scripts/abstain/allwrong_report.py --arm A=$S/A/seed0 --arm A1=$S/A/seed1 --arm A2=$S/A/seed2 --arm C=$S/C/$s --arm F=$S/F/$s --out runs/allwrong/report-full"
echo "./kev.sh scripts/abstain/allwrong_report.py$(for a in Ah Ch Fh HBh Rh R2h R2Bh Sh STh RYh Dh Hh PBh PRh PUh P2h P2Bh PLh; do printf ' --arm %s=%s/%s/%s' $a $S $a $s; done) --out runs/allwrong/report-half"
