#!/usr/bin/env bash
# Wave 1: arms A B C D E (then F after C) for each seed. Usage: scripts/pref/wave1.sh [GPUS]   e.g. "0" (default) or "0,1"
# env: SEEDS="0 1 2" (default "0"), RUN_C0=1 adds C0, RUN_F=0 skips F, plus run_arm.sh env (FORCE, BATCH_FALLBACK, BCE_W, F_EPOCHS, TEST)
set -uo pipefail
cd "$(dirname "$0")/../.."            # kev/
IFS=, read -ra G <<< "${1:-0}"
SEEDS=${SEEDS:-0}
run() { scripts/pref/run_arm.sh "$1" "$2" "$3" || echo "FAILED: arm $1 seed $2" >&2; }   # one failure does not stop the queue
queue() {  # gpu seed arms...
  local gpu=$1 seed=$2; shift 2
  for a in "$@"; do run "$a" "$seed" "$gpu"; done
}
for s in $SEEDS; do
  if [ ${#G[@]} -ge 2 ]; then
    # queue 1: A, C, F (F needs C, so they share a queue)   queue 2: B, D, E (+ C0)
    q1=(A C); [ "${RUN_F:-1}" = 1 ] && q1+=(F)
    q2=(B D E); [ "${RUN_C0:-0}" = 1 ] && q2+=(C0)
    queue "${G[0]}" "$s" "${q1[@]}" & queue "${G[1]}" "$s" "${q2[@]}" & wait
  else
    q=(A B C D E); [ "${RUN_C0:-0}" = 1 ] && q+=(C0); [ "${RUN_F:-1}" = 1 ] && q+=(F)
    queue "${G[0]}" "$s" "${q[@]}"
  fi
done
