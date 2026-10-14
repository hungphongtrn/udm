#!/usr/bin/env bash
# Wave 1: arms C A B D E (+ F after C) for each seed, on any number of GPUs. Usage: scripts/pref/wave1.sh [GPUS]   e.g. "0" (default), "0,1", "0,1,2"
# env: SEEDS="0 1 2" (default "0"), RUN_C0=1 adds C0, RUN_F=0 skips F, plus run_arm.sh env (FORCE, BATCH_FALLBACK, BCE_W, F_EPOCHS, TEST)
# One worker per GPU; each takes the next unclaimed arm when it is free, so an arm that is only evaluated (already trained)
# frees its GPU early. F needs C's checkpoint: the worker that ran C runs F next. Seeds run one after another.
set -uo pipefail
cd "$(dirname "$0")/../.."            # kev/
IFS=, read -ra G <<< "${1:-0}"
SEEDS=${SEEDS:-0}
run() { scripts/pref/run_arm.sh "$1" "$2" "$3" || echo "FAILED: arm $1 seed $2" >&2; }   # one failure does not stop the queue
worker() {  # gpu seed claims arms...
  local gpu=$1 seed=$2 claims=$3; shift 3
  for a in "$@"; do
    mkdir "$claims/$a" 2>/dev/null || continue                    # atomic claim: another GPU has this arm
    run "$a" "$seed" "$gpu"
    if [ "$a" = C ] && [ "${RUN_F:-1}" = 1 ]; then run F "$seed" "$gpu"; fi
  done
}
for s in $SEEDS; do
  arms=(C A B D E); [ "${RUN_C0:-0}" = 1 ] && arms+=(C0)       # C first: F follows it on the same GPU
  claims=$(mktemp -d)
  for g in "${G[@]}"; do worker "$g" "$s" "$claims" "${arms[@]}" & done
  wait; rm -rf "$claims"
done
