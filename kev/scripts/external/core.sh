#!/usr/bin/env bash
# Lu & Wu's Decision Model Benchmark (core profile) on our checkpoints, end to end (scripts/external/README.md).
#   ./kev.sh scripts/external/core.sh                 # setup only: their package, frozen inputs, contamination gate
#   GPU=<uuid or index> ./kev.sh scripts/external/core.sh A Ah    # then, per arm: our adapter, their scorer, our policies
# Arms are names under runs/pref/<ARM>/seed$SEED. env: GPU (default 0; an index is nvidia-smi order), SEED (0),
# DMB_HOME (default $HOME/dmb: their clone, venv and dataset cache). Every finished step is skipped on a rerun; an
# adapter run dir without its runs/dmb/<name>.done marker is an interrupted run and is refused (delete it to rerun).
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
DMB_HOME=${DMB_HOME:-$HOME/dmb}; SEED=${SEED:-0}
SRC=$DMB_HOME/Decision_Model_Benchmark; VENV=$DMB_HOME/venv; RES=$SRC/decision_benchmark/resources
INPUTS=runs/dmb/core-inputs; OVERLAP=runs/dmb/overlap-core.json
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID} CUDA_VISIBLE_DEVICES=${GPU:-0}
log() { echo "[$(date -Is)] $*"; }

mkdir -p "$DMB_HOME" runs/dmb
if [ ! -d "$SRC" ]; then log "clone v0.2.0 -> $SRC"; git clone --depth 1 --branch v0.2.0 https://github.com/luckykevvv/Decision_Model_Benchmark "$SRC"; fi
if [ ! -x "$VENV/bin/decision-benchmark" ]; then
  log "venv -> $VENV"; python3 -m venv "$VENV"; "$VENV/bin/pip" install -q -e "$SRC[data]"
fi
DMB=$VENV/bin/decision-benchmark
[ "$($DMB --version 2>&1)" = "0.2.0" ] || { echo "expected decision-benchmark 0.2.0, got: $($DMB --version 2>&1)" >&2; exit 1; }
$DMB verify
if [ ! -f "$INPUTS/manifest.json" ]; then
  log "prepare core --with-text -> $INPUTS"
  $DMB prepare --profile core --with-text --output-dir "$INPUTS" --cache-dir "$DMB_HOME/cache/datasets"
fi
if [ ! -f "$OVERLAP" ]; then
  log "contamination gate -> $OVERLAP"
  .venv/bin/python scripts/external/overlap.py --cases "$INPUTS" --suite evals/v7/decision-v7 --split train --out "$OVERLAP"
fi

for arm in "$@"; do
  ckpt=runs/pref/$arm/seed$SEED; name=$(echo "$arm" | tr '[:upper:]' '[:lower:]')-s$SEED-core
  run=runs/dmb/$name; scored=runs/dmb/$name-scored; pol=runs/dmb/$name-policies
  [ -f "$ckpt/head.pt" ] || { log "skip $arm: $ckpt not trained"; continue; }
  if [ ! -f "$run.done" ]; then
    [ ! -e "$run" ] || { echo "$run exists without $run.done (interrupted adapter run): delete it to rerun $arm" >&2; exit 1; }
    log "$arm: adapter on gpu $CUDA_VISIBLE_DEVICES -> $run"
    .venv/bin/python scripts/external/luwu_run.py --inputs "$INPUTS" --ckpt "$ckpt" --out "$run"
    touch "$run.done"
  fi
  if [ ! -f "$scored/metrics.json" ]; then
    log "$arm: their scorer -> $scored"; $DMB score --profile core --run-dir "$run" --output-dir "$scored"
  fi
  if [ ! -f "$pol/policies.json" ]; then
    log "$arm: policies -> $pol"
    .venv/bin/python scripts/external/luwu_policies.py --run "$run" --overlap "$OVERLAP" --subsets "$RES/subsets.json" \
      --reference "$RES" --out "$pol"
  fi
  log "$arm: done ($pol/policies.md)"
done
