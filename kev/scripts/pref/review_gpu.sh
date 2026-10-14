#!/usr/bin/env bash
# GPU, only what is missing: H's dev eval (rule 1 screen), H's and HL's RewardBench 2. Skips anything already done.
# Usage: GPU=<index> scripts/pref/review_gpu.sh   (default card 0; check nvidia-smi for a free one first)
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -f runs/pref/H/seed0/eval/transfer-v4-development/report.json ] || scripts/pref/eval_arm.sh runs/pref/H/seed0   # ~1 h
for a in H HL; do
  [ -f runs/pref/$a/seed0/rewardbench2.json ] || \
    .venv/bin/python scripts/pref/rewardbench2.py --run runs/pref/$a/seed0 --out runs/pref/$a/seed0/rewardbench2.json   # ~35 min each
done
echo "gpu.sh done"
