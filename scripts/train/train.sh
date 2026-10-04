#!/usr/bin/env bash
# scripts/train/train.sh <config.yaml> [--resume auto|DIR] [key.sub=value ...]
# NPROC>1 launches multi-GPU DDP via torchrun (e.g. NPROC=2 scripts/train/train.sh configs/scrm_qwen3_5_4b_h200.yaml).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: train.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  py -m torch.distributed.run --standalone --nproc_per_node "$NPROC" -m scrm.train --config "$CFG" "$@"
else
  py -m scrm.train --config "$CFG" "$@"
fi
