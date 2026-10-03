#!/usr/bin/env bash
# scripts/train/eval.sh <ckpt_dir> [--split test] [--filter source_split=ood] [--out metrics.json] [key=value ...]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CKPT="${1:?usage: eval.sh <ckpt_dir> [args]}"; shift
cd "$REPO_ROOT"
py -m scrm.evaluate --ckpt "$CKPT" "$@"
