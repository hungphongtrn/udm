#!/usr/bin/env bash
# scripts/train/train.sh <config.yaml> [--resume auto|DIR] [key.sub=value ...]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: train.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
py -m scrm.train --config "$CFG" "$@"
