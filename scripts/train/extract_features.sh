#!/usr/bin/env bash
# scripts/train/extract_features.sh <config.yaml> [key.sub=value ...]
# Frozen-backbone feature cache (Experiment 2), e.g.
#   scripts/train/extract_features.sh configs/features_qwen3_5_4b.yaml features.max_sets.train=2000
# Resumable: finished shards (listed in <features.out_dir>/manifest.json) are skipped on re-run.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: extract_features.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
py -m scrm.features --config "$CFG" "$@"
