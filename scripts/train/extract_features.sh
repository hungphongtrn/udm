#!/usr/bin/env bash
# scripts/train/extract_features.sh <config.yaml> [key.sub=value ...]
# Frozen-backbone feature cache (Experiment 2), e.g.
#   scripts/train/extract_features.sh configs/features_qwen3_5_4b.yaml features.max_sets.train=2000
# First Ctrl-C/SIGTERM finishes and commits the current shard; repeat the command to resume.
# A second signal exits immediately; unfinished files are ignored and that shard is recomputed.
# Keep cache identity unchanged; batching caps may change. Never overwrite to resume.
# vLLM uses its isolated environment; FEATURES_PYTHON can select the HF reference environment.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CFG="${1:?usage: extract_features.sh <config.yaml> [overrides]}"; shift
cd "$REPO_ROOT"
FEATURES_PYTHON="${FEATURES_PYTHON:-$REPO_ROOT/.venv-vllm/bin/python}"
if [[ ! -x "$FEATURES_PYTHON" ]]; then
  printf 'Missing feature interpreter: %s\nSee docs/TRAINING.md, Frozen feature extraction, for setup.\n' "$FEATURES_PYTHON" >&2
  exit 1
fi
# No uv/shell process between the caller and Python: stop signals reach the checkpoint handler.
exec "$FEATURES_PYTHON" -m scrm.features --config "$CFG" "$@"
