#!/usr/bin/env bash
# scripts/train/pack_features.sh [out_dir]
# One frozen-feature cache for the loss grid: the complete HF train split + the first VAL_SHARDS (512 sets each,
# source-balanced prefix) vLLM validation shards. Symlinks only; rerunnable.
#   TRAIN_FEATS=outputs/features_qwen3_5_4b VAL_FEATS=outputs/features_qwen3_5_4b_vllm_eval VAL_SHARDS=8
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
OUT="${1:-outputs/features_qwen3_5_4b_pack}"
py -m scrm.feature_pack --train "${TRAIN_FEATS:-outputs/features_qwen3_5_4b}" \
    --validation "${VAL_FEATS:-outputs/features_qwen3_5_4b_vllm_eval}" --val-shards "${VAL_SHARDS:-8}" --out "$OUT"
