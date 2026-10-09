#!/usr/bin/env bash
# Full Decision Index 0.2.1 for frozen-backbone heads: the backbone featurizes the whole suite ONCE (resumable
# shards), then every head of the loss grid is scored on those features with the kit's own scorer.
#   scripts/train/dindex_frozen.sh [selected.json|head.pt ...]
# Env: BACKEND (hf: each question's prompt encoded once, options branch off its cache - the train cache's backend;
#      vllm: one request per option, ~10x+ more compute on DI's many-option questions), FEATS_CFG
#      (configs/features_qwen3_5_4b.yaml), DI_FEATS (outputs/dindex_feats_qwen3_5_4b_<backend>),
#      OUT (outputs/dindex_frozen/qwen3_5_4b_pack), SHARD_REQUESTS (1024), DI_DIR (../decision-index),
#      FEATS_OVERRIDES (hf: "features.max_batch_size=32 features.cache_tokens=65536"; vllm: bench_vllm.sh winner).
# Needs scripts/train/dindex_setup.sh first. Ctrl-C between shards is safe; rerun to resume.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
HEADS=("$@"); [ ${#HEADS[@]} -gt 0 ] || HEADS=(outputs/lossgrid/qwen3_5_4b_pack/selected.json)
BACKEND="${BACKEND:-hf}"
DI_FEATS="${DI_FEATS:-outputs/dindex_feats_qwen3_5_4b_$BACKEND}"
export DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
PY="${FEATURES_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python}"
# shellcheck disable=SC2086
"$PY" -m scrm.dindex_frozen featurize --config "${FEATS_CFG:-configs/features_qwen3_5_4b.yaml}" --out "$DI_FEATS" \
    --shard-requests "${SHARD_REQUESTS:-1024}" "features.backend=$BACKEND" ${FEATS_OVERRIDES:-}
"$PY" -m scrm.dindex_frozen score --feats "$DI_FEATS" --heads "${HEADS[@]}" \
    --out "${OUT:-outputs/dindex_frozen/qwen3_5_4b_pack}"
