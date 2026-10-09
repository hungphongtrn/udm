#!/usr/bin/env bash
# Decision Index 0.2.1 for frozen-backbone heads: the backbone featurizes the suite ONCE (resumable shards), then every
# head of the loss grid is scored on those features with the kit's own scorer.
#   scripts/train/dindex_frozen.sh [selected.json|head.pt ...]
# Env: SAMPLE (default $DI_DIR/sample-1000.jsonl.gz from dindex_setup.sh: fixed stratified sample, index = estimate;
#      SAMPLE= (empty) = full suite, ~150k requests, about a day on one 3090 even with BACKEND=hf),
#      BACKEND (hf: each question's prompt encoded once, options branch off its cache - the train cache's backend;
#      vllm: one request per option, ~10x+ more compute on DI's many-option questions), FEATS_CFG
#      (configs/features_qwen3_5_4b.yaml), DI_FEATS (outputs/dindex_feats_qwen3_5_4b_<backend>_<sample|full>),
#      OUT (outputs/dindex_frozen/qwen3_5_4b_pack_<sample|full>), SHARD_REQUESTS (128 with SAMPLE, else 1024: one
#      progress line + resume point per shard), DI_DIR (../decision-index),
#      FEATS_OVERRIDES (hf: "features.max_batch_size=32 features.cache_tokens=65536"; vllm: bench_vllm.sh winner).
# Needs scripts/train/dindex_setup.sh first. Ctrl-C between shards is safe; rerun to resume.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
HEADS=("$@"); [ ${#HEADS[@]} -gt 0 ] || HEADS=(outputs/lossgrid/qwen3_5_4b_pack/selected.json)
BACKEND="${BACKEND:-hf}"
export DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
SAMPLE="${SAMPLE-$DI_DIR/sample-1000.jsonl.gz}"
if [ -n "$SAMPLE" ]; then SCOPE="$(basename "$SAMPLE" .jsonl.gz)"; SAMPLE_ARGS=(--sample "$SAMPLE"); SHARD_DEFAULT=128
else SCOPE=full; SAMPLE_ARGS=(); SHARD_DEFAULT=1024; fi
DI_FEATS="${DI_FEATS:-outputs/dindex_feats_qwen3_5_4b_${BACKEND}_$SCOPE}"
PY="${FEATURES_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python}"
# shellcheck disable=SC2086
"$PY" -m scrm.dindex_frozen featurize --config "${FEATS_CFG:-configs/features_qwen3_5_4b.yaml}" --out "$DI_FEATS" \
    --shard-requests "${SHARD_REQUESTS:-$SHARD_DEFAULT}" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} "features.backend=$BACKEND" \
    ${FEATS_OVERRIDES:-}
"$PY" -m scrm.dindex_frozen score --feats "$DI_FEATS" --heads "${HEADS[@]}" \
    --out "${OUT:-outputs/dindex_frozen/qwen3_5_4b_pack_$SCOPE}"
