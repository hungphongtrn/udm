#!/usr/bin/env bash
# End-to-end frozen-backbone loss study, each stage resumable (rerun the same command after an interruption):
#   1. pack_features.sh   HF train cache + vLLM validation prefix -> outputs/features_qwen3_5_4b_pack
#   2. lossgrid.sh        Decision Index suite featurized once (vLLM), then every loss subset x seeds (checkpoint =
#                         lowest validation loss), then every head scored on the full Decision Index 0.2.1
#                         (config dindex.enabled; with it off, run dindex_frozen.sh <output_dir>/selected.json after)
#   scripts/train/frozen_pipeline.sh
# Env: WORKERS (lossgrid parallel runs; all share the one GPU, each holding the train + validation features on it),
# GRID_CFG (configs/lossgrid_qwen3_5_4b.yaml), FEATS_OVERRIDES (fastest bench_vllm.sh setting), plus those of the stage
# scripts. Run bench_vllm.sh first (alone on the GPU) to choose FEATS_OVERRIDES.
# Stages run one after another, so vLLM (featurizing) never shares the card with head training.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
GRID_CFG="${GRID_CFG:-configs/lossgrid_qwen3_5_4b.yaml}"
"$HERE/pack_features.sh"
"$HERE/lossgrid.sh" "$GRID_CFG"
