#!/usr/bin/env bash
# End-to-end frozen-backbone loss study, each stage resumable (rerun the same command after an interruption):
#   1. pack_features.sh   HF train cache + vLLM validation prefix -> outputs/features_qwen3_5_4b_pack
#   2. lossgrid.sh        every loss subset x seeds; checkpoint = lowest validation loss; -> selected.json + heads
#   3. dindex_frozen.sh   full Decision Index 0.2.1: backbone once over the suite, then every selected head
#   scripts/train/frozen_pipeline.sh
# Env: WORKERS (lossgrid parallel runs; all share the one GPU, the memory-mapped features via the page cache, and the
# CPU cores, split evenly), GRID_CFG (configs/lossgrid_qwen3_5_4b.yaml), BACKEND / FEATS_OVERRIDES (stage 3, see
# dindex_frozen.sh), plus those of the three stage scripts.
# Stages run one after another, so the stage-3 backbone never shares the card with head training.
set -euo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
GRID_CFG="${GRID_CFG:-configs/lossgrid_qwen3_5_4b.yaml}"
"$HERE/pack_features.sh"
"$HERE/lossgrid.sh" "$GRID_CFG"
"$HERE/dindex_frozen.sh" outputs/lossgrid/qwen3_5_4b_pack/selected.json
