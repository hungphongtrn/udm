#!/usr/bin/env bash
# Official full-suite Decision Index 0.2.1 run of a checkpoint (resumable), then its rank among the board entrants.
#   scripts/train/dindex_eval.sh <ckpt_dir> [out_dir] [--option max_len=16384] [--limit N] ...
# Needs scripts/train/dindex_setup.sh first. Unanswerable requests (longer than max_len) are `unsupported` = wrong.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CKPT="${1:?usage: dindex_eval.sh <ckpt_dir> [out_dir] [args]}"; shift
OUT="${1:-$CKPT/decision_index_full}"; [ $# -gt 0 ] && shift
DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
cd "$REPO_ROOT"
python -m decision_index pipeline --engine scrm.dindex:SCRMEngine --option "ckpt=$CKPT" \
    --suite-dir "$DI_DIR/suite-0.2" --edition 0.2.1 --out "$OUT" --compact "$@"
python -m scrm.dindex --scores "$OUT/scores.json" --board "$DI_DIR/tests/fixtures/board-0.2.1.json"
