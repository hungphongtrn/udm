#!/usr/bin/env bash
# Decision Index baseline of untrained LMs (scrm.base_eval.BaseLMEngine: hybrid-safe option log-prob scoring).
#   scripts/train/base_eval.sh                      # Qwen3.5-4B (instruct) and -Base on the training-time sample
#   MODELS="Qwen/Qwen3.5-4B" scripts/train/base_eval.sh
#   FULL=1 MODELS="Qwen/Qwen3.5-4B" scripts/train/base_eval.sh     # full suite, board-comparable (hours)
# Env: MAX_TOKENS (context limit; longer prompts are `unsupported` = wrong; default 65536 like the v2 train-time eval),
#      OPTION_BATCH (multi-token option keys scored per cache copy, halves on OOM), OUT_ROOT, DI_DIR.
# Resumable: re-running continues each model's results.jsonl. Sample index = train-time `dindex/index`.
# Needs scripts/train/dindex_setup.sh first (suite + sample under $DI_DIR).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
MODELS="${MODELS:-Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B-Base}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
OPTION_BATCH="${OPTION_BATCH:-16}"
OUT_ROOT="${OUT_ROOT:-outputs/base_eval}"
OPTS=(--option "max_tokens=$MAX_TOKENS" --option "option_batch=$OPTION_BATCH")
for M in $MODELS; do
  OUT="$OUT_ROOT/$(basename "$M")$([ -n "${FULL:-}" ] && echo _full || echo _sample)"
  echo "== $M -> $OUT"
  if [ -n "${FULL:-}" ]; then
    py -m decision_index pipeline --engine scrm.base_eval:BaseLMEngine --model "$M" "${OPTS[@]}" \
        --suite-dir "$DI_DIR/suite-0.2" --edition 0.2.1 --out "$OUT" --compact "$@"
    py -m scrm.dindex --scores "$OUT/scores.json" --board "$DI_DIR/tests/fixtures/board-0.2.1.json"
  else
    py -m scrm.base_eval --model "$M" "${OPTS[@]}" --suite-dir "$DI_DIR/suite-0.2" \
        --rows "$DI_DIR/sample-1000.jsonl.gz" --edition 0.2.1 --out "$OUT" "$@"
  fi
done
