#!/usr/bin/env bash
# Score a trained run on the all-wrong minimal pairs. Usage: scripts/abstain/eval_allwrong.sh RUN_DIR   (env: CUDA_VISIBLE_DEVICES)
#                                                         scripts/abstain/eval_allwrong.sh --build    (data only, CPU)
# Builds runs/allwrong/data/<suite>-development once (build_allwrong.py, CPU), then one kev.benchmark --data pass per
# condition into RUN_DIR/eval/allwrong-<suite>/<condition>/. A condition dir without report.json was interrupted and is
# redone; finished ones are skipped. Records over the context are rejected (rejected.json), never truncated; the report
# keeps only pairs every arm scored in every condition. G's temperature comes from RUN_DIR/eval/decision-v7-development/
# reward_report.json (scripts/pref/eval_arm.sh). Several runs in parallel: --build first (scripts/abstain/queue.sh does),
# else two of them may build the same data dir at once.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
RUN=${1:?run dir or --build}
SPECS="evals/v7/decision-v7:decision-v7 evals/v4/transfer-v4:transfer-v4"
for spec in $SPECS; do
  suite=${spec%%:*} name=${spec##*:}
  data=runs/allwrong/data/$name-development
  [ -f "$data/manifest.json" ] || { rm -rf "$data"; .venv/bin/python scripts/abstain/build_allwrong.py --suite "$suite" --out "$data"; }
done
[ "$RUN" = --build ] && exit 0
for spec in $SPECS; do
  name=${spec##*:} data=runs/allwrong/data/${spec##*:}-development
  for cond in answerable allwrong answerable_none allwrong_none; do
    out=$RUN/eval/allwrong-$name/$cond
    [ -f "$out/report.json" ] || { rm -rf "$out"; .venv/bin/python -m kev.benchmark --run "$RUN" --data "$data/$cond.jsonl" --out "$out"; }
  done
done
