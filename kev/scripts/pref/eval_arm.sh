#!/usr/bin/env bash
# Score a trained run. Usage: scripts/pref/eval_arm.sh RUN_DIR      (env: TEST=1, CUDA_VISIBLE_DEVICES)
# Splits (kev.benchmark --split): development (default), calibration (fit G on it), test (LOCKED; needs --allow-test).
# Output: RUN_DIR/eval/<suite>-<split>/{rows.json,report.json,reward_report.json}. benchmark refuses an existing --out, so done splits are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
RUN=${1:?run dir}
V7=evals/v7/decision-v7; T4=evals/v4/transfer-v4
bench() {  # suite name split   (split "test" => --allow-test and no --split: benchmark picks the locked test itself)
  local out=$RUN/eval/$2-$3 sp="--split $3"; [ "$3" = test ] && sp="--allow-test"
  [ -f "$out/report.json" ] || .venv/bin/python -m kev.benchmark --run "$RUN" --suite "$1" $sp --out "$out"
  local cal=$RUN/eval/$2-calibration   # G: fit on this run's calibration split of the same suite
  if [ "$3" != calibration ] && [ -f "$cal/rows.json" ]; then .venv/bin/python scripts/pref/reward_report.py "$out" --calib "$cal"
  else .venv/bin/python scripts/pref/reward_report.py "$out"; fi
}
bench $V7 decision-v7 calibration                     # first, so the development reports can include G
bench $T4 transfer-v4 calibration
bench $V7 decision-v7 development
bench $T4 transfer-v4 development
if [ "${TEST:-0}" = 1 ]; then
  # !!! LOCKED TEST SPLIT. Read once, for the final verdict only; never for selection or search. Runs only with TEST=1. !!!
  bench $V7 decision-v7 test
  bench $T4 transfer-v4 test
fi
