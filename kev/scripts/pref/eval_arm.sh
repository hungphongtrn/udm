#!/usr/bin/env bash
# Score a trained run. Usage: scripts/pref/eval_arm.sh RUN_DIR      (env: TEST=1, CUDA_VISIBLE_DEVICES)
# Splits (kev.benchmark --split): development (default), calibration (fit G on it), test (LOCKED; needs --allow-test).
# Output: RUN_DIR/eval/<suite>-<split>/{rows.json,report.json,reward_report.json}. benchmark refuses an existing --out, so done splits are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
RUN=${1:?run dir}
V7=evals/v7/decision-v7; T4=evals/v4/transfer-v4
CAL=$RUN/eval/decision-v7-calibration
bench() {  # suite name split [calibration dir: fit G on it]   (split "test" => --allow-test and no --split: benchmark picks the locked test itself)
  local out=$RUN/eval/$2-$3 sp="--split $3" cal=${4:-}; [ "$3" = test ] && sp="--allow-test"
  [ -f "$out/report.json" ] || .venv/bin/python -m kev.benchmark --run "$RUN" --suite "$1" $sp --out "$out"
  .venv/bin/python scripts/pref/reward_report.py "$out" ${cal:+--calib "$cal"}
}
# transfer-v4 is eval-only: its calibration split is empty by design (manifest: calibration "shared"), so it is scored with
# G fitted on decision-v7's calibration split, as kev.experiment scores a transfer suite at the primary suite's temperature.
bench $V7 decision-v7 calibration                     # first: every other split's G is fit on it
bench $V7 decision-v7 development "$CAL"
bench $T4 transfer-v4 development "$CAL"
if [ "${TEST:-0}" = 1 ]; then
  # !!! LOCKED TEST SPLIT. Read once, for the final verdict only; never for selection or search. Runs only with TEST=1. !!!
  bench $V7 decision-v7 test "$CAL"
  bench $T4 transfer-v4 test "$CAL"
fi
