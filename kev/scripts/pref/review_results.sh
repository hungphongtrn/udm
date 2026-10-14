#!/usr/bin/env bash
# CPU only (~10-20 min: the H retemper rescores the subset twice). Usage: scripts/pref/review_results.sh   (env: DI_DIR)
# Writes kev/review/*.log: run state, reward_report tables (pointwise arms), DI pair/flips of H vs v7-base / D eager /
# D graphed, H retempered at its G temperature, rb2_pooled on every RewardBench 2 JSON. Then review_pack.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."            # kev/
KEV=$PWD PY=$PWD/.venv/bin/python DI_DIR=${DI_DIR:-$HOME/decision-index}
export PYTHONPATH=$KEV:$KEV/scripts/pref${PYTHONPATH:+:$PYTHONPATH}   # this worktree's kev wins over the editable install
SUB=("$PY" "$KEV/scripts/pref/dindex_subset.py")
mkdir -p review

# 0. state: code commit, which kev is imported, GPUs, what is still running, which outputs exist per run
{ git log --oneline -1; "$PY" -c 'import kev; print(kev.__file__)'; readlink -f runs
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  ps -eo pid,etime,args | grep -E 'run_arm|kev.train|kev.benchmark|rewardbench2|dindex|chat_speed' | grep -v grep
  for r in runs/pref/*/seed*/; do
    printf '%-24s eval: %s| rb2: %s| di: %s\n' "$r" "$(ls "$r"eval 2>/dev/null | tr '\n' ' ')" \
      "$(ls "$r" | grep '^rewardbench2' | tr '\n' ' ')" "$(ls "$r" | grep '^dindex' | tr '\n' ' ')"
  done
  ls -d runs/dindex/*/
} > review/state.log 2>&1

# 1. text tables for the pointwise arms (reward_report.py rewrites reward_report.json with the same numbers)
for a in H HP HL HB HBL; do
  cal=runs/pref/$a/seed0/eval/decision-v7-calibration
  for s in decision-v7-development transfer-v4-development; do
    d=runs/pref/$a/seed0/eval/$s
    if [ -f "$d/report.json" ]; then echo "== $d"; "$PY" scripts/pref/reward_report.py "$d" --calib "$cal"
    else echo "== $d: no report.json"; fi
  done
done > review/reward-reports.log 2>&1

# 2. Decision Index: H paired with v7-base, D eager, D graphed (index, areas, benchmarks); answer flips vs D
V7=$KEV/runs/dindex/kev4b-v7base-sub15000 D=$KEV/runs/pref/D/seed0/dindex-0.3-sub15000
DG=$KEV/runs/dindex/D0-graphed-sub15000 H=$KEV/runs/pref/H/seed0/dindex-0.3-sub15000
( cd "$DI_DIR" || exit 1
  for A in "$V7" "$D" "$DG"; do echo "== pair $A -> H"; "${SUB[@]}" pair "$A" "$H"; done
  for A in "$D" "$DG"; do echo "== flips $A vs H"; "${SUB[@]}" flips "$A" "$H"; done
) > review/dindex-pairs.log 2>&1

# 3. H's DI at T=1 and at its own fitted T (G), paired like D's check (D at its fitted T=1.47)
RR=runs/pref/H/seed0/eval/decision-v7-development/reward_report.json
if [ -f "$RR" ]; then
  HT=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["G"]["T"]; print(("%.2f" % t).rstrip("0").rstrip("."))' "$RR")
  ( cd "$DI_DIR" || exit 1
    "${SUB[@]}" retemper subset-0.3-15000.jsonl.gz "$H" 1 "$HT"
    echo "== v7-base vs H at T=$HT"; "${SUB[@]}" pair "$V7" "$H/T$HT"
    echo "== D at T=1.47 vs H at T=$HT"; "${SUB[@]}" pair "$D/T1.47" "$H/T$HT"
  ) > review/dindex-retemper-H.log 2>&1
else
  echo "no $RR: run gpu.sh first, then rerun this script" > review/dindex-retemper-H.log
fi

# 4. pooled AUROC / sigmoid ECE on every RewardBench 2 JSON (Platt from that run's G when it has one)
for j in runs/pref/*/seed*/rewardbench2*.json; do
  case $j in *.pooled.json) continue ;; esac
  rr=$(dirname "$j")/eval/decision-v7-development/reward_report.json
  echo "== $j"
  if [ -f "$rr" ]; then "$PY" scripts/pref/rb2_pooled.py "$j" --report "$rr"; else "$PY" scripts/pref/rb2_pooled.py "$j"; fi
done > review/rb2-pooled.log 2>&1

if grep -l -E 'Traceback|Error' review/*.log; then echo "^ these logs have errors"; else echo "review/ done, no errors"; fi
