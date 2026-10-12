#!/usr/bin/env bash
# Pack every pref run's logs, training metrics and eval outputs (no weights) for download. Usage: scripts/pref/pack_results.sh [OUT.tgz]
# Prints, per run, which eval splits are complete (report.json + reward_report.json) and how many calibration rows G was fit on.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
OUT=${1:-pref-results-$(date +%Y%m%d-%H%M).tgz}
git rev-parse HEAD > runs/pref/COMMIT
LIST=$(mktemp)
find runs/pref -name .ipynb_checkpoints -prune -o -type f \( -name COMMIT -o -name '*.train.log' -o -path '*/seed*/train.log' \
  -o -path '*/seed*/training_*.json' -o -path '*/eval/*' \) -print | sort > "$LIST"
tar czf "$OUT" -T "$LIST"
for run in runs/pref/*/seed*/; do
  for split in "$run"eval/*/; do
    rr=$split/reward_report.json
    g=$( [ -f "$rr" ] && python3 -c 'import json,sys; g=json.load(open(sys.argv[1])).get("G"); print("G n_calib=%s" % g["n_calib"] if g else "no G")' "$rr" || echo "no reward_report")
    printf '%-28s %-26s %-10s %s\n' "${run#runs/pref/}" "$(basename "$split")" "$([ -f "$split/report.json" ] && echo ok || echo INCOMPLETE)" "$g"
  done
done
rm -f "$LIST"
echo "wrote $OUT ($(du -h --apparent-size "$OUT" | cut -f1), $(tar tzf "$OUT" | wc -l) files)"
