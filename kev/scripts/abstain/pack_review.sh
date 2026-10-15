#!/usr/bin/env bash
# Final-review bundle of the all-wrong queue and everything its docs compare against. From the repo root:
#   ./kev.sh scripts/abstain/pack_review.sh [OUT.tgz]
# First reruns the two all-wrong reports (CPU, overwritten in place) on every arm whose all-wrong evals exist, then packs
# into one tgz: per pref run the logs, training metrics, eval outputs, RewardBench 2 JSONs and Decision Index subset dirs;
# runs/allwrong/report*/ and the all-wrong data manifests; runs/queue/*.log (G0, G4, every arm) and runs/*.log;
# runs/dindex/ (comparators); runs/dmb/ (Lu & Wu adapter runs, scores, policies) minus cases.jsonl / requests.jsonl,
# which are byte-identical to runs/dmb/core-inputs (their digests are in each manifest.json). No weights, no datasets.
# env: SEED (0), REPORTS (1; 0 packs the existing reports unchanged).
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
SEED=${SEED:-0}; S=runs/pref; s=seed$SEED
OUT=${1:-review-$(date +%Y%m%d-%H%M).tgz}
has_allwrong() { compgen -G "$1/eval/allwrong-*/allwrong/report.json" > /dev/null; }
report() {  # report OUT NAME=RUN...: the arms with all-wrong evals; the others are named and skipped
  local out=$1 args=() spec; shift
  for spec in "$@"; do
    if has_allwrong "${spec#*=}"; then args+=(--arm "$spec"); else echo "report $out: skip ${spec%%=*} (no all-wrong evals)"; fi
  done
  [ ${#args[@]} -gt 0 ] || return 0
  .venv/bin/python scripts/abstain/allwrong_report.py "${args[@]}" --out "$out" 2>&1 | tee "$out.stdout.txt"
  mv "$out.stdout.txt" "$out/stdout.txt"
}
if [ "${REPORTS:-1}" = 1 ]; then
  report runs/allwrong/report-full A=$S/A/seed0 A1=$S/A/seed1 A2=$S/A/seed2 C=$S/C/$s F=$S/F/$s
  half=(); for a in Ah Ch Fh HBh Rh R2h R2Bh Sh STh RYh Dh Hh PBh PRh PUh P2h P2Bh PLh; do half+=("$a=$S/$a/$s"); done
  report runs/allwrong/report-half "${half[@]}"
fi
git describe --always --dirty > runs/COMMIT
LIST=$(mktemp)
find runs -name .ipynb_checkpoints -prune -o -path runs/dmb/core-inputs -prune -o -type f \
  ! -name '*.pt' ! -name '*.safetensors' ! -name '*.bin' \( \
    -path "$S/*" \( -name COMMIT -o -name '*.train.log' -o -path '*/seed*/train.log' -o -path '*/seed*/training_*.json' \
      -o -path '*/eval/*' -o -path '*/seed*/dindex-*' -o -name '*.manifest.json' -o -name 'rewardbench2*.json' \) \
    -o -path 'runs/allwrong/report*' -o -path 'runs/allwrong/data/*/manifest.json' \
    -o -path 'runs/queue/*.log' -o -regex 'runs/[^/]*\.log' -o -path runs/COMMIT -o -path 'runs/dindex/*' \
    -o -path 'runs/dmb/*' ! -name cases.jsonl ! -name requests.jsonl \) -print | sort > "$LIST"
tar czf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "queue: $(ls runs/queue/*.done 2>/dev/null | wc -l) done, failed: $(cd runs/queue && ls *.failed 2>/dev/null | tr '\n' ' ')"
echo "gates: $(cd runs/queue && grep -h -E 'G[04] (PASS|FAIL)' G0-parity.log G4-*.log 2>/dev/null | tr '\n' ' ')"
for r in "$S"/*/seed*/; do
  [ -f "$r/head.pt" ] || continue
  printf '%-14s eval %-60s rb2 %-3s dindex %s\n' "${r#$S/}" "$(cd "$r" && ls -d eval/*/ 2>/dev/null | sed 's#eval/##; s#/$##' | tr '\n' ' ')" \
    "$([ -f "$r/rewardbench2.json" ] && echo ok || echo -)" "$(cd "$r" && ls -d dindex-* 2>/dev/null | tr '\n' ' ')"
done
echo "wrote $OUT ($(du -h --apparent-size "$OUT" | cut -f1), $(tar tzf "$OUT" | wc -l) files)"
