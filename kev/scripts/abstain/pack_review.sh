#!/usr/bin/env bash
# Final-review bundle of the all-wrong queue and everything its docs compare against. From the repo root:
#   ./kev.sh scripts/abstain/pack_review.sh [OUT.tgz]
# First reruns the two all-wrong reports (CPU, overwritten in place) on every arm whose all-wrong evals exist, then packs
# into one tgz: per pref run the logs, training metrics, eval outputs, RewardBench 2 JSONs and Decision Index subset dirs;
# runs/allwrong/report*/ and the all-wrong data manifests; runs/queue/*.log (G0, G4, every arm) and runs/*.log;
# runs/dindex/ (comparators); runs/dmb/ (Lu & Wu adapter runs, scored metrics, policies). No weights, no datasets.
# Per-row bulk is left out unless FULL=1 (~15 MB compressed per dmb arm, plus every DI results.jsonl): dmb
# predictions.jsonl / raw_rewards.jsonl / policies scores.jsonl and DI results.jsonl*; the review reads the summaries
# next to them (metrics.json, report.md, policies.json/md, scores.json, benchmark-summary.json). dmb cases.jsonl /
# requests.jsonl are always left out: byte-identical to runs/dmb/core-inputs (digests in each manifest.json).
# env: SEED (0), REPORTS (1; 0 packs the existing reports unchanged), FULL (0).
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
BULK=(); [ "${FULL:-0}" = 1 ] || BULK=(! -name predictions.jsonl ! -name raw_rewards.jsonl ! -name scores.jsonl ! -name 'results.jsonl*')
LIST=$(mktemp)
find runs -name .ipynb_checkpoints -prune -o -path runs/dmb/core-inputs -prune -o -type f \
  ! -name '*.pt' ! -name '*.safetensors' ! -name '*.bin' \( \
    -path "$S/*" \( -name COMMIT -o -name '*.train.log' -o -path '*/seed*/train.log' -o -path '*/seed*/training_*.json' \
      -o -path '*/eval/*' -o -path '*/seed*/dindex-*' ${BULK[@]+"${BULK[@]}"} -o -name '*.manifest.json' -o -name 'rewardbench2*.json' \) \
    -o -path 'runs/allwrong/report*' -o -path 'runs/allwrong/data/*/manifest.json' \
    -o -path 'runs/queue/*.log' -o -regex 'runs/[^/]*\.log' -o -path runs/COMMIT -o -path 'runs/dindex/*' ${BULK[@]+"${BULK[@]}"} \
    -o -path 'runs/dmb/*' ! -name cases.jsonl ! -name requests.jsonl ${BULK[@]+"${BULK[@]}"} \) -print | sort > "$LIST"
tar czf "$OUT" -T "$LIST"
echo "largest dirs (uncompressed MB):"
tr '\n' '\0' < "$LIST" | du -b --files0-from=- | awk '{n=split($2,p,"/"); k=p[1]"/"p[2]"/"p[3]; s[k]+=$1} END{for(k in s) printf "%9.1f %s\n", s[k]/1e6, k}' | sort -rn | head -12
rm -f "$LIST"
echo "queue: $(ls runs/queue/*.done 2>/dev/null | wc -l) done, failed: $(cd runs/queue && ls *.failed 2>/dev/null | tr '\n' ' ')"
echo "gates: $(cd runs/queue && grep -h -E 'G[04] (PASS|FAIL)' G0-parity.log G4-*.log 2>/dev/null | tr '\n' ' ')"
for r in "$S"/*/seed*/; do
  [ -f "$r/head.pt" ] || continue
  printf '%-14s eval %-60s rb2 %-3s dindex %s\n' "${r#$S/}" "$(cd "$r" && ls -d eval/*/ 2>/dev/null | sed 's#eval/##; s#/$##' | tr '\n' ' ')" \
    "$([ -f "$r/rewardbench2.json" ] && echo ok || echo -)" "$(cd "$r" && ls -d dindex-* 2>/dev/null | tr '\n' ' ')"
done
echo "wrote $OUT ($(du -h --apparent-size "$OUT" | cut -f1), $(tar tzf "$OUT" | wc -l) files)"
