#!/usr/bin/env bash
# Pack runs/pref and runs/dindex (no weights, no per-row DI answers) plus kev/review/ and the *.log files in kev/ and in
# the checkout runs/ lives in. Usage: scripts/pref/review_pack.sh   (env: ROOT = dir holding the real runs/; default:
# where kev/runs points). Output: kev/review-<date>.tgz.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
ROOT=${ROOT:-$(dirname "$(readlink -f runs)")}
HERE=$PWD OUT=$PWD/review-$(date +%Y%m%d-%H%M).tgz LIST=$(mktemp)
[ -d "$ROOT/runs/pref" ] || { echo "no $ROOT/runs/pref: set ROOT=<dir holding runs/>"; exit 1; }
mkdir -p review
cd "$ROOT"
# weights, optimizer state, per-row DI answers (results.jsonl, the retemper copies) and kit work dirs stay behind
find runs/pref runs/dindex \( -name .ipynb_checkpoints -o -name 'checkpoint*' -o -name work \) -prune -o -type f \
  ! -name '*.safetensors' ! -name '*.bin' ! -name '*.pt' ! -name '*.pth' ! -name '*.ckpt' ! -name '*.gguf' \
  ! -name 'results.jsonl*' ! -name '*.part.gz' -size -100M -print | sort > "$LIST"
# what was left out for size (so nothing is silently missing)
find runs/pref runs/dindex -type f -size +100M ! -name '*.safetensors' ! -name '*.bin' ! -name '*.pt' ! -name 'results.jsonl*' \
  -exec ls -lh {} + > "$HERE/review/pack-skipped-large.txt" 2>/dev/null || true
tar czf "$OUT" -T "$LIST" \
  -C "$HERE" review $(cd "$HERE" && ls *.log 2>/dev/null) \
  -C "$ROOT" $(ls *.log 2>/dev/null)
rm -f "$LIST"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(tar tzf "$OUT" | wc -l) files)"
