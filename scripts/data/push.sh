#!/usr/bin/env bash
# Append $OUT to hungphongtrn/udm-massive-typed: prints the dry-run plan first, then asks before uploading.
# Needs HF_TOKEN (or huggingface-cli login). Set YES=1 to skip the confirmation.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
py -m scrm_data.push --out "$OUT" --dry-run "$@"
if [ "${YES:-0}" != "1" ]; then
  read -r -p "Upload to the Hub now? [y/N] " ans
  [ "$ans" = "y" ] || { echo "aborted"; exit 1; }
fi
py -m scrm_data.push --out "$OUT" "$@"
