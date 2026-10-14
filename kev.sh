#!/usr/bin/env bash
# Run a Kev command from the repo root. The code, its .venv and runs/ stay in kev/; paths after the command are
# kev-relative (runs/..., evals/...).
#   ./kev.sh scripts/pref/run_arm.sh H 0 0                      # shell scripts run as-is
#   ./kev.sh scripts/pref/rb2_pooled.py runs/pref/H/seed0/rewardbench2.json   # *.py runs with kev/.venv/bin/python
#   ./kev.sh .venv/bin/python -m pytest tests/test_pref.py       # anything else runs inside kev/
set -euo pipefail
[ $# -gt 0 ] || { sed -n '2,7p' "$0"; exit 1; }
cd "$(dirname "$0")/kev"
case "$1" in
  *.py) exec .venv/bin/python "$@" ;;
  *) exec "$@" ;;
esac
