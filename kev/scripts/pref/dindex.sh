#!/usr/bin/env bash
# Decision Index 0.3 (public index) for a Kev checkpoint, through scripts/pref/dindex_engine.py.
# Usage (from anywhere; paths resolve against kev/):
#   scripts/pref/dindex.sh setup                  clone the kit (pinned), install it into kev/.venv, build the suite (~7 GB dl, 17 GB disk)
#   scripts/pref/dindex.sh sample RUN [NAME]      100-request stratified sample: smoke + latency; out: RUN/dindex-0.3-sample
#   scripts/pref/dindex.sh full   RUN [NAME]      the whole suite (140,620 requests), scored at the end; out: RUN/dindex-0.3
#   scripts/pref/dindex.sh index  RUN [NAME]      only the rows the index reads (~110k: no display-only benchmarks, no
#                                                 excluded rows); same index.json value as `full`, scores.json `complete`
#                                                 false (not a board submission); out: RUN/dindex-0.3 (shared with `full`)
# RUN: a checkpoint dir (runs/pref/D/seed0) or a Hub id (jaredpalmer/kev-4b@v7-base: the published base-recipe comparator).
# NAME: output dir name override (needed for Hub ids): out goes to runs/dindex/NAME[-sample] instead.
# env: DI_DIR (kit checkout, default ~/decision-index), TEMP (default: checkpoint's own; e.g. 1.47 = D seed0's G),
#      DTYPE (bf16 default | fp32), CUDA_VISIBLE_DEVICES.
# Resume: rerun the same command; the kit keeps answered rows and retries only `error` rows.
set -euo pipefail
cd "$(dirname "$0")/../.."            # kev/
KEV=$PWD PY=$PWD/.venv/bin/python
DI_DIR=${DI_DIR:-$HOME/decision-index}
DI_COMMIT=9eb2dbe2a358004c8782c66e40a83ac07b953fec   # kit edition 0.3, the commit the engine was written against
# PYTHONUNBUFFERED: the kit prints its events without flushing, so through tee / a log file they would sit in the buffer
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONPATH=$KEV/scripts/pref${PYTHONPATH:+:$PYTHONPATH}

cmd=${1:?setup | sample RUN | full RUN | index RUN}
if [ "$cmd" = setup ]; then
  [ -d "$DI_DIR/.git" ] || git clone https://github.com/apolinario/decision-index "$DI_DIR"
  git -C "$DI_DIR" fetch -q origin && git -C "$DI_DIR" checkout -q $DI_COMMIT
  uv pip install --python "$PY" -e "$DI_DIR[rebuild]"
  # the suite needs the gated HLE dataset: accept its terms on https://huggingface.co/datasets/cais/hle and `hf auth login` first
  cd "$DI_DIR"
  if [ ! -f suite-0.3/gsm8k-rows.jsonl.gz ]; then   # the last file `suite import` stages
    "$PY" -m decision_index suite rebuild --work work
    "$PY" -m decision_index suite import \
      --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
      --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
      --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
  fi
  "$PY" -m decision_index suite verify
  exit
fi

RUN=${2:?checkpoint dir or Hub id}
[ -d "$RUN" ] && RUN=$(cd "$RUN" && pwd)   # the kit runs from DI_DIR, so a local checkpoint needs an absolute path
if [ -n "${3:-}" ]; then OUT=$KEV/runs/dindex/$3
elif [ -d "$RUN" ]; then OUT=$RUN/dindex-0.3
else echo "a Hub id needs NAME (3rd arg)" >&2; exit 1; fi
opts=(--engine dindex_engine:KevEngine --option "run=$RUN" --option "dtype=${DTYPE:-bf16}")
[ -n "${TEMP:-}" ] && opts+=(--option "temperature=$TEMP")
cd "$DI_DIR"
case $cmd in
  sample)
    [ -f sample-100.jsonl.gz ] || "$PY" -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
    "$PY" -m decision_index run "${opts[@]}" --rows sample-100.jsonl.gz --out "$OUT-sample"
    "$PY" -m decision_index score --results "$OUT-sample/results.jsonl" ;;
  full)
    mkdir -p "$OUT"
    "$PY" -m decision_index pipeline "${opts[@]}" --compact --out "$OUT" 2>&1 | tee -a "$OUT/pipeline.log" ;;
  index)
    # the index reads only the five areas' benchmarks, on non-excluded rows (decision_index/scoring/index02.py);
    # the display-only ones (not_in_index: MMLU, RouterBench, ARC-E/C, SGD, SimpleBench) are shown, never counted
    [ -f index-rows-0.3.jsonl.gz ] || "$PY" - <<'EOF'
import gzip, json, pathlib, decision_index
from decision_index.suite.io import Suite, dumps
spec = json.loads((pathlib.Path(decision_index.__file__).parent / "data/index-0.3.json").read_text())
ids = {n for a in spec["areas"] for n in a["benchmarks"]}
with gzip.open("index-rows-0.3.jsonl.gz.part", "wt", encoding="utf-8") as f:
    n = sum(f.write(dumps(r) + "\n") > 0 for r in Suite("suite-0.3", "0.3").rows(apply_exclusions=True) if r["_evaluation"]["catalog_id"] in ids)
pathlib.Path("index-rows-0.3.jsonl.gz.part").rename("index-rows-0.3.jsonl.gz")
print(json.dumps({"index_rows": n, "benchmarks": len(ids)}))
EOF
    mkdir -p "$OUT"
    "$PY" -m decision_index pipeline "${opts[@]}" --rows index-rows-0.3.jsonl.gz --compact --out "$OUT" 2>&1 | tee -a "$OUT/pipeline.log" ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
