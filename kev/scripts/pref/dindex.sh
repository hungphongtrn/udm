#!/usr/bin/env bash
# Decision Index 0.3 (public index) for a Kev checkpoint, through scripts/pref/dindex_engine.py.
# Usage (from anywhere; paths resolve against kev/):
#   scripts/pref/dindex.sh setup                  clone the kit (pinned), install it into kev/.venv, build the suite (~7 GB dl, 17 GB disk)
#   scripts/pref/dindex.sh sample RUN [NAME]      100-request stratified sample: smoke + latency; out: RUN/dindex-0.3-sample
#   scripts/pref/dindex.sh full   RUN [NAME]      the whole suite (140,620 requests), scored at the end; out: RUN/dindex-0.3
#   scripts/pref/dindex.sh subset RUN [NAME]      the shared comparison subset (N env, default 15000 rows of the index's
#                                                 benchmarks; same rows for every checkpoint), scored on the subset by
#                                                 dindex_subset.py; out: RUN/dindex-0.3-sub$N; rows RUN/dindex-0.3 (a
#                                                 `full` run, finished or stopped) already answered are copied, not rerun
#   scripts/pref/dindex.sh check  D_RUN           the whole external check, published Kev 4B first: board Kev-4B r10 on the
#                                                 subset + compare with its board row (does our engine reproduce the
#                                                 board?), kev-4b@v7-base, then D_RUN; both retempered over TGRID and
#                                                 paired at shipped T (v7-base's own vs D_T), raw (T=1) and best-on-subset
#                                                 T. Stop D_RUN's `full` run first (one GPU). Rerun to resume.
#                                                 env: D_T (D's fitted T, default 1.47 = seed 0's G), TGRID (T values),
#                                                 TEMP (D_RUN's serving T only: match its full run so its rows are reused)
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

cmd=${1:?setup | sample RUN | full RUN | subset RUN | check D_RUN}
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

if [ "$cmd" = check ]; then
  SELF=$KEV/scripts/pref/dindex.sh D=${2:?D checkpoint dir}
  export N=${N:-15000}
  D=$(cd "$D" && pwd)
  if pgrep -f "decision_index pipeline .*run=$D( |$)" >/dev/null; then
    echo "D's full run is still running; stop it first (its answered rows are reused): pkill -f 'decision_index pipeline .*run=$D'" >&2; exit 1
  fi
  R10=jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101   # the board's "Kev 4B r10" (later commits: model card only)
  D_T=${D_T:-1.47} TGRID=${TGRID:-0.5 0.7 1 1.2 1.47 1.7 2 2.41 3 4}
  [[ " $TGRID " == *" $D_T "* ]] || TGRID="$TGRID $D_T"
  SUBPY=("$PY" "$KEV/scripts/pref/dindex_subset.py") V7=$KEV/runs/dindex/kev4b-v7base-sub$N DS=$D/dindex-0.3-sub$N
  TEMP= "$SELF" subset "$R10" kev4b-r10
  (cd "$DI_DIR" && "${SUBPY[@]}" compare "$KEV/runs/dindex/kev4b-r10-sub$N" kev-4b-r10)
  TEMP= "$SELF" subset jaredpalmer/kev-4b@v7-base kev4b-v7base
  "$SELF" subset "$D"
  cd "$DI_DIR"
  # shellcheck disable=SC2086   # TGRID splits into one argument per T
  "${SUBPY[@]}" retemper "subset-0.3-$N.jsonl.gz" "$V7" $TGRID
  # shellcheck disable=SC2086
  "${SUBPY[@]}" retemper "subset-0.3-$N.jsonl.gz" "$DS" $TGRID
  echo "== shipped: v7-base at its own T vs D at T=$D_T"; "${SUBPY[@]}" pair "$V7" "$DS/T$D_T"
  echo "== raw: both at T=1";                         "${SUBPY[@]}" pair "$V7/T1" "$DS/T1"
  echo "== best T of the grid on the subset (optimistic)"; "${SUBPY[@]}" pair "$V7/best" "$DS/best"
  exit
fi

RUN=${2:?checkpoint dir or Hub id}
[ -d "$RUN" ] && RUN=$(cd "$RUN" && pwd)   # the kit runs from DI_DIR, so a local checkpoint needs an absolute path
if [ -n "${3:-}" ]; then OUT=$KEV/runs/dindex/$3
elif [ -d "$RUN" ]; then OUT=$RUN/dindex-0.3
else echo "a Hub id needs NAME (3rd arg)" >&2; exit 1; fi
optv=("run=$RUN" "dtype=${DTYPE:-bf16}")
[ -n "${TEMP:-}" ] && optv+=("temperature=$TEMP")
opts=(--engine dindex_engine:KevEngine)
for o in "${optv[@]}"; do opts+=(--option "$o"); done
cd "$DI_DIR"
case $cmd in
  sample)
    [ -f sample-100.jsonl.gz ] || "$PY" -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
    "$PY" -m decision_index run "${opts[@]}" --rows sample-100.jsonl.gz --out "$OUT-sample"
    "$PY" -m decision_index score --results "$OUT-sample/results.jsonl" ;;
  full)
    mkdir -p "$OUT"
    "$PY" -m decision_index pipeline "${opts[@]}" --compact --out "$OUT" 2>&1 | tee -a "$OUT/pipeline.log" ;;
  subset)
    N=${N:-15000}; ROWS=subset-0.3-$N.jsonl.gz; SUB=$OUT-sub$N
    [ -f "$ROWS" ] || "$PY" "$KEV/scripts/pref/dindex_subset.py" make "$N" "$ROWS"
    # rows this checkpoint already answered in a full run with the same options are reused, not rerun
    [ ! -f "$OUT/environment.json" ] || "$PY" "$KEV/scripts/pref/dindex_subset.py" seed "$ROWS" "$OUT" "$SUB" "${optv[@]}"
    mkdir -p "$SUB"
    "$PY" -m decision_index run "${opts[@]}" --rows "$ROWS" --compact --out "$SUB" 2>&1 | tee -a "$SUB/run.log"
    "$PY" "$KEV/scripts/pref/dindex_subset.py" score "$ROWS" "$SUB/results.jsonl" "$SUB" 2>&1 | tee -a "$SUB/run.log" ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
