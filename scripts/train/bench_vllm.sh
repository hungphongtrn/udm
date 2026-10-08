#!/usr/bin/env bash
# Is the vLLM backbone fast enough for the full Decision Index? Times `dindex_frozen featurize` on the first
# LIMIT suite requests (the exact final workload) under several engine settings, each in a fresh process and a fresh
# throwaway cache under $BENCH, then prints steady-state throughput, prefix-cache hit rate and full-suite ETA.
#   scripts/train/bench_vllm.sh                 # default sweep below
#   scripts/train/bench_vllm.sh 'mine|features.vllm.max_num_seqs=64'   # custom 'tag|overrides' settings
# Env: LIMIT (512 requests; 2 shards, the 2nd = steady state), FEATS_CFG, BENCH (outputs/vllm_bench), DI_DIR.
# One GPU: do not run it while the loss grid or another vLLM process holds the card.
# Pick the fastest tag and pass its overrides to dindex_frozen.sh via FEATS_OVERRIDES.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_ROOT"
LIMIT="${LIMIT:-512}"; BENCH="${BENCH:-outputs/vllm_bench}"; CFG="${FEATS_CFG:-configs/features_qwen3_5_4b.yaml}"
export DI_DIR="${DI_DIR:-$REPO_ROOT/../decision-index}"
PY="${FEATURES_PYTHON:-${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin/python}"
V=features.vllm
SETTINGS=("$@")
# Sized for one 24 GB RTX 3090 (the config's 16k-token budget / 32 seqs is what the extraction ran with). Every
# sequence pins Qwen3.5 Mamba state, so more seqs trade KV/prefix-cache room for parallelism; an OOM shows as failed.
[ ${#SETTINGS[@]} -gt 0 ] || SETTINGS=(
  "base|"
  "nocache|$V.enable_prefix_caching=false"
  "seqs64|$V.max_num_seqs=64"
  "seqs128|$V.max_num_seqs=128"
  "graphs|$V.enforce_eager=false"
  "graphs_seqs64|$V.enforce_eager=false $V.max_num_seqs=64"
  "seqs64_win1024|$V.max_num_seqs=64 $V.request_batch_size=1024"
)
mkdir -p "$BENCH"
for s in "${SETTINGS[@]}"; do
  tag="${s%%|*}"; ov="${s#*|}"
  rm -rf "${BENCH:?}/$tag"
  echo "=== $tag: ${ov:-<config defaults>}"
  # shellcheck disable=SC2086
  if ! "$PY" -m scrm.dindex_frozen featurize --config "$CFG" --out "$BENCH/$tag" --limit "$LIMIT" \
      --shard-requests $((LIMIT / 2)) $V.log_stats=true $ov 2>&1 | tee "$BENCH/$tag.log"; then
    echo "!!! $tag failed (see $BENCH/$tag.log)"
  fi
  echo "$ov" > "$BENCH/$tag.overrides"
done
"$PY" - "$BENCH" "$DI_DIR/suite-0.2" <<'EOF'
import json, os, sys
from decision_index.suite.io import Suite
bench, suite_dir = sys.argv[1:]
total = sum(1 for _ in Suite(suite_dir, "0.2.1").rows(apply_exclusions=True))
print(f"\nfull suite: {total} requests; steady state = last shard of each run")
print(f"{'tag':34s} {'req/s':>7s} {'rows/s':>7s} {'uniq tok/s':>10s} {'subm tok/s':>10s} {'cache hit':>9s} {'ideal':>6s} {'ETA h':>6s}  overrides")
for tag in sorted(t for t in os.listdir(bench) if os.path.isdir(os.path.join(bench, t))):
    p = os.path.join(bench, tag, "manifest.json")
    sh = json.load(open(p))["shards"] if os.path.exists(p) else {}
    if not sh:
        print(f"{tag:34s} failed"); continue
    s = sh[sorted(sh)[-1]]
    t = s["seconds"]
    pc = s.get("prefix_cache") or {}
    hit = f"{pc['hits'] / max(pc['queries'], 1):.1%}" if pc else "-"
    ov = open(os.path.join(bench, tag + ".overrides")).read().strip()
    print(f"{tag:34s} {s['n_requests'] / t:7.1f} {s['n_rows'] / t:7.0f} {s['unique_tokens'] / t:10.0f} "
          f"{s['tokens'] / t:10.0f} {hit:>9s} {1 - s['unique_tokens'] / max(s['tokens'], 1):6.1%} "
          f"{total / (s['n_requests'] / t) / 3600:6.2f}  {ov}")
EOF
