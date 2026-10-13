#!/usr/bin/env bash
# Source me. Single source of truth for the arms' flags. Usage: source arms.sh; arm_flags ARM SEED
# Released Kev-4B recipe (experiments/q35-4b.json, README "first stage of Kev-4B"); --device cuda is explicit because
# kev.train refuses --dtype bf16 unless --device is literally cuda (it checks before resolving auto).
# Effective batch 8 records per optimizer step for every arm; only the microbatch (records per forward) differs. Chat-format
# arms (C D E F H HP) run every option row of a question, so batch 4 runs out of memory on a 24 GB 3090 even with
# expandable_segments: they use 2 x accum 4. The gradient does not depend on the split (train.py divides each pass's loss by
# the step's record count) except for the --reg arms (D E H HP), which compute it per pass: keep those microbatches the same across seeds.
# LoRA dropout (0.05) draws different masks for a different split: noise like a different seed, not a bias.
# BATCH_FALLBACK=1 halves every arm's microbatch (Kev 2 x 4, chat 1 x 8) if that still runs out.
# ROW_BUDGET=N (chat arms only, e.g. 2048): kev.train --row_budget N, the step below one record: a record whose question rows
# (kev.train.pass_tokens, row form) exceed N padded tokens runs as several forward/backward passes over groups of its
# questions. Nothing is dropped or truncated and the gradient is the same sum (each part carries its share of the record's
# mean); only one pass's activations are alive at a time. A single question always fits a pass. The --reg arms compute it per
# pass, so it then averages over smaller groups: set ROW_BUDGET identically for every --reg run (D E H HP, and every seed) or not at all.
if [ "${BATCH_FALLBACK:-0}" = 1 ]; then _KEV="--batch 2 --accum 4" _CHAT="--batch 1 --accum 8"
else _KEV="--batch 4 --accum 2" _CHAT="--batch 2 --accum 4"; fi
if [ "${ROW_BUDGET:-0}" != 0 ]; then _CHAT="$_CHAT --row_budget $ROW_BUDGET"; fi
COMMON="--suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
--epochs 2 --lr 5e-5 --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --weights_dtype bf16 --device cuda"
ARMS_ALL="A B C0 C D E F H HP"
arm_flags() {
  local arm=$1 seed=$2
  local c="--head scalar --input_format chat --loss pref $_CHAT"       # arm C
  local cp="--head scalar --input_format chat_pointwise --loss pref $_CHAT"   # arm H: C's recipe, pointwise prompt
  case $arm in
    A)  echo "$COMMON $_KEV --seed $seed" ;;                              # Kev baseline: pointer + CE
    B)  echo "$COMMON $_KEV --seed $seed --loss pref" ;;
    C0) echo "$COMMON $_KEV --seed $seed --head scalar --loss pref" ;;    # scalar head, Kev input format
    C)  echo "$COMMON --seed $seed $c" ;;
    D)  echo "$COMMON --seed $seed $c --reg bsr --reg_w 1e-3" ;;
    E)  echo "$COMMON --seed $seed $c --reg l2 --reg_w 1e-3" ;;
    H)  # H = arm D in the pointwise chat format (docs/hypotheses/pointwise-chat-input.md): the option listing is dropped
        # from the prompt, the one difference from D. Same recipe, seed and microbatch (D/E's --reg is per pass).
        echo "$COMMON --seed $seed $cp --reg bsr --reg_w 1e-3" ;;
    HP) # Ranking study: H + general preference pairs from an open dataset. Convert it first, then pass the JSONL as
        # PREF_DATA (`PREF_DATA=runs/pref/data/pref-helpsteer3.jsonl`, the path the docs use; runs/ is gitignored and the
        # manifest next to it is the provenance). The suite pins the base revision and holdout and --replay mixes
        # PREF_REPLAY (default all 12,576) of its training records back in, so HP trains on H's data plus the pairs.
        [ -n "${PREF_DATA:-}" ] || { echo "arm HP needs PREF_DATA=<converted preference JSONL> (scripts/pref/convert_pref_data.py)" >&2; return 1; }
        echo "$COMMON --seed $seed $cp --reg bsr --reg_w 1e-3 --data $PREF_DATA --replay ${PREF_REPLAY:-12576}" ;;
    F)  # Stage 2: warm start from C, +BCE. Overrides COMMON's --epochs (argparse: last wins) with F_EPOCHS (default 1).
        # lr deliberately kept at 5e-5: train.py builds a fresh optimizer + OneCycleLR for each run, so the delta run gets its own
        # warmup/decay over its own steps; we change one thing (the BCE term) relative to C.
        echo "$COMMON --seed $seed $c --init_from runs/pref/C/seed$seed --bce_w ${BCE_W:-0.5} --epochs ${F_EPOCHS:-1}" ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
}
