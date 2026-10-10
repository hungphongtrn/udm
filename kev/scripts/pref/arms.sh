#!/usr/bin/env bash
# Source me. Single source of truth for the arms' flags. Usage: source arms.sh; arm_flags ARM SEED
# Released Kev-4B recipe (experiments/q35-4b.json, README "first stage of Kev-4B"); --device cuda is explicit because
# kev.train refuses --dtype bf16 unless --device is literally cuda (it checks before resolving auto).
# BATCH_FALLBACK=1 -> --batch 2 --accum 4 (same effective batch 8) if batch 4 OOMs on 24GB.
if [ "${BATCH_FALLBACK:-0}" = 1 ]; then _B="--batch 2 --accum 4"; else _B="--batch 4 --accum 2"; fi
COMMON="--suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
--epochs 2 --lr 5e-5 $_B --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --weights_dtype bf16 --device cuda"
ARMS_ALL="A B C0 C D E F"
arm_flags() {
  local arm=$1 seed=$2
  local c="--head scalar --input_format chat --loss pref"       # arm C
  case $arm in
    A)  echo "$COMMON --seed $seed" ;;                              # Kev baseline: pointer + CE
    B)  echo "$COMMON --seed $seed --loss pref" ;;
    C0) echo "$COMMON --seed $seed --head scalar --loss pref" ;;    # scalar head, Kev input format
    C)  echo "$COMMON --seed $seed $c" ;;
    D)  echo "$COMMON --seed $seed $c --reg bsr --reg_w 1e-3" ;;
    E)  echo "$COMMON --seed $seed $c --reg l2 --reg_w 1e-3" ;;
    F)  # Stage 2: warm start from C, +BCE. Overrides COMMON's --epochs (argparse: last wins) with F_EPOCHS (default 1).
        # lr deliberately kept at 5e-5: train.py builds a fresh optimizer + OneCycleLR for each run, so the delta run gets its own
        # warmup/decay over its own steps; we change one thing (the BCE term) relative to C.
        echo "$COMMON --seed $seed $c --init_from runs/pref/C/seed$seed --bce_w ${BCE_W:-0.5} --epochs ${F_EPOCHS:-1}" ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
}
