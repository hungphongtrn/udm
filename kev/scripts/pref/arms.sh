#!/usr/bin/env bash
# Source me. Single source of truth for the arms' flags. Usage: source arms.sh; arm_flags ARM SEED
# Released Kev-4B recipe (experiments/q35-4b.json, README "first stage of Kev-4B"); --device cuda is explicit because
# kev.train refuses --dtype bf16 unless --device is literally cuda (it checks before resolving auto).
# Effective batch 8 records per optimizer step for every arm; only the microbatch (records per forward) differs. Chat-format
# arms (C D E F H HP HL HB HBL) run every option row of a question, so batch 4 runs out of memory on a 24 GB 3090 even with
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
ARMS_ALL="A B C0 C D E F H HP HL HB HBL Ah Ch Fh HBh Rh R2h Sh STh RYh Dh Hh PBh PRh PUh P2h PLh"
# Half-data screen (docs/hypotheses/allwrong-abstention.md): an arm name ending in a lowercase "h" trains on the
# HALF_FRAC (default 0.5) per-source subsample of the training records (kev.train --train_frac, keyed on --seed, so every
# h arm of a seed, and a warm start from one, sees the same records). Everything else is the full arm's recipe.
_H="--train_frac ${HALF_FRAC:-0.5}"
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
    HL) # Reranker loss: H's grader (chat_pointwise, scalar per option) trained with Kev's CE over the softmax of a question's
        # option rewards (RankLLaMA's listwise contrastive loss) instead of Bradley-Terry + BSR. Serving is unchanged: one
        # scalar per option; probabilities come from the softmax at the temperature G fits on the calibration split.
        echo "$COMMON --seed $seed --head scalar --input_format chat_pointwise --loss ce $_CHAT" ;;
    HB) # Absolute grader (docs/hypotheses/pointwise-chat-input.md, stage 1): each option asked "Is this option a correct
        # answer? Answer Yes or No." on its own (chat_yesno), the scalar head started at the LM head's W_yes - W_no (so step 0
        # reads the base's own Yes/No logit), trained with class-balanced per-option BCE (positives and negatives weigh half
        # each, none wordings masked): a reward on one scale across questions, no softmax over the question.
        echo "$COMMON --seed $seed --head scalar --input_format chat_yesno --head_init yesno --loss bce $_CHAT" ;;
    HBL) # HB + 0.25 x Kev's listwise CE over the question's option rewards, to keep top-1 while BCE sets the scale.
        echo "$COMMON --seed $seed --head scalar --input_format chat_yesno --head_init yesno --loss bce --ce_w 0.25 $_CHAT" ;;
    F)  # Stage 2: warm start from C, +BCE. Overrides COMMON's --epochs (argparse: last wins) with F_EPOCHS (default 1).
        # lr deliberately kept at 5e-5: train.py builds a fresh optimizer + OneCycleLR for each run, so the delta run gets its own
        # warmup/decay over its own steps; we change one thing (the BCE term) relative to C.
        echo "$COMMON --seed $seed $c --init_from runs/pref/C/seed$seed --bce_w ${BCE_W:-0.5} --epochs ${F_EPOCHS:-1}" ;;
    Ah)  echo "$(arm_flags A "$seed") $_H" ;;
    Ch)  echo "$(arm_flags C "$seed") $_H" ;;
    HBh) echo "$(arm_flags HB "$seed") $_H" ;;
    Fh)  echo "$COMMON --seed $seed $c --init_from runs/pref/Ch/seed$seed --bce_w ${BCE_W:-0.5} --epochs ${F_EPOCHS:-1} $_H" ;;
    Rh)  # RCR (Bai et al. 2023): (1 - a) per-option sigmoid CE + a ListCE over sigmoid rewards, a = 0.5, in one stage.
         # Linear head so it differs from F only in the loss (K4).
         echo "$COMMON --seed $seed --head scalar --input_format chat --loss rcr --rcr_alpha 0.5 $_CHAT $_H" ;;
    R2h) # Rh with the MLP head (Linear 1024 -> SiLU -> Linear, zero-init output)
         echo "$COMMON --seed $seed --head mlp --input_format chat --loss rcr --rcr_alpha 0.5 $_CHAT $_H" ;;
    Sh)  # R2h with a = 0: per-option sigmoid CE only (the absolute stage); STh's stage 1
         echo "$COMMON --seed $seed --head mlp --input_format chat --loss rcr --rcr_alpha 0 $_CHAT $_H" ;;
    STh) # Stage 2 on Sh: Bradley-Terry + 0.5 x BCE for one epoch (F's stage 2 recipe on the absolute-first model)
         echo "$COMMON --seed $seed --head mlp --input_format chat --loss pref $_CHAT --init_from runs/pref/Sh/seed$seed --bce_w ${BCE_W:-0.5} --epochs ${F_EPOCHS:-1} $_H" ;;
    RYh) # RCR on the yes/no prompt, linear head started at W_yes - W_no (vs HBh: the loss; vs Rh: the prompt + init)
         echo "$COMMON --seed $seed --head scalar --input_format chat_yesno --head_init yesno --loss rcr --rcr_alpha 0.5 $_CHAT $_H" ;;
    # PF-RCR wave (docs/hypotheses/allwrong-abstention.md, phase 1-2): the option listing leaves the prompt (chat_pointwise,
    # each question's prefix runs once and its options are branches from it, kev.shared_prefix), then one thing changes
    # per step. Dh/Hh carry D's --reg bsr, computed per pass: run both at the same microbatch (BATCH_FALLBACK) and ROW_BUDGET.
    Dh)  echo "$(arm_flags D "$seed") $_H" ;;                                    # listing + BT + BSR (Hh's reference)
    Hh)  echo "$(arm_flags H "$seed") $_H" ;;                                    # H-BT: Dh without the listing
    PBh) echo "$COMMON --seed $seed --head scalar --input_format chat_pointwise --loss bce $_CHAT $_H" ;;   # H-BCE: K-balanced BCE alone
    PRh) # H-RCR: (1 - a) K-balanced BCE + a ListCE(sigmoid), a = 0.5 (vs PBh: + the listwise term; vs Hh: the loss)
         echo "$COMMON --seed $seed --head scalar --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point balanced $_CHAT $_H" ;;
    PUh) # H-RCR-U: PRh with the unbalanced mean BCE (Rh's pointwise term) - the K-balance ablation
         echo "$COMMON --seed $seed --head scalar --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point mean $_CHAT $_H" ;;
    P2h) # H-RCR-2L: PRh with the MLP head
         echo "$COMMON --seed $seed --head mlp --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point balanced $_CHAT $_H" ;;
    PLh) # H-RCR-LM: PRh's loss on the yes/no prompt, linear head started at W_yes - W_no. --head_init yesno needs the
         # chat_yesno prompt (the Yes/No question is what makes W_yes - W_no meaningful), so prompt and init change together.
         echo "$COMMON --seed $seed --head scalar --input_format chat_yesno --head_init yesno --loss rcr --rcr_alpha 0.5 --rcr_point balanced $_CHAT $_H" ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
}
