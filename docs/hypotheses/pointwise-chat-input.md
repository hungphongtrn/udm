# Idea: pointwise chat input (no option list)

Branch `idea/pointwise-chat-input`, cut from `hyp/preference-reward-model` after its wave-1 seed-0 results. Not
pre-registered and not implemented: the arm, rule and code below are a proposal to finish before any run.

## Claim

In the preference reward model's chat format, each option row currently shows every option
(`Options:\n- o_1\n- o_2 ...\n\nGrade this option: o_k`, `kev.model.chat_splits`). Dropping the `Options:` block makes the
reward a function of (state, instruction, o_k) only, the pointwise score Bradley-Terry assumes. Claim: this keeps top1
within 1 point of the listed format while making the reward independent of the option set and its order.

## Arm

H = arm D of [`preference-reward-model.md`](preference-reward-model.md) (scalar head, pref loss, BSR 1e-3, chat format)
with the option list removed. Compare H vs D only: the list is the one difference. Same recipe, seed, and microbatch as D
(1 x accum 8, `BATCH_FALLBACK=1`, since `--reg` is per pass).

## Expected costs (why it may fail)

- Relative judgements lose their contrast: many-class choice (banking77, 77 intents), score scales (sst5, amazon: "3"
  means little without 1-5), multiple-choice items written to be read side by side (mmlu).
- The none option (`p_none_pair` 0.25) has no referent: "Grade this option: none of these" without the list.
  none_present accuracy is already the weakest number on transfer-v4 (0.53-0.64 at seed 0).
- Compute: the list sits in the shared prefix (run once per question), so the saving is small.
- Calibration: the readout is still a softmax over the question's rewards, so no gain is expected there; a temperature
  already makes the pointer + CE baseline the best calibrated arm.

## Proposed rule (fix before running)

- Keep H if its top1 is within 1 point of D on both dev sets (decision-v7, transfer-v4 development); seed 0 first, three
  seeds if it passes.
- It must also show the gain it is for: lower permutation flip rate or better transfer-v4 top1.
- Report score-type questions and the none_present / none_absent variants separately.

## Code to write

- `kev/kev/model.py`: a new `INPUT_FORMATS` entry (e.g. `chat_pointwise`) whose `chat_splits` prefix omits the
  `Options:` block; recorded in the checkpoint like `chat`, so evaluation reads the same prompt.
- `kev/scripts/pref/arms.sh`: arm H's flags.
- `kev/tests/test_pref.py`: the prefix carries no option text and is identical across a question's options.

Cost: one chat-format run, ~9 h on a 3090.
