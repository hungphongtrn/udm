# Hypothesis: an anchored reward knows when no option is right (arm F vs A on all-wrong questions)

Branch `idea/allwrong-abstention`, cut from `hyp/preference-reward-model` at `4668509` (its arms, checkpoints and
wave-1 numbers: `docs/hypotheses/preference-reward-model.md`). No training: every number comes from the existing seed-0
checkpoints `runs/pref/{A,C,D,F}/seed0`. Code: `kev/scripts/abstain/` (`build_allwrong.py`, `eval_allwrong.sh`,
`allwrong_report.py`), `kev/tests/test_allwrong.py`. Everything below (arms, readouts, thresholds, rule) is fixed before any
run.

## Claim

A softmax over options is shift-invariant: rewards (2, 1.1, 1.2) and (12, 11.1, 11.2) give the same distribution, so a
softmax-trained model (arm A: pointer head + CE) reports the same confidence for "A is best and every option is fine" and
"A is best and every option is bad". Its scale is not lost (bigger gaps, sharper softmax), only its level.
Bradley-Terry (arm C) is shift-invariant too, and BSR (arm D) pushes the mean reward toward 0, so neither is trained to
carry a level. Arm F (C + per-option BCE on sigmoid(r_i) against [i = label]) is: its rewards are anchored to an absolute
"this option is right" probability.

**Claim:** on questions whose right option has been removed, F's absolute reward (max r, equivalently sigmoid(max r))
detects that no option is right better than A's softmax confidence, and that gain comes from the anchor, not from reading
a logit instead of a probability.

Why it matters: in a cascade, a question whose options are all wrong should escalate. Detecting that without a "none"
option in the set is something a softmax decision model cannot do from its probabilities alone.

## What each comparison isolates

The tempting comparison (F's sigmoid(max r) vs A's max softmax) mixes three things. Each control removes one:

| Comparison | Question it answers |
|---|---|
| **F.maxr - A.maxprob** | The headline: anchored absolute reward vs softmax confidence. |
| A.maxr - A.maxprob, A.lse - A.maxprob | Does A's *raw logit* already carry the level (the energy-score effect: CE does not constrain the level, but nothing forces it to be uninformative)? If yes, the headline gain is the readout, not F's training. |
| F.maxr - A.maxr | The anchor itself: same readout, the two training objectives. |
| F.maxr - F.maxprob | Inside F: how much of F's detection is level (shift) and how much is scale. |
| F.maxr - C.maxr, F.maxr - D.maxr | BCE vs no anchor (C) and vs an anchor at 0 (D: BSR). Prediction: C and D near A.maxprob, D worst. F starts from C, so F - C is the BCE stage alone. |
| F.maxr - A.pnone | The product's existing mechanism: Kev trains with none options (`--p_none 0.1`, `--p_none_pair 0.25`), so A offered a "None of these" option already has a direct abstention signal. If A.pnone matches F.maxr, the anchor buys abstention only where no none option is offered. |

AURC (risk-coverage) mixes detection with top1: an arm with a lower answerable top1 has a worse oracle AURC. AUROC is the
detection-only number and the one the rule uses; AURC and its oracle are reported beside it.

## Eval data (`build_allwrong.py`)

From decision-v7 development and transfer-v4 development: every clean Choice question (not `unknowable`) with a hard
label, >= 3 options and no none-of-the-above key: 448 questions on v7 (banking77, agnews, mnli, trec, dbpedia14 80 each,
legacy_policy 48; K = 3/4/6/10) and 240 on t4 (mmlu, emotion, sciq 80 each; K = 4/6), counted from the frozen partitions.
Each is a minimal pair plus a none pair:

| Condition | Options | Label |
|---|---|---|
| answerable | K - 1: one seeded wrong option removed | the true option |
| allwrong | K - 1: the true option removed | none right (placeholder; never scored as top1) |
| answerable_none | answerable + one none option (seeded wording from `kev.data.NONE_OPTIONS`, seeded position) | the true option |
| allwrong_none | allwrong + the same none option at the same position | the none option |

answerable and allwrong share state, instruction, option count and the order of their shared options, so neither the
count nor a gap in letter keys (mmlu's a-d) separates them. Seed 0 per (seed, record id, question id).

## Readouts (`allwrong_report.py`)

Per question, from the rewards r the benchmark records (`rows.json` logits):

| Readout | Definition | Note |
|---|---|---|
| maxprob | max softmax(r) | what a softmax cascade thresholds |
| maxprob_G | max softmax(r / T), T = the arm's G fit on decision-v7 calibration | temperature changes the order across questions |
| maxr | max r | = sigmoid(max r) = Platt sigmoid(a max r + b) for ranking |
| lse | logsumexp(r) | energy score |
| pnone | softmax(r)[none] in the none conditions | the existing mechanism |

Metrics per (suite, arm, readout): AUROC (allwrong = positive), paired win rate (a question's answerable twin more
confident than its allwrong twin), and on the pool answerable + allwrong (one each per question; answering an allwrong
question is always wrong): AURC, accuracy at coverage 0.3/0.5/0.7/0.9/1.0, the oracle AURC. Only pairs every arm scored in
every condition count (rejected = over context, never truncated; counts printed). Differences: paired bootstrap over
questions, 2,000 resamples, 95% percentile CI (question sampling only, not seed variance).

## Decision rule (seed 0; v7 = decision-v7 development, t4 = transfer-v4 development)

1. **Headline.** Supported if, on both v7 and t4, AUROC(F.maxr) - AUROC(A.maxprob) >= 0.05 with a CI above 0, and F's
   answerable top1 is not more than 2 points below A's (the parent study's screen; wave 1 had F - A = -1.58 on full v7).
   Refuted if the difference is <= 0 on either set. Inconclusive otherwise.
2. **Attribution to the anchor.** Only if 1 holds: AUROC(F.maxr) - AUROC(A.maxr) >= 0.03 with a CI above 0 on both sets.
   If 1 holds and 2 fails, the result is recorded as "the level is in A's logits too; the gain is the readout", not as
   an anchor result.
3. **Against the existing mechanism (reported, not part of 1-2).** F.maxr (no none option) vs A.pnone (none offered).
   F.maxr >= A.pnone - 0.02 on both sets: the anchor matches the none option without needing one in the set.
   Otherwise: the none option already does it where it can be offered.
4. **Predictions recorded before the run:** C.maxr and D.maxr within 0.05 of A.maxprob, D.maxr the lowest of the
   maxr readouts; F.maxprob below F.maxr. These are not verdicts.

If 1-2 hold, follow-ups (not pre-registered here): seeds 1-2 of F (needs C seeds 1-2), the pointwise BCE arm HB from
`idea/pointwise-chat-input` on the same pairs, and the risk-coverage operating point on decision-v7 calibration.

## Commands (training machine, from `kev/`)

```bash
.venv/bin/python -m pytest tests/test_allwrong.py
# data once (CPU), before any eval_allwrong.sh: they would race to build it
.venv/bin/python scripts/abstain/build_allwrong.py --suite evals/v7/decision-v7 --out runs/allwrong/data/decision-v7-development
.venv/bin/python scripts/abstain/build_allwrong.py --suite evals/v4/transfer-v4 --out runs/allwrong/data/transfer-v4-development
# one GPU per arm; needs runs/pref/<arm>/seed0/eval/decision-v7-development/reward_report.json (eval_arm.sh) for maxprob_G
CUDA_VISIBLE_DEVICES=0 nohup scripts/abstain/eval_allwrong.sh runs/pref/A/seed0 > runs/allwrong/A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup scripts/abstain/eval_allwrong.sh runs/pref/F/seed0 > runs/allwrong/F.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup scripts/abstain/eval_allwrong.sh runs/pref/C/seed0 > runs/allwrong/C.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup scripts/abstain/eval_allwrong.sh runs/pref/D/seed0 > runs/allwrong/D.log 2>&1 &
.venv/bin/python scripts/abstain/allwrong_report.py --arm A=runs/pref/A/seed0 --arm C=runs/pref/C/seed0 \
  --arm D=runs/pref/D/seed0 --arm F=runs/pref/F/seed0 --out runs/allwrong/report
```

Outputs: `runs/allwrong/report/report.json` (every number above), `rc-decision-v7.svg` / `rc-transfer-v4.svg`
(accuracy vs coverage; solid = maxr, dashed = maxprob).

## Risks

- F trained one extra epoch from C on the same data: a detection gain over C could be more training, not the BCE term.
  F - C cannot separate them; a C + 1 epoch arm without BCE would (not run).
- Training already contains all-wrong sets, but only with a none option present (augment's `p_none` and none pairs): A
  and F have seen "the true option is gone" with a none option to pick, never without. The no-none condition is out of
  their training distribution for both.
- One seed, one checkpoint per arm; CIs are question sampling only.
- K = 3 questions become 2-option pairs; allwrong mnli is "neither of the remaining two relations", a natural case;
  banking77's 10-option subsets lose one of ten.

## Results

Not run.
