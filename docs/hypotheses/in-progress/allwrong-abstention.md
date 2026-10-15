# Hypothesis: an anchored reward knows when no option is right (all-wrong detection, readout and training screen)

Status: in-progress (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

Branch `hyp/allwrong-abstention`, cut from `dev` at `a80515f`. The code was built on `idea/allwrong-abstention`, cut from
`hyp/preference-reward-model` at `4668509` (its arms, checkpoints and wave-1 numbers:
`docs/hypotheses/done/preference-reward-model.md`), with `idea/pointwise-chat-input` merged (arm HB, the chat_yesno
prompt). Two parts: D1 reads the existing seed-0 checkpoints `runs/pref/{A,C,F}/seed0` (no training); D2-D7 are
a half-data training screen of anchored recipes; D8-D11 the prefix-forked RCR wave (PF-RCR, below). Code:
`kev/scripts/abstain/` (`build_allwrong.py`, `eval_allwrong.sh`, `allwrong_report.py`, `fork_check.py`, `queue.sh`),
`kev/scripts/pref/arms.sh` (the h arms), `kev/kev/pref.py` (`rcr_loss`), `kev/kev/model.py` (`MLPHead`),
`kev/kev/train.py` (`--loss rcr`, `--rcr_alpha`, `--rcr_point`, `--rcr_beta`, `--head mlp`, `--train_frac`); tests in
`kev/tests/test_allwrong.py` and `kev/tests/test_pref.py`. Everything below (arms, readouts, thresholds, rules) is fixed
before any run.

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
| F.maxr - C.maxr | BCE vs no anchor. Prediction: C near A.maxprob. F starts from C, so F - C is the BCE stage alone. |
| F.maxr - A.pnone | The product's existing mechanism: Kev trains with none options (`--p_none 0.1`, `--p_none_pair 0.25`), so A offered a "None of these" option already has a direct abstention signal. If A.pnone matches F.maxr, the anchor buys abstention only where no none option is offered. |

Arm D (BSR) is dropped from D1 (K1): an anchor at 0 is not a level, and every D question is answered by C and F. Its half
version Dh returns in the PF-RCR wave only as Hh's listing reference.

AURC (risk-coverage) mixes detection with top1: an arm with a lower answerable top1 has a worse oracle AURC. AUROC is the
detection-only number and the one the rule uses; AURC and its oracle are reported beside it.

## Eval data (`build_allwrong.py`)

From decision-v7 development and transfer-v4 development: every clean Choice question (not `unknowable`) with a hard
label, >= 3 options and no none-of-the-above key: 448 questions on v7 (banking77, agnews, mnli, trec, dbpedia14 80 each,
contrastive 48; original K = 77 / 4 / 3 / 6 / 14 / 3) and 240 on t4 (mmlu, emotion, sciq 80 each; K = 4 / 6 / 4),
counted from the frozen partitions. The report's K strata: small K <= 4 (agnews, mnli, contrastive; mmlu, sciq), mid
5-14 (trec, dbpedia14; emotion), large >= 15 (banking77, v7 only).
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
| maxr_k | max r - log(n - 1), n = the options scored in the row | the K-balanced level: `kev.pref.rcr_loss` point=balanced has optimum r* = logit(q) + log(K-1), so this removes that shift; n = 1 (log 0) is defined as max r |
| noisy_or | 1 - prod_i (1 - sigmoid(r_i)), in log space | confidence that some option is right |
| mass | sum_i sigmoid(r_i) | single-label option mass (Li et al. 2609.33209, their sum); ~1 on answerable rows, ~0 on allwrong, whatever K. Per arm its answerable/allwrong mean and median are reported |
| lse | logsumexp(r) | energy score |
| pnone | softmax(r)[none] in the none conditions | the existing mechanism |

maxr_k, noisy_or and mass are the calibrated confidence only for a scalar (sigmoid) arm; for the pointer arm (A/Ah) they
are computed from its logits the same way and reported, but its sigmoid is not a calibrated probability.

Metrics per (suite, arm, readout): AUROC (allwrong = positive), FPR@95TPR (answerable rows flagged at the threshold that
flags 95% of allwrong rows), AUROC per K stratum, paired win rate (a question's answerable twin more confident than its
allwrong twin), and on the pool answerable + allwrong (one each per question; answering an allwrong question is always
wrong): AURC, accuracy at coverage 0.3/0.5/0.7/0.9/1.0, the oracle AURC. Per arm also: answerable top1 per stratum and
the shared-option |d reward| (an option scored in both the answerable and the allwrong set: 0 up to rounding for a
set-independent pointwise grader, nonzero for the listing prompt). Only pairs every arm scored in every condition count
(rejected = over context, never truncated; counts printed). Differences: paired bootstrap over questions, 2,000
resamples, 95% percentile CI (question sampling only, not seed variance).

## Training screen (D2-D7): half data, seed 0

Every arm below is an existing or new recipe trained on the same seeded half of the decision-v7 training records
(`--train_frac 0.5`, a per-source sample keyed on the seed, so all h arms and their warm starts see the same records),
otherwise the full arm's flags (`scripts/pref/arms.sh`: base Qwen3.5-4B-Base, 2 epochs, lr 5e-5, LoRA, `--p_none_pair
0.25`). A screen: the winner is rerun on full data with more seeds; a Decision Index subset is a final guard after that,
not part of this screen.

| Arm | Prompt | Head | Loss | Role |
|---|---|---|---|---|
| Ah | Kev | pointer | CE | baseline: maxprob, maxr, lse, and pnone (K9) |
| Ch | chat | linear | Bradley-Terry | no anchor |
| Fh | chat | linear | Ch + 1 epoch BT + 0.5 BCE (`--init_from Ch`) | the two-stage anchor (arm F) |
| HBh | chat_yesno | linear, W_yes - W_no init | class-balanced BCE | absolute grader (arm HB) |
| Rh | chat | linear | RCR, alpha 0.5 | one-stage anchor: (1 - a) sigmoid CE + a ListCE (Bai et al. 2023) |
| R2h | chat | MLP (1024, SiLU) | RCR, alpha 0.5 | Rh with the MLP head |
| Sh | chat | MLP | RCR, alpha 0 (sigmoid CE only) | STh's stage 1 |
| STh | chat | MLP | Sh + 1 epoch BT + 0.5 BCE (`--init_from Sh`) | absolute first, then pairwise |
| RYh | chat_yesno | linear, W_yes - W_no init | RCR, alpha 0.5 | RCR on the yes/no prompt |

Fixed choices: base model, not instruct (K2); MLP head for the new arms (K3) except Rh (the linear control against F)
and RYh (the yes/no init needs a linear head) (K4); top1 is read on decision-v7 development, never test (K5); AUROC
decides, AURC is reported beside it (K6); maxr is the anchored arms' readout (K7); the minimal pairs are the eval (K8);
STh's stage 2 uses F's BCE weight 0.5 (K10). RCR deviation from the paper: the pointwise term is the mean over a
question's options, not the sum (`kev.pref.rcr_loss`); none rows stay in the listing prompt and are dropped on the
pointwise ones, as `--loss bce` does.

## Decision rules (seed 0; v7 = decision-v7 development, t4 = transfer-v4 development)

Terms: X **beats** Y when dAUROC(X - Y) >= 0.03 with the 95% CI above 0 on both v7 and t4. X is **not worse** than Y
when the CI's lower end is >= -0.02 on both. The **top1 guard**: an arm's v7 development top1 (reward_report.json
`raw.overall.top1`, printed by the report) is at most 2 points below A's (full) or Ah's (half). Development NLL is
reported beside it; it breaks ties only.

- **D1 readout (full A, C, F; no training).** Headline supported if F.maxr - A.maxprob >= 0.05 with a CI above 0 on both
  sets and F passes the top1 guard; refuted if <= 0 on either; inconclusive otherwise. If supported, A.maxr - A.maxprob
  and A.lse - A.maxprob say whether A's logits carry the level too (if A.maxr is within 0.03 of F.maxr, the gain is the
  readout, not the anchor). C.maxr - C.maxprob is the same question for the unanchored pairwise arm.
- **D2 one-stage anchor.** Rh replaces F's two stages if Rh is not worse than Fh, beats Ch, and passes the guard. Rh beats
  Fh: RCR is the recipe. Fh beats Rh: two stages carry something RCR does not.
- **D3 staging.** STh beats R2h: pointwise first, then pairwise, matters; otherwise R2h (one run) is preferred. STh vs Fh
  is reported (order of the stages; the heads differ).
- **D4 head.** The MLP head is adopted if R2h beats Rh, or is not worse with a higher dev top1. R2h - Sh is the listwise
  term's share (alpha 0.5 vs 0), reported.
- **D5 prompt.** RYh vs HBh: RCR vs class-balanced BCE on the yes/no prompt; RYh vs Rh: yes/no prompt + init vs the
  listing prompt. The better of each pair by the beats rule; neither beating: not worse both ways, the cheaper (Rh) stays.
- **D6 threshold transfer.** For each arm, readout and target risk (0.05 and 0.10), the lowest threshold with that risk
  on the v7 pool (answerable + allwrong) is applied unchanged to the t4 pool and to each K stratum of both pools. An
  anchored readout transfers if, at risk 0.10, its t4 risk is <= 0.15 and its t4 coverage is >= 0.8 x the coverage a
  threshold fit on t4 would reach; reported for every arm, judged on the D2-D5 winner against Ah.maxprob. The strata
  (and risk 0.05) are reported, not judged. Reported beside the fitted threshold, not judged: two zero-fit policies
  (accept iff maxr > 0, i.e. sigmoid 0.5, and iff maxr_k > 0) on both pools and every stratum; and a Lu & Wu 2610.03387
  threshold rule per arm and curve readout (accept iff score > threshold) -- the threshold maximising allwrong detection
  subject to a false rejection rate of answerable rows <= 5% (deterministic: the candidates are the pool's own scores,
  a score equal to the threshold is rejected, and the largest qualifying score is taken), fit on the v7 pool and on a
  fixed seeded 80-question sample of it (their 80-pair protocol), applied frozen to t4 and the strata, with the false
  rejection and detection printed on the fit and apply sets.
- **D7 against the none option.** The D2-D5 winner's maxr (no none option in the set) vs Ah.pnone (none offered): it
  matches the existing mechanism if AUROC >= Ah.pnone - 0.02 on both sets.

Predictions recorded before the run (not verdicts): C.maxr within 0.05 of A.maxprob; F.maxprob below F.maxr; Rh within
0.02 of Fh; Sh the highest AUROC and the lowest top1 of the MLP arms.

## PF-RCR wave (D8-D11): the option listing out of the prompt, then one change per step

A grader that sees one option at a time (`chat_pointwise`: state + instruction, then the option and the reward token;
no listing) is invariant to the option count and order, so the same reward serves a K = 1 best-of-N verifier and a
K = 77 question. On Qwen3.5 each question's prefix runs once and its K options are branches continuing it (gradients
included, `kev/kev/shared_prefix.py`; inference: `DecisionModel.chat_hidden` / CUDA graphs); every chat arm trains this
way already. The pointwise term of RCR needs balancing for that: the mean sigmoid CE over a question's options is
minimised at sigmoid(r_i) = P(option i right), so a K = 77 question's rewards sit ~log 76 below a K = 2 question's and one
threshold cannot serve both. `--rcr_point balanced` gives the positives and the wrong options half the question each
(alpha 0 = `--loss bce`); its optimum is r* = logit(q) + log(K-1), so the level still moves with K (the line r = 0 is one
class-balanced boundary, not one threshold: it corresponds to q = 1/K), and the readout `maxr_k` = max r - log(n-1)
removes that shift, which is what makes a threshold comparable across K. `--rcr_beta` (0 here) would weigh wrong options
by exp(beta r_j), detached.

| Phase | Arm | Prompt | Head | Loss | Changes vs |
|---|---|---|---|---|---|
| 0 | (A seeds 0-2, C, F full) | - | - | - | all-wrong eval (E0.2: maxr / lse of A) and Decision Index subset of A seeds (E0.1), no training |
| 1 | Dh | chat (listing) | linear | BT + BSR | D on half data: Hh's reference |
| 1 | Hh | chat_pointwise | linear | BT + BSR | Dh: the listing removed (H-BT) |
| 2 | PBh | chat_pointwise | linear | K-balanced BCE | pointwise only (H-BCE) |
| 2 | PRh | chat_pointwise | linear | RCR, balanced point, alpha 0.5 | PBh: + ListCE; Hh: the loss (H-RCR) |
| 2 | PUh | chat_pointwise | linear | RCR, mean point, alpha 0.5 | PRh: no K balance (H-RCR-U) |
| 2 | P2h | chat_pointwise | MLP | as PRh | PRh: the head (H-RCR-2L) |
| 2 | PLh | chat_yesno | linear, W_yes - W_no init | as PRh | PRh: the init, and with it the prompt (H-RCR-LM) |

`--rcr_alpha` weighs the listwise term here (the plan's alpha weighs the pointwise one; 0.5 is the same either way).
PLh changes two things: `--head_init yesno` reads W_yes - W_no, which is only meaningful after the Yes/No question, so
the yes/no prompt comes with it; RYh vs Rh already measured that pair on the listing prompt.

Gates (each a printed line of `fork_check.py` or a column of the report):

- **G0 fork exactness** (before training; `fork_check.py parity --kernels reference`, Qwen3.5-0.8B-Base, 10 development
  records including a banking77 one, fp32, checkpointing on): rewards within 1e-4 and the relative error of every LoRA and
  head gradient within 1e-3 of one naive causal row per option, with transformers' PyTorch DeltaNet kernels and no TF32.
  With the bound CUDA kernels (fla, causal-conv1d) even naive vs naive1 misses these thresholds, so that run is only
  read against its floor; causal-conv1d is CUDA-only, so `--device cpu` cannot give the reference either. A failure stops
  the PF arms' interpretation, not the queue.
- **G1 ranking** = the top1 guard: v7 development top1 within 2 points of Ah (t4 top1 printed beside it).
- **G2 detection**: AUROC and FPR@95TPR on the minimal pairs (v7 and t4). The allwrong condition is a real forward pass
  without the true option for every arm; for a pointwise arm it equals masking the true option's reward (checked by the
  shared-option |d reward|).
- **G3 threshold transfer**: D6, with the K strata.
- **G4 K = 1**: `fork_check.py k1` on each pointwise checkpoint: every option scored alone vs the K together, max |d
  sigmoid| < 1e-3 and no argmax change.

Rules (half data, seed 0; beats / not worse / guard as above):

- **D8 topology.** Hh replaces Dh as the scalar vehicle if Hh.maxr is not worse than Dh.maxr, Hh passes the guard and
  its banking77 (large stratum) answerable top1 is not below Dh's by more than 2 points. Hh.maxprob - Dh.maxprob is
  reported.
- **D9 objective.** PRh is the pointwise recipe if it beats Hh and is not worse than PBh with the guard passed; PBh
  beats PRh: the listwise term costs detection. PRh.maxr_k - PUh.maxr is the K balance at the calibrated level (maxr_k
  removes PRh's log(K-1) optimum shift, so this compares like with like; PRh.maxr - PUh.maxr is the same comparison on
  the raw level and is reported beside it): adopted if PRh.maxr_k beats PUh.maxr or is not worse with a smaller
  small-vs-large gap in D6 stratum coverage at risk 0.10. PRh - Rh (pointwise vs listing prompt, same RCR alpha,
  different point term) is reported.
- **D10 head.** P2h replaces PRh if it beats it, or is not worse with a higher dev top1; PLh the same against PRh.
- **D11 against Kev.** The D9-D10 winner's maxr vs Ah.maxprob and Ah.pnone, as D7.

Not built: a two-stage GradCache (prefix under no_grad, suffix chunks backpropagated to a detached prefix state, then a
second prefix pass with grad). It runs the prefix twice, not once; it is exact only up to summation order; and memory is
not the constraint here (arm C peaked at 19.7 GB at one record per pass, with per-layer checkpointing through the fork).
`--row_budget` already splits a record whose question rows do not fit a pass.

Predictions (not verdicts): Hh within 1 point of Dh's top1 overall and above it on banking77; PRh.maxr over-accepts on
the large stratum at the v7 threshold, PRh.maxr_k and PUh.maxr do not; P2h and PLh within noise of PRh.

## Commands (training machine, from `kev/`)

```bash
.venv/bin/python -m pytest tests/test_allwrong.py tests/test_pref.py
# G0 alone (minutes; queue.sh runs it first):
.venv/bin/python scripts/abstain/fork_check.py parity --kernels reference --base Qwen/Qwen3.5-0.8B-Base --format chat_pointwise --n 10
# everything (data build, G0, D1/E0 evals, the 16 h arms with eval_arm.sh + eval_allwrong.sh, the Decision Index subset
# of A seeds 0-2 (DI=0 skips), G4) on GPUs 0,1,2:
nohup scripts/abstain/queue.sh > runs/queue.log 2>&1 &
# progress: runs/queue.log; per arm runs/queue/<arm>.log and .done/.failed; gates: runs/queue/G0-parity.log, G4-*.log.
# Then the two reports (queue.sh prints them):
.venv/bin/python scripts/abstain/allwrong_report.py --arm A=runs/pref/A/seed0 --arm A1=runs/pref/A/seed1 \
  --arm A2=runs/pref/A/seed2 --arm C=runs/pref/C/seed0 --arm F=runs/pref/F/seed0 --out runs/allwrong/report-full
.venv/bin/python scripts/abstain/allwrong_report.py $(for a in Ah Ch Fh HBh Rh R2h Sh STh RYh Dh Hh PBh PRh PUh P2h PLh; \
  do printf -- '--arm %s=runs/pref/%s/seed0 ' $a $a; done) --out runs/allwrong/report-half
```

Lanes (`queue.sh`; chat arms at `BATCH_FALLBACK=1`, 1 record x accum 8 as C-F trained): GPU 0 G0, the D1/E0 all-wrong
evals, Ah, Ch, Hh, Dh, PLh, then the A Decision Index subsets; GPU 1 PRh, PUh, PBh, Sh, STh, HBh; GPU 2 P2h, Rh, R2h,
RYh, then Fh once Ch is done; then G4. Wall time [estimate from C-F's full-data 9.1 h and A's 2.4 h]: ~4.5 h per half
chat arm, ~1.2 h Ah, ~2.3 h per stage 2, ~2 h per Decision Index subset: ~25-30 h. A single arm by hand:
`BATCH_FALLBACK=1 scripts/pref/run_arm.sh PRh 0 1` (arm, seed, GPU index). Outputs: `runs/allwrong/report-*/report.json`
(every number above; `dev` per suite top1/NLL/sigmoid ECE; `threshold_transfer.risks.<risk>.<arm>.<readout>` with
`fit`, `apply`, `apply_oracle_coverage`, `strata`; `threshold_transfer.zero_fit.<arm>.<readout>` and
`threshold_transfer.frr_policy[_80].<arm>.<readout>` with the zero-fit and Lu & Wu policies), `rc-decision-v7.svg` /
`rc-transfer-v4.svg` (accuracy vs coverage; solid = maxr, dashed = maxprob).

## Risks

- F trained one extra epoch from C on the same data: a detection gain over C could be more training, not the BCE term
  (the same holds for Fh - Ch and STh - Sh). Rh vs Fh compares equal BCE exposure only roughly (2 vs 2 + 1 epochs).
- Half data lowers top1 for every h arm; the guard compares h arms with Ah, never with the full arms.
- Training already contains all-wrong sets, but only with a none option present (augment's `p_none` and none pairs): A
  and F have seen "the true option is gone" with a none option to pick, never without. The no-none condition is out of
  their training distribution for both.
- One seed, one checkpoint per arm; CIs are question sampling only.
- K = 3 questions become 2-option pairs; allwrong mnli is "neither of the remaining two relations", a natural case;
  banking77 keeps 76 of its 77 options, so its stratum is one source: a large-K effect and a banking77 effect are the
  same number here.
- Dh and Hh compute BSR per pass: both run at the same microbatch (queue.sh: `BATCH_FALLBACK=1`).

## External context

- Decision 1.0 (https://vllm-sr.ai/decision-paper.pdf) sec 3.2: "This supports variable output cardinality without
  enforcing permutation invariance."; sec 5: "Neither architecture enforces candidate-order invariance, and a closed
  candidate set cannot express an abstention outcome that the application has not supplied." The second sentence is
  why this eval is built rather than assumed: detection here reads the reward, not a supplied none option.
- Lu & Wu 2610.03387 report cross-task threshold transfer failure for their Jev policy (69.3% false rejection transferring
  DBpedia -> Emotion); D6 and the `frr_policy` numbers are our copy of that test, fit on v7 and applied frozen to t4.
- Zhong et al. 2609.39496: menu rejection collapse is arithmetic-dependent; their pNONE threshold fix is our `pnone`
  readout (offer a none option and threshold its mass).
- Li et al. 2609.33209: single-label coherence -- the readout `mass` sums the per-option sigmoids sigmoid(r_i), ~1 on an
  answerable row and ~0 on an allwrong one; its per-arm mean/median summary is the coherence check.
- Azizi et al. 2610.12292 (option-channel attack): option isolation does NOT defend it -- the attack edits the scored
  option's own name or the judged text. Stated as a non-claim: nothing here measures or claims robustness to that.

## Results

- **G0 PASS** (2026-10-14, jason, one 3090, `--kernels reference`): fork vs naive max |d reward| 2.03e-05, gradient
  relative error 8.13e-06 (worst tensor 1.89e-05), against naive1's floor of 2.48e-05 / 8.17e-06 (worst 2.08e-05); 10
  questions, K 2/4/77. The bound-kernel run on CUDA printed FAIL with naive1 itself at 3.25e-03 / 1.40e-03 and the fork
  at 7.60e-03 / 2.19e-03: TF32-class kernel rounding, not a fork error.

### Queue partial (2026-10-15, seed 0; `kev/runs/allwrong/report-full`, `report-half` run on the 12 finished h arms)

Finished: Ah Ch Rh R2h R2Bh Dh Hh PBh PRh PUh P2h P2Bh, and the D1 evals of A seeds 0-2 / C / F. Still queued: PLh, Sh,
STh, RYh, Fh, HBh, then G4 and RewardBench 2, so D2 (needs Fh), D3, D5 and D10's PLh half are open. Numbers are AUROC on
the minimal pairs, v7 / t4, dAUROC with the 95% CI; top1 is `reward_report.json raw.overall.top1` as the report prints it.

| Arm | v7 / t4 dev top1 | maxr AUROC v7 / t4 | | Arm | v7 / t4 dev top1 | maxr AUROC v7 / t4 |
|---|---|---|---|---|---|---|
| Ah | 85.21 / 77.59 | .9633 / .8539 | | Hh | 86.16 / 77.13 | .9451 / .8481 |
| Ch | 86.16 / 78.66 | .9339 / .8441 | | PBh | 85.76 / 78.51 | .9603 / .8460 |
| Rh | 85.60 / 76.07 | .9594 / .8412 | | PRh | 86.63 / 77.13 | .9595 / .8393 |
| R2h | 86.39 / 76.98 | .9616 / .8455 | | PUh | 86.31 / 75.91 | .9575 / .8476 |
| R2Bh | 85.76 / 77.59 | .9539 / .8492 | | P2h | 86.16 / 77.74 | .9572 / .8483 |
| Dh | 86.08 / 77.90 | .9197 / .8577 | | P2Bh | 86.23 / 77.29 | .9390 / .8368 |

Ah.maxprob is .8943 / .7349 and Ah.pnone .9683 / .8265. Every h arm passes the top1 guard (lowest is Ah itself).

- **Readout: the scalar reward beats the slot probability on every arm read** (the "A.maxr − A.maxprob" row of the
  comparison table): `maxr − maxprob` A +.0336 [+.0225, +.0451] / +.1041 [+.0790, +.1303], Ah +.0690 [+.0530, +.0858] /
  +.1190 [+.0930, +.1449], C +.0996 / +.1310, F +.0927 / +.1474 (CIs in `report-full`). Only A on v7 is
  under 0.05. It also transfers a threshold where `maxprob` does not (D6: Ah.maxr passes, Ah.maxprob fails at 0.78x), and
  it holds on Lu & Wu's benchmark (local detection `maxr` .66-.70 vs `maxprob` .42-.54, `external-luwu-benchmark.md`).
  It ties the none option rather than beating it (F.maxr − A.pnone −.0103 / +.0136), and unlike the none option it needs
  no none option in the prompt.
- **What the level looks like per pair** (`scripts/abstain/allwrong_examples.py`, Ah vs P2h seed 0, pairs Ah answers
  right; medians, answerable → allwrong):

  | Suite (pairs) | Arm | maxr | maxprob | mass |
  |---|---|---|---|---|
  | v7 (410) | Ah | +6.02 → −7.84 | 1.000 → .762 | 1.00 → .00 |
  | v7 (410) | P2h | +5.86 → −6.41 | 1.000 → .703 | 1.00 → .00 |
  | t4 (187) | Ah | +3.88 → −3.71 | .998 → .784 | 1.01 → .04 |
  | t4 (187) | P2h | +5.33 → −3.93 | .999 → .773 | 1.00 → .03 |

  When the right option is dropped, every remaining score stays low (mass ~1 → ~0, both arms, the pointer arm too), while
  softmax hands the freed mass to the best wrong option. Pairs where softmax stays confident and maxr does not (v7): mnli
  9430 Ah maxprob .993 on the allwrong row, maxr +3.31 → −10.05; mnli 1093 Ah 1.000, +6.25 → −8.53 (P2h .877, +4.93 →
  −7.07); agnews 536 Ah .963, +5.62 → −7.81. P2h's shared options keep their reward to ~0.02 (e.g. agnews 536 world
  −10.21 / −10.19), as a pointwise arm should; Ah's move with the set, both ways (dbpedia 17540 officeholder −2.90 →
  −10.14; dbpedia 35579 plant −13.24 → −4.79). Flag rates on the same 410 v7 pairs, each readout thresholded to keep
  95% of the answerable rows: Ah maxr (−0.267) flags .920 of the allwrong rows, Ah maxprob (.925) .639; P2h maxr
  (+1.419) .946, P2h maxprob (.879) .717. The reward flags 25-28 points more than softmax in both arms. Of Ah's 33
  maxr misses, only 8 have the wrong option that only the allwrong row holds on top (P2h 9 of 22). Those 8 are the
  arm believing that option over the gold one (dbpedia 35579 gold naturalplace Ah −10.58, allwrong-only village +8.54;
  mnli 5166 neutral −7.85, contradiction +6.31; trec 99 entity −4.18, number +4.11). That is model error or a
  mislabelled / ambiguous item, which no readout of the option set can flag. The other 25 (P2h 13) are a wrong
  option present in both rows scoring above the threshold; the script's "maxr misses with a shared option on top"
  section prints them (not yet read). The script's first print did not list the allwrong-only option; fixed since.
- **D1 (anchored training) inconclusive.** F.maxr − A.maxprob +.0254 [+.0123, +.0391] / +.0953 [+.0655, +.1276]:
  positive on both, below the 0.05 bar on v7; F's top1 86.16 passes the guard vs A 87.74. The anchor adds nothing over
  the readout: A.maxr (.9721 / .8525) is at or above F.maxr (.9640 / .8438). A.maxprob across seeds 0-2: v7 .9342-.9445,
  t4 .7379-.7485 (the seed spread is ~0.01).
- **D2 partial (Fh pending).** Rh − Ch +.0255 [+.0157, +.0352] / −.0029 [−.0149, +.0101]: Rh does not beat Ch.
- **D4: MLP head adopted** on the not-worse branch: R2h − Rh +.0022 [−.0021, +.0067] / +.0042 [−.0078, +.0159], top1
  86.39 vs 85.60.
- **D6 at risk 0.10** (t4 coverage / risk / t4-fit coverage): Ah.maxprob .158 / .066 / .202 (0.78x, fails); Ah.maxr .250 /
  .075 / .258 passes; Rh.maxr .281 / .133 / .260, R2h.maxr .256 / .106 / .252, PRh.maxr .258 / .121 / .208, P2h.maxr .240 /
  .130 / .231 pass; PUh.maxr risk .200, PRh.maxr_k .180 and P2h.maxr_k .212 fail on risk. Judging D6 needs the D2-D5
  winner (open).
- **D7 / D11 match the none option** (≥ Ah.pnone − 0.02 on both): Rh.maxr − Ah.pnone −.0089 / +.0148, R2h −.0066 /
  +.0190, PRh −.0088 [−.0173, −.0008] / +.0129.
- **D8 refuted: Dh stays.** Hh.maxr − Dh.maxr +.0254 [+.0162, +.0351] / −.0096 [−.0235, +.0049]: the t4 lower end misses
  −0.02, and banking77 answerable top1 is Hh .775 vs Dh .8125 (−3.75 points, n 80,
  `eval/decision-v7-development/report.json`). Hh.maxprob − Dh.maxprob +.0316 / +.0049.
- **D9: no pointwise recipe.** PRh − Hh +.0144 [+.0077, +.0217] / −.0087 [−.0177, +.0004]: no beat. PRh is not worse
  than PBh (−.0008 / −.0066 [−.0145, +.0012]). K balance not adopted: PRh.maxr_k − PUh.maxr −.0047 / −.0140 [−.0220,
  −.0059] misses not-worse on t4. PRh − Rh +.0001 / −.0019.
- **D10 (P2h half): P2h does not replace PRh.** P2h − PRh −.0023 [−.0086, +.0038] / +.0089 [+.0007, +.0178], top1 86.16
  vs 86.63. P2h's leave-one-out verdict: `../done/pointwise-rcr-mlp.md`.
- **Predictions:** C.maxr within 0.05 of A.maxprob holds on v7 (+.011) and fails on t4 (+.107); F.maxprob below F.maxr
  holds; Hh within 1 point of Dh's top1 holds, above it on banking77 fails; PRh.maxr over-accepts the large stratum at the
  v7 threshold (risk .250 vs small .083) while PRh.maxr_k and PUh.maxr do not (.095 each): holds; P2h within noise of PRh
  holds.
- **Shared-option |Δ reward|** on pointwise arms: mean 3-8e-3 but max .08-.17 on v7 and .558 for PRh on t4, more than bf16
  rounding should give; listing arms 4-14. G4 (`fork_check.py k1`) decides whether PRh's numbers stand.
- Pointwise arms rarely pick a none option when it is offered on an allwrong row (v7 .61-.69, t4 .23-.37) against
  .85-.91 on the listing arms: none rows are dropped from pointwise training.
