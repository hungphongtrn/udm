# Question: do a few target labels restore the reward's abstention threshold on a new task family?

Status: done, supported (see ../README.md). Commands below are kev-relative: run them from the repo root as
./kev.sh <command>.
Branch: `hyp/few-label-target-threshold`, cut from `hyp/allwrong-abstention` (not `dev`: the report script and the
all-wrong evals it reads are unmerged there). No training.

## Question

The scalar reward's level (`maxr`) flags all-wrong questions far better than softmax (`maxprob`) at the same answer
rate, but the level drifts between task families: the threshold that keeps 95% of answerable rows is −0.267 on
decision-v7 and −1.991 on transfer-v4 (Ah). Is that drift a per-domain offset, which a handful of labelled target
questions can re-fit, or does the reward rank all-wrong rows worse on the new family, so that no threshold recovers
the in-domain detection? And does the reward still beat softmax after both are re-fit on the same few labels?

## Why now / evidence

On the pairs Ah answers right, each readout thresholded to keep 95% of their answerable rows
(`kev/scripts/abstain/allwrong_examples.py`, seed 0; `hyp/allwrong-abstention`
`docs/hypotheses/in-progress/allwrong-abstention.md` §Results, "What the level looks like per pair"):

|Suite (pairs)|Ah maxr|Ah maxprob|P2h maxr|P2h maxprob|
|---|---|---|---|---|
|v7 (410)|.920 (t −0.267)|.639 (t .925)|.946 (t +1.419)|.717 (t .879)|
|t4 (187)|.647 (t −1.991)|.337 (t .643)|.658 (t −2.353)|.273 (t .579)|

- The thresholds move with the suite, so a threshold fit on v7 and applied to t4 does not keep 95% of t4's answerable
  rows. That is the D6 / Lu & Wu transfer problem (`external-luwu-benchmark.md`, on `hyp/allwrong-abstention`:
  `maxr_k` passes 2/6 cross-task rows; Emotion as target over-rejects).
- The literature says transfer under shift degrades and carries no guarantee without assumptions on the shift (Ovadia
  et al. 2019, arXiv 1906.02530; Eisenstein et al. 2023, 2312.09244 for reward models; Ben-David et al. 2010; Fang et
  al. 2022, 2210.14707). The practical alternative is a small labelled calibration set from the target (conformal
  prediction: Angelopoulos & Bates, 2107.07511). This question measures how small "small" is for this reward.
- The motivation is a generalised reward: one score whose level means "this option is right" across tasks. If a few
  labels close the gap, the level carries across families up to an offset, and the per-domain cost is a few dozen
  labels, not retraining.

## Arms

No new arms. Readouts of runs that already have `eval/allwrong-decision-v7` and `eval/allwrong-transfer-v4`:

|Arm|Run|Role|
|---|---|---|
|Ah|`runs/pref/Ah/seed0`|pointer + CE (Kev): softmax-trained baseline|
|P2h|`runs/pref/P2h/seed0`|`chat_pointwise` + MLP + RCR: the anchored pointwise reward|

Readouts: `maxr` (primary), `maxprob` (the softmax comparator), `maxr_k` and `maxprob_G` (reported, not judged).

## Protocol

`kev/scripts/abstain/allwrong_report.py`, section "few-label target fit" (added on this branch). The pool, per arm and
readout, is every transfer-v4 question the arm scored: its answerable row and its allwrong twin (all questions, not
only those the arm answers right as in the table above, so the ceiling here differs from that table).

- Policy (Lu & Wu 2610.03387, as in D6): accept iff readout > t, with t chosen to maximise allwrong detection subject to
  answerable false rejection ≤ 5% on the fit set (`fit_frr_threshold`).
- Fit on n = 20 / 40 / 80 transfer-v4 questions (both rows of each), drawn without replacement, 200 seeded draws per n
  (seed 0). Score detection and false rejection on the held-out transfer-v4 questions. Report the mean and p10 / p90
  over draws.
- Beside it: the v7-fit threshold frozen on the whole transfer-v4 pool ("frozen", the zero-label baseline), and the
  transfer-v4 pool's own in-sample fit ("ceiling", what the readout's ranking allows).

## Decision rule (pre-registered)

Judged on `maxr` for Ah and for P2h separately, at n = 40 (the held-out means):

1. **Supported** if all three hold:
   - (a) mean held-out false rejection ≤ 0.08 (5% target plus small-sample slack);
   - (b) mean held-out detection ≥ ceiling − 0.05;
   - (c) the arm's `maxr` detection exceeds its `maxprob` detection at n = 40 by ≥ 0.05.
2. **Refuted** if at n = 80 either holds:
   - mean held-out detection is below ceiling − 0.10 (labels do not recover the in-domain level);
   - `maxr` detection ≤ `maxprob` detection (re-fitting erases the reward's advantage).
3. Otherwise **inconclusive**. Also report the smallest n whose mean detection is within 0.05 of the ceiling with
   false rejection ≤ 0.08.

The ceiling itself is also read against v7: if the t4 ceiling is far below v7's own detection (e.g. Ah .647 vs .920
in the table above), the gap is ranking, not offset, and no number of labels closes it. That reading is reported, not
judged.

## Commands

```bash
git fetch origin && git checkout hyp/few-label-target-threshold
mkdir -p kev/runs/allwrong
./kev.sh scripts/abstain/allwrong_report.py --arm Ah=runs/pref/Ah/seed0 --arm P2h=runs/pref/P2h/seed0 \
    --out runs/allwrong/report-fewlabel | tee kev/runs/allwrong/report-fewlabel.txt
```

CPU only, numpy; minutes **[estimate]**. Paste the "few-label target fit" block (`report.json` key
`threshold_transfer.target_fit` holds every number).

## Results

Run 2026-10-15 on the training machine, command above, seed 0 for both arms and for the draws. Pools: 448 v7 and 240
t4 questions scored by both arms (answerable row + allwrong twin each). No deviations from the plan.

Lu & Wu policy (answerable false rejection ≤ 5%), allwrong detection / answerable false rejection; few-label cells are
held-out means over 200 draws, p10–p90 of detection in brackets:

|Arm · readout|v7 in-sample|v7 frozen → t4|t4 ceiling|n=20|n=40|n=80|
|---|---|---|---|---|---|---|
|Ah · maxr|.826 / .049|.621 / .154|.467 / .050|.495 / .082|.483 / .069 [.385, .575]|.464 / .057|
|Ah · maxprob|.507 / .049|.508 / .242|.200 / .050|.228 / .083|.208 / .068 [.110, .305]|.190 / .056|
|P2h · maxr|.862 / .049|.704 / .221|.421 / .050|.463 / .074|.457 / .070 [.344, .585]|.435 / .059|
|P2h · maxprob|.484 / .049|.387 / .142|.171 / .050|.228 / .083|.211 / .074 [.085, .355]|.173 / .057|

Rule, at n = 40 on `maxr`:

|Clause|Ah|P2h|
|---|---|---|
|(a) false rejection ≤ .08|.069 ✓|.070 ✓|
|(b) detection ≥ ceiling − .05|.483 ≥ .417 ✓|.457 ≥ .371 ✓|
|(c) `maxr` − `maxprob` ≥ .05|+.275 ✓|+.246 ✓|
|Refuted at n = 80?|no (.464 vs ceiling .467; `maxr` > `maxprob`)|no (.435 vs .421)|
|Smallest n within .05 of ceiling, frr ≤ .08|40 (n = 20 frr .082)|20|

Readings:

- **The frozen v7 threshold over-rejects on t4**: 15% (Ah) and 22% (P2h) of answerable rows, 3–4× the 5% budget.
  Its higher t4 detection (.621 / .704) is bought with those rejections. 40 target questions bring false rejection
  back to ~7%, and 80 bring it to ~6%. The fit still runs above 5% on held-out rows (small-sample optimism of an
  in-sample quantile), and single draws reach p90 .115 at n = 40. A hard 5% guarantee needs a finite-sample
  (conformal-rank) threshold, not this policy.
- **The t4 gap is ranking, not offset** (the reported, unjudged reading). At the same 5% false rejection, `maxr`
  detects .826 / .862 of allwrong rows on v7 but at most .467 / .421 on t4. No t4 threshold closes that gap, so the
  level does not carry across families up to an offset. Part of it is offset (fixed by labels), and roughly
  .36–.44 detection is lost in the ranking itself.
- **The reward still beats softmax after re-fitting**: `maxr` detects 2.2–2.3× what `maxprob` does at n = 40. On
  t4, `maxprob` cannot exceed .20 / .17 detection at 5% false rejection.
- `maxr_k`, `noisy_or`, `mass` and `lse` sit within .02 of `maxr` at every n (not judged). Ah and P2h are within
  .03 of each other.
- An 80-question v7 sample is worse than the full v7 fit as a transfer source: t4 false rejection is .338 (Ah) and
  .404 (P2h) `maxr`. Few labels help only when they come from the target family.

## Verdict

**Supported** (pre-registered rule): for both arms, 40 labelled t4 questions recover the t4 ceiling within .05 at
≤ 8% false rejection, and `maxr` keeps a ≥ .05 lead over `maxprob` (+.25–.28). Scope: this establishes that the
threshold is cheap to re-fit per family. It does not establish a generalised reward level: the t4 ceiling is
.36–.44 below v7's detection, so most of the cross-family loss is in the reward's ranking on t4. Raising that
ceiling is a training question (a reward that ranks t4 allwrong rows lower), not a calibration one. Single seed
per arm.
