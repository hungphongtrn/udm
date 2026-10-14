# Question: does the absolute grader (HB / HBL) beat H's pooled AUROC?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

Pre-registered in `done/pointwise-chat-input.md` (§"The absolute-grader prompt" and rule 5's absolute-grader clause);
the arms exist and were never run.

## Question

Does asking one option at a time "Is this option a correct answer? Answer Yes or No." (`chat_yesno`) — the scalar head
started at the base LM head's `W_yes − W_no` (`--head_init yesno`), trained with class-balanced per-option BCE — give a
reward on **one scale across questions**: a higher pooled AUROC than H (and than H + Platt, which has the same AUROC)
on transfer-v4 development and RewardBench 2, at dev top1 within 1 point of H? HBL = HB + 0.25 × Kev's listwise CE
keeps HB's AUROC while recovering any top1 HB loses (arm table, `done/pointwise-chat-input.md` §Arms).

## Why now / evidence

The comparator is now measured, so the bars are numbers, not assumptions (`done/pointwise-chat-input.md` §Readings,
"Absolute-grader arms (HB / HBL) are compared with H, now measured"):

| HB's bar | Value | Source |
|---|---|---|
| transfer-v4 pooled AUROC (H) | .898 → HB needs **≥ .908** | §Readings, §Development sets |
| RewardBench 2 pooled AUROC (H) | .855 → HB needs **≥ .865** | §Readings, §RewardBench 2 |
| v7 dev top1 (H) | 85.84 → guard floor **84.84** | §Development sets |
| t4 dev top1 (H) | 78.51 → guard floor **77.51** | §Development sets |
| Platt cannot move a pooled AUROC | `auroc_pooled` is rank-based | §Metrics |

Supporting evidence for the recipe: absolute/sigmoid losses were the best frozen-head family (36.33 ± 0.97 vs ~32 for
CE-only, `docs/studies/README.md` conclusion 2), and arm F's BCE stage calibrated sigmoid(r) without costing ranking
(v7 sigmoid ECE .063 → .015, t4 .159 → .107, top1 within noise, `done/preference-reward-model.md` §Readings). The
softmax-listwise objective on the pointwise prompt is the one already refuted (HL: t4 −1.22 vs H, RB2 0.6332 vs
0.6653), which is why HBL carries a 0.25 weight and not HL's full loss. HB's prior is also the k-option reading
sigmoid(r − log(k − 1)) (`kev.pref.balanced_bce_loss`), i.e. the reward estimates a log likelihood ratio.

## Arms

| Arm | Flags | Status |
|---|---|---|
| HB | `--head scalar --input_format chat_yesno --head_init yesno --loss bce` (half data: `--train_frac`) | built (`arms.sh`), not run |
| HBL | HB + `--ce_w 0.25` (one flag) | built (`arms.sh`), not run |

HB differs from H in three coupled things (prompt, head init, loss) deliberately — the Yes/No question is what makes
`W_yes − W_no` meaningful, and the init is what puts BCE's absolute target on a scale the base already reads; HB is a
recipe test against H, not an ablation (`done/pointwise-chat-input.md` §Arms). The `none` wordings
(`kev.data.NONE_TEXTS`) are left out of the loss: "none of these" is a statement about the set, not a gradable option.

## Decision rule (pre-registered, seed 0; from `done/pointwise-chat-input.md` rule 5)

- **Supported** if HB's pooled AUROC beats H's by ≥ 0.01 on transfer-v4 development AND on RewardBench 2 (overall,
  `rb2_pooled.py`) — with H's measured values: **≥ .908 and ≥ .865** — with HB's top1 not more than 1 point below H's on
  either dev set (**≥ 84.84 v7, ≥ 77.51 t4**).
- **Refuted** if HB's pooled AUROC is not above H's on either, or its top1 is more than 1 point below H on either dev set.
- **Inconclusive** otherwise.
- HBL vs HB: HBL within 0.01 of HB's pooled AUROC **plus** HBL's top1 ≥ HB's.
- Calibration (secondary): HB's `ece_sigmoid_prior` vs H + Platt's `ece_sigmoid` on decision-v7 development.
- Per-subset prediction, recorded before the run: the gain shows on the verifiable subsets (Factuality, Math, Precise
  IF) and not on Focus or Ties.
- Next stages, **only if HB is supported** (not pre-registered): HBX (HB on long-form candidate data) and preference
  fine-tuning from HB's checkpoint with its BCE kept as an anchor.

## Cost [estimate]

- HB / HBL training + dev eval: ~9 h per arm on one 3090 (H's chat cost; the prompt is pointwise in both formats,
  `done/pointwise-chat-input.md` §Compute).
- RewardBench 2: ~5-8 h per arm (HB scores each candidate alone: 8,977 one-candidate requests) plus the 64-prompt probe
  ~10 min. Two arms with RB2 ≈ **28-35 h** of one 3090.

## Commands

```bash
BATCH_FALLBACK=1 ./kev.sh scripts/pref/run_arm.sh HB 0 0      # train + eval_arm.sh (dev sets + calibration)
BATCH_FALLBACK=1 ./kev.sh scripts/pref/run_arm.sh HBL 0 1

# pooled AUROC on both pools (the rule's two numbers), H's are in done/pointwise-chat-input.md §RewardBench 2:
./kev.sh scripts/pref/rewardbench2.py --run runs/pref/HB/seed0 --out runs/pref/HB/seed0/rewardbench2.json
./kev.sh scripts/pref/rb2_pooled.py runs/pref/HB/seed0/rewardbench2.json \
  --report runs/pref/HB/seed0/eval/decision-v7-development/reward_report.json
```

(also in `kev/scripts/pref/README.md` §"Pointwise chat arms".)

## Results

Not run (`done/pointwise-chat-input.md` §Results: "HB / HBL not run").
