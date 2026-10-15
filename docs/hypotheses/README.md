# Hypotheses index

Every question this repo is asking about the reward model, one doc each. **Lifecycle:** a question starts as
`todo/<slug>.md` — claim, arms (or "not built"), a pre-registered measurable decision rule, a cost estimate and the
commands if they exist; it moves to `in-progress/` with `git mv` when its first run launches, and to `done/` with
`git mv` once its decision rule has been applied and the numbers are recorded. A doc is never copied between folders,
and it keeps its claim, arms, pre-registered decision rule, commands and results (each doc carries a `Status:` line
under its title). Every number below comes from a doc, code or data file in this repo — the source is cited beside it;
anything unknown is said to be unknown, never estimated silently. Costs are marked **[estimate]**.

## Index

| Doc | Status | Question | Verdict / next |
|---|---|---|---|
| [`done/preference-reward-model.md`](done/preference-reward-model.md) | done | Does a scalar Bradley-Terry reward head on Kev's recipe match the pointer + CE baseline on top1, calibration and reward scale? | **Inconclusive** (rule 5): best scalar arm D, 3-seed v7 top1 −0.79 vs A (margin −0.5), t4 +0.15; externally D seed 0 scores 36.91 on the DI 0.3 subset vs 33.60 for v7-base. Next: A on the same subset ([`todo/di-attribution-a-subset.md`](todo/di-attribution-a-subset.md)) and C0 ([`todo/c0-head-vs-format.md`](todo/c0-head-vs-format.md)). |
| [`done/pointwise-chat-input.md`](done/pointwise-chat-input.md) | done | Does dropping the `Options:` listing from the chat prompt (pointwise `chat_pointwise`) hold dev accuracy while buying set-independence and ranking? | **Format refuted** at seed 0 (rule 1: v7 85.84 vs D 87.42, −1.58; t4 78.51 vs 79.73, −1.22); HP (preference pairs) refuted, RB2 −1.36 vs H; HL (listwise CE) refuted, t4 −1.22; the probe Δ = 0 / 0-flip clauses hold. HB / HBL were never run → [`todo/absolute-grader-hb.md`](todo/absolute-grader-hb.md); H seeds 1-2 → [`todo/h-seeds-exploratory.md`](todo/h-seeds-exploratory.md). |
| [`done/pointwise-rcr-mlp.md`](done/pointwise-rcr-mlp.md) | done | Is P2h (`chat_pointwise` + MLP head + RCR with the balanced point term) the contribution, and which of its three components carries the gain? | **Not the contribution** (rule 1, seed 0): beats no leave-one-out arm; not worse than R2Bh / P2Bh / PRh but v7 top1 86.16 < PRh 86.63. Only the RCR loss is above 0 on both sets (P2h − P2Bh +.018 / +.012, under 0.03). G4 (K = 1, reference kernels, fp32) passed on every pointwise arm. |
| [`in-progress/allwrong-abstention.md`](in-progress/allwrong-abstention.md) | in-progress | Does an anchored reward know when *no* option is right (minimal pairs, no none option) better than a softmax confidence, and can one training stage carry it? | **Readout (pre-registered comparison, no rule of its own):** the scalar reward `maxr` beats softmax `maxprob` on every arm read (A +.034 / +.104, Ah +.069 / +.119, C/F +.09-.15), transfers a threshold where `maxprob` fails (D6, Ah), and ties the none option without needing one. **Anchored training (D1) inconclusive:** F.maxr − A.maxprob +.025 v7, and A.maxr ≥ F.maxr. D2: Rh ≈ Fh, neither beats Ch on t4; D3 R2h preferred (STh ≈ Fh); D4 MLP head adopted; D8 refuted (Dh stays); D5 closed (RYh ≈ HBh ≈ Rh; listing prompt stays); D2-D5 winner R2h, which transfers its threshold (D6: t4 coverage .256, risk .106) and matches the none option (D7: −.007 / +.019); D9 no pointwise recipe; D10 P2h not adopted. G4 PASS on all nine pointwise arms (reference kernels, fp32). Open: RB2 on the h arms. |
| [`todo/absolute-grader-hb.md`](todo/absolute-grader-hb.md) | todo | Does asking each option "Is this option a correct answer? Yes/No." (HB, head = base `W_yes − W_no`, class-balanced BCE) give a better pooled AUROC than H? | Bars measured from H: HB is supported only at t4 pooled AUROC ≥ .908 and RB2 ≥ .865, with top1 ≥ 84.84 v7 / 77.51 t4. HB / HBL never run. |
| [`todo/h-seeds-exploratory.md`](todo/h-seeds-exploratory.md) | todo (exploratory) | Is H's true gap to D 0.9 or 1.6 points on v7 (seed-0 screen vs 3-seed mean)? | Rule 2 forbids seeds 1-2 for the verdict after the seed-0 failure; the two runs would annotate the refuted verdict, not move it. |
| [`todo/d-order-averaging.md`](todo/d-order-averaging.md) | todo | Does averaging D's reward over k shuffled option orders remove RewardBench 2's mild primacy? | Not run; D's seed-0 shuffled run shows top-scored share 27 → 22% and win −10 points from 1st to 4th (χ² 12.1 on 3 df, p .007). |
| [`todo/c0-head-vs-format.md`](todo/c0-head-vs-format.md) | todo | Did the scalar head or the chat format cost C its 0.5 / 1.4 points vs B (Kev format)? | C0 was never run ("head and format are not separated", preference doc §Readings); C vs A on v7 is −1.19 with the CI excluding 0. |
| [`todo/di-attribution-a-subset.md`](todo/di-attribution-a-subset.md) | todo | Is D's DI 0.3 subset +3.31 over v7-base the pref loss / scalar head, or the training run? | **Loss effect (D's recipe as a whole), all 3 A seeds:** A 34.54 / 33.05 / 33.12, mean 33.57 ± 0.84 = v7-base 33.60; D 36.91 is +3.34 over the A mean. Per-benchmark pairs for A seeds 0 / 2 not yet run. |
| [`todo/di-full-suite.md`](todo/di-full-suite.md) | todo | A board-comparable full-suite DI run (140,620 requests) for the final head. | Not run: the subset index is a paired estimate, not a board number; 15-30 h on one 3090 **[estimate]**. |
| [`todo/di-contamination-audit.md`](todo/di-contamination-audit.md) | todo | Is our training data contaminated against Decision Index items? | `docs/studies/external_comparison.md` §3: "investigated, not measured"; needs canonical-content-key + normalised-state matching, per benchmark / source / split. |
| [`todo/serving-ideas.md`](todo/serving-ideas.md) | todo | Which recorded chat-serving idea pays off, and what does a graphed D subset actually cost? | D's eager subset was 27,246 s for 15,000 rows, 3.7x the Kev format; the graphed re-measure is "not run yet" (preference doc §Inference speed). |
| [`todo/frozen-head-followups.md`](todo/frozen-head-followups.md) | todo | Do the frozen-head levers left unrun beat L24 + sigmoid (DI sample 36.33 ± 0.97)? | `docs/studies/README.md` open items + `frozen_lossgrid.md`: "Not run: L24 vs last at 3 seeds, concatenated L24 + last features, `augment_variants=true`, larger heads, full-suite DI." |
| [`in-progress/external-luwu-benchmark.md`](in-progress/external-luwu-benchmark.md) | in-progress | On Lu & Wu's public abstention benchmark (TREC, DBpedia, Emotion, CLINC150), does our selected arm's threshold transfer across tasks where Jev's does not (69.3% false rejection DBpedia → Emotion)? | `maxr_k` cross-task passing rows: A, C, D, F, P2h 2/6; PRh, Ah, P2Bh 3/6 (Emotion as target over-rejects on every arm); Ah.`maxprob_T` 0/6. Half-data pointwise arms (PRh, P2h) do not beat Ah's logit on false rejection and detection together. The gain is local (`maxr` detection .66-.70 vs `maxprob` .42-.55). Rule 3 passes for every arm scored. |
| [`todo/coherence-option-channel.md`](todo/coherence-option-channel.md) | todo | Does pointwise scoring make confidences coherent and immune to injected / renamed distractor options (Li et al.; option-channel attack)? | Probes not built; reorder and add/remove are already measured (`docs/data-eval.md` §2g rows 10-12). |

## GPU plan (3x 3090)

Training machine: 3x RTX 3090 24 GB, one experiment per GPU, checkout `kev/` holds `runs/`. The next queue is
`scripts/abstain/queue.sh` (lanes below are its, with the P2h leave-one-out arms in their pre-registered positions).
Costs **[estimate]** from the allwrong doc §Commands: ~4.5 h per half chat arm, ~1.2 h `Ah`, ~2.3 h per stage-2 arm,
~2 h per Decision Index subset, ~25-30 h for the whole queue.

| Lane | Order | Covers |
|---|---|---|
| GPU 0 | G0 fork parity (0.8B, minutes) → `Hh` → `Dh` → all-wrong evals of the full `A` seeds 0-2 / `C` / `F` (D1, E0.2, no training) → `Ah` → `Ch` → `PLh` → DI subsets of `A` seeds 0-2 (E0.1; `DI=0` skips) | G0 gate, the D8 topology pair, the D1 readout on full arms, the half-data baseline pair ah/ch, PLh (D10) and the DI baseline D's +3.31 needs |
| GPU 1 | `PRh` → `P2Bh` → `PUh` → `PBh` → `Sh` → `STh` → `HBh` | D9 objective (RCR vs BT vs K-balance), P2h's loss ablation, the absolute stage-1/2 pair, the yes/no grader |
| GPU 2 | `P2h` → `R2Bh` → `R2h` → `Rh` → `RYh` → `Fh` (waits for `Ch`'s marker; stage 2 on `Ch`) | **P2h first** (the priority arm), then its listing ablation, then the head/loss ladder on the listing prompt |

After the queue: G4 (`fork_check.py k1 --kernels reference` on Hh, PBh, PRh, PUh, P2h, P2Bh, PLh, RYh, HBh — every
pointwise arm; each option scored alone vs all K together), then the two reports `queue.sh` prints (`allwrong_report.py --out runs/allwrong/report-full`
and `report-half`); `report-half`'s defaults already carry `P2h.maxr-R2Bh.maxr`, `P2h.maxr-P2Bh.maxr` and
`P2h.maxr-Dh.maxr`.

## Conventions

- Commands are **kev-relative**: run them from the repo root through the wrapper, `./kev.sh <command>` (a `*.py` first
  argument runs with `kev/.venv/bin/python`, anything else is exec'd as-is, paths stay kev-relative).
- A doc's decision rule is fixed before any run; a verdict is reported as supported / refuted / inconclusive, with the
  rule and the numbers that decided it.
- Cite the file a number came from. Unknown is unknown.
