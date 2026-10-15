# Question: is P2h the contribution (pointwise prompt + MLP head + RCR), and which component carries the gain?

Status: done (see ../README.md): P2h is not the contribution at seed 0. Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

**Priority: this is the novel contribution.** Pre-registered here before any P2h run; arms already built.

## Question

P2h = the pointwise chat prompt (no option listing) + the MLP head + RCR with the balanced point term:

```
--head mlp --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point balanced
```

(`kev/scripts/pref/arms.sh` `P2h`). Does the *combination* beat each of its three leave-one-out arms — the listing
restored, the RCR loss replaced by BT + BSR, the head linear — and which component carries the gain?

- Head: `MLPHead`, `Linear(d, 1024)` → SiLU → `Linear(1024, 1)`, output layer zero-initialised so step 0 gives r = 0
  for every option (`kev/kev/model.py:360-373`).
- Loss: `rcr_loss` = (1 − α)·point + α·ListCE(sigmoid), α = 0.5; `--rcr_point balanced` gives a question's positives and
  wrong options half the question each (α = 0, β = 0 is `balanced_bce_loss`), where the mean point term is minimised at
  sigmoid(r_i) = P(option i right) and so a K = 77 question's rewards sit ~log 76 below a K = 2 question's
  (`kev/kev/pref.py:82-99`, `in-progress/allwrong-abstention.md` §PF-RCR wave).

P2h already sits in the PF-RCR wave as "PRh: the head (H-RCR-2L)" (phase 2, rule D10). This doc frames the combination
as the claim and states the ablations it must beat.

## Why now / evidence

The three components target different, measured failure modes (all seed 0, from `done/pointwise-chat-input.md`):

| Evidence | Number | Source |
|---|---|---|
| Dropping the listing costs decision accuracy | H v7 85.84 / t4 78.51 vs D seed 0 87.42 / 79.73; D 3-seed means 86.76 / 78.81 | §Verdict, §Development sets |
| Dropping it buys exact set-independence | probe mean / max \|Δ\| 0 / 0, 0 of 64 flips; 0 permutation flips on both dev sets vs D 1 of 36 on t4 | §RewardBench 2, §Verdict |
| Ranking is a wash | RB2 0.6653 vs D 0.6641 (+0.12) | §RewardBench 2 |
| A listwise loss on the same prompt gains nothing | HL v7 +0.79, t4 −1.22 vs H; RB2 0.6332 vs H 0.6653 | §Verdict, §RewardBench 2 |
| The listing is what readouts depend on | A's top-scored share by shown position .458 / .217 / .176 / .149; D mild primacy (top-scored 27 → 22%, win −10 points 1st → 4th, χ² 12.1 on 3 df, p .007); pointwise none by construction | §Position effect on RewardBench 2 |
| What bounded the reward scale so far is the BSR regularizer, not the head | D seed 0 \|r\| p99 7.1 vs A 15.0 (fitted T 1.44 vs 1.78), stable across 3 seeds; P2h keeps BT + BSR only in its P2Bh ablation, so RCR's point term is the untested scale control | `done/preference-reward-model.md` §Seeds 1-2, §Readings |

None of the three has been measured on the pointwise prompt together with the other two.

## Arms

| Arm | Role | Flags | Status |
|---|---|---|---|
| **P2h** | the claim | `--head mlp --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point balanced` | built (`arms.sh`) |
| R2Bh | listing-removal ablation = R2h with `--rcr_point balanced` | `--head mlp --input_format chat --loss rcr --rcr_alpha 0.5 --rcr_point balanced` | built (`arms.sh`) |
| P2Bh | RCR-loss ablation = Hh with the MLP head | `--head mlp --input_format chat_pointwise --loss pref --reg bsr --reg_w 1e-3` | built (`arms.sh`) |
| PRh | MLP-head ablation (linear head, same prompt and loss) | `--head scalar --input_format chat_pointwise --loss rcr --rcr_alpha 0.5 --rcr_point balanced` | built; D10's head control |
| Hh / Dh | half-data references (pointwise / listing, BT + BSR, linear head) | `arm_flags H` / `arm_flags D` + `--train_frac` | built |
| PLh | the yes/no variant of PRh's loss (D10's other arm) | `--head scalar --input_format chat_yesno --head_init yesno --loss rcr --rcr_alpha 0.5 --rcr_point balanced` | built |

R2h alone is confounded as a listing ablation: it is `--head mlp --input_format chat --loss rcr --rcr_alpha 0.5` with
the default `--rcr_point mean` (`arms.sh` `R2h`), so P2h − R2h would move the prompt and the point term together;
R2Bh holds the point term fixed. R2Bh is a listing arm, so it is not in the G4 K = 1 check (pointwise arms only).

## Decision rule (pre-registered, seed 0, half data)

Terms from `in-progress/allwrong-abstention.md` §Decision rules, unchanged: X **beats** Y when dAUROC(X − Y) ≥ 0.03 with
the 95% CI above 0 on both v7 and t4 (AUROC on the minimal pairs, readout `maxr`); X is **not worse** when the CI's
lower end is ≥ −0.02 on both; the **top1 guard** is v7 development top1 (`reward_report.json raw.overall.top1`) at most
2 points below `Ah`'s.

1. **Contribution.** P2h is the contribution if it beats each leave-one-out arm (R2Bh, P2Bh, PRh), or is not worse than
   each and has a higher v7 dev top1. If it beats some and not others, the contribution note must name which component
   carries the gain and which is inside noise; "P2h wins" alone is not the claim.
2. **Per-component readings**, reported whether or not (1) holds: P2h − R2Bh is the listing removal under the MLP head
   and the balanced point term; P2h − P2Bh is the RCR loss under the pointwise prompt; P2h − PRh is the MLP head (D10's
   own rule: adopted if P2h beats PRh, or is not worse with a higher dev top1).
3. **Beside the pairs:** dev top1 on decision-v7 and transfer-v4 development (with `none_present` / `none_absent`),
   RewardBench 2 (six-subset mean and pooled AUROC via `rb2_pooled.py`), threshold transfer at risk 0.10 with the K
   strata (D6 / gate G3), and the shared-option |Δ reward| — 0 up to rounding for a pointwise arm, nonzero for a listing
   arm; the join is the report's.
4. **Gates:** G0 fork parity before training (a failure stops the PF arms' interpretation, not the queue) and G4 K = 1 on
   every pointwise checkpoint (max |Δ sigmoid| < 1e-3 and no argmax change).
5. The 0.03 / 0.02 / 2-point thresholds are the screen's, fixed before these arms existed; do not re-tune them here.

## Cost [estimate]

~4.5 h per half chat arm on one 3090 (allwrong doc §Commands); the MLP head adds a `Linear(d, 1024)` and a
`Linear(1024, 1)` applied to each option's hidden state (memory, not wall time, at this scale [INFERENCE]). Three
leave-one-out arms ≈ 13.5 h; R2Bh and P2Bh are two of them, so ~9 h is new over the already-planned queue.

## Commands

Lanes for these arms (queue.sh, in `../README.md`): P2h first on GPU 2, P2Bh second on GPU 1, R2Bh second on GPU 2 —
i.e. GPU 0 `G0 parity → Hh → Dh → rest of the GPU 0 lane`; GPU 1 `PRh → P2Bh → PUh → PBh → Sh → STh → HBh`;
GPU 2 `P2h → R2Bh → R2h → Rh → RYh → Fh`.

```bash
# the whole queue (GPU0/1/2 lanes in ../README.md); report-half's defaults already include
# P2h.maxr-R2Bh.maxr, P2h.maxr-P2Bh.maxr and P2h.maxr-Dh.maxr (kev/scripts/abstain/allwrong_report.py)
nohup ./kev.sh scripts/abstain/queue.sh > kev/runs/queue.log 2>&1 &

# one arm by hand (arm, seed, GPU index):
BATCH_FALLBACK=1 ./kev.sh scripts/pref/run_arm.sh P2h 0 2

# the leave-one-out table, once the half-data runs exist (queue.sh prints this line too):
./kev.sh scripts/abstain/allwrong_report.py $(for a in Ah Ch Fh HBh Rh R2h R2Bh Sh STh RYh Dh Hh PBh PRh PUh P2h P2Bh PLh; do printf -- '--arm %s=runs/pref/%s/seed0 ' $a $a; done) --out runs/allwrong/report-half
```

## Results

2026-10-15, seed 0, half data (`kev/runs/allwrong/report-half`; full tables in `../in-progress/allwrong-abstention.md`
§Results). AUROC `maxr`, v7 / t4; dAUROC with the 95% CI:

| Arm | v7 / t4 dev top1 | maxr AUROC v7 / t4 | P2h − arm, v7 | P2h − arm, t4 |
|---|---|---|---|---|
| **P2h** | 86.16 / 77.74 | .9572 / .8483 | | |
| R2Bh | 85.76 / 77.59 | .9539 / .8492 | +.0033 [−.0059, +.0122] | −.0010 [−.0149, +.0137] |
| P2Bh | 86.23 / 77.29 | .9390 / .8368 | +.0181 [+.0108, +.0265] | +.0115 [+.0018, +.0224] |
| PRh | 86.63 / 77.13 | .9595 / .8393 | −.0023 [−.0086, +.0038] | +.0089 [+.0007, +.0178] |
| Dh | 86.08 / 77.90 | .9197 / .8577 | +.0375 [+.0268, +.0493] | −.0094 [−.0232, +.0058] |

- **Rule 1: not the contribution.** P2h beats no leave-one-out arm (largest gain +.0181 vs P2Bh, under 0.03). It is not
  worse than all three, but its v7 dev top1 (86.16) is below PRh's (86.63) and P2Bh's (86.23).
- **Rule 2:** removing the listing (P2h − R2Bh) is inside noise on both sets. The RCR loss (P2h − P2Bh) is the one
  component with a CI above 0 on both sets, +.018 / +.012, below the 0.03 bar. The MLP head (P2h − PRh) is inside noise
  on v7 and +.009 on t4; D10 does not adopt it (not worse, lower top1).
- **Rule 3:** top1 guard passes (Ah 85.21). Threshold transfer at risk 0.10 (t4 coverage / risk / t4-fit coverage):
  P2h.maxr .240 / .130 / .231; P2h.maxr_k .304 / .212 fails risk. Shared-option |Δ reward| on v7: max .081, mean .0041
  (PRh .084), not the 0 a pointwise arm should give up to rounding; RewardBench 2 not run yet.
- **Gates:** G0 passed (allwrong doc §Results). G4 (K = 1) is still queued; a G4 failure would void the pointwise
  numbers above, not reverse this verdict.
- The Lu & Wu benchmark still scores P2h with Ah, PRh and P2Bh (`../in-progress/external-luwu-benchmark.md`), to see
  whether its pointwise score carries a threshold across tasks; that does not reopen this doc's rule.
