# Question: is D's Decision Index +3.31 over v7-base the pref loss, or the training run?

Status: todo (see ../README.md). Commands below are kev-relative: run them from the repo root as ./kev.sh <command>.

No training: A seed 0 already exists; the run is the subset scoring only.

## Question

D seed 0 scores **36.91** on the shared 15,000-row Decision Index 0.3 subset, **+3.31** over the published base recipe
(`jaredpalmer/kev-4b@v7-base`, 33.60) and level with the board's r10 (36.69, board 37.95). But the comparison is
cross-model: v7-base is a different training run, not our arm A. Question: how much of the +3.31 survives when **A**
(the same Kev recipe, retrained on bf16 weights, our own run) is scored on the same rows?

## Why now / evidence

| Number | Value | Source |
|---|---|---|
| DI 0.3 subset index | r10 36.69 (board 37.95) / v7-base 33.60 / D seed 0 **36.91** | `done/preference-reward-model.md` §Decision Index 0.3 |
| D − v7-base per benchmark | 27 up / 9 down / 1 tied, mean +0.034, median +0.032, sign test p = 0.004 | same |
| where D loses | BANKING77 −0.067 (the task D also loses on v7 dev, −5.8) | same |
| dev and index disagree in sign | D trails A on v7 dev (−0.79, 3 seeds) yet beats v7-base on the index | §Readings |
| the pending measurement | "Pending to attribute the gain: A seed 0 on the same subset (`dindex.sh subset runs/pref/A/seed0`), paired with D." | same |
| noise scale to quote | "Sample noise is ~±1-2 index points [INFERENCE]" on the 1000-request sample, 15x smaller than this subset | `docs/studies/README.md` §Headline results |
| weak-proxy caveat | in-distribution validation top1 is a weak proxy for the index | `docs/studies/README.md` conclusion 1 |

This is the repo's one external number for the preference recipe, and the parent verdict (rule 5, **inconclusive** on dev
top1) does not rest on it — so attributing it decides whether the DI section is evidence for the pref loss or evidence
that retraining alone moves the index.

## Arms

No training: `runs/pref/A/seed0` (Kev format, pointer + CE, trained). D's subset run exists
(`runs/pref/D/seed0/dindex-0.3-sub15000`).

## Decision rule (pre-registered)

- Score A seed 0 on the same `$DI_DIR/subset-0.3-15000.jsonl.gz` through the same engine and pair it with D (the kit's
  own scorers, chance levels and areas, suite restricted to those rows).
- **Training-run effect** if A ≥ D − 1 (the ±1-2-point scale at its optimistic end): the retrained recipe reproduces D's
  index without the pref loss, so the +3.31 is not the loss.
- **Loss effect** if A ≤ v7-base + 1: the gain is not reproduced by the retrained baseline, so the pref loss / scalar
  head carries it.
- **Both** in between: report the split; no threshold read.
- Report beside it: the per-benchmark tally for A − D and A − v7-base, A's unsupported-row count (D 0; A is the Kev
  format, so expect the format's refusals), and the r10 reproduction numbers as the engine check (24 accuracy benchmarks
  z mean −0.03, sd 1.14, none beyond 3).
- The parent verdict and the pointwise refutation do not depend on this; a "training-run effect" reading must be stated
  in the parent doc's DI section as weakening its one external number.

## Cost [estimate]

A is a Kev-format checkpoint served without graphs: the two comparators cost 6,048 s (r10) and 7,367 s (v7-base) for
15,000 rows on one 3090, so **~1.7-2.0 h** for A [estimate from those walls], plus the one-time kit setup (7 GB
download, 17 GB disk) and `hf auth login` + HLE terms acceptance if not already done
(`kev/scripts/pref/README.md` §Decision Index 0.3).

## Commands

```bash
./kev.sh scripts/pref/dindex.sh setup                     # once: clone the kit, build the suite; needs hf auth + HLE terms
./kev.sh scripts/pref/dindex.sh subset runs/pref/A/seed0  # A on the same 15,000 rows
# A - D (pair prints B - A for `pair OUT_A OUT_B`):
./kev.sh scripts/pref/dindex_subset.py pair runs/pref/D/seed0/dindex-0.3-sub15000 runs/pref/A/seed0/dindex-0.3-sub15000
```

## Results

**Partial (2026-10-15; seeds 0-1 of 3, seed 2 running; `kev/runs/queue/dindex-A-seed{0,1}.log`,
`runs/pref/A/seed1/dindex-0.3-sub15000/index.json`).** Same 15,000 rows and engine as D and v7-base.

| Model | Index | Raw | Breadth | Unsupported |
|---|---:|---:|---:|---:|
| D seed 0 (chat, scalar head, pref loss) | 36.91 | | | 0 |
| A seed 0 (Kev format, pointer + CE) | 34.54 | 50.58 | 33.06 | 13 |
| A seed 1 | 33.05 | 49.87 | 31.51 | not printed |
| v7-base | 33.60 | | | 12 |

- **Rule read on seed 0: loss effect, by 0.06.** A 34.54 ≤ v7-base + 1 = 34.60, and below D − 1 = 35.91. Seed 1 (33.05)
  reads the same way with room. D − A = +2.37 (seed 0), +3.86 (seed 1, `dindex_subset.py pair`).
- The two A seeds differ by 1.49 points, the ±1-2 noise scale this rule assumed, so the seed-0 call sits inside one
  seed's spread. D has one seed; its own spread is unmeasured.
- "Loss effect" here means D's recipe as a whole: D differs from A in prompt format (chat), head (scalar) and loss
  (pref + BSR) at once, the same confound as C vs B (C0 not run). It is not the loss alone.
- A seed 1 − D per benchmark: 9 up, 26 down, 2 tied (9, 56). D's largest margins over A are the benchmarks where it
  beat v7-base: RAGTruth (59) −.219, BPoMP (20) −.192, Humicroedit (21) −.150; Arts & Human Taste −.071, Language
  −.065. A leads where D lost to v7-base: BANKING77 (4) +.067, BFCL (1) +.046. So D's per-benchmark gain pattern is not
  something retraining the Kev recipe reproduces.
- Not yet: seed 2; A − v7-base per benchmark (`pair runs/dindex/kev4b-v7base-sub15000 runs/pref/A/seed<S>/...`); A
  seed 0 − D per benchmark.
