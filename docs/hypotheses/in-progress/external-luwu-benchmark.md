# External abstention benchmark: Lu & Wu's Decision Model Benchmark

Status: in-progress (see ../README.md). Commands are kev-relative: run them from the repo root as ./kev.sh <command>.

## Question

On a public abstention benchmark with published baselines (Lu & Wu, arXiv 2610.03387, release 0.2.0,
`github.com/luckykevvv/Decision_Model_Benchmark`: TREC, DBpedia, Emotion, CLINC150; ground-truth omission, retrieval
miss, out-of-scope), does the all-wrong screen's selected arm keep a threshold across tasks where Kev-style softmax
models do not?

## Why now / evidence

- Their headline: a Jev threshold calibrated on DBpedia falsely rejects 69.3% of covered Emotion test inputs, and an
  Emotion threshold moved to DBpedia loses detection (paper abstract, quoted in the study notes summarised in
  `docs/data-eval.md` §2g row 7). The paper states it is descriptive and proposes no rejection method.
- Our internal version of the same test is D6 in `../in-progress/allwrong-abstention.md` (v7 → t4); this doc is its
  external replication against published Jev / Laya / Qwen numbers.
- The adapter is built: `kev/scripts/external/` (`luwu_run.py`, `luwu_policies.py`, `overlap.py`, README).

## Arms

No training. Score existing checkpoints (`kev/scripts/external/README.md` "Arms to run first"): `A` seed 0 (done), the
full-data `C`, `D`, `F` (done), then the half-data group `P2h`, `Ah`, `PRh`, `P2Bh` (launched 2026-10-15). P2h is not
the screen's selected arm (`../done/pointwise-rcr-mlp.md`: not the contribution at seed 0); it runs here to test whether
its pointwise scores carry a threshold across tasks better than its controls, with Ah as the matched half-data baseline.

## Decision rule (pre-registered)

1. **Contamination gate first.** `overlap.py` against decision-v7 train; any dataset over the 1% gate is dropped from
   every comparison (TREC and DBpedia are our trainable sources, so expect flags). Report what was dropped.
2. **Transfer.** On each source → target transfer row that survives (1), the selected arm's `maxr_k` (or `maxr`)
   threshold, fit by their 80-text equal-budget protocol, is **supported** if its target false rejection of covered
   inputs is ≤ 15% with detection ≥ the target-local threshold's detection − 0.10, on a majority of surviving rows;
   **refuted** if its false rejection exceeds Jev's published value on the same row on a majority of rows; otherwise
   inconclusive. `Ah.maxprob_T` is reported on the same rows as the in-house softmax reference.
3. Their scorer's `metrics.json` must validate the run (one prediction per request, offered-key probabilities sum to 1);
   a failed validation voids the arm.

## Cost [estimate]

`core` profile: 1,456 cases, 39,168 requests (`kev/scripts/external/README.md`); a few GPU-hours per arm on one 3090
[estimate, not measured]. Needs their package in its own venv (`~/dmb/venv`, made by `core.sh`).

## Commands

Training machine, from the repo root. `core.sh` does setup (clone v0.2.0, venv, verify, prepare, contamination gate),
then per arm our adapter, their scorer and our policies, skipping finished steps (manual steps: the scripts' README):

```bash
./kev.sh scripts/external/core.sh                                                       # setup + overlap gate (CPU)
GPU=<free GPU> nohup ./kev.sh scripts/external/core.sh A Ah > kev/runs/dmb-core.log 2>&1 &   # outputs runs/dmb/<arm>-s0-core*
```

## Results

**Contamination gate (2026-10-14, `kev/runs/dmb/overlap-core.json`, 12,576 decision-v7 train records, 1,456 core
cases).** TREC flagged and dropped from every comparison per rule 1; the other three pass with zero matches.

|Dataset|Cases|Matched|Rate|Calibration|Test|Flagged|
|---|--:|--:|--:|--:|--:|---|
|ag_news|200|0|0%|0/80|0/120|no|
|dbpedia|700|0|0%|0/280|0/420|no|
|emotion|286|0|0%|0/120|0/166|no|
|trec|270|23|8.5%|22/120|1/150|**yes**|

- All 23 TREC matches are exact (7 also by containment) against our `trec` source, and every case id equals the
  matched record's index (`trec:calibration:128` ↔ `trec/train/128`): their TREC calibration split is drawn from
  TREC train, of which decision-v7 holds ~1.3k questions, so 18% of it overlaps. The one test match (`trec:test:251`)
  is presumably a question duplicated across TREC train and test [inference].
- DBpedia and AG News are also decision-v7 sources (`dbpedia14`, `agnews`) but share no case text: same label sets and
  domain, different examples. Comparisons on them are in-domain for our arms, not contaminated; Emotion is the only
  out-of-domain core dataset left.
- Surviving transfer rows: those among DBpedia, Emotion and AG News. `luwu_policies.py --overlap` drops TREC by default
  (`core.sh` uses that); `--drop-overlapping-cases` (only the 23 ids) is a sensitivity view only, not the decision.

**A seed 0 (2026-10-14; `kev/runs/dmb/a-s0-core{-scored,-policies}`).** A is the Kev baseline (pointer head + CE,
`scripts/pref/arms.sh`), not a rewarder: its `r` is a softmax-trained logit, so `maxr`/`maxr_k` here test the Kev
logit, not the rewarder claim. It is the in-house counterpart of the paper's softmax models and the pipeline check; the
rewarder rows come from the selected arm (D9) and the trained full-data rewarder arms C/F.

- Rule 3 passes: 39,168 predictions for 39,168 requests, 0 interface failures, their scorer accepted the run.
- Our `luwu_policies.py` reproduces their scorer exactly where both compute the same thing: every transfer row's
  threshold, test D and test F for `maxprob_T`/`entropy_T`/`pnone_T` match their `negative_max`/`normalized_entropy`/
  `none_probability` (e.g. DBpedia k2 → Emotion k2 `negative_max`: -0.95970, D 0.837, F 0.578 in both). Their
  "frozen-threshold calibration" table differs on DBpedia/Emotion because it fits on the whole calibration split
  (280/120 texts), not the 80-text subset the transfer protocol uses.
- Cross-task transfer, k2 natural, the 6 surviving rows (D = detection on target test, F = false rejection, local =
  target-local detection; pass = F ≤ 0.15 and D ≥ local − 0.10):

|Source → target|`maxr_k` D / F (local)|`maxprob_T` D / F (local)|`pnone_T` D / F (local)|
|---|---|---|---|
|AG News → DBpedia|0.995 / 0.019 (0.976) pass|0.812 / 0.002 (0.962) fail|0.988 / 0.019 (0.981) pass|
|AG News → Emotion|0.765 / **0.361** (0.367) fail|0.614 / **0.337** (0.187) fail|0.771 / **0.398** (0.301) fail|
|DBpedia → AG News|**0.667** / 0.008 (0.883) fail|0.942 / 0.150 (0.683) edge|0.817 / 0.025 (0.917) edge|
|DBpedia → Emotion|0.566 / **0.223** (0.367) fail|0.837 / **0.578** (0.187) fail|0.669 / **0.277** (0.301) fail|
|Emotion → AG News|**0.475** / 0.000 (0.883) fail|**0.158** / 0.017 (0.683) fail|**0.250** / 0.000 (0.917) fail|
|Emotion → DBpedia|0.938 / 0.007 (0.976) pass|**0.326** / 0.000 (0.962) fail|0.912 / 0.007 (0.981) pass|
|**Passing rows**|**2/6**|0–1/6|2–3/6|

  "edge" = on the boundary at 3 decimals (0.150 vs ≤ 0.15; 0.817 vs 0.917 − 0.10). Had A been the selected arm,
  `maxr_k` would not be supported (2/6); refuted-vs-inconclusive needs Jev's false rejection per row, which
  `policies.md` now prints beside Jev's detection (`luwu_policies.py`, reference columns "D / F").
- A's logit does not carry one threshold across tasks: the 80-text `maxr_k` thresholds are 3.74 (AG News), 5.89
  (DBpedia), 7.17 (Emotion). A low source threshold over-rejects Emotion (F 0.22–0.36); Emotion's high one under-detects
  elsewhere (AG News D 0.475). `maxr_k` cuts the headline DBpedia → Emotion false rejection from 0.578 (`maxprob_T`;
  the paper reports 0.693 for Jev) to 0.223, still above the 0.15 bar.
- Emotion is A's weak task, not just a shifted scale: full-label accuracy 0.556 (AG News 0.930, DBpedia 0.989), and
  local detection is 0.10–0.37 for every readout. Their F counts every rejected covered input, including the ones A
  gets wrong, so part of Emotion's F is rejections that are arguably right.
- Within-task transfer (count k2 → k3/k4/k5/k13 and name natural → neutral, 13 rows): `maxr_k` passes 13/13,
  `maxprob_T` 10/13 (Emotion k2 → k4/k5 F 0.31/0.38; Emotion k5 F 0.157), `pnone_T` 10/13 (DBpedia k2 → k13 D 0.705;
  Emotion k2 → k4/k5 D ≤ 0.006).

**C, D, F seed 0 (2026-10-15; `kev/runs/dmb/{c,d,f}-s0-core{-scored,-policies}`).** Rule 3 passes for all three
(39,168 predictions each, 0 interface failures). Cross-task rows, k2 natural, `maxr_k` D / F; pass as above; target
local detection in parentheses only where it decides the row:

|Source → target|A|C|D|F|
|---|---|---|---|---|
|AG News → DBpedia|0.995 / 0.019 pass|pass|pass|pass|
|AG News → Emotion|0.765 / **0.361**|0.873 / **0.554**|0.873 / **0.560**|0.783 / **0.422**|
|DBpedia → AG News|**0.667** / 0.008|pass|pass|pass|
|DBpedia → Emotion|0.566 / **0.223**|0.789 / **0.428**|0.825 / **0.470**|0.819 / **0.488**|
|Emotion → AG News|**0.475** / 0.000|**0.450** / 0.008|**0.283**|**0.508**|
|Emotion → DBpedia|0.938 / 0.007 pass|**0.860** (0.998)|**0.798** (0.993)|**0.893** (0.995)|
|**Passing rows**|2/6|2/6|2/6|2/6|
|Rows with F above Jev's published `negative_max` F|3/6|3/6|3/6|5/6|

- No arm passes rule 2's bar: each passes 2 of 6 rows, and C/D/F pass a different two (AG News ↔ DBpedia) than A.
  Training for an absolute score (C, F) or a direct one (D) did not give a task-independent threshold: Emotion as target
  over-rejects (F 0.42-0.56, worse than A's 0.22-0.36), Emotion as source under-detects AG News.
- Other readouts, cross-task passing rows: `maxprob_T` A 1, C 1, D 2, F 0; `pnone_T` A 3, C 2, D 2, F 3.
- Within-task transfer (13 rows): `maxr_k` passes A 13, C 10, D 11, F 9; `maxprob_T` A 10, C 11, D 9, F 11; `pnone_T`
  A 10, C 8, D 9, F 8.
- Local (fit and apply on one task, 16 groups), mean detection `maxr` A .672, C .678, D .661, F .695 against `maxprob`
  A .543, C .421, D .461, F .433: the raw-score readout is the gain on every arm, as on the all-wrong pairs.

**P2h seed 0 (2026-10-15; `kev/runs/dmb/p2h-s0-core{-scored,-policies}`).** Rule 3 passes (39,168 predictions, 0
interface failures; their scorer accepted the run). Cross-task rows, k2 natural, `maxr_k` D / F, as above; Jev's
published `negative_max` F on the same row in the last column:

|Source → target|P2h `maxr_k` D / F (local)|Pass|Jev F|
|---|---|---|---|
|AG News → DBpedia|0.990 / 0.033 (0.998)|pass|0.002|
|AG News → Emotion|0.837 / **0.476** (0.271)|fail|0.120|
|DBpedia → AG News|0.908 / 0.025 (0.883)|pass|0.142|
|DBpedia → Emotion|0.855 / **0.554** (0.271)|fail|0.434|
|Emotion → AG News|**0.442** / 0.017 (0.883)|fail|0.008|
|Emotion → DBpedia|**0.893** / 0.002 (0.998)|fail (by .005)|0.002|
|**Passing rows**|**2/6**| | |
|Rows with F above Jev's|4/6| | |

- Same pattern as C/D/F: passes AG News ↔ DBpedia, fails every Emotion row. The 80-text `maxr_k` thresholds are 2.42
  (AG News), 1.54 (DBpedia), 7.23 (Emotion): the pointwise RCR reward does not put Emotion on the other tasks' scale
  either. Emotion as target over-rejects (F 0.48 / 0.55, between A's 0.22-0.36 and C/D's 0.43-0.56); Emotion as source
  under-detects AG News (0.442 vs local 0.883). F is above Jev's on 4/6 rows, a majority, which would meet rule 2's
  refuted clause had P2h been the selected arm (A 3/6, F 5/6).
- Other readouts, cross-task passing rows: `maxprob_T` 1, `pnone_T` 2.
- Within-task transfer (13 rows): `maxr_k` 11 (fails Emotion k2 → k5 D 0.042 vs local 0.301; Emotion k4 natural →
  neutral D 0.217 vs 0.325), `maxprob_T` 11 (Emotion k2 → k4/k5 F 0.355/0.458), `pnone_T` 10 (DBpedia k2 → k13 D 0.874
  vs 0.990; Emotion k2 → k4/k5 D ≤ 0.024).
- Local (16 groups), mean detection `maxr` .686 vs `maxprob` .519: the raw-score gain holds (+.167; A +.129, F
  +.262).
- Native with-NONE decisions (their scorer): P2h picks NONE on 26-39% of answerable Emotion test texts (≤ 3.3% on AG
  News / DBpedia, 0-2% TREC), and Emotion full-set accuracy drops from 0.517 (`full_plain`) to 0.437-0.462
  (`full_with_none`): offering NONE costs Emotion accuracy rather than catching its errors.

Ah, PRh, P2Bh still running (`kev/runs/dmb-core-h.log`). Question for P2h vs Ah (open until Ah lands): on the
cross-task rows, is P2h's `maxr_k` false rejection at most Ah's while its detection is higher?
