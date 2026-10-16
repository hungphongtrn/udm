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
- Our internal version of the same test is D6 in `../done/allwrong-abstention.md` (v7 → t4); this doc is its
  external replication against published Jev / Laya / Qwen numbers.
- The adapter is built: `kev/scripts/external/` (`luwu_run.py`, `luwu_policies.py`, `overlap.py`, README).

## Arms

No training. Score existing checkpoints (`kev/scripts/external/README.md` "Arms to run first"): `A` seed 0 (done), the
full-data `C`, `D`, `F` (done), then the half-data group `P2h`, `Ah`, `PRh`, `P2Bh` (all done 2026-10-15). P2h is not
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

**PRh seed 0 (2026-10-15; `kev/runs/dmb/prh-s0-core-policies`).** The pasted policies report carried no arm label; it
is PRh by its temperature: the AG News k2 thresholds `maxprob` 0.930 and `maxprob_T` 0.709 give T = 2.905, PRh's
fitted 2.908 (Ah's is 1.7-1.85). Rule 3 (scorer validation) was not in the paste. Same rows and columns as P2h:

|Source → target|PRh `maxr_k` D / F (local)|Pass|Jev F|
|---|---|---|---|
|AG News → DBpedia|0.995 / 0.029 (0.993)|pass|0.002|
|AG News → Emotion|0.729 / **0.331** (0.325)|fail|0.120|
|DBpedia → AG News|0.875 / 0.025 (0.892)|pass|0.142|
|DBpedia → Emotion|0.693 / **0.301** (0.325)|fail|0.434|
|Emotion → AG News|**0.683** / 0.025 (0.892)|fail|0.008|
|Emotion → DBpedia|0.971 / 0.002 (0.993)|pass|0.002|
|**Passing rows**|**3/6**| | |
|Rows with F above Jev's|3/6| | |

- 3/6 is not a majority, so rule 2 would be inconclusive with PRh as the selected arm. F is above Jev's on 3/6 rows,
  also not a majority. PRh is the first arm to pass both A's Emotion → DBpedia row and C/D/F's AG News ↔ DBpedia pair.
- Against P2h (same half data, linear vs MLP pointwise head): PRh's Emotion-target false rejection is lower (0.331 /
  0.301 vs 0.476 / 0.554) and Emotion-source detection higher (0.683 / 0.971 vs 0.442 / 0.893); the other rows match
  within 0.005. The 80-text `maxr_k` thresholds are 2.57 (AG News), 2.87 (DBpedia), 5.05 (Emotion): Emotion still sits
  off-scale, but at 2x the other tasks rather than P2h's 3-5x.
- Other readouts, cross-task passing rows: `maxprob_T` 1 (DBpedia → AG News), `pnone_T` 2 (AG News ↔ DBpedia).
- Within-task transfer (13 rows): `maxr_k` 11 (fails Emotion k2 → k5 D 0.181 vs local 0.331; Emotion k5 natural →
  neutral F 0.169), `maxprob_T` 10 (Emotion k2 → k4/k5 F 0.373/0.434; Emotion k2 natural → neutral D 0.102 vs 0.223),
  `pnone_T` 9 (DBpedia k2 → k13 D 0.600 vs 0.986; Emotion k2 → k4/k5 D ≤ 0.024; Emotion k4 natural → neutral F 0.151).
- Local (16 groups), mean detection `maxr` .701 vs `maxprob` .547 (+.154).

**Ah seed 0 (2026-10-15; `kev/runs/dmb/ah-s0-core-policies`).** Identified by the AG News k2 `maxprob_T` threshold
−0.808, which the user's `grep` of that file printed; rule 3 was not in the paste. Ah is the half-data Kev pointer arm,
so `maxr_k` tests its logit, as for A. Same rows and columns as P2h, with `Ah.maxprob_T` (rule 2's in-house softmax
reference) beside it:

|Source → target|Ah `maxr_k` D / F (local)|Pass|Ah `maxprob_T` D / F (local)|Pass|Jev F|
|---|---|---|---|---|---|
|AG News → DBpedia|1.000 / 0.024 (0.995)|pass|**0.571** / 0.000 (0.967)|fail|0.002|
|AG News → Emotion|0.813 / **0.422** (0.416)|fail|0.458 / **0.277** (0.253)|fail|0.120|
|DBpedia → AG News|0.858 / 0.025 (0.917)|pass|0.917 / **0.242** (0.375)|fail|0.142|
|DBpedia → Emotion|0.693 / **0.313** (0.416)|fail|0.952 / **0.735** (0.253)|fail|0.434|
|Emotion → AG News|**0.733** / 0.008 (0.917)|fail|**0.175** / 0.025 (0.375)|fail|0.008|
|Emotion → DBpedia|0.981 / 0.010 (0.995)|pass|**0.283** / 0.000 (0.967)|fail|0.002|
|**Passing rows**|**3/6**| |**0/6**| | |
|Rows with F above Jev's|3/6 (Emotion → AG News ties at 0.008)| |4/6| | |

- Ah's logit passes the same 3/6 rows as PRh, and its softmax confidence passes none. `maxprob_T` repeats the paper's
  failure: DBpedia → Emotion F 0.735 (Jev 0.693 in the paper, 0.434 in their published row). The 80-text `maxr_k`
  thresholds are 1.86 (AG News), 3.26 (DBpedia), 5.81 (Emotion): Emotion's is 3x AG News's, so Emotion as target
  still over-rejects (F 0.42 / 0.31).
- **Open question answered: no.** On the cross-task `maxr_k` rows neither pointwise arm has false rejection ≤ Ah's
  together with higher detection, except DBpedia → AG News (PRh 0.875 / 0.025, P2h 0.908 / 0.025 vs Ah 0.858 / 0.025).
  PRh's lower Emotion-target F (0.331 / 0.301 vs 0.422 / 0.313) comes with lower detection (0.729 / 0.693 vs
  0.813 / 0.693), and Emotion → AG News detection is lower (0.683 vs 0.733). P2h is above Ah's F on both Emotion-target
  rows (0.476 / 0.554). At half data, the pointwise reward does not transfer thresholds better than the Kev logit.
- Other readouts, cross-task passing rows: `pnone_T` 3 (AG News ↔ DBpedia, Emotion → DBpedia).
- Within-task transfer (13 rows): `maxr_k` 13/13 (as A), `maxprob_T` 8 (DBpedia k2 → k13 F 0.150 is on the edge;
  fails Emotion k2 → k4/k5 F 0.367/0.452, Emotion k2 natural → neutral F 0.151, Emotion k4/k5 natural → neutral D
  0.066/0.084 vs local 0.193/0.205), `pnone_T` 11 (Emotion k2 → k4/k5 D 0.042/0.012).
- Local (16 groups), mean detection `maxr` .702 vs `maxprob` .495 (+.206; PRh +.154, P2h +.167).

**P2Bh seed 0 (2026-10-15; `kev/runs/dmb/p2bh-s0-core{-scored,-policies}`).** Rule 3 passes (39,168 predictions for
39,168 requests, 0 interface failures, scorer accepted); the same holds for the Ah and PRh runs in the archive, so their
"rule 3 not in the paste" above is now confirmed. P2Bh is Hh with the MLP head (Bradley-Terry + BSR on the pointwise
prompt, no RCR). Same rows and columns as P2h:

|Source → target|P2Bh `maxr_k` D / F (local)|Pass|Jev F|
|---|---|---|---|
|AG News → DBpedia|0.974 / 0.021 (0.969)|pass|0.002|
|AG News → Emotion|0.747 / **0.355** (0.307)|fail|0.120|
|DBpedia → AG News|0.833 / 0.025 (0.875)|pass|0.142|
|DBpedia → Emotion|0.705 / **0.301** (0.307)|fail|0.434|
|Emotion → AG News|**0.592** / 0.017 (0.875)|fail|0.008|
|Emotion → DBpedia|0.886 / 0.000 (0.969)|pass|0.002|
|**Passing rows**|**3/6**| | |
|Rows with F above Jev's|3/6| | |

- The same three rows as PRh and Ah pass; Emotion as target over-rejects and Emotion as source under-detects AG News
  (0.592; PRh 0.683, Ah 0.733, P2h 0.442). The 80-text `maxr_k` thresholds are 0.088 (AG News), 0.487 (DBpedia),
  3.321 (Emotion): Emotion sits 7-38x off the other tasks' scale, further than any other arm.
- Against P2h (RCR vs BT + BSR, both MLP pointwise): P2Bh's Emotion-target F is lower (0.355 / 0.301 vs 0.476 / 0.554)
  and Emotion → DBpedia passes (0.886 vs P2h's 0.893 failing by .005, on a lower local 0.969), and Emotion → AG News
  detection is higher (0.592 vs 0.442; both fail). Neither loss puts Emotion on scale.
- Other readouts, cross-task passing rows: `maxprob_T` 1, `pnone_T` 3.
- Within-task transfer (13 rows): `maxr_k` 10 (fails Emotion k2 → k4/k5 D 0.145/0.114 vs local 0.247/0.380; Emotion
  k5 natural → neutral F 0.157), `maxprob_T` 10 (DBpedia k2 → k13 F 0.302; Emotion k2 → k4/k5 F 0.277/0.386),
  `pnone_T` 10 (DBpedia k2 → k13 D 0.564 vs 0.983; Emotion k2 → k4/k5 D ≤ 0.018).
- Local (16 groups), mean detection `maxr` .686 vs `maxprob` .518 (+.168).
