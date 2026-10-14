# External abstention benchmark: Lu & Wu's Decision Model Benchmark

Status: todo (see ../README.md). Commands are kev-relative: run them from the repo root as ./kev.sh <command>.

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

No training. Score existing checkpoints, in this order (`kev/scripts/external/README.md` "Arms to run first"): `Ah`
seed 0, `A` seed 0, then the arm the all-wrong screen selects (D9-D10 winner; P2h if
`../todo/pointwise-rcr-mlp.md` supports it).

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

Not run.
