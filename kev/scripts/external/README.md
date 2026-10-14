# Running Lu & Wu's Decision Model Benchmark on our checkpoints

Three scripts bridge our checkpoints to the frozen benchmark of Lu & Wu (arXiv 2610.03387,
`github.com/luckykevvv/Decision_Model_Benchmark`, release 0.2.0). Nothing here touches the benchmark's code:
`luwu_run.py` writes a run directory their `decision-benchmark score` accepts, `luwu_policies.py` reimplements their
threshold protocol locally (so the arms can be compared before/independently of their scorer), and `overlap.py` is the
contamination gate between their texts and our training data.

Their protocol, as released: a threshold is fit on the **80 fixed texts per task** of
`resources/subsets.json:equal_budget_calibration` (text-equal weights), maximising detection subject to empirical false
rejection <= 5% under the strict rule `score > threshold`; transfer applies the source task's threshold frozen to the
target task's test split; intervals are 2000 text-cluster bootstrap draws at their seed 20261005. Our `policies.json`
carries their released numbers from `resources/reference/core-{jev,laya,qwen}/metrics.json` beside ours.

## Environment

Their package runs in its own venv; our side runs in our own venv as every other `kev/scripts` entry point does.
`core.sh` runs every step below in order, skipping finished ones (env `GPU`, `SEED`, `DMB_HOME`, default `~/dmb`):

```bash
./kev.sh scripts/external/core.sh                                     # setup: clone, venv, verify, prepare, overlap
GPU=0 nohup ./kev.sh scripts/external/core.sh A Ah > kev/runs/dmb-core.log 2>&1 &   # + adapter, scorer, policies per arm
```

The manual steps, with `core.sh`'s paths:

```bash
git clone --depth 1 --branch v0.2.0 https://github.com/luckykevvv/Decision_Model_Benchmark ~/dmb/Decision_Model_Benchmark
python3 -m venv ~/dmb/venv
~/dmb/venv/bin/pip install -e "$HOME/dmb/Decision_Model_Benchmark[data]"      # [data] = dataset download helpers
~/dmb/venv/bin/decision-benchmark --version                                  # 0.2.0
```

`luwu_run.py` and `overlap.py` never import their package; only `decision-benchmark prepare` and
`decision-benchmark score` do.

## 1. Frozen inputs (their package, once per profile)

```bash
~/dmb/venv/bin/decision-benchmark prepare --profile core --with-text \
    --output-dir runs/dmb/core-inputs --cache-dir ~/dmb/cache/datasets      # from kev/
```

`--with-text` is required: without it the cases carry `case_representation: 'text-index-v1'` and both our scripts refuse
them (their own `run_model` refuses them too). `--profile intent` instead of `core` for the CLINC150 scope-gate data.

## 2. Our adapter: score a checkpoint (our venv)

```bash
./kev.sh scripts/external/luwu_run.py \
    --inputs runs/dmb/core-inputs --ckpt runs/pref/Ah/seed0 --out runs/dmb/ah-s0-core
```

It writes `cases.jsonl` / `requests.jsonl` byte-identical to the prepared ones, `manifest.json`
(`protocol: paper1-confirmation-v2` plus a `kev` block: checkpoint, temperature, G fit, context, load options,
environment, script sha256), `predictions.jsonl` with exactly one row per frozen request (`status: ok` with `choice` and
the offered-key `probabilities` map = `softmax(r / T)`, or `status: interface_failure` with null choice/probabilities),
and the `raw_rewards.jsonl` sidecar (`request_id` -> per-key raw reward `r`) that `luwu_policies.py` scores `maxr`,
`maxr_k` and `noisy_or` from.

`T` is the arm's fitted G temperature (`RUN/eval/decision-v7-development/reward_report.json`, `G.T`), the same fit
`allwrong_report.py:load_T` uses; `--temperature` overrides it and the run records which was used. Requests of one case
share its state and are scored in chunks (`--budget-tokens`, a chars/4 estimate); a chunk that overflows the context is
bisected and a single request that still overflows becomes a failure row, never a truncation. `--context serving` admits
requests under the serving limits instead of the training ones. Keep `KEV_TEMPERATURE` unset: it replaces the head
temperature and moves `r` itself (the script warns). `--limit N` is a smoke run only (the run dir is not scorable: their
validator wants complete pairs and every case's full-set baselines). Re-running into an existing `--out` is refused, as
their `new_output` does.

Cost: `core` is 1456 cases and 39168 requests; the requests of one case differ only in instruction/criteria lengths, so
tune `--budget-tokens` until a case fits one or two chunks (the log line reports chunks per case).

## 3. Their scorer (their venv)

```bash
./kev.sh ~/dmb/venv/bin/decision-benchmark score --profile core \
    --run-dir runs/dmb/ah-s0-core --output-dir runs/dmb/ah-s0-core-scored
```

It validates the run (`protocol`, both file digests, exactly one prediction per request, offered-key probabilities
summing to one) and writes `metrics.json` with their own `negative_max` / `normalized_entropy` / `none_probability`
readouts off our probability map.

## 4. Our policies, side by side with their released metrics (our venv)

```bash
./kev.sh scripts/external/luwu_policies.py \
    --run runs/dmb/ah-s0-core --overlap runs/dmb/overlap-core.json \
    --subsets ~/dmb/Decision_Model_Benchmark/decision_benchmark/resources/subsets.json \
    --reference ~/dmb/Decision_Model_Benchmark/decision_benchmark/resources \
    --out runs/dmb/ah-s0-core-policies
```

Writes `policies.json` (every group's `calibration_split` and 80-text `equal_budget` thresholds plus their rate families,
all transfer rows with the source threshold frozen and the target's own local threshold for reference, the trimmed
reference metrics, and the contamination block), `policies.md` (the tables, our numbers beside theirs) and
`scores.jsonl` (per-request readouts). `--replicates` defaults to 2000 and `--seed` to their 20261005.

Readouts, and the sign meaning "reject": `maxprob` (no temperature), `maxprob_T`, `entropy_T`, `maxr`, `maxr_k`
(`max r - log(n - 1)`, `n` = options actually scored in that row), `noisy_or` (`1 - prod(1 - sigmoid(r))` in log space),
and `pnone` / `pnone_T` from the with-NONE calls. Our `maxprob_T` / `entropy_T` / `pnone_T` are their `negative_max` /
`normalized_entropy` / `none_probability`, so those are the three readouts their reference columns are printed for.

## 5. Contamination gate (our venv, before trusting any comparison)

```bash
./kev.sh scripts/external/overlap.py \
    --cases runs/dmb/core-inputs --suite evals/v7/decision-v7 --split train \
    --out runs/dmb/overlap-core.json
```

Unit-level check: each frozen case text against our training records' rendered state, state string leaves, instructions
and options — exact normalised match, and containment both ways (the contained side needs `--min-chars`, default 32).
Per dataset it reports the case count, matched cases, rate, the matching unit's source/kind/mode and examples; a dataset
whose rate exceeds `--gate` (default 1%) is `flagged`. `luwu_policies.py --overlap FILE` then drops a flagged dataset
from the comparisons entirely, or, with `--drop-overlapping-cases`, only the flagged case ids.

## Arms to run first

1. `runs/pref/Ah/seed0` — the arm the abstention plan is built around (Kev pointer, Hh prompt), pointwise, the D1
   reference the readouts `maxprob` / `maxr` / `lse` / `pnone` were defined against.
2. `runs/pref/A/seed0` — A seed 0, the second reference point (pointer head + CE).
3. The arm D9 selects (PRh if it beats Hh and clears the top1 guard; `docs/hypotheses/in-progress/allwrong-abstention.md`, D9), once
   its `seed0` sweep is scored.

One `luwu_run.py` invocation per arm; each arm's `T` comes from its own `reward_report.json`, so a single `--temperature`
across arms would make the probability maps incomparable. `maxr`, `maxr_k` and `noisy_or` are the calibrated confidence
only for a scalar (sigmoid) arm; for the pointer arms (A/Ah) they are computed from the same logits and reported, but
their sigmoid is not a calibrated probability (the plan's caveat), so compare those two arms' `maxprob`/`maxprob_T`
first.

## Assumptions and open ends

* Our runtime renders an option as `name: definition` (`kev.api.option_text`) and passes their `instruction` verbatim; the
  option order and the NONE wording are exactly the frozen `criteria`.
* `manifest.model` becomes `{"provider": "kev-local", "run": ...}`; their `analyze.py` keys one informational note off
  `provider == 'TypeSafe'`, so their tables will carry the Laya sentence for our runs.
* Their `qwen` reference has no core transfer rows and no test numbers (its groups are ineligible), so their columns are
  `n/a` for it — expected, not a bug in the run.
