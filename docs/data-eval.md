# Data and evaluation: source of truth

Single source of truth for (1) the training-data budget and (2) the controlled evaluation set. Everything here is
gathered from the repo; every number carries its source path inline. Unknowns are marked **not recorded** — nothing is
invented.

Conventions:
- Commands are written to run from the repo root through the wrapper `./kev.sh` (it `cd`s into `kev/`; a `*.py` first
  argument runs with `kev/.venv/bin/python`, anything else is exec'd as-is, and all paths stay kev-relative). Spec:
  parent `README.md` / `kev.sh`.
- Training machine: 3× RTX 3090 24 GB, user `jason`, checkout `/home/jason/udm/kev` holds `runs/`. Three experiments run
  at once, one per GPU.
- Code lives in `kev/` (vendored Kev fork; provenance `kev/UPSTREAM.md`). Docs live at repo root.

Hypothesis docs (linked at their post-restructure paths):
- `docs/hypotheses/done/preference-reward-model.md` — arms A–G, D/E/F/H results, DI subset.
- `docs/hypotheses/done/pointwise-chat-input.md` — arms H/HP/HL/HB/HBL, RewardBench 2, position effect.
- `docs/hypotheses/in-progress/allwrong-abstention.md` — all-wrong abstention, PF-RCR wave.
- `docs/studies/data.md` — dataset build/push, row counts, context/truncation, epoch arithmetic.

---

## 1. Training data budget

The suite (`--suite evals/v7/decision-v7`) supplies the training partition for every arm. The train partition is
**12,576 records / 15,576 questions** (`kev/evals/v7/decision-v7/manifest.json` `files.train.jsonl`: `records` 12576,
`questions` 15576, `sha256` `7ed5254b…`).

| Arm(s) | Training source | Records / epoch | Epochs | Requested records | Optimizer steps @ eff. batch 8 | Tokens | Source |
|---|---|---:|---:|---:|---:|---|---|
| A, B, C0, C, D, E, H, HL, HB, HBL | decision-v7 train | 12,576 | 2 | 25,152 | 3,144 | **not recorded** | `kev/evals/v7/decision-v7/manifest.json`; `kev/scripts/pref/arms.sh` (`--epochs 2`, `COMMON`) |
| F | decision-v7 train | 12,576 | 1 (`F_EPOCHS` default 1) | 12,576 | 1,572 | **not recorded** | `kev/scripts/pref/arms.sh` (arm `F`: `--epochs ${F_EPOCHS:-1}`) |
| HP | decision-v7 train + HelpSteer3 pairs | 12,576 + 12,000 = 24,576 | 2 | 49,152 | 6,144 | **not recorded** | `kev/scripts/pref/arms.sh` (arm `HP`: `--data $PREF_DATA --replay ${PREF_REPLAY:-12576}`); `docs/hypotheses/done/pointwise-chat-input.md` (`--cap 12000`) |
| h-screen arms (Ah, Ch, Fh, HBh, Rh, R2h, Sh, STh, RYh, Dh, Hh, PBh, PRh, PUh, P2h, PLh) | decision-v7 train, half | 6,288 | 2 (Fh, STh: 1) | 12,576 (Fh, STh: 6,288) | 1,572 (Fh, STh: 786) | **not recorded** | `kev/scripts/pref/arms.sh` (`_H="--train_frac ${HALF_FRAC:-0.5}"`); `docs/hypotheses/in-progress/allwrong-abstention.md` |

- Optimizer-step counts are arithmetic: `records × epochs ÷ 8`. Effective batch 8 records per optimizer step for every
  arm (`kev/scripts/pref/arms.sh` header; `docs/hypotheses/done/preference-reward-model.md`: "batch 4 x accum 2" Kev
  format, "batch 2 × accum 4" chat arms, `BATCH_FALLBACK=1` halves to 2×4 / 1×8 — the effective batch is 8 either way).
- Half-data screen (`HALF_FRAC` default 0.5): a per-source subsample of every training record keyed on `--seed`, so
  every h arm of a seed — and a warm start from one — sees the same records (`kev/kev/train.py` `--train_frac`;
  `kev/scripts/pref/arms.sh` comment; `docs/hypotheses/in-progress/allwrong-abstention.md`).
- Training tokens are not recorded anywhere in the repo (**not recorded**). RewardBench 2 records per-run input-token
  totals in its JSON (`kev/scripts/pref/rewardbench2.py` — "input-token totals").

### Where the decision-v7 train partition comes from

- **Suite**: `evals/v7/decision-v7`. Its train partition is **not in git** (all partitions > 10 MiB are gitignored,
  `kev/kev/suite.py` `GIT_LIMIT`); `load_split` fetches it from the Hub dataset **`jaredpalmer/kev-suites` @ revision
  `cc4bac803e73112689ec327ffa481c519cbc7a05`** and verifies it against the manifest sha256
  (`kev/kev/suite.py` `SUITES_DATASET`, `SUITES_REVISION`, `fetch_partition`, `load_split`). Only `development`,
  `calibration`, `test` and `manifest.json` ship in git for v7.
- **Source datasets + revisions**: pinned in `manifest.dataset_revisions` (`kev/evals/v7/decision-v7/manifest.json`):
  `CogComp/trec` `65752bf5…`, `legacy-datasets/banking77` `f5412156…`, `stanfordnlp/imdb` `e6281661…`,
  `fancyzhx/ag_news` `eb185aad…`, `SetFit/amazon_reviews_multi_en` `ec73b665…`, `fancyzhx/dbpedia_14` `9abd46cf…`,
  `google/boolq` `35b264d0…`, `SetFit/sst5` `e51bdcd8…`, `nyu-mll/multi_nli` `da70db2a…`,
  `Yelp/yelp_review_full` `c1f9ee93…`. Trainable sources: agnews, amazon, banking77, boolq, dbpedia14, imdb, mnli,
  sst5, trec, `legacy_policy`, `compositional`; eval-only sources: mmlu, emotion, tweet_offensive, qnli, paws, sciq,
  contrastive, `legacy_holdout`, `composition_holdout` (`manifest.trainable_sources` / `eval_only_sources`;
  validated by `kev/kev/suite.py:validate_training`).
- **Admission rules + headroom**: a clean record is admitted only if it fits the common context with
  `ADMISSION_BRANCH_HEADROOM = 64` branch tokens to spare (`fits(rec, *tokenizers, max_branch=MAX_BRANCH - 64)`,
  `kev/kev/suite.py` `select_unique`, `ADMISSION_BRANCH_HEADROOM`), and normalized exact-state deduplication is applied
  across partitions (`select_unique`, `normalise_text`/`text_digest`; `manifest.selection` in v2 freezes). The
  same rule is what the pointwise doc describes as "the rule every frozen suite was admitted under"
  (`docs/hypotheses/done/pointwise-chat-input.md`). v7's manifest context is `max_state 384`, `max_branch 1024`,
  `max_packed 2048`, `truncate false` (`kev/evals/v7/decision-v7/manifest.json` → `context`;
  `kev/kev/model.py` `MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048`). v7 is **version 3** and carries a
  `protocol` block (train/transfer/locked shapes, render styles, synthetic records per arm) **but no `selection` /
  `admission_branch_headroom` field**; the exact builder that froze v7 is **not recorded** in this repo (the in-repo
  `kev.suite.freeze` emits version 2, `kev/kev/suite.py`).
- **v7 lineage**: `manifest.parent_files` names decision-v2, transfer-v2 and public-pool-v4, and
  `protocol.inherited_eval = "evals/v4/decision-v4"` (`kev/evals/v7/decision-v7/manifest.json`). No build doc in the
  repo describes the v2→v7 assembly (**not recorded**).
- **Chat-row slack**: a chat arm's row (prefix + branch) may exceed the Kev row limit by
  `CHAT_ROW_SLACK = 128` tokens; the trainer re-filters `--data` with the same norm
  (`kev/kev/model.py` `CHAT_ROW_SLACK`; `docs/hypotheses/done/pointwise-chat-input.md`;
  `kev/scripts/pref/README.md` "Pointwise chat arms").

### HelpSteer3 conversion (arm HP)

`kev/scripts/pref/convert_pref_data.py` converts `nvidia/HelpSteer3`, split `preference` (**38,459 rows**, CC-BY-4.0)
into Kev labelled requests + a manifest, doing seeded key assignment, exact-match decontamination against RewardBench 2
prompts, and `--cap 12000` (the size of decision-v7's training partition) (`docs/hypotheses/done/pointwise-chat-input.md`
"The preference data (arm HP)"; `kev/scripts/pref/README.md`).

| Fact | Value | Source |
|---|---|---|
| Dataset / split / rows | `nvidia/HelpSteer3`, `preference`, 38,459 | `docs/hypotheses/done/pointwise-chat-input.md` |
| Revision | resolved at conversion time and recorded in the run manifest | `docs/hypotheses/done/pointwise-chat-input.md` |
| Dropped: `overall_preference == 0` ("about the same") | 2,160 rows | `docs/hypotheses/done/pointwise-chat-input.md` |
| Dropped: empty responses | dropped (count **not recorded** in git) | `docs/hypotheses/done/pointwise-chat-input.md` |
| Non-tie pairs | 36,299 (38,459 − 2,160) | arithmetic from the two numbers above |
| Admitted pool / per-limit drops | reported by the manifest; **not recorded** in git (`runs/pref/data/*.manifest.json` is gitignored) | `kev/scripts/pref/README.md`; `docs/hypotheses/done/pointwise-chat-input.md` |
| Fit estimate | ~13,000 of 36,299 fit the admission limits [INFERENCE] | `docs/hypotheses/done/pointwise-chat-input.md` |
| `--cap` / `--replay` for HP | 12,000 pairs + 12,576 suite records = 24,576 records/epoch | `docs/hypotheses/done/pointwise-chat-input.md`; `kev/scripts/pref/arms.sh` |
| Decontamination | exact normalized match against RewardBench 2 prompts | `docs/hypotheses/done/pointwise-chat-input.md` |

### Base model, LoRA and optimizer

| Item | Value | Source |
|---|---|---|
| Base | `Qwen/Qwen3.5-4B-Base` rev `1001bb4d826a52d1f399e183466143f4da7b741b` | `kev/scripts/pref/arms.sh` (`COMMON`); `docs/hypotheses/done/preference-reward-model.md` |
| LoRA | r 16, alpha 32, dropout 0.05 | `docs/hypotheses/done/preference-reward-model.md` "Fixed across arms" |
| Learning rate / schedule | 5e-5, OneCycleLR; stage-2 arms rebuild a fresh OneCycleLR over their own steps | `kev/scripts/pref/arms.sh`; `docs/hypotheses/done/preference-reward-model.md` |
| Optimizer | AdamW, weight decay 0.01, gradient clipping | `docs/hypotheses/done/preference-reward-model.md` |
| Precision | bf16 autocast + `--weights_dtype bf16` + gradient checkpointing | `kev/scripts/pref/arms.sh` (`COMMON`); `docs/hypotheses/done/preference-reward-model.md` |
| Regularization | `--p_none_pair 0.25`; option permutation re-drawn each epoch | `kev/scripts/pref/arms.sh`; `docs/hypotheses/done/preference-reward-model.md` |
| Checkpoint selection | final checkpoint only, no selection; seeds 0 first, then 1, 2 | `docs/hypotheses/done/preference-reward-model.md` |
| Suite admission bases (eval suites) | `Qwen/Qwen3-0.6B-Base` `da87bfb6…`, `Qwen/Qwen3-4B-Base` `906bfd4b…` (the bases the *eval suites* were admitted under) | `kev/evals/v7/decision-v7/manifest.json` `base_revisions` |

### Budget rule

Only the arm's **declared factor** may change between arms; everything else is fixed.

- May change (declared per arm in `kev/scripts/pref/arms.sh`, which is the source of truth if it differs from a doc
  table): head / input format / loss / regularizer (`--head`, `--input_format`, `--loss`, `--reg`, `--reg_w`, `--ce_w`,
  `--bce_w`, `--rcr_*`, `--head_init`), the `--data` file + `--replay` (HP only), `--init_from` for stage-2 arms, and
  `--train_frac` for the h-screen (`k` arms). Each doc table states that difference explicitly, e.g. "H and D differ in
  exactly one flag" (`docs/hypotheses/done/pointwise-chat-input.md`).
- Never changes: base model + revision, LoRA r/alpha/dropout, lr/schedule/optimizer, epochs per arm (except a stage-2
  arm's own 1 epoch, declared), effective batch 8, `p_none_pair 0.25`, the training suite and its train partition, the
  splits, and (for the `--reg`/`--row_budget` per-pass arms D, E, H, HP, and the Dh/Hh pair) the microbatch split — the
  README warns to keep those identical across seeds or not at all (`kev/scripts/pref/README.md` Env vars; arms.sh header).
- Comparability caveat recorded upstream: `--weights_dtype bf16` deviates from the released recipe, so arm A is "Kev
  recipe on bf16 weights", not the published checkpoint (`docs/hypotheses/done/preference-reward-model.md`).

---

## 2. Evaluation set (controlled)

Split policy for every suite: **development** = decisions; **calibration** = fit G (temperature + Platt) only, never a
verdict; **test** = locked, read once for the final verdict, never for selection/search
(`kev/scripts/pref/eval_arm.sh`; `kev/kev/suite.py` `load_split` refuses test without `--allow-test`).

### a. Kev eval data

Suites: `evals/v7/decision-v7` (primary) and `evals/v4/transfer-v4` (eval-only, out-of-source).

| Suite / split | What it measures | n | Counts source |
|---|---|---|---|
| decision-v7 development | in-distribution decision accuracy / calibration | clean **1,264 questions** | `docs/hypotheses/done/preference-reward-model.md`; manifest `records` 1204 / `questions` 1468 |
| decision-v7 calibration | G fit only (T + Platt) | **1,148 questions** | `kev/evals/v7/decision-v7/manifest.json` `files.calibration.jsonl`; `docs/hypotheses/done/pointwise-chat-input.md` (`n = 1148`) |
| decision-v7 test | locked final verdict | **1,176 records / 1,440 questions** | `kev/evals/v7/decision-v7/manifest.json` `files.test.jsonl` |
| transfer-v4 development | out-of-source transfer | clean **656 questions** | `docs/hypotheses/done/preference-reward-model.md`; manifest `records` 764 / `questions` 764 |
| transfer-v4 test | locked final verdict | **764 records / 764 questions** | `kev/evals/v4/transfer-v4/manifest.json` `files.test.jsonl` |
| transfer-v4 calibration / train | empty by design (eval-only) | 0 / 0 | `kev/evals/v4/transfer-v4/manifest.json` (`files`, `eval_only: true`); `kev/scripts/pref/README.md` |

- Command (one finished arm): `./kev.sh scripts/pref/eval_arm.sh runs/pref/<ARM>/seed<SEED>`; the locked test adds
  `TEST=1` (prefix the env var). `eval_arm.sh` scores decision-v7 calibration first (it supplies G for every other
  split), then decision-v7 development and transfer-v4 development, then the test splits when `TEST=1`
  (`kev/scripts/pref/eval_arm.sh`).
- Output files: `runs/pref/<ARM>/seed<N>/eval/<suite>-<split>/{rows.json, report.json, reward_report.json,
  predictions.jsonl}` (`kev/scripts/pref/eval_arm.sh`; `kev/scripts/pref/README.md` Outputs).
- Metrics:
  - `kev.benchmark` → `report.json`: `clean` (top1, NLL, ECE-softmax), `tasks` (per task), `variants`
    (`none_present` / `none_absent`, and `permuted`), `paired_flip`, `permutation.flip_rate`, `temperature`, and
    `calibrated_clean` (`kev/kev/benchmark.py`: `grouped_metrics(clean, "task")`, `grouped_metrics(rows, "variant")`,
    `paired_flip`, `"permutation": {"n", "mean_max_delta", …, "flip_rate"}`).
  - `scripts/pref/reward_report.py` → `reward_report.json`: per type + overall `n, top1, pairwise_acc, r_mean/r_std/r_min/
    r_max/abs_r_p99, label_minus_other, nll, ece_softmax, ece_sigmoid, auroc_pooled, ece_sigmoid_prior`; `raw` at T=1,
    and `G` = fitted `T` / Platt `(a,b)` with `n_calib` and the calibrated metrics (`kev/scripts/pref/reward_report.py`).
    The permutation flip rate is copied into `reward_report.json` from `report.json` (`reward_report.py` `main`).
  - Arm G (post-hoc calibration) is not a training run: `G` is fit inside `reward_report.py`
    (`docs/hypotheses/done/preference-reward-model.md` arm G; `kev/scripts/pref/README.md`).
- Cost / wall: eval ~15–30 min per arm [estimate], plus training (`kev/scripts/pref/README.md` Compute). Training wall
  per run: A/B 2.4 h, C/D/E 9.1 h, F 4.6 h (one RTX 3090) (`docs/hypotheses/done/preference-reward-model.md` Results).

### b. Decision Index 0.3 subset

- What it measures: a paired estimate of the public Decision Index 0.3 on a fixed stratified subset — an external check,
  not a board number and not part of the pre-registered dev decision rule
  (`docs/hypotheses/done/preference-reward-model.md` "Decision Index 0.3"; `kev/scripts/pref/dindex_subset.py`).
- n: **15,000 rows**, the **same file for every checkpoint** (`$DI_DIR/subset-0.3-15000.jsonl.gz`);
  `dindex.sh subset` builds it once via `dindex_subset.py make` (round-robin over the index's 37 benchmarks, ~400 cases
  each) and reuses it (`kev/scripts/pref/dindex.sh`; `kev/scripts/pref/README.md` "Decision Index 0.3").
- Kit: cloned to **`~/decision-index`** (`DI_DIR` default), pinned commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`
  (`kev/scripts/pref/dindex.sh`).
- Command: `./kev.sh scripts/pref/dindex.sh subset runs/pref/<ARM>/seed<SEED>` (default `GRAPHS` = on for chat
  checkpoints; `GRAPHS=0` = eager); one-shot `./kev.sh scripts/pref/dindex.sh check <D_RUN>` runs r10, v7-base, D,
  retemper and pairs. Comparators: `./kev.sh scripts/pref/dindex.sh subset jaredpalmer/kev-4b@v7-base kev4b-v7base`.
  Pairing: `./kev.sh scripts/pref/dindex_subset.py pair <A> <B>`; flips: `... flips <A> <B>`; retemper:
  `... retemper <ROWS> <OUT> T…` (`kev/scripts/pref/README.md`).
- Output: `RUN/dindex-0.3-sub<N>/{results.jsonl, index.json, run.log}`; comparisons write
  `runs/dindex/<NAME>-sub15000` and `board-compare-<engine>.json` (`kev/scripts/pref/dindex.sh`).
- Metrics: index, raw index, breadth, per-benchmark deltas, paired `flips` (choice argmax / noul side-of-0.5 changes,
  max |Δp|), retempered index per T (`kev/scripts/pref/dindex_subset.py`).
- Reference scores on the same 15,000 rows | value | source:
  | Model | Index | Source |
  |---|---:|---|
  | `kev-4b@v7-base` | 33.60 | `docs/hypotheses/done/preference-reward-model.md` |
  | D seed 0, eager | 36.91 | `docs/hypotheses/done/preference-reward-model.md` |
  | D seed 0, graphed | 37.01 | `docs/hypotheses/done/pointwise-chat-input.md` |
  | H seed 0 | 35.78 (raw 51.91, breadth 34.89) | `docs/hypotheses/done/pointwise-chat-input.md` |
  | board kev-4b r10 (reproduction check) | 36.69 (board 37.95) | `docs/hypotheses/done/preference-reward-model.md` |
- Cost / wall: eager chat path 27,246 s (≈7.6 h) for D on 15,000 rows; graphed ≈15–18.6k s (≈4.2–5.2 h); Kev format
  6,048 s (r10) / 7,367 s (v7-base) (`docs/hypotheses/done/preference-reward-model.md` "Inference speed";
  `docs/hypotheses/done/pointwise-chat-input.md` "Compute"). One GPU at a time.

### c. RewardBench 2

- What it measures: external best-of-4 ranking + Ties (`kev/scripts/pref/rewardbench2.py` docstring).
- n: **1,865 prompts, 8,977 candidates**; subsets `Factuality, Precise IF, Math, Safety, Focus, Ties`; overall = mean of
  the six subsets (`kev/scripts/pref/rewardbench2.py`: `SUBSETS`, Scoring transcribed from `allenai/reward-bench` sha
  `5b958c9d…`; `BENCHMARK = "allenai/reward-bench-2"`, split `test`).
- Scoring: the chosen response wins when it is one of the highest-scored (credit shared at a tied top); Ties uses
  `0.30 tied accuracy + 0.30 reference accuracy + 0.20 correctness_preferred + 0.20 correctness_preferred_hard + 0.01
  correctness_margin_score`, the `ref:N` row of a tied prompt being its reference
  (`kev/scripts/pref/rewardbench2.py`).
- Set-relative formats (A/D): candidates passed as one question's options in a seeded per-prompt shuffle
  `--order_seed 0` (`-1` = the dataset's chosen-first order); scores mapped back before scoring
  (`kev/scripts/pref/rewardbench2.py`; `docs/hypotheses/done/pointwise-chat-input.md`).
- Unsupported policy: a candidate past the serving context or OOM is never truncated and never scored around — a
  non-Ties prompt with any unsupported candidate earns 0; a Ties row with one keeps its place in every denominator and
  contributes nothing; counts in the JSON (`kev/scripts/pref/rewardbench2.py` docstring).
- Set-independence probe: `--probe 64` (default) — for 64 best-of-4 prompts, the chosen response's reward scored with
  all four candidates vs with itself + one rejected; reports mean/median/max |Δ| and the pairwise flip rate; pointwise
  arms must be exactly 0 (`kev/scripts/pref/rewardbench2.py`; `docs/hypotheses/done/pointwise-chat-input.md`).
- Pooled metrics: `./kev.sh scripts/pref/rb2_pooled.py <rewardbench2.json> [--report <reward_report.json>]` →
  `*.pooled.json`: per subset + overall `auroc_pooled`, `ece_sigmoid`, `ece_prior`, and (with `--report`) `ece_platt`
  (`kev/scripts/pref/rb2_pooled.py`; `docs/hypotheses/done/pointwise-chat-input.md`).
- Command: `./kev.sh scripts/pref/rewardbench2.py --run runs/pref/<ARM>/seed<SEED> --out
  runs/pref/<ARM>/seed<SEED>/rewardbench2.json` (`kev/scripts/pref/README.md`).
- Output: `rewardbench2.json` (per-prompt + per-subset scores, overall + no-Ties means, unsupported counts, token
  totals, probe) and, after `rb2_pooled.py`, `<stem>.pooled.json` (`kev/scripts/pref/rewardbench2.py`).
- Cost / wall (one RTX 3090, per arm): A 622 s, D 1,527 s, H 1,997 s, HP 1,988 s, HL 1,993 s; probe ~10 min
  (`docs/hypotheses/done/pointwise-chat-input.md` "RewardBench 2"). Pointwise arms issue 8,977 one-candidate requests
  (`~5–8 h` estimate) vs `~1–1.5 h` for set-relative arms (`docs/hypotheses/done/pointwise-chat-input.md` Compute).

### d. All-wrong abstention eval

- What it measures: whether an anchored reward can detect that *no* option is right (an all-wrong set), and whether that
  threshold transfers (`docs/hypotheses/in-progress/allwrong-abstention.md`).
- Data (`kev/scripts/abstain/build_allwrong.py`): from decision-v7 development and transfer-v4 development, every clean
  Choice question (not `unknowable`, hard label, ≥ 3 options, no none key): **448 questions on v7** (banking77, agnews,
  mnli, trec, dbpedia14 80 each, contrastive 48; K = 77/4/3/6/14/3) and **240 on t4** (mmlu, emotion, sciq 80 each;
  K = 4/6/4) (`docs/hypotheses/in-progress/allwrong-abstention.md`). Four conditions per question: `answerable`,
  `allwrong`, `answerable_none`, `allwrong_none`, as minimal pairs (`kev/scripts/abstain/build_allwrong.py`;
  `kev/scripts/abstain/eval_allwrong.sh` runs one `kev.benchmark --data` pass per condition).
- Readouts per question from the recorded rewards `r` (`kev/scripts/abstain/allwrong_report.py`): `maxprob`,
  `maxprob_G` (T = the arm's G fit), `maxr`, `lse`, `pnone`, `pnone_G`.
- Metrics per (suite, arm, readout): AUROC (allwrong positive), FPR@95TPR, paired win rate; on the answerable+allwrong
  pool: AURC, selective accuracy at coverages 0.3/0.5/0.7/0.9/1.0, oracle AURC; AUROC per K stratum (small ≤ 4, mid
  5–14, large ≥ 15, inclusive bounds); shared-option |Δ reward|; paired bootstrap differences (2,000 resamples)
  (`kev/scripts/abstain/allwrong_report.py`; `K_STRATA`, `COVERAGES`).
- Threshold transfer (D6): per arm/readout/target risk (0.05, 0.10), the lowest threshold with that risk on the v7 pool,
  applied unchanged to the t4 pool and each K stratum; fit suite decision-v7, apply suite transfer-v4
  (`kev/scripts/abstain/allwrong_report.py`: `FIT_SUITE`, `APPLY_SUITE`).
- K = 1 check (G4): `fork_check.py k1` scores each option alone vs all K together, `max |Δ sigmoid| < 1e-3` and no argmax
  change (`kev/scripts/abstain/fork_check.py`; `docs/hypotheses/in-progress/allwrong-abstention.md` gate G4).
- Commands: build `./kev.sh scripts/abstain/build_allwrong.py --suite evals/v7/decision-v7 --out
  runs/allwrong/data/decision-v7-development`; score `./kev.sh scripts/abstain/eval_allwrong.sh runs/pref/<ARM>/seed<N>`;
  report `./kev.sh scripts/abstain/allwrong_report.py --arm Ah=runs/pref/Ah/seed0 … --out runs/allwrong/report-half`.
  `run_arm.sh` also runs `eval_allwrong.sh` automatically unless `ALLWRONG=0` (`kev/scripts/pref/run_arm.sh`).
- Output: `RUN/eval/allwrong-<suite>/<condition>/{rows.json, report.json}` and `<out>/{report.json, rc-<suite>.svg}`
  (`kev/scripts/abstain/eval_allwrong.sh`; `kev/scripts/abstain/allwrong_report.py`).
- Cost / wall: three-GPU screen ~25–30 h total [estimate] (`docs/hypotheses/in-progress/allwrong-abstention.md`
  Compute); per half chat arm ~4.5 h, Ah ~1.2 h, stage 2 ~2.3 h, DI subset ~2 h (same doc).

### e. Position-bias analysis (offline, from RB2 shuffled runs)

Method: the set-relative arm's shuffled runs store scores in the dataset's order, and the shuffle is
`random.Random(f"order:0:{id}")`, so each prompt's displayed order can be rebuilt offline. On non-Ties prompts with one
chosen + three rejected (n 1763), report the strict-win rate by chosen position and the top-scored candidate's position,
plus the dataset-order run (chosen always first) as a cross-check. Example: A win .822/.547/.462/.434 (chosen shown
1st–4th), top-scored share .458/.217/.176/.149; D .750/.747/.697/.653 and .272/.263/.247/.218, dataset-order .837/.767
(`docs/hypotheses/done/pointwise-chat-input.md` "Position effect on RewardBench 2"). Pointwise arms have none by
construction (one request per candidate). No separate command — the analysis is derived from `rewardbench2.json` and the
`--order_seed` shuffle in `kev/scripts/pref/rewardbench2.py`.

### f. Serving-path parity (gates, not scores)

- `kev/scripts/pref/chat_speed_check.py`: scores a chat checkpoint twice on the same rows (eager `chat_hidden` vs graphed
  `kev.cuda_graphs.chat`) and prints per-path wall time, max |Δ reward| / |Δ prob|, answer flips, chunked/refused counts,
  projected 15,000-row subset time, and an fp32-reference comparison (`--dtype fp32 … --dump`, then `--against`).
  Command: `./kev.sh scripts/pref/chat_speed_check.py runs/pref/H/seed0 --n 100` (`kev/scripts/pref/chat_speed_check.py`).
- `kev/scripts/abstain/fork_check.py`:
  - `parity` (G0): the forked training forward (`kev.shared_prefix`) vs one naive causal row per option, rewards within
    1e-4 and every LoRA/head gradient within 1e-3 (`--n 10`, `--format chat_pointwise`, 0.8B base). Commands:
    `./kev.sh scripts/abstain/fork_check.py parity --base Qwen/Qwen3.5-0.8B-Base --format chat_pointwise --n 10`.
  - `k1` (G4): each option alone vs all K (see 2d).
  A failed parity stops the PF arms' interpretation, not the queue (`docs/hypotheses/in-progress/allwrong-abstention.md`
  gates G0/G4).

### g. Additional evaluation aspects (to add)

The user's list of further evaluation aspects is **pending**. Each new one gets a row in the same format as (a)–(f):
name, what it measures, n, split policy, command (`./kev.sh …`), output files, metrics read, cost/wall — with a source
path for every number.

---

## 3. Eval control rules

| Rule | Statement | Source |
|---|---|---|
| dev = decisions | development splits are for decisions; calibration only fits G; test is locked | `kev/scripts/pref/eval_arm.sh`; `kev/scripts/pref/README.md` |
| Never tune on dev twice | a dev split is not reused to make the same claim twice (one pre-registered read per claim) | `docs/hypotheses/done/preference-reward-model.md`, `docs/hypotheses/done/pointwise-chat-input.md` (rules fixed before runs) |
| Test read once | the locked test split is read once for the final verdict, never for selection/search | `kev/kev/suite.py` `load_split`; `kev/scripts/pref/eval_arm.sh` (`TEST=1`, "!!!" banner) |
| Same rows for every arm | the DI subset is one file used by every checkpoint; clean dev rows are the same questions for every arm | `kev/scripts/pref/README.md`; `docs/hypotheses/done/preference-reward-model.md` |
| Seeds | seeds 0 first, then 1, 2 for a claim; single-seed differences ≤ 2 points are noise (±0.4–1.5 seed spread) | `docs/hypotheses/done/preference-reward-model.md`; `docs/studies/README.md` |
| Fixed per-pass split | `--reg`/`--row_budget` arms (D, E, H, HP, Dh, Hh) keep the same microbatch across seeds, or not at all | `kev/scripts/pref/arms.sh`; `kev/scripts/pref/README.md` |
| What gets packed | `./kev.sh scripts/pref/pack_results.sh [OUT.tgz]` packs `runs/pref/COMMIT`, `*.train.log` / `seed*/train.log`, `seed*/training_*.json`, `eval/*`, `*.manifest.json`, `rewardbench2*.json` (no weights) and prints per-run split completeness + G's `n_calib` | `kev/scripts/pref/pack_results.sh` |

---

## 4. Per-arm eval checklist (one finished arm, in order)

All commands from the repo root, `./kev.sh`-prefixed; `<ARM>`/`<SEED>` as the arm was trained. `run_arm.sh` already runs
training + the standard evals + the all-wrong eval, so this is the order for an arm (or a re-run) whose checkpoint exists.

```bash
# 0. (once) environment + CPU tests
./kev.sh scripts/pref/setup_3090.sh                         # prints the chat template / rendered prompt
./kev.sh scripts/abstain/fork_check.py parity --base Qwen/Qwen3.5-0.8B-Base --format chat_pointwise --n 10  # G0 (pointwise arms)

# 1. train + standard evals + all-wrong evals (or evaluate an existing run)
./kev.sh scripts/pref/run_arm.sh <ARM> <SEED> <GPU>         # train -> eval_arm.sh -> eval_allwrong.sh; ALLWRONG=0 skips the last
./kev.sh scripts/pref/eval_arm.sh runs/pref/<ARM>/seed<SEED>   # redo/report: decision-v7 cal+dev, transfer-v4 dev

# 2. all-wrong report (cross-arm)
./kev.sh scripts/abstain/allwrong_report.py --arm <ARM>=runs/pref/<ARM>/seed<SEED> --out runs/allwrong/report-<name>

# 3. serving-path parity (chat/pointwise checkpoints)
./kev.sh scripts/pref/chat_speed_check.py runs/pref/<ARM>/seed<SEED> --n 100

# 4. RewardBench 2
./kev.sh scripts/pref/rewardbench2.py --run runs/pref/<ARM>/seed<SEED> --out runs/pref/<ARM>/seed<SEED>/rewardbench2.json
./kev.sh scripts/pref/rb2_pooled.py runs/pref/<ARM>/seed<SEED>/rewardbench2.json --report runs/pref/<ARM>/seed<SEED>/eval/decision-v7-development/reward_report.json

# 5. Decision Index 0.3 subset (secondary/external)
./kev.sh scripts/pref/dindex.sh setup                       # once: kit -> ~/decision-index
./kev.sh scripts/pref/dindex.sh subset runs/pref/<ARM>/seed<SEED>
./kev.sh scripts/pref/dindex.sh check runs/pref/<ARM>/seed<SEED>   # r10 + v7-base + this run, retemper, pairs

# 6. G4 K=1 (pointwise arms)
./kev.sh scripts/abstain/fork_check.py k1 --run runs/pref/<ARM>/seed<SEED> --n 200

# 7. pack (after the checks above)
./kev.sh scripts/pref/pack_results.sh pref-<name>.tgz

# 8. FINAL VERDICT ONLY — locked test split, read once
TEST=1 ./kev.sh scripts/pref/eval_arm.sh runs/pref/<ARM>/seed<SEED>
```

Source for the script sequence: `kev/scripts/pref/README.md` "Commands, in order",
`kev/scripts/pref/run_arm.sh`, `kev/scripts/pref/eval_arm.sh`, `kev/scripts/abstain/queue.sh`.
