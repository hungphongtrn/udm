# Data studies

Schema and per-source mapping: `docs/DATA_SPEC.md` (on `claude/set-conditioned-reward-model-hvsb9s`); trainer contract: `docs/CONTRACT.md` (on `claude/set-conditioned-reward-model-hvsb9s`).

## 1. Build and push (2026-10-02)

Sources converted into the 26-column schema of `hungphongtrn/udm-massive-typed`: `tasksource/tasksource-jev-typed-decisions`,
`samatv256/jev-decisions-v1` (`default`, `general-clean-50k`), `ZefanCai/Open-Jev`; existing MASSIVE / typed rows kept.

- Open-Jev smoke build: 25 s on a 4-core sandbox. A Vast CPU attempt was abandoned before upload (~$0.08 spent).
- tasksource receipt: rows `{train: 2,399,451, validation: 14,451, test: 14,411}`; drops `duplicate_content_in_split 98,702`,
  `noul_tie 1,653`, `no_trainable_pair 27`, `leak_train_content_in_heldout 1,305`; `leakage=0`.
- After push:

| source | train | validation | test |
|---|---:|---:|---:|
| tasksource | 2,399,451 | 14,451 | 14,411 |
| samatv | 5,448,117 | 171,170 | 168,931 |
| openjev | 264,287 | 31,994 | 106,924 |
| samatv_clean50k | 1,491 | 27 | 30 |
| **new rows** | **8,113,346** | **217,642** | **290,296** |

Whole repo: 9,488,376 rows (8,718,074 train / 323,358 validation / 446,944 test), of which 867,092 are the older
slug-less rows. Leakage is only checked against the project's own validation/test, not against Decision Index items
(see [external_comparison.md](external_comparison.md#3-contamination-audit)).

## 2. Prompt rendering (2026-10-02)

Plain chat-format text, no new tokens; labelled blocks (`State:`, `Instruction:`, `Options:` / `Option k:`,
`Grade this choice: Option k: …`), stopping at the assistant header (commit `fc546c0`). IDs, hashes and the
instruction `type` are never rendered. Checked on fixture rows only (20 Open-Jev, 7 tasksource).

## 3. Context length and truncation (2026-10-02)

Measured on 300 sampled rows per source:

- massive, typed, tasksource, openjev: no truncation; p99 prompt ≤ 1,248 tokens.
- samatv: 68 % over 4096 tokens; median state ~4.9k tokens; p90 ~65k.
- At 4096 with a 3072-token state budget ~55 % of samatv states were cut (middle truncation); 8192 was estimated to
  still cut 48 % and OOM on 24 GB.

Decision (`deb6dc6`): no truncation anywhere. A set whose longest graded sequence exceeds `data.render.max_len` is
dropped whole. Consequence: the 24 GB config (max_len 4096) effectively trains on a different, shorter-context mix
than H200 (32768). In eval logs, 72–115 of the drawn samatv rows per 400-item eval did not render at the configured
max_len (replaced by new draws).

Decision Index requests with the real tokenizer: none exceed 16384 tokens; 4.5 % exceed 4096.

## 4. Eval-sampling bug (2026-10-03)

The first long run (4090, `vqn31gtl`) looked like it peaked at step 1000 (val loss 0.098, top1 0.754) and degraded by
step 2000 (0.212, 0.682). Cause: streaming `select_eval_rows` scanned only the first 200k rows and took the first
100 per source in file order, stopping inside samatv:

| source | validation rows available | eval sets used | train weight |
|---|---:|---:|---:|
| massive | 105,716 | 100 | 0.15 |
| Open-Jev | 31,994 | 100 | 0.20 |
| samatv | 171,197 | 36 | 0.30 |
| tasksource | 14,451 | 0 | 0.25 |

236 validation sets in total; ~55 % of the training mix (samatv + tasksource) was effectively unmeasured; test had the
same gap. Fix (`fee8e03`): scan every split file, 100 random sets per source up to 1000, redraw over-long rows up to
10× quota, report per-source draw/drop counts, per-source train loss. Calibration metrics (`conf`, `overconf`,
`ece_top1`, `brier_top1`, `p_best`, `nll_best`) added as tracking only (`679f6a3`). All validation numbers before
`fee8e03` are not comparable with later ones.

## 5. Epoch arithmetic (2026-10-03)

Streaming, weighted source sampling, caps massive 150k + samatv 1M. At 13–15 sets/step, 20k steps cover 260–300k
sets; one pass over just those two capped sources needs ~80k steps. 20k steps is a budget, far below one epoch.

## 6. Training-signal shape (2026-10-06)

Measured on the training mix: BT pairs per set — MASSIVE 59.0, samatv 9.96, tasksource 5.78, typed 2.36, open-jev 2.24.
Share of option pairs that are same-tier (never trained by BT): MASSIVE 96.7 %, tasksource 96.9 %, samatv 87.6 %,
open-jev 61.3 %, typed 48.5 %. Twelve of the 38 DI panel benchmarks share upstream datasets with our training sources
(BANKING77, CLINC150, ContractNLI, ANLI, Humicroedit, WinoGrande, HellaSwag, ESCI, iSarcasmEval, NLI4CT, CLadder, …);
item-level overlap is not measured.

## 7. Vietnamese data search (2026-10-04, not used)

Candidates found but not audited or integrated: `NLPLab-SoICT/Vi-SRS` (120k pairs, Apache-2.0),
`FrostyAshe/vietnews-dpo` (16k), `522H0134-NguyenNhatHuy/vietnamese-dpo-10k` (10k, CC-BY-4.0),
`thanhhoangnvbg/empathAI-dpo-vi` (~6k, MIT), Vietnamese safety classification (3.5k), MultiJail-vi (315),
`NamSyntax/Vietnamese-Legal-QA-RAG` (420); eval-only m-arena Vietnamese slice and `lightblue/mt_bench_vietnamese`.
