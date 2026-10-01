# Data ↔ Training contract (internal, v1)

This file is the shared contract between the data pipeline (`src/scrm_data/`) and
the training code (`src/scrm/`). Both sides MUST follow it. The full user-facing
spec lives in `docs/DATA_SPEC.md` (written by the data pipeline).

## 1. Storage target

* HF dataset repo: `hungphongtrn/udm-massive-typed` (append in place).
* Existing shards `data/{train,validation,test}-000NN-of-000MM.parquet` are NEVER
  modified or deleted.
* New shards are added as `data/{split}-{source_slug}-{NNNNN}-of-{MMMMM}.parquet`
  so the existing config globs `data/train-*`, `data/validation-*`, `data/test-*`
  pick them up automatically.
* Splits: `train`, `validation`, `test` only (HF split names). Column
  `partition_role` uses the same values the existing data uses (`train`, `dev`,
  `test`, verify against the existing shards).
* Every new shard has EXACTLY the existing Arrow schema (same 26 columns, same
  order, same types, including the nested `candidate_rows` list of 5-field structs).
  No new columns.

## 2. Row semantics (one row = one decision set)

| column | meaning for training |
|---|---|
| `state_json` | canonical JSON. A JSON string (plain text state) or a JSON object (structured state). Rendered to text by the trainer. |
| `instruction_json` | canonical JSON. Either a JSON string (the instruction) or an object `{"type": ..., "instructions": str, "criteria"?: {...}}`. |
| `options_json` | canonical JSON object `{choice_id: candidate_text}`. Authoritative candidate set. |
| `tier_json` | canonical JSON list of lists of choice_ids. Tier 0 = best. Every choice_id appears in exactly one tier. Pairs are trained only for tier_i < tier_j. Same-tier pairs are never trained. |
| `probabilities_json` | canonical JSON object `{choice_id: float}` or `{}`. Optional soft target. |
| `candidate_rows` | list ordered by `candidate_order`; gives the canonical candidate order. |
| `label_kind` | how tiers were derived (see §3). |
| `family`, `source_id` | used for train-time mixing caps and per-family metrics. |

Choice_ids and other hashes: replicate the existing recipe (data agent verifies
against existing rows, e.g. `choice_id == sha256(normalized candidate text)`) and
documents it in DATA_SPEC.md.

## 3. Label kinds → tiers (classification framing)

All label kinds become tiers. The ORIGINAL supervision must stay recoverable and
is written into `source_label_or_null` as canonical JSON (see below), and when
relevant into `probabilities_json`.

| source kind | `label_kind` | tiers | `probabilities_json` | `source_label_or_null` (canonical JSON) |
|---|---|---|---|---|
| single winner choice (one-hot) | `source_dataset_label` | `[[winner],[rest]]` | `{}` or one-hot | `{"kind":"choice","target":<raw target>,"winner":[ids]}` |
| soft choice distribution | `soft_choice_distribution` | group by distinct prob, descending | the distribution | `{"kind":"choice_soft","target":<raw probs in source order>}` |
| `score` (ordinal levels) | `ordinal_score_distance_tiers` | tier k = levels at ordinal distance k from the true level (e.g. true=2 of 0..4 → `[[2],[1,3],[0,4]]`) | raw target over levels if given | `{"kind":"score","levels":[level texts in ordinal order],"level_choice_ids":[...],"target":<raw>,"true_level_index":int or null,"expected_level":float or null}` |
| `noul` (independent multi-label probs) | `independent_labels_binary_tiers` | `[[p>=0.5],[p<0.5]]`; if one side is empty the row is kept only if probabilities differ (then tiers by prob) else dropped | the independent probs (NOT normalized, do not sum to 1) | `{"kind":"noul","target":<raw independent probs in source order>,"threshold":0.5}` |
| tool/route selection (samatv256) | `agent_choice_target` | `[[target],[rest]]` (ordered_targets → more tiers if meaningful) | `{}` | `{"kind":"agent_choice","target":{...raw target struct...},"labels":{...raw labels struct...},"decision_type":...}` |

Rows with fewer than 2 candidates or no trainable pair (all candidates in one tier)
are dropped and counted in the build report.

## 4. Trainer-side expectations

* Trainer loads the HF repo with `datasets.load_dataset("hungphongtrn/udm-massive-typed")`
  (streaming or local cache) and only needs: `state_json`, `instruction_json`,
  `options_json`, `tier_json`, `candidate_rows`, `family`, `source_id`,
  `label_kind`, `decision_set_id`.
* Prompt rendering: instruction text (+ criteria if present), state rendered as
  text (string as-is; object → pretty JSON), then each candidate wrapped in the
  boundary tokens. Candidate order is randomly shuffled every time in training,
  and kept canonical in eval.
* Pair mask: `M[i,j] = tier[i] < tier[j]`.
