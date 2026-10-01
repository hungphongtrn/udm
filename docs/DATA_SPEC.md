# SCRM data specification

Target: HF dataset `hungphongtrn/udm-massive-typed` (schema_version `udm.unified-huggingface-dataset/v2`), extended in
place with new shards. This document describes the schema, the hash/encoding recipes (what was verified against the
existing data and what had to be defined), label kinds, per-source mappings, split/leakage policy and drop reasons.
Code: `src/scrm_data/`; the binding data/training interface is `docs/CONTRACT.md`.

Every row is **one decision set**: a `state`, an `instruction`, a set of candidate texts and best-first `tiers`. All
sources (NLI, QA, ratings, agent tool selection, controlled simulators, ...) are framed purely as classification /
ranking over candidates.

## 1. Pipeline overview

```
HF source parquet --(hf_hub_download | HTTPS range reads)--> unit workers (ProcessPool, pyarrow iter_batches)
   converter: raw row -> 26-column row | drop(reason)
   -> content-key dedupe / leakage filter -> zstd parquet parts -> merged to ~400 MB shards
   <out>/data/{split}-{slug}-{NNNNN}-of-{MMMMM}.parquet   <out>/receipts/{slug}.json   <out>/keys/...
build -> validate -> push (additive) ; scripts in scripts/data/
```

Source slugs: `tasksource`, `samatv` (config `default`), `samatv_clean50k` (config `general-clean-50k`), `openjev`.
Stage order inside a build: (1) eval units (validation/test, Open-Jev calibration/validation/test/ood), (2) train units
(filtered against all held-out content keys), (3) `samatv_clean50k` (deduplicated against `samatv`).
Units are source files (samatv files are further cut into row-group ranges of ~256 MB). Finished units leave
`_work/<slug>/<unit>/DONE.json`; re-running skips them (`--force` ignores). `--limit-rows N` reads at most N raw rows per
unit (smoke tests; receipts are flagged `partial_build`).

## 2. Schema (26 columns, identical to the existing shards)

Verified: the Arrow schema of every new shard equals the schema of an existing remote shard
(`validate.py` fetches only the remote parquet footer; list child field is `element` in both).

| # | column | type | meaning |
|---|---|---|---|
| 1 | `decision_set_id` | string | sha256 identifying the row (recipe 3.4) |
| 2 | `source_id` | string | upstream HF repo (`tasksource/tasksource-jev-typed-decisions`, `samatv256/jev-decisions-v1`, `ZefanCai/Open-Jev`) |
| 3 | `source_revision` | string | pinned upstream commit sha |
| 4 | `source_config` | string | upstream config (`default`, `general-clean-50k`, or the Open-Jev config name) |
| 5 | `source_split` | string | RAW upstream split: `train`/`validation`/`test`; Open-Jev keeps `calibration`, `ood`; clean50k `train` |
| 6 | `source_row_id` | string | upstream row id |
| 7 | `source_parent_id` | string | upstream group id (tasksource `group_id`, samatv `trajectory_id`, Open-Jev `group_id`) |
| 8 | `family` | string | mixing/metrics family, section 7 |
| 9 | `partition_role` | string | `train` / `dev` / `test` (HF split `validation` <-> `dev`; verified on existing shards) |
| 10 | `lineage_key` | string | leakage-grouping key (tasksource/Open-Jev group id; samatv `<dataset_id>:<trajectory_id>`) |
| 11 | `state_json` | string | canonical JSON: a JSON string (plain text state) or object (structured state) |
| 12 | `instruction_json` | string | canonical JSON object `{"type": choice\|score\|noul, "instructions": str}` (new sources never carry `criteria`: the candidates ARE the criteria) |
| 13 | `options_json` | string | canonical JSON object `{choice_id: candidate_text}` (authoritative candidate set; keys sorted) |
| 14 | `tier_json` | string | canonical JSON list of lists of choice_ids, tier 0 = best, ids sorted inside a tier |
| 15 | `probabilities_json` | string | canonical JSON `{choice_id: float}` (6 decimals) or `{}` |
| 16 | `candidate_count` | int32 | `len(options)` |
| 17 | `label_kind` | string | how tiers were derived, section 5 |
| 18 | `source_label_or_null` | string | canonical JSON with the ORIGINAL supervision (section 6) |
| 19 | `instruction_template_id` | string | `tasksource.<kind>.<variant>`, `openjev.<kind>`, `jev_agent.choice_v1` |
| 20 | `option_permutation_seed_or_null` | string | always null (candidate order is the upstream order) |
| 21 | `raw_record_sha256` | string | fingerprint of the raw upstream record (recipe 3.5) |
| 22 | `license` | string | upstream license string (tasksource: per-row `license`; Open-Jev `cc0-1.0`; samatv per-row, default `cc-by-4.0`) |
| 23 | `known_pretraining_overlap` | string | null |
| 24 | `contamination_status` | string | `not_assessed` |
| 25 | `quarantine_reason_or_null` | string | null |
| 26 | `candidate_rows` | list<struct{record_id, decision_set_id, choice_id, candidate_order int32, normalized_text_sha256}> | one entry per candidate, ordered by `candidate_order` = canonical (upstream) order |

## 3. Encoding and hash recipes

### 3.1 Verified exactly against existing rows (LocalLLaMA/typed-decisions + MASSIVE rows)
* **Canonical JSON** for `state_json`, `instruction_json`, `options_json`, `tier_json`, `probabilities_json`:
  `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` (checked on rows of 3 typed families
  and MASSIVE incl. Urdu text: `ensure_ascii=False` required). We additionally pass `allow_nan=False`.
* **`choice_id == normalized_text_sha256 == sha256(utf-8 text)`** of the option text stored in `options_json`
  (28,400 / 28,400 typed option texts and all 60 MASSIVE labels matched). Hence `sha256(options[cid]) == cid`
  holds for every row and is checked by `validate.py`.
* `options_json` keys are sorted (consequence of canonical JSON); `tier_json`: tier 0 first, ids sorted inside a tier
  (e.g. score row `[[argmax],[rest sorted by id]]`); `candidate_rows[*].candidate_order` is the **source option
  order** (score levels in ordinal order, MASSIVE labels alphabetical), not the sorted-id order.
* `partition_role` vocabulary: `train`, `dev` (for HF split `validation`), `test`. `source_split` uses `validation`.
* Existing label kinds: `source_dataset_label` (MASSIVE) and `hard_ranking_with_retained_choice_probabilities`
  (typed-decisions; score/noul flattened to a single top tier). New data uses the kinds of section 5.
* Existing `lineage_key` = parent id (typed: `customer_service_000007`) or a canonical JSON object (MASSIVE).

### 3.2 Not recoverable, defined by this pipeline (recipe tag `scrm-data-recipe/v1`)
`decision_set_id`, `record_id`, `raw_record_sha256` of the existing data could not be reproduced (hundreds of thousands
of field/serialisation combinations of the row's own fields and of the raw LocalLLaMA record were tried, no hit), and the
existing text normalisation is unobservable (every existing candidate text is invariant under strip, NFC, NFKC and
whitespace collapsing). The new definitions are deterministic and collision-safe; they intentionally do not claim
compatibility with the original hashes.

### 3.3 Text normalisation
`normalize_text`: Unicode NFC, `\r\n`/`\r` -> `\n`, strip leading/trailing whitespace. Inner whitespace is kept (code,
tables). The STORED option text is the normalised text, so `choice_id = sha256(stored text)`.
Candidates with identical normalised text are collapsed (first occurrence kept) if their targets agree, otherwise the
row is dropped (`duplicate_options_conflict`); duplicate score levels drop the row (`duplicate_levels`); empty candidate
text drops the row (`empty_option_text`). Collapsed index pairs are recorded in `source_label_or_null` as
`collapsed_duplicate_indices` and the raw target keeps the original (pre-collapse) alignment.

### 3.4 `decision_set_id`, `record_id`
```
decision_set_id = sha256(canon({"recipe":"scrm-data-recipe/v1","source_id","source_config","source_split","source_row_id","raw_record_sha256"}))
record_id       = sha256(canon({"decision_set_id","choice_id","recipe":"scrm-data-recipe/v1"}))
```
`canon` = canonical JSON above. `decision_set_id` is unique over the whole dataset (checked globally by `validate.py`).

### 3.5 `raw_record_sha256`
* tasksource: `sha256(canon({13 raw columns}))`
* samatv default: `sha256(canon({"id", "content_hash"}))`; clean50k: `sha256(canon({"id","config"}))`
* Open-Jev: `sha256(record_json)` (the exact original JSON record, UTF-8)

### 3.6 Content key (dedupe and leakage)
`sha256(state_json \x1f "<type>\x1d<instruction text>" \x1f "\x1e".join(sorted(candidate texts)))`, truncated to 64 bits.
Independent of candidate order and labels; `type` keeps the same ordinal task asked as `choice` and as `score` apart. It
can be recomputed from stored columns (`canonical.content_key_from_columns`), which is how keys of the existing
repo rows are exported (`python -m scrm_data.keys`).

## 4. Tier semantics and how pairs are formed
* `tier_json = [[tier 0 ids (best)], [tier 1 ids], ...]`; every candidate is in exactly one tier; there are at least 2
  non-empty tiers (otherwise the row is dropped, `no_trainable_pair`).
* Pairs are `(i, j)` with `tier(i) < tier(j)` (i preferred); same-tier pairs are never trained. Pair mask:
  `M[i,j] = tier[i] < tier[j]`.
* Candidate order in `candidate_rows` is canonical; trainers shuffle it during training.

## 5. Label kinds (classification framing of every source label)

| `label_kind` | used for | tiers | `probabilities_json` |
|---|---|---|---|
| `source_dataset_label` | one-hot choice (single winner) | `[[winner],[rest]]` | `{}` |
| `soft_choice_distribution` | soft / multi-winner choice (annotator votes, tied optimal moves) | distinct probability groups, descending (rounded to 1e-9 for grouping) | the distribution (6 decimals) |
| `ordinal_score_distance_tiers` | `score` (ordinal levels) | tier k = levels at ordinal distance k from the true level, e.g. true=2 of 0..4 -> `[[2],[1,3],[0,4]]` | raw target over levels (6 decimals) |
| `independent_labels_binary_tiers` | `noul` (yes/no probability; also generic independent multi-label) | yes/no: `[[yes],[no]]` if p_yes > 0.5 else `[[no],[yes]]`; multi-label: `[[p>=0.5],[p<0.5]]` (if one side is empty: by probability, dropped if all equal) | yes/no: `{no: 1-p, yes: p}`; multi-label: independent, NOT normalised |
| `agent_choice_target` | samatv tool/route selection | `[[target],[rest]]` | `{}` |

Details and choices:
* **score**: the true level is the unique argmax of the target; if the argmax is not unique, tiers are formed by
  `|level - expected_level|` and `true_level_index` is null. Level order = candidate order = source option order
  (lowest first). Soft targets (mean ratings split between two levels) keep their distribution in `probabilities_json`.
* **noul**: candidates are always the two texts `no` (order 0) and `yes` (order 1). tasksource gives `[p]` (p = probability
  of yes); Open-Jev gives `[p_no, p_yes]` over options `["no","yes"]` (reversed option order is handled). `p_yes == 0.5`
  is a tie and is dropped (`noul_tie`). The independent multi-label branch exists and is unit tested but no current source
  uses it.
* **choice** with all-equal targets (e.g. empty tic-tac-toe board, uniform 1/9) is dropped (`no_trainable_pair`).
* ordered_targets of samatv are NOT used for tiers (they are source-ordered multi-call targets, not a ranking); their
  candidate ids are kept in `source_label_or_null.ordered_target_candidate_ids`.

## 6. `source_label_or_null` shapes (canonical JSON; original supervision is always recoverable)

| kind | shape |
|---|---|
| choice (one-hot) | `{"kind":"choice","target":[raw floats in source option order],"winner":[choice_id]}` |
| choice_soft | `{"kind":"choice_soft","target":[raw floats in source order]}` |
| score | `{"kind":"score","levels":[level texts, ordinal order],"level_choice_ids":[ids, same order],"target":[raw],"true_level_index":int or null,"expected_level":float}` |
| noul | `{"kind":"noul","target":[raw],"threshold":0.5,"options":["no","yes"],"p_yes":float}` (Open-Jev `options` = source order) |
| agent_choice | `{"kind":"agent_choice","target":{raw target struct},"labels":{raw labels struct},"decision_type":str,"candidate_ids":[source candidate ids in candidate order],"ordered_target_candidate_ids":[...],"provenance":{dataset_id,dataset_revision,source_config,raw_split,trajectory_id,step_index,decision_ordinal},"supervision_evidence":str,"quality_weight":float}` |

All shapes may also carry source metadata (tasksource: `source_task`, `variant`, `question_id`, `license_use`,
`source_split_label`; Open-Jev: `task`, `config`, `original_line_number`) and `collapsed_duplicate_indices`.

Recovering the original target:
* **score** (tasksource `AES2-essay-scoring`, 6 levels): `sl = json.loads(row["source_label_or_null"])`;
  `sl["target"]` is the original distribution aligned with `sl["levels"]` / `sl["level_choice_ids"]`; the expected level
  (mean rating) is `sl["expected_level"]`; `{cid: p for cid, p in zip(sl["level_choice_ids"], sl["target"])}` equals
  `probabilities_json`.
* **noul** (`target=[0.8]`): `sl == {"kind":"noul","options":["no","yes"],"p_yes":0.8,"target":[0.8],"threshold":0.5,...}`;
  tiers `[[id("yes")],[id("no")]]`, `probabilities_json = {id(no):0.2, id(yes):0.8}`.

## 7. Sources, mappings, families

### 7.1 `tasksource/tasksource-jev-typed-decisions` (slug `tasksource`, rev `8173a06c`)
Files `data/{train-0000k-of-00013,validation-00000-of-00001,test-00000-of-00001}.parquet` (2.5M / 15k / 15k).
* `state` -> `state_json` (JSON string; empty state is legitimate, kept as `""`); `question` -> `instruction.instructions`; `kind` ->
  `instruction.type`.
* `choice`: options -> candidates (source per-row order), `target` one-hot/soft -> section 5.
  `score`: options = ordered levels, `target` distribution over levels. `noul`: no options, target `[p]` -> `no`/`yes`.
* `id` -> `source_row_id`, `group_id` -> `source_parent_id` and `lineage_key`; file split -> `source_split`
  (`validation` file -> `partition_role` `dev`; the raw `dev` string is kept as `source_split_label`).
* `family = tasksource:<source>` (670 sources, e.g. `tasksource:super_glue/copa`); `license` = row license,
  `license_use` (commercial / non-commercial / unspecified) is kept in `source_label_or_null`.
* Upstream already removed val/test rows seen in train; the pipeline re-checks (section 8).

### 7.2 `samatv256/jev-decisions-v1`
`default`: `data/{train/train-000kk-of-00011,validation/...,test/...}.parquet` (~12M rows, 21.6 GB train).
Usable rows: `training.choice_eligible` and target candidate id present in the candidates and >= 2 distinct candidate
texts (everything else: value-only / completion-only / unsupported rows, ~half of the corpus, is skipped at Arrow level and
counted `not_choice_eligible`).
* `state_json` = `{"system":..,"user_goal":..,"history":[{"role":..,"payload":<parsed payload_json or raw string>}],"environment":<parsed environment_json>}`; empty/null keys omitted.
* candidate text = canonical JSON `{"name","description"?,"parameters"?}` (parsed `parameters_json`; `metadata_json` and target
  arguments are NOT shown to the model, per the dataset card); candidate ids are kept in `source_label_or_null.candidate_ids`.
* instruction: `{"type":"choice","instructions":"Given the current state and available options,\nwhich option should be selected?"}` (the dataset's fixed question).
* output split = the row's `training_split` (`val` -> `validation`), which is group-aware by trajectory in the source
  (zero trajectory overlap per the dataset card); `lineage_key = "<dataset_id>:<trajectory_id>"`.
* `family = jev_agent:<upstream dataset>[/<raw split or config>]`, e.g. `jev_agent:Nemotron-SFT-Agentic-v2/search`.

`general-clean-50k` (slug `samatv_clean50k`, train only, 50,000 sanitised Choice rows derived from the default train
partition; decision ids unchanged): state/candidates mapped identically (`answer_options` -> candidate text; `target_index` ->
winner). Rows whose decision id OR content key already exists in `samatv` are dropped (`duplicate_of_samatv`) - the
config is a sanitised subset of the default train rows, so after a full build very little survives. Because the source has no
trajectory id, surviving rows are split 96/2/2 by hash of the CONTENT key (identical content never straddles splits).

### 7.3 `ZefanCai/Open-Jev` (slug `openjev`, rev `c67699e1`)
12 configs x splits `train/calibration/validation/test/ood` -> files `data/<config>/<split>-00000-of-00001.parquet`.
* split map: `train`->train; `calibration`,`validation`->validation; `test`,`ood`->test. `source_split` keeps the raw name,
  so e.g. OOD rows are excluded with `source_split != 'ood'`, calibration with `source_split != 'calibration'`.
* `state_json` = re-canonicalised original state (string or object); `metadata_json` (privileged labels) is never copied;
  `raw_record_sha256 = sha256(record_json)`.
* `kind=choice` -> choice over `options` (the options text may include the label, e.g. `invoice: An issued bill...`);
  `kind=noul` -> yes/no 2 candidates; `kind=score` -> ordinal levels (painting HSL grades, IR relevance 0-3, severity...).
* `release-v2-redistributable` is an exact subset (same ids, records, splits; verified for all 113,568 rows) of
  `browser-drone-expansion-v1-redistributable`, so it is skipped when its superset is converted (`skipped_configs_exact_subset`
  in the receipt); building it alone is possible with `--openjev-configs release-v2-redistributable`.
* Duplicated heads (e.g. workflow-controls variants `v0`/`v1` that differ only in metadata) collapse via the content key.
* `family = openjev:<source>` (`openjev:painting-geometry-v1`, `openjev:workflow-controls-v1/customer_service`,
  `openjev:mailroom-control-v1`, ...); the group id is the lineage key (Open-Jev guarantees group separation between splits).
* Per-config mapping notes: yes/no probability heads (all `noul`) -> `no`/`yes`; choice distributions (game moves,
  mailroom categories, amount/phone/email candidate selection, citation supported/contradicted/insufficient, sponsor segment
  categories) -> candidates; entity-alignment and IR `score` heads -> ordinal tiers; control-only configs
  (`context-retention`, `silent-failure`) contain only noul heads.

## 8. Split policy and leakage handling
* Source splits are respected (tasksource, samatv, Open-Jev as mapped above). Only `samatv_clean50k` is hash split.
* Content keys (3.6) of all validation/test rows of every built source (and, if exported with
  `python -m scrm_data.keys` / `--existing-keys`, of the existing repo val/test rows, MASSIVE excluded by default) form the
  held-out set. Train rows whose key is in it are dropped (`leak_train_content_in_heldout`).
* Within a unit identical content keys are deduplicated (`duplicate_content_in_split`); cross-unit duplicates are not removed
  but reported.
* Every receipt has a `leakage_audit` computed from the key files: `train_rows_in_heldout` (must be 0 when sources were built in
  one run), val/test overlaps, per-split duplicate keys. A source built later than another can still have held-out rows that
  overlap earlier sources' train rows - re-run `python -m scrm_data.report` to see it.
* Group awareness: samatv by trajectory (source split), Open-Jev/tasksource by group id (source guarantees); the content-key filter
  is additional.

## 9. Drop reasons (counted in receipts)
`not_choice_eligible`, `no_target_candidate`, `target_not_in_candidates`, `fewer_than_2_candidates`, `no_trainable_pair`
(single tier / uniform targets), `noul_tie`, `noul_no_distinct_probabilities`, `noul_target_not_single_probability`,
`noul_unexpected_options`, `target_length_mismatch`, `invalid_target`, `zero_target_mass`, `duplicate_options_conflict`,
`duplicate_levels`, `empty_option_text`, `unknown_kind`, `invalid_unicode`, `non_finite_number`,
`duplicate_content_in_split`, `leak_train_content_in_heldout`, `duplicate_of_samatv`.

## 10. Filtering cookbook
```python
ds = load_dataset("hungphongtrn/udm-massive-typed", split="train")
no_ood_cal = ds.filter(lambda s: s not in ("ood", "calibration"), input_columns="source_split")
commercial = ds.filter(lambda l: '"license_use":"commercial"' in l, input_columns="source_label_or_null")  # tasksource rows
only_score = ds.filter(lambda k: k == "ordinal_score_distance_tiers", input_columns="label_kind")
per_family_cap = ...  # use the `family` column (tasksource:<task>, openjev:<task>, jev_agent:<dataset>)
```

## 11. Real example rows (smoke build; long strings truncated with `…`)

tasksource noul (`label_verification`, target `[0.0]`):
```json
{"decision_set_id":"75859541…","source_id":"tasksource/tasksource-jev-typed-decisions","source_split":"train",
 "source_row_id":"AdjectiveScaleProbe-nli-c125da07dd:train:186:noul-label-verification","family":"tasksource:AdjectiveScaleProbe-nli",
 "partition_role":"train","state_json":"\"text_A: A 216°C oven is hot. A 136°C oven is cold.\\ntext_B: A 124°C oven is hot.\"",
 "instruction_json":"{\"instructions\":\"Does text_A entail text_B, contradict it, or neither? Is \\\"entailment\\\" the correct answer?\",\"type\":\"noul\"}",
 "options_json":"{\"8a798890…\":\"yes\",\"9390298f…\":\"no\"}","tier_json":"[[\"9390298f…\"],[\"8a798890…\"]]",
 "probabilities_json":"{\"8a798890…\":0.0,\"9390298f…\":1.0}","candidate_count":2,"label_kind":"independent_labels_binary_tiers",
 "source_label_or_null":"{\"kind\":\"noul\",\"license_use\":\"unspecified\",\"options\":[\"no\",\"yes\"],\"p_yes\":0.0,\"question_id\":\"noul-label-verification\",\"source_split_label\":\"train\",\"source_task\":\"AdjectiveScaleProbe-nli\",\"target\":[0.0],\"threshold\":0.5,\"variant\":\"label_verification\"}",
 "instruction_template_id":"tasksource.noul.label_verification","license":"unspecified","contamination_status":"not_assessed"}
```
tasksource score (6 essay grades; true level 3 of 0..5): `label_kind=ordinal_score_distance_tiers`,
`tier_json=[[id("4 out of 6")],[id("3 out of 6"),id("5 out of 6")],...]` (levels at distance 0,1,2,...),
`source_label_or_null={"expected_level":3.0,"kind":"score","level_choice_ids":[...],"levels":["1 out of 6",...],"target":[0,0,0,1,0,0],"true_level_index":3,...}`.

tasksource soft choice (ChaosNLI votes, options in source order `entailment, neutral, contradiction`, target `[0.82,0.18,0.0]`):
`label_kind=soft_choice_distribution`, `tier_json=[[id(entailment)],[id(neutral)],[id(contradiction)]]`,
`probabilities_json={"456cf73c…":0.0,"7e2372f4…":0.18,"f9906324…":0.82}`,
`source_label_or_null={"kind":"choice_soft","target":[0.82,0.18,0.0],"source_task":"chaos-mnli-ambiguity/votes",...}`.

Open-Jev `ir-control-v1` score head (raw split `calibration` -> `partition_role` dev, HF split validation):
```json
{"source_id":"ZefanCai/Open-Jev","source_config":"ir-control-v1","source_split":"calibration","source_row_id":"q-4899749a7cbaacbc:8:relevance",
 "source_parent_id":"ir:7dd1306cf238","family":"openjev:ir-control-v1","partition_role":"dev",
 "state_json":"{\"passage\":\"System: Relay-7dd1306cf238\\nProfile: burst\\nStatus: current\\nRetry ceiling: not specified attempts…",
 "instruction_json":"{\"instructions\":\"How completely does this passage answer the query?\",\"type\":\"score\"}",
 "options_json":"{\"11c5363b…\":\"3: Current guidance for the requested system and profile provides both requested values.\",…}",
 "candidate_count":4,"label_kind":"ordinal_score_distance_tiers","instruction_template_id":"openjev.score","license":"cc0-1.0",
 "source_label_or_null":"{\"config\":\"ir-control-v1\",\"expected_level\":1.0,\"kind\":\"score\",\"level_choice_ids\":[\"eecd5150…\",\"23669ece…\",…],\"levels\":[\"0: About a different system…\",\"1: About the requested system but no applicable current answer…\",…],\"target\":[0.0,1.0,0.0,0.0],\"true_level_index\":1,…}"}
```
Open-Jev noul (`target=[1.0,0.0]` over `["no","yes"]` -> p_yes 0, tiers `[[no],[yes]]`):
`source_label_or_null={"config":"ir-control-v1","kind":"noul","options":["no","yes"],"original_line_number":1,"p_yes":0.0,"target":[1.0,0.0],"task":"ir-control-v1","threshold":0.5}`.

samatv agent choice (15 candidates, `family=jev_agent:Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`, split val -> validation):
```json
{"source_row_id":"770e0b2f…","source_parent_id":"149","lineage_key":"nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1:149",
 "state_json":"{\"history\":[{\"payload\":{\"content\":\"You are a customer service agent …","instruction_json":"{\"instructions\":\"Given the current state and available options,\\nwhich option should be selected?\",\"type\":\"choice\"}",
 "options_json":"{\"10c92fd0…\":\"{\\\"description\\\":\\\"Modify scheduled service appointment. …\\\",\\\"name\\\":\\\"modify_service\\\",\\\"parameters\\\":{…}}\",…}",
 "tier_json":"[[\"b3930f28…\"],[\"10c92fd0…\",…14 ids]]","probabilities_json":"{}","candidate_count":15,"label_kind":"agent_choice_target",
 "source_label_or_null":"{\"candidate_ids\":[\"tool::authenticate_user\",…],\"decision_type\":\"tool_choice\",\"kind\":\"agent_choice\",\"labels\":{…\"source_pass_rate\":0.28125…},\"target\":{\"action_name\":\"get_new_service_information\",\"arguments_json\":\"{}\",\"candidate_id\":\"tool::get_new_service_information\",\"label_json\":null},…}"}
```

## 12. Measured behaviour (4-core sandbox)
* Open-Jev: all 12 configs (44 units, 403k decision sets) converted in 25 s; 1,358 `no_trainable_pair` (uniform/tied targets),
  2,068 `duplicate_content_in_split`, 0 leakage. Output 123 MB.
* tasksource / samatv: ~2,300 samatv rows/s/core (conversion), ~10k tasksource rows/s/core; samatv output ~2.2 KB/row
  compressed (zstd) -> ~14 GB for the ~6.2M choice-eligible rows; plan for ~25 GB total output and ~22 GB of source download
  (use `--delete-source` or `--stream` to avoid keeping sources).
* Smoke (2,000 raw rows per unit, remote range reads): see `scripts/data/smoke.sh`.
