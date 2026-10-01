"""Format-conformance tests: every source / config / label kind is converted with the REAL converter from a
hand-written raw fixture shaped like the real source row, and the output is checked against the exact
udm-massive-typed schema and hand-computed expectations. Offline."""
import collections
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scrm_data import canonical as C
from scrm_data.schema import rows_to_table
from scrm_data.sources import openjev, samatv, tasksource

# Independent reference constant (not imported from scrm_data.schema): names, order, types of the existing shards.
REF = [
    ("decision_set_id", pa.string()), ("source_id", pa.string()), ("source_revision", pa.string()),
    ("source_config", pa.string()), ("source_split", pa.string()), ("source_row_id", pa.string()),
    ("source_parent_id", pa.string()), ("family", pa.string()), ("partition_role", pa.string()),
    ("lineage_key", pa.string()), ("state_json", pa.string()), ("instruction_json", pa.string()),
    ("options_json", pa.string()), ("tier_json", pa.string()), ("probabilities_json", pa.string()),
    ("candidate_count", pa.int32()), ("label_kind", pa.string()), ("source_label_or_null", pa.string()),
    ("instruction_template_id", pa.string()), ("option_permutation_seed_or_null", pa.string()),
    ("raw_record_sha256", pa.string()), ("license", pa.string()), ("known_pretraining_overlap", pa.string()),
    ("contamination_status", pa.string()), ("quarantine_reason_or_null", pa.string()),
    ("candidate_rows", pa.list_(pa.struct([("record_id", pa.string()), ("decision_set_id", pa.string()),
                                           ("choice_id", pa.string()), ("candidate_order", pa.int32()),
                                           ("normalized_text_sha256", pa.string())]))),
]


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def conform(res, split, role, texts, tiers, label_kind, probs, candidate_order=None):
    """Common assertions. `texts` = expected candidate texts in canonical order; `tiers` = expected tiers as
    lists of TEXTS (best first)."""
    row = res.row
    assert row is not None, res.reason
    assert res.split == split and row["partition_role"] == role
    # exact Arrow schema
    t = rows_to_table([row])
    assert [(f.name, f.type) for f in t.schema] == REF or \
        [(f.name, str(f.type)) for f in t.schema] == [(n, str(ty)) for n, ty in REF]
    assert t.schema.field("candidate_count").type == pa.int32()
    assert t.schema.field("candidate_rows").type.value_type.field("candidate_order").type == pa.int32()
    # options / hash recipe
    opts = json.loads(row["options_json"])
    assert opts == {sha(x): x for x in texts}
    assert row["options_json"] == json.dumps(opts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # tiers: exact and a partition
    exp_tiers = [sorted(sha(x) for x in tier) for tier in tiers]
    got = json.loads(row["tier_json"])
    assert got == exp_tiers
    assert sorted(c for tier in got for c in tier) == sorted(opts)
    # probabilities, kind
    assert json.loads(row["probabilities_json"]) == ({sha(k): v for k, v in probs.items()} if probs else {})
    assert row["label_kind"] == label_kind
    # candidates
    assert row["candidate_count"] == len(opts) == len(row["candidate_rows"]) == len(texts)
    assert [c["candidate_order"] for c in row["candidate_rows"]] == list(range(len(texts)))
    assert [c["choice_id"] for c in row["candidate_rows"]] == [sha(x) for x in texts]
    assert all(c["normalized_text_sha256"] == c["choice_id"] and c["decision_set_id"] == row["decision_set_id"]
               for c in row["candidate_rows"])
    # byte-stable canonical JSON
    for col in ("state_json", "instruction_json", "tier_json", "probabilities_json", "source_label_or_null"):
        assert C.dumps(json.loads(row[col])) == row[col]
    assert C.dumps(json.loads(row["source_label_or_null"])) == row["source_label_or_null"]
    return row


def sl(row):
    return json.loads(row["source_label_or_null"])


# ---------------------------------------------------------------- tasksource
def ts(**kw):
    r = {"state": "My body cast a shadow over the grass. What was the cause of this?", "kind": "choice",
         "id": "super_glue-copa-1:train:0", "options": ["The grass was cut.", "The sun was rising.", "It rained."],
         "target": [0.0, 1.0, 0.0], "question": "Choose the criterion that best answers the question.",
         "source": "super_glue/copa", "variant": "direct", "split": "train",
         "group_id": "super_glue-copa-1:train:0", "question_id": "decision", "license": "cc-by-4.0",
         "license_use": "commercial"}
    r.update(kw)
    return r


def test_tasksource_choice_single_winner():
    res = tasksource.convert(ts(), "train")
    row = conform(res, "train", "train", ["The grass was cut.", "The sun was rising.", "It rained."],
                  [["The sun was rising."], ["The grass was cut.", "It rained."]], "source_dataset_label", None)
    d = sl(row)
    assert d["kind"] == "choice" and d["target"] == [0.0, 1.0, 0.0] and d["winner"] == [sha("The sun was rising.")]
    assert row["instruction_json"] == '{"instructions":"Choose the criterion that best answers the question.","type":"choice"}'
    assert row["state_json"] == json.dumps(ts()["state"], ensure_ascii=False)
    assert row["family"] == "tasksource:super_glue/copa" and row["source_split"] == "train"


def test_tasksource_soft_choice():
    res = tasksource.convert(ts(options=["entailment", "neutral", "contradiction"], target=[0.82, 0.18, 0.0],
                                kind="choice"), "train")
    row = conform(res, "train", "train", ["entailment", "neutral", "contradiction"],
                  [["entailment"], ["neutral"], ["contradiction"]], "soft_choice_distribution",
                  {"entailment": 0.82, "neutral": 0.18, "contradiction": 0.0})
    assert sl(row)["kind"] == "choice_soft" and sl(row)["target"] == [0.82, 0.18, 0.0]


def test_tasksource_score_distance_tiers_true2_of_5():
    levels = ["0", "1", "2", "3", "4"]
    res = tasksource.convert(ts(kind="score", options=levels, target=[0.0, 0.0, 1.0, 0.0, 0.0],
                                question="Rate it."), "train")
    row = conform(res, "train", "train", levels, [["2"], ["1", "3"], ["0", "4"]], "ordinal_score_distance_tiers",
                  {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0})
    d = sl(row)
    assert d["kind"] == "score" and d["levels"] == levels and d["level_choice_ids"] == [sha(x) for x in levels]
    assert d["target"] == [0.0, 0.0, 1.0, 0.0, 0.0] and d["true_level_index"] == 2 and d["expected_level"] == 2.0


def test_tasksource_score_soft_mean_rating():
    res = tasksource.convert(ts(kind="score", options=["1", "2", "3", "4"], target=[0.0, 0.0, 0.6, 0.4]), "train")
    row = res.row
    d = sl(row)
    assert d["true_level_index"] == 2 and abs(d["expected_level"] - 2.4) < 1e-9
    assert json.loads(row["tier_json"]) == [[sha("3")], sorted([sha("2"), sha("4")]), [sha("1")]]


def test_tasksource_noul_raw_probability_preserved():
    res = tasksource.convert(ts(kind="noul", options=[], target=[0.8], question="Is it true?"), "validation")
    row = conform(res, "validation", "dev", ["no", "yes"], [["yes"], ["no"]], "independent_labels_binary_tiers",
                  {"no": 0.2, "yes": 0.8})
    d = sl(row)
    assert d["kind"] == "noul" and d["target"] == [0.8] and d["threshold"] == 0.5 and d["p_yes"] == 0.8
    res = tasksource.convert(ts(kind="noul", options=[], target=[0.1]), "test")
    conform(res, "test", "test", ["no", "yes"], [["no"], ["yes"]], "independent_labels_binary_tiers",
            {"no": 0.9, "yes": 0.1})


def test_tasksource_drops_counted():
    drops = collections.Counter()
    cases = [ts(options=["only"], target=[1.0]),                         # fewer than 2 candidates
             ts(target=[0.3, 0.3, 0.3]),                                  # no trainable pair
             ts(kind="noul", options=[], target=[0.5])]                   # noul tie (all mass equal)
    for c in cases:
        r = tasksource.convert(c, "train")
        assert r.row is None
        drops[r.reason] += 1
    assert drops == {"fewer_than_2_candidates": 1, "no_trainable_pair": 1, "noul_tie": 1}


# ---------------------------------------------------------------- Open-Jev
def oj(**kw):
    r = {"id": "cfg:1:head", "group_id": "cfg:1", "split": "train", "source": "demo-control-v1", "kind": "choice",
         "question": "Pick the category.", "options": ["billing: bills", "account: logins", "other: rest"],
         "target": [0.0, 1.0, 0.0], "state_json": '{"b":2,"a":"x"}', "metadata_json": '{"secret":1}',
         "record_json": '{"id":"cfg:1:head"}', "original_line_number": 3}
    r.update(kw)
    return r


@pytest.mark.parametrize("raw,split,role", [("train", "train", "train"), ("calibration", "validation", "dev"),
                                            ("validation", "validation", "dev"), ("test", "test", "test"),
                                            ("ood", "test", "test")])
def test_openjev_choice_all_splits(raw, split, role):
    res = openjev.convert(oj(), raw, "demo-control-v1")
    texts = ["billing: bills", "account: logins", "other: rest"]
    row = conform(res, split, role, texts, [["account: logins"], ["billing: bills", "other: rest"]],
                  "source_dataset_label", None)
    assert row["source_split"] == raw and row["state_json"] == '{"a":"x","b":2}'
    assert "secret" not in json.dumps(row)               # privileged metadata never copied
    assert sl(row)["winner"] == [sha("account: logins")] and sl(row)["target"] == [0.0, 1.0, 0.0]


def test_openjev_choice_soft_tied_optimal_moves():
    res = openjev.convert(oj(options=["0", "1", "2", "3"], target=[0.5, 0.5, 0.0, 0.0]), "train", "c")
    row = conform(res, "train", "train", ["0", "1", "2", "3"], [["0", "1"], ["2", "3"]], "soft_choice_distribution",
                  {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0})
    assert sl(row)["target"] == [0.5, 0.5, 0.0, 0.0]


def test_openjev_noul_yes_no():
    res = openjev.convert(oj(kind="noul", options=["no", "yes"], target=[0.25, 0.75]), "test", "c")
    row = conform(res, "test", "test", ["no", "yes"], [["yes"], ["no"]], "independent_labels_binary_tiers",
                  {"no": 0.25, "yes": 0.75})
    d = sl(row)
    assert d["kind"] == "noul" and d["target"] == [0.25, 0.75] and d["threshold"] == 0.5 and d["p_yes"] == 0.75


def test_openjev_score_ordinal():
    levels = ["0: low", "1: mid", "2: high", "3: top"]
    res = openjev.convert(oj(kind="score", options=levels, target=[0.0, 0.0, 1.0, 0.0]), "calibration", "c")
    row = conform(res, "validation", "dev", levels, [["2: high"], ["1: mid", "3: top"], ["0: low"]],
                  "ordinal_score_distance_tiers", {x: p for x, p in zip(levels, [0.0, 0.0, 1.0, 0.0])})
    d = sl(row)
    assert d["levels"] == levels and d["level_choice_ids"] == [sha(x) for x in levels]
    assert d["true_level_index"] == 2 and d["expected_level"] == 2.0 and d["target"] == [0.0, 0.0, 1.0, 0.0]


def test_openjev_drops_counted():
    drops = collections.Counter()
    for c in [oj(options=["a"], target=[1.0]), oj(options=["0", "1", "2"], target=[1 / 3] * 3),
              oj(kind="noul", options=["no", "yes"], target=[0.5, 0.5])]:
        r = openjev.convert(c, "train", "c")
        assert r.row is None
        drops[r.reason] += 1
    assert drops == {"fewer_than_2_candidates": 1, "no_trainable_pair": 1, "noul_tie": 1}


# ---------------------------------------------------------------- samatv
def sam(**kw):
    r = {
        "id": "dec1", "source": "nvidia/Nemotron-SFT-Agentic-v2", "decision_type": "tool_choice",
        "status": "trainable", "unsupported_reason": None, "content_hash": "ch1",
        "state": {"system": "You are an agent.", "user_goal": "Find news.", "environment_json": None,
                  "history": [{"role": "user", "payload_json": '{"content":"hi"}'}]},
        "candidates": [
            {"id": "tool::news", "name": "news", "description": "Get news", "parameters_json": '{"type":"object"}', "metadata_json": "{}"},
            {"id": "tool::stocks", "name": "stocks", "description": "Get stocks", "parameters_json": "{}", "metadata_json": "{}"},
            {"id": "tool::search", "name": "search", "description": None, "parameters_json": "{}", "metadata_json": "{}"}],
        "target": {"candidate_id": "tool::stocks", "action_name": "stocks", "arguments_json": "{}", "label_json": None},
        "ordered_targets": [],
        "labels": {"complete": None, "trajectory_success": None, "value_target": None, "reward": None,
                   "score": None, "source_pass_rate": None, "outcome_evidence": None},
        "provenance": {"dataset_id": "nvidia/Nemotron-SFT-Agentic-v2", "dataset_revision": "r1", "source_config": "default",
                       "raw_split": "tool_calling", "source_row_index": 5, "source_identity": "i", "trajectory_id": "t1",
                       "step_index": 1, "decision_ordinal": 0, "adapter_version": "2.0.0",
                       "license_metadata_json": '{"license":"cc-by-4.0"}', "source_metadata_json": "{}"},
        "training": {"choice_eligible": True, "completion_eligible": False, "value_eligible": False,
                     "score_eligible": False, "boolean_eligible": False, "arguments_eligible": True, "use_for_bc": True,
                     "use_for_value": False, "supervision_evidence": "verified_demonstration",
                     "choice_ineligible_reason": None, "quality_weight": 1.0},
        "training_split": "test",
    }
    r.update(kw)
    return r


def test_samatv_default_agent_choice_eligible():
    raw = sam()
    res = samatv.convert_default(raw, "test")
    texts = ['{"description":"Get news","name":"news","parameters":{"type":"object"}}',
             '{"description":"Get stocks","name":"stocks"}', '{"name":"search"}']
    row = conform(res, "test", "test", texts, [[texts[1]], [texts[0], texts[2]]], "agent_choice_target", None)
    d = sl(row)
    assert d["kind"] == "agent_choice" and d["decision_type"] == "tool_choice"
    assert d["target"] == raw["target"] and d["labels"] == raw["labels"]
    assert d["candidate_ids"] == ["tool::news", "tool::stocks", "tool::search"]
    assert json.loads(row["state_json"]) == {"system": "You are an agent.", "user_goal": "Find news.",
                                             "history": [{"role": "user", "payload": {"content": "hi"}}]}
    assert row["family"] == "jev_agent:Nemotron-SFT-Agentic-v2/tool_calling"
    assert row["lineage_key"] == "nvidia/Nemotron-SFT-Agentic-v2:t1" and row["license"] == "cc-by-4.0"


@pytest.mark.parametrize("ts_,split,role", [("train", "train", "train"), ("val", "validation", "dev"), ("test", "test", "test")])
def test_samatv_training_split_mapping(ts_, split, role):
    res = samatv.convert_default(sam(training_split=ts_), "train")
    assert res.split == split and res.row["partition_role"] == role


def test_samatv_ineligible_and_degenerate_dropped_and_counted():
    drops = collections.Counter()
    inel = sam()
    inel["training"] = dict(inel["training"], choice_eligible=False)
    one = sam(candidates=sam()["candidates"][:1])
    missing = sam(target={"candidate_id": "tool::nope", "action_name": "x", "arguments_json": None, "label_json": None})
    for c in (inel, one, missing):
        r = samatv.convert_default(c, "train")
        assert r.row is None
        drops[r.reason] += 1
    assert drops == {"not_choice_eligible": 1, "fewer_than_2_candidates": 1, "target_not_in_candidates": 1}


def test_samatv_clean50k_agent_choice():
    raw = {"id": "c1", "source": "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
           "state": {"system": None, "user_goal": "goal", "environment": None,
                     "history": [{"role": "user", "content": '"hello"'}]},
           "question": samatv.QUESTION, "question_type": "choice",
           "answer_options": [{"type": "action", "label": "a", "description": "da", "schema_json": "{}"},
                              {"type": "action", "label": "b", "description": "db", "schema_json": "{}"}],
           "target_index": 1}
    res = samatv.convert_clean(raw, "train")
    texts = ['{"description":"da","name":"a"}', '{"description":"db","name":"b"}']
    row = conform(res, res.split, {"train": "train", "validation": "dev", "test": "test"}[res.split], texts,
                  [[texts[1]], [texts[0]]], "agent_choice_target", None)
    assert row["source_config"] == "general-clean-50k"
    assert json.loads(row["state_json"])["history"] == [{"role": "user", "payload": "hello"}]


# ---------------------------------------------------------------- trainer round trip
def test_round_trip_through_trainer_loader(tmp_path):
    from scrm.render import parse_row

    results = [
        tasksource.convert(ts(), "train"),
        tasksource.convert(ts(options=["e", "n", "c"], target=[0.82, 0.18, 0.0], id="x2"), "train"),
        tasksource.convert(ts(kind="score", options=list("01234"), target=[0, 0, 1.0, 0, 0], id="x3"), "train"),
        tasksource.convert(ts(kind="noul", options=[], target=[0.8], id="x4"), "train"),
        openjev.convert(oj(), "ood", "c"),
        openjev.convert(oj(kind="noul", options=["no", "yes"], target=[0.25, 0.75], id="o2"), "train", "c"),
        openjev.convert(oj(kind="score", options=["0: a", "1: b", "2: c"], target=[0, 1.0, 0], id="o3"), "train", "c"),
        samatv.convert_default(sam(), "test"),
    ]
    rows = [r.row for r in results]
    assert all(r is not None for r in rows)
    path = tmp_path / "train-demo-00000-of-00001.parquet"
    pq.write_table(rows_to_table(rows), path)
    back = pq.read_table(path).to_pylist()
    assert len(back) == len(rows)
    for r in back:
        ex = parse_row(r)
        assert ex is not None, f"trainer dropped {r['family']} {r['label_kind']}"
        assert ex.decision_set_id == r["decision_set_id"] and ex.family == r["family"]
        assert len(ex.texts) == r["candidate_count"] and min(ex.tiers) == 0 and len(set(ex.tiers)) >= 2
        order = [c["choice_id"] for c in sorted(r["candidate_rows"], key=lambda c: c["candidate_order"])]
        assert ex.choice_ids == order
        top = {cid for cid, t in zip(ex.choice_ids, ex.tiers) if t == 0}
        assert top == set(json.loads(r["tier_json"])[0])
