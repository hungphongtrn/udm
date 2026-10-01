import json

import pytest
from data_contract_helpers import assert_contract_row

from scrm_data import canonical as C
from scrm_data.schema import SCHEMA, rows_to_table, schemas_equivalent
from scrm_data.sources import openjev, samatv, tasksource
from scrm_data.validate import validate_rows


def _check(res, split):
    assert res.row is not None, res.reason
    assert res.split == split
    assert_contract_row(res.row, split)
    errs, ok = validate_rows([res.row], split, canonical_every=1)
    assert errs == [], errs
    t = rows_to_table([res.row])
    assert schemas_equivalent(t.schema, SCHEMA)
    assert t.num_columns == 26
    return res.row


# ------------------------------------------------------------------ tasksource
def ts_row(**kw):
    r = {"state": "A cat sat.", "kind": "choice", "id": "x:train:1", "options": ["yes it did", "no it did not", "maybe"],
         "target": [1.0, 0.0, 0.0], "question": "Did it sit?", "source": "demo/task", "variant": "direct",
         "split": "train", "group_id": "x:train:1", "question_id": "decision", "license": "cc-by-4.0",
         "license_use": "commercial"}
    r.update(kw)
    return r


def test_tasksource_choice_one_hot():
    row = _check(tasksource.convert(ts_row(), "train"), "train")
    opts = json.loads(row["options_json"])
    assert [opts[c["choice_id"]] for c in row["candidate_rows"]] == ["yes it did", "no it did not", "maybe"]
    tiers = json.loads(row["tier_json"])
    assert tiers[0] == [C.choice_id_of("yes it did")] and len(tiers[1]) == 2
    assert row["label_kind"] == "source_dataset_label" and row["probabilities_json"] == "{}"
    sl = json.loads(row["source_label_or_null"])
    assert sl["kind"] == "choice" and sl["winner"] == [C.choice_id_of("yes it did")] and sl["target"] == [1.0, 0.0, 0.0]
    assert sl["license_use"] == "commercial"
    assert row["family"] == "tasksource:demo/task" and row["partition_role"] == "train"
    assert json.loads(row["state_json"]) == "A cat sat."
    assert json.loads(row["instruction_json"]) == {"instructions": "Did it sit?", "type": "choice"}


def test_tasksource_validation_role_is_dev():
    row = _check(tasksource.convert(ts_row(split="dev"), "validation"), "validation")
    assert row["partition_role"] == "dev" and row["source_split"] == "validation"


def test_tasksource_soft_choice():
    row = _check(tasksource.convert(ts_row(target=[0.2, 0.5, 0.3]), "train"), "train")
    assert row["label_kind"] == "soft_choice_distribution"
    probs = json.loads(row["probabilities_json"])
    assert probs[C.choice_id_of("no it did not")] == 0.5
    tiers = json.loads(row["tier_json"])
    assert [len(t) for t in tiers] == [1, 1, 1]
    assert json.loads(row["source_label_or_null"])["target"] == [0.2, 0.5, 0.3]


def test_tasksource_score_metadata_roundtrip():
    r = ts_row(kind="score", options=["1 (bad)", "2", "3", "4 (good)"], target=[0.0, 0.0625, 0.625, 0.3125])
    row = _check(tasksource.convert(r, "train"), "train")
    assert row["label_kind"] == "ordinal_score_distance_tiers"
    sl = json.loads(row["source_label_or_null"])
    assert sl["levels"] == ["1 (bad)", "2", "3", "4 (good)"]
    assert sl["target"] == [0.0, 0.0625, 0.625, 0.3125]
    assert sl["true_level_index"] == 2
    ids = sl["level_choice_ids"]
    assert [json.loads(row["options_json"])[i] for i in ids] == sl["levels"]
    tiers = json.loads(row["tier_json"])
    assert tiers[0] == [ids[2]] and sorted(tiers[1]) == sorted([ids[1], ids[3]]) and tiers[2] == [ids[0]]
    # original target is recoverable from probabilities_json too
    probs = json.loads(row["probabilities_json"])
    assert [probs[i] for i in ids] == sl["target"]


def test_tasksource_noul_metadata_roundtrip():
    r = ts_row(kind="noul", options=[], target=[0.8], question="Is it true?")
    row = _check(tasksource.convert(r, "train"), "train")
    assert row["label_kind"] == "independent_labels_binary_tiers"
    opts = json.loads(row["options_json"])
    assert sorted(opts.values()) == ["no", "yes"]
    sl = json.loads(row["source_label_or_null"])
    assert sl["target"] == [0.8] and sl["threshold"] == 0.5 and sl["p_yes"] == 0.8
    assert json.loads(row["tier_json"]) == [[C.choice_id_of("yes")], [C.choice_id_of("no")]]
    probs = json.loads(row["probabilities_json"])
    assert probs[C.choice_id_of("yes")] == 0.8 and abs(probs[C.choice_id_of("no")] - 0.2) < 1e-9


def test_tasksource_drops():
    assert tasksource.convert(ts_row(kind="noul", options=[], target=[0.5]), "train").reason == "noul_tie"
    assert tasksource.convert(ts_row(target=[0.3, 0.3, 0.3]), "train").reason == "no_trainable_pair"
    assert tasksource.convert(ts_row(options=["a", "a", "b"], target=[1.0, 0.0, 0.0]), "train").reason == "duplicate_options_conflict"
    r = tasksource.convert(ts_row(options=["a", "a", "b"], target=[0.5, 0.5, 0.0]), "train")
    assert r.row is not None and r.row["candidate_count"] == 2          # agreeing duplicates are collapsed
    assert tasksource.convert(ts_row(options=["a"], target=[1.0]), "train").reason == "fewer_than_2_candidates"
    assert tasksource.convert(ts_row(target=[1.0]), "train").reason == "target_length_mismatch"


def test_tasksource_empty_state_kept():
    row = _check(tasksource.convert(ts_row(state=""), "train"), "train")
    assert json.loads(row["state_json"]) == ""


# ------------------------------------------------------------------ open-jev
def oj_row(**kw):
    r = {"id": "g:1:head", "group_id": "g:1", "split": "train", "source": "demo-control-v1", "kind": "choice",
         "question": "Pick one", "options": ["a: first", "b: second"], "target": [0.0, 1.0],
         "state_json": json.dumps({"z": 1, "a": ["x"]}), "metadata_json": "{}", "record_json": '{"id":"g:1:head"}',
         "original_line_number": 7}
    r.update(kw)
    return r


def test_openjev_choice_and_split_mapping():
    for raw, out in [("train", "train"), ("calibration", "validation"), ("validation", "validation"),
                     ("test", "test"), ("ood", "test")]:
        row = _check(openjev.convert(oj_row(), raw, "demo-control-v1"), out)
        assert row["source_split"] == raw and row["source_config"] == "demo-control-v1"
    row = _check(openjev.convert(oj_row(), "ood", "demo-control-v1"), "test")
    assert row["partition_role"] == "test"
    assert json.loads(row["state_json"]) == {"a": ["x"], "z": 1}
    assert row["state_json"] == '{"a":["x"],"z":1}'
    assert row["raw_record_sha256"] == C.sha256_hex('{"id":"g:1:head"}')
    assert row["license"] == "cc0-1.0" and row["family"] == "openjev:demo-control-v1"


def test_openjev_noul_yes_no():
    r = oj_row(kind="noul", options=["no", "yes"], target=[0.25, 0.75])
    row = _check(openjev.convert(r, "train", "c"), "train")
    sl = json.loads(row["source_label_or_null"])
    assert sl["kind"] == "noul" and sl["target"] == [0.25, 0.75] and sl["p_yes"] == 0.75 and sl["options"] == ["no", "yes"]
    assert json.loads(row["tier_json"])[0] == [C.choice_id_of("yes")]
    # reversed option order still reads p(yes) correctly
    r = oj_row(kind="noul", options=["yes", "no"], target=[0.1, 0.9])
    row = _check(openjev.convert(r, "train", "c"), "train")
    assert json.loads(row["tier_json"])[0] == [C.choice_id_of("no")]
    assert openjev.convert(oj_row(kind="noul", options=["no", "yes"], target=[0.5, 0.5]), "train", "c").reason == "noul_tie"


def test_openjev_score():
    r = oj_row(kind="score", options=["0: low", "1: mid", "2: high"], target=[0.0, 1.0, 0.0])
    row = _check(openjev.convert(r, "validation", "c"), "validation")
    sl = json.loads(row["source_label_or_null"])
    assert sl["levels"] == ["0: low", "1: mid", "2: high"] and sl["true_level_index"] == 1
    ids = sl["level_choice_ids"]
    assert json.loads(row["tier_json"]) == [[ids[1]], sorted([ids[0], ids[2]])]


def test_openjev_uniform_choice_dropped():
    r = oj_row(options=["0", "1", "2"], target=[1 / 3] * 3)
    assert openjev.convert(r, "train", "c").reason == "no_trainable_pair"


def test_openjev_plan_skips_exact_subset():
    units, skipped = openjev.plan(configs=["release-v2-redistributable", "browser-drone-expansion-v1-redistributable"])
    assert skipped == {"release-v2-redistributable": "browser-drone-expansion-v1-redistributable"}
    assert {u.config for u in units} == {"browser-drone-expansion-v1-redistributable"}
    assert len(units) == 5
    units, skipped = openjev.plan(configs=["release-v2-redistributable"])
    assert not skipped and len(units) == 5


# ------------------------------------------------------------------ samatv
def sam_row(**kw):
    r = {
        "id": "abc", "source": "nvidia/Nemotron-SFT-Agentic-v2", "decision_type": "tool_choice", "status": "trainable",
        "content_hash": "h1",
        "state": {"system": "sys", "user_goal": "find news", "environment_json": None,
                  "history": [{"role": "user", "payload_json": json.dumps({"content": "hi"})},
                              {"role": "tool", "payload_json": "not json"}]},
        "candidates": [
            {"id": "tool::a", "name": "get_news", "description": "News", "parameters_json": '{"type":"object"}', "metadata_json": "{}"},
            {"id": "tool::b", "name": "get_stocks", "description": None, "parameters_json": "{}", "metadata_json": "{}"},
            {"id": "tool::c", "name": "search", "description": "Search", "parameters_json": "", "metadata_json": "{}"},
        ],
        "target": {"candidate_id": "tool::b", "action_name": "get_stocks", "arguments_json": "{}", "label_json": None},
        "ordered_targets": [], "labels": {"complete": None, "trajectory_success": True, "value_target": 0.5, "reward": None,
                                           "score": None, "source_pass_rate": None, "outcome_evidence": "x"},
        "provenance": {"dataset_id": "nvidia/Nemotron-SFT-Agentic-v2", "dataset_revision": "r", "source_config": "default",
                       "raw_split": "search", "source_row_index": 3, "source_identity": "i", "trajectory_id": "traj9",
                       "step_index": 2, "decision_ordinal": 0, "adapter_version": "2", "license_metadata_json": '{"license":"cc-by-4.0"}',
                       "source_metadata_json": "{}"},
        "training": {"choice_eligible": True, "supervision_evidence": "verified_demonstration", "quality_weight": 1.0},
        "training_split": "val",
    }
    r.update(kw)
    return r


def test_samatv_default_agent_choice():
    row = _check(samatv.convert_default(sam_row(), "validation"), "validation")
    assert row["label_kind"] == "agent_choice_target" and row["probabilities_json"] == "{}"
    sl = json.loads(row["source_label_or_null"])
    assert sl["kind"] == "agent_choice" and sl["decision_type"] == "tool_choice"
    assert sl["target"]["candidate_id"] == "tool::b" and sl["labels"]["value_target"] == 0.5
    assert sl["candidate_ids"] == ["tool::a", "tool::b", "tool::c"]
    tiers = json.loads(row["tier_json"])
    texts = json.loads(row["options_json"])
    assert json.loads(texts[tiers[0][0]])["name"] == "get_stocks" and len(tiers[1]) == 2
    state = json.loads(row["state_json"])
    assert state["history"][0] == {"role": "user", "payload": {"content": "hi"}}
    assert state["history"][1]["payload"] == "not json" and "environment" not in state
    assert row["family"] == "jev_agent:Nemotron-SFT-Agentic-v2/search"
    assert row["lineage_key"] == "nvidia/Nemotron-SFT-Agentic-v2:traj9" and row["source_parent_id"] == "traj9"
    assert row["partition_role"] == "dev" and row["license"] == "cc-by-4.0"


def test_samatv_drops():
    r = sam_row()
    r["training"] = dict(r["training"], choice_eligible=False)
    assert samatv.convert_default(r, "train").reason == "not_choice_eligible"
    r = sam_row(target={"candidate_id": "tool::zzz", "action_name": "x", "arguments_json": None, "label_json": None})
    assert samatv.convert_default(r, "train").reason == "target_not_in_candidates"
    r = sam_row(candidates=sam_row()["candidates"][:1])
    assert samatv.convert_default(r, "train").reason == "fewer_than_2_candidates"
    # identical candidate definitions collapse; target maps to the kept one
    c = sam_row()["candidates"]
    c.append(dict(c[1], id="tool::b2"))
    r = sam_row(candidates=c, target={"candidate_id": "tool::b2", "action_name": "get_stocks", "arguments_json": None, "label_json": None})
    res = samatv.convert_default(r, "train")
    assert res.row["candidate_count"] == 3
    tiers = json.loads(res.row["tier_json"])
    assert json.loads(json.loads(res.row["options_json"])[tiers[0][0]])["name"] == "get_stocks"


def test_samatv_clean_split_by_content_hash():
    raw = {"id": "i1", "source": "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
           "state": {"system": None, "user_goal": "goal", "environment": None, "history": [{"role": "user", "content": "\"x\""}]},
           "question": samatv.QUESTION, "question_type": "choice",
           "answer_options": [{"type": "action", "label": "a", "description": "d", "schema_json": "{}"},
                              {"type": "action", "label": "b", "description": "e", "schema_json": "{}"}],
           "target_index": 1}
    res = samatv.convert_clean(raw, "train")
    row = _check(res, res.split)
    assert res.split in ("train", "validation", "test")
    assert row["source_config"] == "general-clean-50k"
    assert samatv.convert_clean(raw, "train").split == res.split          # deterministic
    tiers = json.loads(row["tier_json"])
    assert json.loads(json.loads(row["options_json"])[tiers[0][0]])["name"] == "b"


def test_schema_has_26_columns_in_order():
    assert len(SCHEMA) == 26 and SCHEMA.names[0] == "decision_set_id" and SCHEMA.names[-1] == "candidate_rows"
