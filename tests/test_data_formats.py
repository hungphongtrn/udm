"""Format-conformance tests. Every source/config/label-kind path runs the REAL converter on a raw fixture shaped
like the real upstream row (tests/fixtures/*.json), calls `assert_contract_row` on the output and compares exact,
hand-computed expectations (tiers, probabilities, metadata, split). Offline."""
import collections
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_contract_helpers import assert_contract_row, sha

from scrm_data import build
from scrm_data.schema import rows_to_table
from scrm_data.sources import openjev, samatv, tasksource

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name):
    with open(os.path.join(FX, name)) as f:
        return json.load(f)


TS = load("tasksource_rows.json")
OJ = load("openjev_rows.json")
SAM = load("samatv_rows.json")


def expect_checks(row, split, exp, texts):
    parsed = assert_contract_row(row, split)
    assert [c["choice_id"] for c in row["candidate_rows"]] == [sha(t) for t in texts]
    assert parsed["options_json"] == {sha(t): t for t in texts}
    assert parsed["tier_json"] == [sorted(sha(t) for t in tier) for tier in exp["tiers"]]
    assert row["label_kind"] == exp["label_kind"]
    want = {sha(k): v for k, v in exp["probs"].items()} if exp.get("probs") else {}
    assert parsed["probabilities_json"] == want
    return parsed


# ------------------------------------------------------------------------------ tasksource
@pytest.mark.parametrize("fx", TS, ids=[f"{f['raw']['kind']}-{f['raw']['source']}" for f in TS])
def test_tasksource_fixture_rows(fx):
    raw, exp = fx["raw"], fx["expect"]
    res = tasksource.convert(raw, fx["split"])
    assert res.row is not None, res.reason
    assert res.split == exp["split"]
    texts = ["no", "yes"] if raw["kind"] == "noul" else raw["options"]
    parsed = expect_checks(res.row, exp["split"], exp, texts)
    row = res.row
    assert row["source_id"] == "tasksource/tasksource-jev-typed-decisions" and row["source_split"] == fx["split"]
    assert row["source_row_id"] == raw["id"] and row["source_parent_id"] == raw["group_id"] == row["lineage_key"]
    assert row["family"] == f"tasksource:{raw['source']}" and row["license"] == raw["license"]
    assert parsed["state_json"] == raw["state"]
    assert parsed["instruction_json"] == {"instructions": raw["question"], "type": raw["kind"]}
    sl = parsed["source_label_or_null"]
    for k, v in exp["sl"].items():
        if k == "winner":
            v = [sha(x) for x in v]
        assert sl[k] == v, k
    assert sl["license_use"] == raw["license_use"] and sl["variant"] == raw["variant"]
    if raw["kind"] == "score":
        assert sl["levels"] == raw["options"] and sl["target"] == raw["target"]
        assert sl["level_choice_ids"] == [sha(x) for x in raw["options"]]


def test_tasksource_empty_state_is_kept_as_empty_json_string():
    fx = next(f for f in TS if f["raw"]["state"] == "")
    row = tasksource.convert(fx["raw"], "train").row
    assert row["state_json"] == '""'


def test_tasksource_noul_probability_keeps_raw_unnormalised_target():
    fx = next(f for f in TS if f["raw"]["target"] == [0.7])
    row = tasksource.convert(fx["raw"], "train").row
    assert json.loads(row["source_label_or_null"])["target"] == [0.7]
    assert json.loads(row["probabilities_json"]) == {sha("no"): 0.3, sha("yes"): 0.7}


# ------------------------------------------------------------------------------ Open-Jev (every config format)
@pytest.mark.parametrize("fx", OJ, ids=[f"{f['config']}-{f['raw']['source']}-{f['raw']['kind']}-{i}" for i, f in enumerate(OJ)])
def test_openjev_fixture_rows(fx):
    raw, exp = fx["raw"], fx["expect"]
    res = openjev.convert(raw, fx["raw_split"], fx["config"])
    assert res.row is not None, res.reason
    assert res.split == exp["split"]
    texts = ["no", "yes"] if raw["kind"] == "noul" else raw["options"]
    parsed = expect_checks(res.row, exp["split"], exp, texts)
    row = res.row
    assert row["source_id"] == "ZefanCai/Open-Jev" and row["source_config"] == fx["config"]
    assert row["source_split"] == fx["raw_split"]                      # raw split name is preserved (ood/calibration)
    assert row["source_row_id"] == raw["id"] and row["lineage_key"] == raw["group_id"]
    assert row["family"] == f"openjev:{raw['source']}" and row["license"] == "cc0-1.0"
    assert row["raw_record_sha256"] == sha(raw["record_json"])
    assert parsed["state_json"] == json.loads(raw["state_json"])
    assert parsed["instruction_json"] == {"instructions": raw["question"], "type": raw["kind"]}
    assert "teacher" not in json.dumps(row)                            # privileged metadata is never copied
    sl = parsed["source_label_or_null"]
    assert sl["target"] == raw["target"] and sl["config"] == fx["config"] and sl["task"] == raw["source"]
    if raw["kind"] == "score":
        tgt = raw["target"]
        assert sl["levels"] == raw["options"] and sl["level_choice_ids"] == [sha(x) for x in raw["options"]]
        assert sl["true_level_index"] == int(np.argmax(tgt))
        assert abs(sl["expected_level"] - sum(i * p for i, p in enumerate(tgt))) < 1e-9
    if raw["kind"] == "noul":
        assert sl["threshold"] == 0.5 and sl["p_yes"] == raw["target"][1] and sl["options"] == ["no", "yes"]


def test_openjev_covers_every_config_and_kind():
    assert {f["config"] for f in OJ} == set(openjev.CONFIGS)
    assert {(f["raw"]["kind"]) for f in OJ} == {"choice", "score", "noul"}
    assert {f["raw_split"] for f in OJ} >= {"train", "calibration", "validation", "test", "ood"}


# ------------------------------------------------------------------------------ samatv
CAND_TEXTS = [
    '{"description":"Verify user identity.","name":"authenticate_user","parameters":{"properties":{"email":{"type":"string"}},"type":"object"}}',
    '{"description":"Return new service details.","name":"get_new_service_information"}',
    '{"name":"transfer_to_human_agent"}',
]


def test_samatv_default_eligible_agent_choice():
    raw = SAM["eligible"]
    res = samatv.convert_default(raw, "validation")
    assert res.split == "validation"
    exp = {"tiers": [[CAND_TEXTS[1]], [CAND_TEXTS[0], CAND_TEXTS[2]]], "label_kind": "agent_choice_target", "probs": None}
    parsed = expect_checks(res.row, "validation", exp, CAND_TEXTS)
    sl = parsed["source_label_or_null"]
    assert sl["kind"] == "agent_choice" and sl["decision_type"] == "tool_choice"
    assert sl["target"] == raw["target"] and sl["labels"] == raw["labels"]
    assert sl["candidate_ids"] == [c["id"] for c in raw["candidates"]]
    assert sl["provenance"]["trajectory_id"] == "149" and sl["supervision_evidence"] == "explicit_expected_action"
    assert parsed["state_json"] == {"system": "You are a customer service agent.", "user_goal": "I need info on new service.",
                                    "history": [{"role": "user", "payload": {"content": "Hello"}}]}
    assert parsed["instruction_json"] == {"instructions": samatv.QUESTION, "type": "choice"}
    row = res.row
    assert row["family"] == "jev_agent:Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
    assert row["lineage_key"] == raw["provenance"]["dataset_id"] + ":149" and row["source_parent_id"] == "149"
    assert row["source_split"] == "validation" and row["source_row_id"] == "770e0b2f" and row["license"] == "cc-by-4.0"


def test_samatv_ineligible_and_single_candidate_dropped_and_counted():
    drops = collections.Counter()
    for k in ("ineligible", "single_candidate"):
        r = samatv.convert_default(SAM[k], "train")
        assert r.row is None
        drops[r.reason] += 1
    assert drops == {"not_choice_eligible": 1, "fewer_than_2_candidates": 1}


def test_samatv_clean50k_agent_choice():
    raw = SAM["clean50k"]
    res = samatv.convert_clean(raw, "train")
    texts = ['{"description":"Live headlines.","name":"get_news_headlines","parameters":{"type":"object"}}',
             '{"description":"Top gainers.","name":"get_special_tickers"}']
    exp = {"tiers": [[texts[1]], [texts[0]]], "label_kind": "agent_choice_target", "probs": None}
    parsed = expect_checks(res.row, res.split, exp, texts)
    assert res.split in ("train", "validation", "test") and res.row["source_config"] == "general-clean-50k"
    assert parsed["state_json"] == {"user_goal": "Plan a quiet art retreat.",
                                    "history": [{"role": "user", "payload": {"content": "Hi"}}]}
    assert parsed["source_label_or_null"]["target"] == {"target_index": 1}


# ------------------------------------------------------------------------------ converter-level drop cases
def test_drop_cases_are_dropped_and_counted():
    base = TS[0]["raw"]
    noul = TS[4]["raw"]
    cases = {
        "fewer_than_2_candidates": dict(base, options=["x"], target=[1.0]),
        "no_trainable_pair": dict(base, options=["a", "b", "c"], target=[1 / 3] * 3),
        "noul_tie": dict(noul, target=[0.5]),
    }
    drops = collections.Counter()
    for reason, raw in cases.items():
        r = tasksource.convert(raw, "train")
        assert r.row is None and r.reason == reason
        drops[r.reason] += 1
    assert sum(drops.values()) == 3
    oj = OJ[0]["raw"]
    assert openjev.convert(dict(oj, options=["a"], target=[1.0]), "train", "c").reason == "fewer_than_2_candidates"
    assert openjev.convert(dict(oj, options=["a", "b"], target=[0.5, 0.5]), "train", "c").reason == "no_trainable_pair"
    ny = next(f["raw"] for f in OJ if f["raw"]["kind"] == "noul")
    assert openjev.convert(dict(ny, target=[0.5, 0.5]), "train", "c").reason == "noul_tie"


# ------------------------------------------------------------------------------ build path + trainer round trip
OJ_SCHEMA = pa.schema([("id", pa.string()), ("group_id", pa.string()), ("split", pa.string()), ("source", pa.string()),
                       ("kind", pa.string()), ("question", pa.string()), ("options", pa.list_(pa.string())),
                       ("target", pa.list_(pa.float64())), ("state_json", pa.string()), ("metadata_json", pa.string()),
                       ("record_json", pa.string()), ("original_line_number", pa.int64())])
TS_SCHEMA = pa.schema([("state", pa.string()), ("kind", pa.string()), ("id", pa.string()),
                       ("options", pa.list_(pa.string())), ("target", pa.list_(pa.float64())),
                       ("question", pa.string()), ("source", pa.string()), ("variant", pa.string()),
                       ("split", pa.string()), ("group_id", pa.string()), ("question_id", pa.string()),
                       ("license", pa.string()), ("license_use", pa.string())])


def _write(path, rows, schema):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_trainer_round_trip_through_real_build_path(tmp_path):
    from scrm.collator import collate
    from scrm.config import load_config
    from scrm.data import EvalSet, select_eval_rows
    from scrm.render import Renderer
    from scrm.tiny import make_tiny_tokenizer
    from scrm.tokens import CAND_END, prepare_tokenizer

    root, out = tmp_path / "src", tmp_path / "out"
    # tasksource: all fixture rows written into the file of their split (+ an empty-state row etc.)
    for sp in ("train", "validation", "test"):
        rows = [f["raw"] for f in TS if f["split"] == sp]
        _write(str(root / f"data/{sp}-00000-of-00001.parquet" if sp != "train" else root / "data/train-00000-of-00013.parquet"),
               rows, TS_SCHEMA)
    by = collections.defaultdict(list)
    for f in OJ:
        by[(f["config"], f["raw_split"])].append(f["raw"])
    cfgs = sorted({c for c, _ in by})
    for cfg in cfgs:
        for sp in openjev.RAW_SPLITS:                       # every config has all five split files (some empty)
            _write(str(root / f"data/{cfg}/{sp}-00000-of-00001.parquet"), by.get((cfg, sp), []), OJ_SCHEMA)
    assert build.main(["--source", "tasksource", "openjev", "--out", str(out), "--workers", "2", "--local-root", str(root),
                       "--openjev-configs", *cfgs, "--unit-bytes", "0",
                       "--unit-filter", "(^(train-00000-of-00013|validation-00000-of-00001|test-00000-of-00001)$|__)",
                       ]) == 0
    # samatv rows go through the converter and are written as shards next to the build output
    sam_rows = collections.defaultdict(list)
    for k, fn in (("eligible", lambda r: samatv.convert_default(r, "validation")),
                  ("clean50k", lambda r: samatv.convert_clean(r, "train"))):
        res = fn(SAM[k])
        sam_rows[res.split].append(res.row)
    for sp, rows in sam_rows.items():
        pq.write_table(rows_to_table(rows), out / "data" / f"{sp}-samatv-00000-of-00001.parquet")
    for sp in ("train", "validation", "test"):                    # make sure every split has at least one file
        assert list((out / "data").glob(f"{sp}-*.parquet")), sp

    tok, _, _ = prepare_tokenizer(make_tiny_tokenizer())
    cfg = load_config(None, [f"data.local_dir={out}", "data.streaming=false", "data.num_workers=0",
                             "data.render.max_len=4096", "data.render.cand_max_tokens=48",
                             "data.render.max_candidates=64"])
    renderer = Renderer(tok, cfg["data"]["render"])
    end_id = tok.convert_tokens_to_ids(CAND_END)
    total = 0
    families = set()
    for sp in ("train", "validation", "test"):
        rows = select_eval_rows(cfg["data"], sp, {}, None, None, streaming=False)
        assert rows
        es = EvalSet(rows, renderer, cfg["data"])
        assert es.n_dropped == 0 and len(es.items) == len(rows)
        for row, it in zip(sorted(rows, key=lambda r: r["decision_set_id"]),
                           sorted(es.items, key=lambda i: i.decision_set_id)):
            assert row["decision_set_id"] == it.decision_set_id
            assert int((it.input_ids == end_id).sum()) == row["candidate_count"]       # every candidate rendered
            assert len(it.cand_pos) == row["candidate_count"]
            families.add(it.family)
        for b in es.batches():
            assert b["pair_mask"].flatten(1).any(1).all()                              # trainable pairs exist
        total += len(rows)
    # release-v2 is skipped because its superset (browser-drone expansion) is converted in the same run
    n_release = sum(f["config"] == "release-v2-redistributable" for f in OJ)
    assert total == len(TS) + len(OJ) - n_release + 2
    assert any(f.startswith("tasksource:") for f in families) and any(f.startswith("openjev:") for f in families)
    assert any(f.startswith("jev_agent:") for f in families)
