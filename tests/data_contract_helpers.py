"""Shared contract assertions for converted rows (reference schema is an independent constant)."""
import hashlib
import json

import pyarrow as pa

REFERENCE_SCHEMA = pa.schema([
    ("decision_set_id", pa.string()), ("source_id", pa.string()), ("source_revision", pa.string()),
    ("source_config", pa.string()), ("source_split", pa.string()), ("source_row_id", pa.string()),
    ("source_parent_id", pa.string()), ("family", pa.string()), ("partition_role", pa.string()),
    ("lineage_key", pa.string()), ("state_json", pa.string()), ("instruction_json", pa.string()),
    ("options_json", pa.string()), ("tier_json", pa.string()), ("probabilities_json", pa.string()),
    ("candidate_count", pa.int32()), ("label_kind", pa.string()), ("source_label_or_null", pa.string()),
    ("instruction_template_id", pa.string()), ("option_permutation_seed_or_null", pa.string()),
    ("raw_record_sha256", pa.string()), ("license", pa.string()), ("known_pretraining_overlap", pa.string()),
    ("contamination_status", pa.string()), ("quarantine_reason_or_null", pa.string()),
    ("candidate_rows", pa.list_(pa.struct([
        pa.field("record_id", pa.string()), pa.field("decision_set_id", pa.string()),
        pa.field("choice_id", pa.string()), pa.field("candidate_order", pa.int32()),
        pa.field("normalized_text_sha256", pa.string())]))),
])
KIND_OF_LABEL = {"source_dataset_label": "choice", "soft_choice_distribution": "choice_soft",
                 "ordinal_score_distance_tiers": "score", "independent_labels_binary_tiers": "noul",
                 "agent_choice_target": "agent_choice"}
ROLE_OF_SPLIT = {"train": "train", "validation": "dev", "test": "test"}
JSON_COLS = ("state_json", "instruction_json", "options_json", "tier_json", "probabilities_json", "source_label_or_null")


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def assert_contract_row(row, split=None):
    """Raises AssertionError if `row` violates the output contract. `split` (train/validation/test) if known."""
    t = pa.Table.from_pylist([row], schema=REFERENCE_SCHEMA)
    assert t.schema.equals(REFERENCE_SCHEMA)
    assert list(row) == REFERENCE_SCHEMA.names
    assert [f.name for f in t.schema] == REFERENCE_SCHEMA.names and len(t.schema) == 26
    assert t.schema.field("candidate_count").type == pa.int32()
    cr_t = t.schema.field("candidate_rows").type.value_type
    assert [f.name for f in cr_t] == ["record_id", "decision_set_id", "choice_id", "candidate_order",
                                       "normalized_text_sha256"] and cr_t.field("candidate_order").type == pa.int32()
    parsed = {}
    for c in JSON_COLS:
        parsed[c] = json.loads(row[c])
        assert canon(parsed[c]) == row[c], f"{c} not canonical"
    opts, tiers, probs = parsed["options_json"], parsed["tier_json"], parsed["probabilities_json"]
    assert isinstance(opts, dict) and len(opts) >= 2
    for cid, text in opts.items():
        assert cid == sha(text), "choice_id != sha256(text)"
    flat = [c for tier in tiers for c in tier]
    assert sorted(flat) == sorted(opts) and len(flat) == len(set(flat)), "tiers must partition options"
    assert len(tiers) >= 2 and all(tier for tier in tiers)
    assert all(tier == sorted(tier) for tier in tiers), "ids inside a tier must be sorted"
    assert isinstance(probs, dict) and set(probs) <= set(opts)
    n = len(opts)
    cr = row["candidate_rows"]
    assert row["candidate_count"] == n == len(cr)
    assert [c["candidate_order"] for c in cr] == list(range(n))
    assert {c["choice_id"] for c in cr} == set(opts)
    for c in cr:
        assert c["normalized_text_sha256"] == c["choice_id"] and c["decision_set_id"] == row["decision_set_id"]
        assert len(c["record_id"]) == 64 and c["record_id"] != c["choice_id"]
    assert len({c["record_id"] for c in cr}) == n
    assert row["partition_role"] in ("train", "dev", "test")
    if split:
        assert row["partition_role"] == ROLE_OF_SPLIT[split]
    assert row["label_kind"] in KIND_OF_LABEL
    sl = parsed["source_label_or_null"]
    kind = KIND_OF_LABEL[row["label_kind"]]
    assert sl["kind"] == kind
    if kind == "choice":
        assert {"target", "winner"} <= set(sl) and len(sl["winner"]) == 1 and sl["winner"][0] in opts
        assert tiers[0] == sl["winner"] and probs == {}
    elif kind == "choice_soft":
        assert "target" in sl and probs
    elif kind == "score":
        assert {"levels", "level_choice_ids", "true_level_index", "expected_level", "target"} <= set(sl)
        ids = sl["level_choice_ids"]
        assert [opts[i] for i in ids] == sl["levels"] and [c["choice_id"] for c in cr] == ids
        assert len(sl["target"]) == n
        if sl["true_level_index"] is not None:
            ti = sl["true_level_index"]
            assert tiers[0] == [ids[ti]]
            for k, tier in enumerate(tiers):
                assert tier == sorted(ids[j] for j in range(n) if abs(j - ti) == k)
    elif kind == "noul":
        assert {"target", "threshold"} <= set(sl) and sl["threshold"] == 0.5
        assert "p_yes" not in sl or sorted(sl["options"]) == ["no", "yes"]
        if "p_yes" in sl:
            assert sorted(opts.values()) == ["no", "yes"]
            yes = sha("yes")
            assert tiers == ([[yes], [sha("no")]] if sl["p_yes"] > 0.5 else [[sha("no")], [yes]])
            assert abs(probs[yes] - sl["p_yes"]) < 1e-6
    elif kind == "agent_choice":
        assert {"target", "labels", "decision_type"} <= set(sl) and probs == {}
        assert len(tiers[0]) == 1
    return parsed
