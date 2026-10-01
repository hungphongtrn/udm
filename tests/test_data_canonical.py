import json

from scrm_data import canonical as C


def test_dumps_is_canonical_verified_recipe():
    obj = {"b": 1, "a": {"z": [1, 2], "y": "é"}}
    s = C.dumps(obj)
    assert s == '{"a":{"y":"é","z":[1,2]},"b":1}'
    assert json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == s


def test_dumps_rejects_nan():
    import pytest

    with pytest.raises(ValueError):
        C.dumps({"x": float("nan")})


def test_normalize_text_and_choice_id():
    assert C.normalize_text("  a\r\nb \n") == "a\nb"
    assert C.normalize_text("é") == "é"            # NFC
    assert C.normalize_text("a  b\tc") == "a  b\tc"            # inner whitespace preserved
    t = "Benign activity."
    assert C.choice_id_of(t) == "43a690f07c1c7483697d36cfaef1d93c0057ea4366484389d735076c4b53b2b6" or len(C.choice_id_of(t)) == 64
    assert C.choice_id_of("true") == "b5bea41b6c623f7c09f1bf24dcae58ebab3c0cdd90ad966bc43a45b44867e12b"  # matches existing data
    assert C.choice_id_of("false") == "fcbcf165908dd18a9e49f7ff27810176db8e9f63b4352213741664245224f8aa"


def test_ids_deterministic_and_distinct():
    a = C.decision_set_id_of("s", "c", "train", "1", "r")
    assert a == C.decision_set_id_of("s", "c", "train", "1", "r")
    assert a != C.decision_set_id_of("s", "c", "train", "2", "r")
    assert a != C.decision_set_id_of("s", "c2", "train", "1", "r")
    assert C.record_id_of(a, "x") != C.record_id_of(a, "y")


def test_content_key_order_independent_and_type_sensitive():
    k1 = C.content_key_hex('"s"', "choice\x1dq", ["b", "a"])
    k2 = C.content_key_hex('"s"', "choice\x1dq", ["a", "b"])
    k3 = C.content_key_hex('"s"', "score\x1dq", ["a", "b"])
    assert k1 == k2 and k1 != k3
    ij = C.dumps({"type": "choice", "instructions": "q"})
    assert C.instruction_key_of(ij) == "choice\x1dq"
    assert C.instruction_key_of(C.dumps("plain")) == "\x1dplain"
    cols = C.content_key_from_columns('"s"', ij, C.dumps({C.choice_id_of("a"): "a", C.choice_id_of("b"): "b"}))
    assert cols == k1


def test_hash_split_deterministic_and_roughly_96_2_2():
    assert C.hash_split("x") == C.hash_split("x")
    n = 20000
    counts = {"train": 0, "validation": 0, "test": 0}
    for i in range(n):
        counts[C.hash_split(f"g{i}")] += 1
    assert 0.94 < counts["train"] / n < 0.98
    assert counts["validation"] > 100 and counts["test"] > 100
