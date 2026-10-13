"""All-wrong minimal pairs (scripts/abstain/build_allwrong.py) and the detection math (scripts/abstain/allwrong_report.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "abstain"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


build_allwrong, report = script("build_allwrong"), script("allwrong_report")


def rec(rid, questions, variant="clean", source="agnews"):
    return {"state": f"s{rid}", "questions": questions, "_meta": {"id": f"{source}/{rid}", "group_id": f"{source}/{rid}", "variant": variant, "source": source}}


def choice(keys, label, **kw):
    return {"type": "choice", "instructions": "pick", "criteria": {k: f"desc {k}" for k in keys}, "label": label, "src": "x", **kw}


def test_pairs_differ_only_in_the_missing_option():
    out, counts = build_allwrong.build([rec(0, {"q": choice("abcde", "c")})], seed=0)
    ans, bad = out["answerable"][0]["questions"]["q"], out["allwrong"][0]["questions"]["q"]
    assert len(ans["criteria"]) == len(bad["criteria"]) == 4
    assert "c" in ans["criteria"] and ans["label"] == "c" and "c" not in bad["criteria"]
    assert out["allwrong"][0]["_meta"]["allwrong"] and not out["answerable"][0]["_meta"]["allwrong"]
    shared = set(ans["criteria"]) & set(bad["criteria"])
    assert len(shared) == 3 and [k for k in ans["criteria"] if k in shared] == [k for k in bad["criteria"] if k in shared] == [k for k in "abcde" if k in shared]
    an, bn = out["answerable_none"][0]["questions"]["q"], out["allwrong_none"][0]["questions"]["q"]
    (nk,) = set(an["criteria"]) - set(ans["criteria"])
    assert nk in build_allwrong.NONE_KEYS and list(an["criteria"]).index(nk) == list(bn["criteria"]).index(nk)
    assert an["criteria"][nk] == bn["criteria"][nk] and an["label"] == "c" and bn["label"] == nk
    assert [k for k in an["criteria"] if k != nk] == list(ans["criteria"]) and [k for k in bn["criteria"] if k != nk] == list(bad["criteria"])
    assert counts == {"records": 1, "questions": 1, "dropped_questions": 0}
    assert build_allwrong.build([rec(0, {"q": choice("abcde", "c")})], seed=0)[0] == out   # seeded per (seed, id, question)


@pytest.mark.parametrize("q", [choice("ab", "a"), choice("abc", "a", target={"a": 1.0}), choice(["a", "b", "other"], "a"),
                               {"type": "noul", "instructions": "x", "label": True, "src": "x"}])
def test_ineligible_questions_dropped(q):
    out, counts = build_allwrong.build([rec(0, {"bad": q, "ok": choice("abc", "b")})], seed=0)
    assert list(out["allwrong"][0]["questions"]) == ["ok"] and counts["dropped_questions"] == 1


def test_variants_and_unknowable_records_skipped():
    recs = [rec(0, {"q": choice("abc", "a")}, variant="permuted"), rec(1, {"q": choice("abc", "a")}, source="unknowable")]
    out, counts = build_allwrong.build(recs, seed=0)
    assert counts["records"] == 0 and all(v == [] for v in out.values())


def test_auroc_ties_and_order():
    assert report.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert report.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert report.auroc(np.array([1.0, 3.0]), np.array([2.0])) == 0.5


def test_risk_coverage_tied_block_is_its_mean():
    """Equal confidences answer as a block: no order-dependent accuracy from tie-breaking."""
    acc = report.risk_coverage([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0])
    assert np.allclose(acc, [1.0, 0.75, 2 / 3, 0.5])


def test_detection_direction():
    """Confidence readouts flag low confidence as all-wrong; pnone flags high none mass."""
    d = {"ans": np.array([0.9, 0.8]), "bad": np.array([0.2, 0.3])}
    assert report.detection(d, "maxprob") == 1.0 and report.detection(d, "pnone") == 0.0


def test_fit_threshold_cuts_only_at_tie_block_ends():
    """The lowest threshold within the risk; a tied block is answered whole, so a cut inside it (risk 0 at the third row)
    is not available; no qualifying prefix -> answer nothing."""
    conf, hit = [0.9, 0.8, 0.5, 0.5, 0.1], [1, 1, 1, 0, 0]
    assert report.fit_threshold(conf, hit, 0.10) == 0.8
    assert report.fit_threshold(conf, hit, 0.25) == 0.5
    assert report.fit_threshold([0.9, 0.2], [0, 1], 0.10) == float("inf")


def test_apply_threshold_splits_answerable_and_allwrong():
    """The pool is answerable rows then allwrong rows (pool()); a threshold carried over reports both answer rates."""
    got = report.apply_threshold([0.85, 0.4, 0.95, 0.3], np.array([1.0, 1.0, 0.0, 0.0]), 0.8, 2)
    assert got == {"threshold": 0.8, "coverage": 0.5, "risk": 0.5, "answered_answerable": 0.5, "answered_allwrong": 0.5}
    assert report.apply_threshold([0.1, 0.2], np.array([1.0, 0.0]), float("inf"), 1)["coverage"] == 0.0
