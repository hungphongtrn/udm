import numpy as np
import pytest
import torch

from scrm.model import build_scrm

di = pytest.importorskip("decision_index")
from decision_index.engines import validate  # noqa: E402

from scrm.dindex import answer_requests, make_renderer, question_example  # noqa: E402

TINY = {"name_or_path": "tiny", "dtype": "float32", "d_set": 32, "set_heads": 4,
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0}}
BCFG = {"max_tokens_per_batch": 4096, "max_batch_size": 64}


@pytest.fixture(scope="module")
def model():
    m, _ = build_scrm(TINY, "cpu")
    m.render_cfg = {"max_len": 256, "max_candidates": 4, "max_graded": 2}   # inference must ignore both caps
    return m.eval()


def req(n_opts=6, state="the state", noul=True):
    q = {"pick": {"type": "choice", "instructions": "Pick one.",
                  "criteria": {f"k{i}": f"option text {i}" for i in range(n_opts)}}}
    if noul:
        q["flag"] = {"type": "noul", "instructions": "Is it hallucinated?"}
    return {"state": state, "questions": q}


def test_responses_are_valid_and_grade_every_option(model):
    r = make_renderer(model, None)
    reqs = [req(6), req(3, state={"k": [1, 2]}, noul=False)]
    out = answer_requests(model, r, reqs, BCFG, torch.device("cpu"), amp=False)
    for q, o in zip(reqs, out):
        assert o["status"] == "ok"
        validate(q["questions"], o["response"])                      # the kit's own response checks
    p = out[0]["response"]["answers"]["pick"]["probabilities"]
    assert list(p) == [f"k{i}" for i in range(6)] and abs(sum(p.values()) - 1) < 1e-6
    # batching does not change answers: same request alone gives the same distribution
    alone = answer_requests(model, r, [reqs[0]], BCFG, torch.device("cpu"), amp=False)[0]
    q = alone["response"]["answers"]["pick"]["probabilities"]
    assert np.allclose([p[k] for k in p], [q[k] for k in p], atol=1e-5)


def test_candidates_follow_option_keys_and_noul_is_no_yes():
    keys, ex = question_example("s", {"type": "choice", "instructions": "x", "criteria": {"b": "beta", "a": ""}})
    assert keys == ["b", "a"] and ex.texts == ["beta", "a"]               # empty description falls back to key
    keys, ex = question_example("s", {"type": "noul", "instructions": "x"})
    assert keys == ex.texts == ["no", "yes"]


def test_overflow_makes_request_unsupported(model):
    r = make_renderer(model, 160)                                     # short request renders to ~110 tokens
    out = answer_requests(model, r, [req(3, state="word " * 200), req(2, state="s", noul=False)], BCFG,
                          torch.device("cpu"), amp=False)
    assert out[0]["status"] == "unsupported" and "max_len" in out[0]["error"]
    assert out[1]["status"] == "ok"
