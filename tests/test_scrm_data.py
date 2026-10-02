import json

import numpy as np
import pytest
import torch

from scrm.collator import collate
from scrm.config import load_config
from scrm.data import EvalSet, Pred, Router, make_batches, make_train_loader, build_groups
from scrm.render import Example, Renderer, parse_row, render_instruction, render_state, subsample_indices
from scrm.synth import write_synth
from scrm.tiny import make_tiny_tokenizer
from scrm.tokens import prepare_tokenizer


@pytest.fixture(scope="module")
def tok():
    t = prepare_tokenizer(make_tiny_tokenizer())
    return t


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return write_synth(str(tmp_path_factory.mktemp("synth")))


def mk_example(n=10, ntop=1, tiers=None, state="some state " * 20, texts=None):
    tiers = tiers or [0] * ntop + [1] * (n - ntop)
    return Example("ds1", "src", "fam", "lk", "train", "do it", state, [f"c{i}" for i in range(n)],
                   texts or [f"cand number {i}" for i in range(n)], tiers)


def test_render_instruction_and_state():
    assert render_instruction("plain") == "plain"
    s = render_instruction({"type": "x", "instructions": "Rank them.", "criteria": {"a": "is a", "b": ["x"]}})
    assert s.startswith("Rank them.") and "- a: is a" in s and '- b: ["x"]' in s
    assert render_state("raw text") == "raw text"
    assert render_state({"k": [1, 2], "z": "é"}) == '{"k": [1, 2], "z": "é"}'


def test_parse_row_string_and_object(synth):
    import pyarrow.parquet as pq
    rows = pq.read_table(f"{synth}/data/train-00000-of-00001.parquet").to_pylist()
    ex0 = parse_row(rows[0])   # massive: string state/instruction
    assert isinstance(ex0.state, str) and not ex0.state.startswith('"')
    assert ex0.instruction == "Select the intent that best matches the utterance."
    ex1 = parse_row(rows[1])   # typed: object state & instruction
    assert ex1.state.startswith("{") and "Criteria:" in ex1.instruction
    # canonical order comes from candidate_rows
    order = [c["choice_id"] for c in sorted(rows[0]["candidate_rows"], key=lambda c: c["candidate_order"])]
    assert ex0.choice_ids == order
    assert min(ex0.tiers) == 0 and sorted(set(ex0.tiers)) == [0, 1]


def test_parse_row_drops_untrainable():
    row = {"options_json": json.dumps({"a": "x", "b": "y"}), "tier_json": json.dumps([["a", "b"]]),
           "state_json": '"s"', "instruction_json": '"i"'}
    assert parse_row(row) is None


def test_collator_shared_prefix_and_per_option_suffixes(tok):
    r = Renderer(tok, {"max_len": 512})
    items = [r.assemble(r.tokenize(mk_example(n)), np.random.default_rng(i), True) for i, n in enumerate([5, 9, 3])]
    b = collate(items, r.pad_id)
    assert len(b["sets"]) == 3
    for it, st in zip(items, b["sets"]):
        n = len(it.suffixes)
        assert st["suffix"].shape[0] == n and torch.equal(st["prefix"], torch.from_numpy(it.prefix))
        for k in range(n):
            rp = int(st["read_pos"][k])
            assert st["suffix"][k, rp] == r.chat_suffix[-1]                       # "\n" after "assistant"
            assert st["suffix_mask"][k, : rp + 1].all() and not st["suffix_mask"][k, rp + 1:].any()
            full = torch.cat([st["prefix"], st["suffix"][k, : rp + 1]]).numpy()
            assert np.array_equal(full, it.seqs[k])                               # prefix + suffix == full row
    B, N = b["candidate_mask"].shape
    assert (B, N) == (3, 9) and b["candidate_mask"].sum() == 17
    assert (b["tiers"][~b["candidate_mask"]] == -1).all() and (b["tiers"][b["candidate_mask"]] >= 0).all()
    assert b["pair_mask"].shape == (3, N, N)
    assert (b["pair_mask"][~b["candidate_mask"]]).sum() == 0
    assert b["n_tokens"] == sum(len(i.prefix) + sum(len(x) for x in i.suffixes) for i in items)


def test_truncation_budget_and_tier0(tok):
    r = Renderer(tok, {"max_len": 160, "cand_max_tokens": 8, "state_max_tokens": 64, "max_candidates": 20,
                       "min_state_tokens": 16})
    big_state = "word " * 5000
    ex = mk_example(60, ntop=2, state=big_state, texts=["a long candidate text " * 10] * 60)
    it = r.assemble(r.tokenize(ex), np.random.default_rng(0), True)
    assert it is not None and max(len(x) for x in it.seqs) <= 160
    assert (it.tiers == 0).sum() == 2 and len(it.order) <= 20 and (it.tiers > 0).any()
    # every row: chat prefix ... "Grade this choice: Option k: <cand>" + chat suffix (ends at the assistant header)
    for x in it.seqs:
        ids = x.tolist()
        assert ids[:len(r.chat_prefix)] == r.chat_prefix and ids[-len(r.chat_suffix):] == r.chat_suffix


def test_example_dropped_when_no_pair_possible(tok):
    r = Renderer(tok, {"max_len": 16, "min_state_tokens": 4, "cand_max_tokens": 64})
    ex = mk_example(4, state="x " * 50, texts=["very long " * 50] * 4)
    it = r.assemble(r.tokenize(ex), None, False)
    assert it is None or max(len(x) for x in it.seqs) <= 16


def test_state_middle_truncation_keeps_head_and_tail(tok):
    r = Renderer(tok, {"max_len": 200, "state_max_tokens": 40})
    state = "HEAD " + "mid " * 300 + " TAIL"
    t = r.tokenize(mk_example(4, state=state))
    it = r.assemble(t, None, False)
    txt = tok.decode(it.seqs[0])
    assert "HEAD" in txt and "TAIL" in txt and txt.count("mid") < 300


def test_pred_router_and_mixing(synth):
    cfg = load_config(None, [f"data.local_dir={synth}", "data.num_workers=0",
                             "data.groups=[{name: massive, match: {family: [massive]}, max_rows: 20, weight: 1}]",
                             "data.other_weight=3"])
    groups = build_groups(cfg["data"], "train", cfg["data"]["filters"], False, 0)
    sizes = {g.name: g.size for g in groups}
    assert sizes["massive"] == 20 and sizes["other"] > 0 and "massive" not in [g.name for g in groups if g.size == 0]
    assert sum(1 for _ in zip(range(50), __import__("scrm.data", fromlist=["Mixer"]).Mixer(groups, 0))) == 50
    import pyarrow as pa
    t = pa.table({"source_id": ["a", "b", "c"], "family": ["x", "y", "x"]})
    assert Pred({"family": ["x"]}, {"source_id": "c"}).mask(t).tolist() == [True, False, False]


def test_train_loader_batches_respect_budget(synth, tok):
    cfg = load_config(None, [f"data.local_dir={synth}", "data.num_workers=0", "data.render.max_len=256",
                             "data.render.cand_max_tokens=16", "data.batch.max_tokens_per_batch=600",
                             "data.batch.max_batch_size=6", "data.batch.bucket_size=16"])
    r = Renderer(tok, cfg["data"]["render"])
    loader, stream = make_train_loader(cfg, r, 0)
    n = 0
    for b in loader:
        assert b["n_padded_tokens"] <= 600 or b["candidate_mask"].size(0) == 1
        assert b["candidate_mask"].size(0) <= 6 and b["pair_mask"].flatten(1).any(1).all()
        n += 1
        if n >= 8:
            break
    assert n == 8


def test_eval_set_deterministic_and_excludes_ood(synth, tok):
    cfg = load_config(None, [f"data.local_dir={synth}", "data.render.max_len=256", "data.render.cand_max_tokens=16",
                             "data.eval_max_rows=1000", "data.eval_max_rows_per_source=10"])
    r = Renderer(tok, cfg["data"]["render"])
    a = EvalSet.from_config(cfg, r)
    b = EvalSet.from_config(cfg, r)
    assert [i.decision_set_id for i in a.items] == [i.decision_set_id for i in b.items]
    assert all(i.source_id != "" for i in a.items)
    from scrm.data import select_eval_rows
    rows = select_eval_rows(cfg["data"], "validation", cfg["data"]["eval_filters"], 1000, 10)
    assert rows and all(x["source_split"] != "ood" for x in rows)
    from collections import Counter
    assert max(Counter(x["source_id"] for x in rows).values()) <= 10


def test_prompt_lists_all_options_and_each_row_grades_one(tok):
    n_vocab = len(tok)
    r = Renderer(tok, {"max_len": 256})
    assert len(tok) == n_vocab                                           # renderer adds no tokens
    ex = Example.from_raw("Pick one.", "the state", ["alpha", "beta", "gamma"])
    it = r.assemble(r.tokenize(ex), None, False, relax=True)
    prompt = ("<|im_start|>user\nState: the state\n\nInstruction: Pick one.\n\nOptions:\n"
              "Option 1: alpha\nOption 2: beta\nOption 3: gamma\n\nGrade this choice: ")
    assert len(it.seqs) == 3
    for k, (x, c) in enumerate(zip(it.seqs, ["alpha", "beta", "gamma"])):
        assert tok.decode(x.tolist()) == prompt + f"Option {k + 1}: {c}<|im_end|>\n<|im_start|>assistant\n"
    off = Renderer(tok, {"max_len": 256, "chat_template": False}).assemble(r.tokenize(ex), None, False, relax=True)
    assert tok.decode(off.seqs[0].tolist()).startswith("State: the state\n\nInstruction: Pick one.\n\nOptions:\nOption 1: alpha\n")


def test_shuffle_relabels_options_in_display_order(tok):
    r = Renderer(tok, {"max_len": 256})
    ex = Example.from_raw("Pick one.", "s", ["alpha", "beta", "gamma", "delta"])
    it = r.assemble(r.tokenize(ex), np.random.default_rng(1), True, relax=True)
    shown = [["alpha", "beta", "gamma", "delta"][i] for i in it.order]
    listing = "".join(f"Option {k + 1}: {c}\n" for k, c in enumerate(shown))
    for k, (x, c) in enumerate(zip(it.seqs, shown)):
        text = tok.decode(x.tolist())
        assert listing in text and text.endswith(f"Grade this choice: Option {k + 1}: {c}<|im_end|>\n<|im_start|>assistant\n")


def test_max_graded_subsamples_rows_but_prompt_shows_all(tok):
    r = Renderer(tok, {"max_len": 1024, "max_graded": 4})
    ex = mk_example(10, ntop=1)
    it = r.assemble(r.tokenize(ex), np.random.default_rng(0), True)
    assert len(it.seqs) == 4 and len(it.kept) == 10 and (it.tiers == 0).sum() == 1 and (it.tiers > 0).any()
    text = tok.decode(it.seqs[0].tolist())
    assert all(f"Option {k}: " in text for k in range(1, 11))          # all 10 options listed in the prompt
    it2 = r.reshuffle(it, np.random.default_rng(5))                     # reshuffle keeps the same graded set
    assert sorted(it2.order.tolist()) == sorted(it.order.tolist())
