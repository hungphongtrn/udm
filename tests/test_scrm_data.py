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


def test_collator_packs_full_sequences_in_set_order(tok):
    r = Renderer(tok, {"max_len": 512})
    items = [r.assemble(r.tokenize(mk_example(n)), np.random.default_rng(i), True) for i, n in enumerate([5, 9, 3])]
    b = collate(items, r.pad_id)
    p = b["pack"]
    seqs = [x for it in items for x in it.seqs]
    assert p["seq_lens"] == [len(x) for x in seqs]
    assert np.array_equal(p["input_ids"].numpy(), np.concatenate(seqs))           # padding-free, set order
    t = 0
    for x in seqs:                                                               # positions restart per sequence
        assert p["position_ids"][t:t + len(x)].tolist() == list(range(len(x)))
        t += len(x)
    assert (p["input_ids"][p["read_idx"]] == r.chat_suffix[-1]).all()           # "\n" after "assistant"
    assert p["row_set"].tolist() == [0] * 5 + [1] * 9 + [2] * 3
    assert p["row_slot"].tolist() == list(range(5)) + list(range(9)) + list(range(3))
    B, N = b["candidate_mask"].shape
    assert (B, N) == (3, 9) and b["candidate_mask"].sum() == 17
    assert (b["tiers"][~b["candidate_mask"]] == -1).all() and (b["tiers"][b["candidate_mask"]] >= 0).all()
    assert b["pair_mask"].shape == (3, N, N)
    assert (b["pair_mask"][~b["candidate_mask"]]).sum() == 0
    assert b["n_tokens"] == len(p["input_ids"]) == sum(i.n_tokens for i in items)


def test_overflow_drops_set_and_never_truncates(tok):
    r = Renderer(tok, {"max_len": 400})
    big = mk_example(6, state="word " * 5000)
    assert r.assemble(r.tokenize(big), None, False) is None                       # state alone > max_len
    assert r.assemble(r.tokenize(big), None, False, relax=True) is None
    state = "HEAD " + "mid " * 20 + " TAIL"
    t = r.tokenize(mk_example(4, state=state))
    it = r.assemble(t, None, False)
    assert it is not None and max(len(x) for x in it.seqs) <= 400
    assert tok.decode(it.seqs[0]).count("mid") == 20                             # state kept whole
    # the same set with a few more tokens than max_len is dropped, not cut
    tight = Renderer(tok, {"max_len": max(len(x) for x in it.seqs) - 1})
    assert tight.assemble(tight.tokenize(mk_example(4, state=state)), None, False) is None


def test_training_set_needs_two_tiers_inference_does_not(tok):
    r = Renderer(tok, {"max_len": 512})
    flat = mk_example(4, tiers=[0, 0, 0, 0])
    assert r.assemble(r.tokenize(flat), None, False) is None
    assert len(r.assemble(r.tokenize(flat), None, False, relax=True).seqs) == 4


def test_reshuffle_never_overflows(tok):
    r = Renderer(tok, {"max_len": 10**6})
    ex = mk_example(12, state="s " * 30)
    it = r.assemble(r.tokenize(ex), None, False)
    worst = max(len(x) for x in it.seqs)
    tight = Renderer(tok, {"max_len": worst})
    it = tight.assemble(tight.tokenize(ex), None, False)
    for k in range(20):
        it2 = tight.reshuffle(it, np.random.default_rng(k)) if it is not None else None
        assert it is None or (it2 is not None and max(len(x) for x in it2.seqs) <= worst)


def test_bfd_packing_respects_budget_and_keeps_sets_whole():
    from scrm.packing import pack_bfd
    lens = [700, 50, 300, 300, 400, 120, 90, 2000, 10]
    bins = pack_bfd(lens, 1000, max_items=3)
    assert sorted(i for b in bins for i in b) == list(range(len(lens)))          # every set exactly once
    for b in bins:
        assert len(b) <= 3 and (sum(lens[i] for i in b) <= 1000 or len(b) == 1)
    assert [7] in bins                                                            # oversized set -> own bin
    assert len(bins) == 4                                                         # 3770 tokens, best fit -> 4 bins


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
                             "data.batch.max_tokens_per_batch=600",
                             "data.batch.max_batch_size=6", "data.batch.bucket_size=16"])
    r = Renderer(tok, cfg["data"]["render"])
    loader, stream = make_train_loader(cfg, r, 0)
    n = 0
    for b in loader:
        assert b["n_tokens"] <= 600 or b["candidate_mask"].size(0) == 1
        assert b["candidate_mask"].size(0) <= 6 and b["pair_mask"].flatten(1).any(1).all()
        n += 1
        if n >= 8:
            break
    assert n == 8


def test_eval_set_deterministic_and_excludes_ood(synth, tok):
    cfg = load_config(None, [f"data.local_dir={synth}", "data.render.max_len=256",
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
