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


def test_collator_packs_shared_prefix_with_option_branches(tok):
    r = Renderer(tok, {"max_len": 512})
    items = [r.assemble(r.tokenize(mk_example(n)), np.random.default_rng(i), True) for i, n in enumerate([5, 9, 3])]
    b = collate(items, r.pad_id)
    p = b["pack"]
    # per set: the shared prefix segment, then one segment per graded option suffix (in pack order)
    segs = [x for it in items for x in [it.prefix, *it.suffixes]]
    assert p["seg_lens"] == [len(x) for x in segs]
    assert np.array_equal(p["input_ids"].numpy(), np.concatenate(segs))          # padding-free, set order
    assert p["seg_prefix"].tolist() == [0] + [0] * 5 + [6] + [6] * 9 + [16] + [16] * 3   # suffix -> its set's prefix
    t = 0
    for it in items:                                                            # prefix positions 0..P-1, suffix P..
        P = len(it.prefix)
        assert p["position_ids"][t:t + P].tolist() == list(range(P))            # the prefix sees its own positions
        t += P
        for x in it.suffixes:                                                   # a suffix continues at len(prefix)
            assert p["position_ids"][t:t + len(x)].tolist() == list(range(P, P + len(x)))
            t += len(x)
    assert (p["input_ids"][p["read_idx"]] == r.chat_suffix[-1]).all()           # "\n" after "assistant"
    assert p["row_seg"].tolist() == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
    assert p["row_set"].tolist() == [0] * 5 + [1] * 9 + [2] * 3
    assert p["row_slot"].tolist() == list(range(5)) + list(range(9)) + list(range(3))
    B, N = b["candidate_mask"].shape
    assert (B, N) == (3, 9) and b["candidate_mask"].sum() == 17
    assert (b["tiers"][~b["candidate_mask"]] == -1).all() and (b["tiers"][b["candidate_mask"]] >= 0).all()
    assert b["pair_mask"].shape == (3, N, N)
    assert (b["pair_mask"][~b["candidate_mask"]]).sum() == 0
    # the shared prefix is encoded once: the pack costs len(prefix) + sum(suffix), not one full sequence per option
    assert b["n_tokens"] == len(p["input_ids"]) == sum(i.n_tokens for i in items)
    assert b["n_tokens"] < sum(len(x) for it in items for x in it.seqs)


def test_collator_chunk_plan_keeps_sets_whole(tok):
    from scrm.collator import row_chunks, set_rows
    r = Renderer(tok, {"max_len": 512})
    items = [r.assemble(r.tokenize(mk_example(n)), np.random.default_rng(i), True) for i, n in enumerate([5, 9, 3])]
    b = collate(items, r.pad_id)
    p = b["pack"]
    rows, cost = set_rows(p)
    assert [len(x) for x in rows] == [5, 9, 3] and cost == [it.n_tokens for it in items]
    assert row_chunks(p, None) == [list(range(17))]                             # no budget -> one chunk
    assert row_chunks(p, 10 ** 9) == [list(range(17))]
    chunks = row_chunks(p, cost[1])                                             # exactly fits set 1: prefix + its 9 suffixes
    assert sorted(x for c in chunks for x in c) == list(range(17))
    assert len(chunks) > 1                                                      # the budget forces several chunks
    for c in chunks:                                                            # every chunk is a whole set (or a set
        sets = {int(p["row_set"][x]) for x in c}                                 # too big for the budget: alone)
        assert all(set(rows[s]) <= set(c) for s in sets)


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


def _bfd_reference(lengths, capacity, max_items=None):
    """The pre-fix algorithm as a naive linear scan (bins keyed by remaining space = capacity - used; among the bins
    with the smallest sufficient space, the one that reached that space earliest wins - TRL's `space_to_bin`
    deque). `pack_bfd` must reproduce its bins with a data-bounded search tree."""
    lens = [max(int(x), 1) for x in lengths]
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    bins: list[list[int]] = []
    used: list[int] = []
    when: list[int | None] = []            # when each bin reached its current remaining space (None: not searchable)
    clock = 0
    for idx in order:
        L = lens[idx]
        best = best_space = None
        if L <= capacity:
            for b in range(len(bins)):
                if when[b] is None:
                    continue
                sp = capacity - used[b]
                if sp >= L and (best is None or sp < best_space or (sp == best_space and when[b] < when[best])):
                    best, best_space = b, sp
        if best is None:
            bins.append([idx]); used.append(L); when.append(None)
            best = len(bins) - 1
        else:
            bins[best].append(idx); used[best] += L
        when[best] = None                   # the bin was popped from space_to_bin[its old space]
        if capacity - used[best] > 0 and (max_items is None or len(bins[best]) < max_items):
            clock += 1
            when[best] = clock              # appended to space_to_bin[new space]: newest entry for that space
    return bins


def test_bfd_packing_large_budget_is_data_bounded():
    """Regression: SegmentTree sized itself to `capacity`, so an 'unlimited' budget (10**9) allocated ~17 GB and
    OOM-killed the suite. The tree must be bounded by the item sizes instead."""
    from scrm.packing import chunk_plan, pack_bfd
    assert pack_bfd([700, 50, 300], 10 ** 9) == [[0, 2, 1]]                      # descending order, one bin
    assert chunk_plan([700, 50, 300], 10 ** 9) == [[0, 1, 2]]
    big = [1] * 20000                                                            # would have needed a 2**31-slot tree
    assert pack_bfd(big, 10 ** 9) == [list(range(20000))]
    assert chunk_plan(big, 10 ** 9) == [list(range(20000))]
    assert pack_bfd(big, 10 ** 9, max_items=8) == [list(range(i, i + 8)) for i in range(0, 20000, 8)]


def test_bfd_packing_matches_naive_reference():
    """The memory fix must not change the bins: identical to a naive best-fit-decreasing scan on moderate inputs
    (including budgets above the total length and the max_items cap)."""
    from scrm.packing import pack_bfd
    rng = np.random.default_rng(0)
    for _ in range(40):
        lens = [int(x) for x in rng.integers(1, 900, size=int(rng.integers(1, 40)))]
        for capacity, max_items in ((1000, None), (1000, 3), (500, 2), (10 ** 9, None), (10 ** 9, 3),
                                    (sum(lens), None), (sum(lens), 2), (max(lens), None)):
            assert pack_bfd(lens, capacity, max_items) == _bfd_reference(lens, capacity, max_items), \
                (lens, capacity, max_items)


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


def test_eval_set_covers_every_source_and_refills_unrenderable(tmp_path, synth, tok):
    """Regression: eval sampling must reach sources stored after large files and replace sets over max_len."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from scrm.data import select_eval_rows
    t = pq.read_table(f"{synth}/data/train-00000-of-00001.parquet")
    src = t.column("source_id")
    big = t.filter(pc.not_equal(src, "samatv256/agent"))      # many rows, several row groups, first in file order
    sam = t.filter(pc.equal(src, "samatv256/agent")).to_pylist()
    for r in sam[::2]:                                          # every other samatv set cannot render at max_len
        r["state_json"] = json.dumps("long state " * 2000)
    (tmp_path / "data").mkdir()
    pq.write_table(big, tmp_path / "data/validation-00000-of-00002.parquet", row_group_size=16)
    pq.write_table(type(t).from_pylist(sam, schema=t.schema), tmp_path / "data/validation-00001-of-00002.parquet",
                   row_group_size=7)
    cfg = load_config(None, [f"data.local_dir={tmp_path}", "data.render.max_len=2048", "data.eval_max_rows=1000",
                             "data.eval_max_rows_per_source=6"])
    r = Renderer(tok, cfg["data"]["render"])
    es = EvalSet.from_config(cfg, r)
    assert es.per_source() == {"AmazonScience/massive": 6, "LocalLLaMA/typed-decisions": 6, "samatv256/agent": 6}
    assert es.dropped_by_source.get("samatv256/agent", 0) > 0
    ids = [i.decision_set_id for i in es.items]
    assert len(set(ids)) == len(ids) and ids == [i.decision_set_id for i in EvalSet.from_config(cfg, r).items]
    # row lookup across row groups returns exactly the stored rows
    rows = select_eval_rows(cfg["data"], "validation", cfg["data"]["eval_filters"], None, None)
    want = {x["decision_set_id"]: x for x in big.to_pylist() + sam if x["source_split"] != "ood"}
    assert {x["decision_set_id"]: x["state_json"] for x in rows} == {k: v["state_json"] for k, v in want.items()}


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
