"""Shared-prefix branching: the branch encoder must reproduce the per-option full-sequence computation exactly.

The oracle (`_oracle_embeds` / `_oracle_fwd_bwd` in test_scrm_model) is the old computation: every option as its own
sequence `prefix + suffix_k` through the backbone. The branch encoder packs one shared prefix segment plus one suffix
segment per option (see scrm.collator / scrm.model), so these tests pin the equivalence of the forward (hidden
states), the rewards and the gradients (LoRA + set head) — including the chunked, gradient-cached and single-token
suffix paths.
"""
import importlib

import pytest
import torch

from scrm.model import build_scrm
from test_scrm_model import TINY, _assert_same, _fwd_bwd, _items, _oracle_embeds, _oracle_fwd_bwd, _sets


def _rel_err(x, ref):
    x, ref = x.detach().float().cpu().flatten(), ref.detach().float().cpu().flatten()
    return ((x - ref).norm() / ref.norm().clamp(min=1e-12)).item()


def _model(**kw):
    model, _ = build_scrm(dict(TINY, set_dropout=0.0, **kw), "cpu")
    return model


def _perturb_lora(model, seed=0):
    """LoRA B starts at 0 (so LoRA A gets no gradient); randomise it so the branch path's grads are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def test_branch_embeddings_match_full_sequence_oracle():
    """Several sets, different option counts and prefix lengths, plus a single-token suffix."""
    model = _model().eval()
    for ns, P, one in [([3, 1, 4], 11, False), ([2, 3], 7, True), ([1, 2], 137, False), ([6], 23, True)]:
        items = _items(ns, P=P, seed=1, one_token=one)
        pack, _ = _sets(ns, P=P, seed=1, one_token=one)
        with torch.no_grad():
            e, o = model.embed(pack), _oracle_embeds(model, items)
        assert e.shape == o.shape
        assert (e - o).abs().max() < 2e-5, (ns, P, one, (e - o).abs().max().item())


def test_branch_rewards_and_grads_match_full_sequence_oracle():
    """Rewards and every parameter gradient (LoRA in both layer types + set head) match the oracle."""
    model = _model()
    model.train()
    _perturb_lora(model)
    items = _items([4, 2, 3], P=19, seed=5, one_token=True)
    pack, cm = _sets([4, 2, 3], P=19, seed=5, one_token=True)
    _assert_same(_oracle_fwd_bwd(model, items, pack, cm), _fwd_bwd(model, pack, cm, None), atol=1e-5)


def test_branch_chunked_matches_single_chunk_forward_and_grad():
    """Splitting the pack into set-budgeted chunks (gradient checkpointing on) changes nothing: a set and its shared
    prefix always land in the same chunk, and chunks stay in pack row order."""
    import scrm.model as M
    model = _model(gradient_checkpointing=True)
    model.train()
    _perturb_lora(model)
    pack, cm = _sets([4, 2], P=37)
    calls = []
    orig = M._branch_chunk
    M._branch_chunk = lambda p, rows: (calls.append(len(rows)), orig(p, rows))[1]
    try:
        one = _fwd_bwd(model, pack, cm, None)
        assert calls == [6]                       # one chunk with every set
        calls.clear()
        two = _fwd_bwd(model, pack, cm, 90)       # 55 + 45 tokens > 90 -> one chunk per set
    finally:
        M._branch_chunk = orig
    assert calls == [4, 2]
    _assert_same(one, two)


def test_branch_plan_covers_every_set_prefix_once_in_pack_order():
    """One chunk = whole sets: each set contributes its prefix segment once, followed by its suffix segments."""
    import scrm.model as M
    model = _model().eval()
    pack, _ = _sets([3, 2], P=13, seed=9)
    calls = []
    orig = M._branch_chunk

    def spy(p, rows):
        ids, pos, plan, read = orig(p, rows)
        calls.append((plan.seg_len, plan.owner))
        return ids, pos, plan, read

    M._branch_chunk = spy
    try:
        with torch.no_grad():
            model.embed(pack)
    finally:
        M._branch_chunk = orig
    assert calls == [([13, 3, 5, 7, 13, 3, 5], [0, 0, 0, 0, 4, 4, 4])]   # prefix once per set, suffixes after it


def test_branch_grad_cache_chunks_are_whole_sets_and_match_standard():
    """Grad cache with a chunk budget below the set cost: pass 1/3 chunks are whole sets (never a partial set) and
    the loss + grads equal the single-pack standard path."""
    from scrm.collator import collate, set_rows
    from scrm.train import accumulate_step
    model = _model()
    model.train()
    _perturb_lora(model)
    items = _items([4, 3, 2, 5], P=23, seed=7)
    b = collate(items)
    rows, cost = set_rows(b["pack"])
    n_valid = max(1, int(b["pair_mask"].flatten(1).any(1).sum()))
    kw = dict(lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"), amp=False,
              n_valid=n_valid, renderer=None, seed=0, step=0, use_perm=False)
    model.zero_grad(set_to_none=True)
    std = accumulate_step(model, [b], grad_cache=False, **kw)
    g_std = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    budget = min(cost) - 1                     # no two sets fit together: one chunk per set
    plans, seen = [], []
    gcmod = importlib.import_module("scrm.gradcache")
    orig_plan, enc = gcmod.chunk_plan, model.embed_indices
    gcmod.chunk_plan = lambda costs, cap: (plans.append(orig_plan(costs, cap)), plans[-1])[1]
    model.embed_indices = lambda pack, idx: (seen.append([int(i) for i in idx]), enc(pack, idx))[1]
    try:
        model.zero_grad(set_to_none=True)
        gc = accumulate_step(model, [b], grad_cache=True, chunk_tokens=budget, **kw)
    finally:
        gcmod.chunk_plan, model.embed_indices = orig_plan, enc
    g_gc = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert plans == [[[0], [1], [2], [3]]]     # one set per chunk
    assert seen[:4] == rows and len(seen) >= 4  # pass 1 encodes the whole sets, in pack order
    for chunk in seen:                          # never a partial set, rows ascending
        whole = [r for r in rows if set(r) <= set(chunk)]
        assert chunk == [r for grp in whole for r in grp]
    assert abs(std["loss"] - gc["loss"]) < 1e-6
    _assert_same((std["last_r"], g_std), (gc["last_r"], g_gc), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="branch varlen kernels are CUDA-only")
def test_branch_gpu_varlen_matches_torch_path_and_oracle(monkeypatch):
    """GPU: the flash-attn varlen + fla/causal-conv1d branch path is as accurate as the bf16 full-sequence oracle.

    bf16 autocast (flash-attn needs half precision) puts ~1e-2 relative noise on gradients, and the branch path and the
    per-option oracle round differently, so they cannot be compared elementwise. Ground truth is the fp32 oracle on
    CPU (exact; the CPU tests pin it to the branch encoder at 1e-5). Every bf16 path (cuda branch, torch branch, oracle)
    is scored by its norm-relative error to that truth per tensor; a branch path may not be worse than
    2x the bf16 oracle's own error (the precision floor), with an absolute floor of 1e-2.

    Run on the training box: `.venv/bin/python -m pytest tests/test_scrm_branching.py -q`.
    """
    import scrm.model as M
    if not M._varlen_kernels(torch.device("cuda")):
        pytest.skip("fla / causal-conv1d not installed")
    model, _ = build_scrm(dict(TINY, set_dropout=0.0), "cuda")
    model.train()
    _perturb_lora(model)
    items = _items([5, 3, 1], P=70, seed=3, one_token=True)
    pack_cpu, cm_cpu = _sets([5, 3, 1], P=70, seed=3, one_token=True)
    cpu_model = _model().train()          # own CPU build: deepcopy().cpu() misses non-module tensors
    cpu_model.load_state_dict(model.state_dict())
    truth = _oracle_fwd_bwd(cpu_model, items, pack_cpu, cm_cpu)   # fp32, no kernels
    pack = {k: v.cuda() if torch.is_tensor(v) else v for k, v in pack_cpu.items()}
    cm = cm_cpu.cuda()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with amp:
        runs = {"cuda-branch": _fwd_bwd(model, pack, cm, None)}
    with amp:
        runs["oracle-bf16"] = _oracle_fwd_bwd(model, items, pack, cm)
    monkeypatch.setattr(M, "_varlen_kernels", lambda d: False)
    with amp:
        runs["torch-branch"] = _fwd_bwd(model, pack, cm, None)

    (r_t, g_t) = truth
    assert all(g.keys() == g_t.keys() for _, g in runs.values()) and any("lora_" in n for n in g_t)
    names = ["rewards"] + list(g_t)
    err = {k: [_rel_err(r, r_t)] + [_rel_err(g[n], g_t[n]) for n in g_t] for k, (r, g) in runs.items()}
    floor = [max(2 * e, 1e-2) for e in err["oracle-bf16"]]
    bad = [(k, i) for k in ("cuda-branch", "torch-branch") for i, e in enumerate(err[k]) if e > floor[i]]
    rows = sorted(range(len(names)), key=lambda i: -max(err["cuda-branch"][i], err["torch-branch"][i]) / floor[i])
    table = "\n".join(f"  oracle-bf16 {err['oracle-bf16'][i]:.2e}  cuda-branch {err['cuda-branch'][i]:.2e}  "
                      f"torch-branch {err['torch-branch'][i]:.2e}  {names[i]}" for i in rows[:12])
    print("norm-relative error vs fp32 oracle (worst first):\n" + table)
    assert not bad, (f"{len(bad)} tensors exceed 2x bf16 oracle error: "
                     f"{sorted({names[i] for _, i in bad})[:6]}\n{table}")
