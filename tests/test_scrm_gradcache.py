"""Gradient caching (`train.grad_cache`) + manual-all-reduce DDP: equivalence with the standard path, RNG replay,
cross-rank gradient reduction, and an end-to-end tiny training run."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scrm.collator import collate
from scrm.config import load_config
from scrm.data import EvalSet
from scrm.model import build_scrm
from scrm.render import Renderer
from scrm.train import accumulate_step, all_reduce_grads, broadcast_params, reduce_ints, train

TINY = {"name_or_path": "tiny", "dtype": "float32", "d_set": 32, "set_heads": 4,
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0}}
RENDER = {"max_len": 256, "max_candidates": 16}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from scrm.synth import write_synth
    return write_synth(str(tmp_path_factory.mktemp("synth")))


def _batch(synth, tok, n=8):
    cfg = load_config(None, [f"data.local_dir={synth}", f"data.eval_max_rows={n}",
                             f"data.eval_max_rows_per_source={n}"])
    r = Renderer(tok, RENDER)
    es = EvalSet.from_config(cfg, r)
    return r, collate(es.items[:n], r.pad_id)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_grads_close(a, b, rtol=1e-4, atol=1e-6):
    assert a.keys() == b.keys() and any("lora_" in n for n in a)
    for n in a:
        assert torch.allclose(a[n], b[n], rtol=rtol, atol=atol), (n, (a[n] - b[n]).abs().max().item())


@pytest.mark.parametrize("branching", [True, False])
def test_grad_cache_matches_standard_multichunk(synth, branching):
    """A chunk budget that splits the pack across chunks (whole sets when branching, single full sequences otherwise)
    gives the same loss and grads as the single-pack path."""
    torch.manual_seed(0)
    model, tok = build_scrm(dict(TINY, set_dropout=0.0, branching=branching), "cpu")
    model.train()
    r, b = _batch(synth, tok, 8)
    n_valid = max(1, int(b["pair_mask"].flatten(1).any(1).sum()))
    kw = dict(lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"), amp=False,
              n_valid=n_valid, renderer=r, seed=0, step=0, use_perm=False)
    model.zero_grad(set_to_none=True)
    std = accumulate_step(model, [b], grad_cache=False, **kw)
    g_std = _grads(model)
    model.zero_grad(set_to_none=True)
    gc = accumulate_step(model, [b], grad_cache=True, chunk_tokens=64, **kw)
    g_gc = _grads(model)
    assert abs(std["loss"] - gc["loss"]) < 1e-5
    _assert_grads_close(g_std, g_gc)


def test_grad_cache_rng_replay_matches_standard(synth):
    """With LoRA dropout > 0 and a single chunk, pass 3 must replay pass 1's dropout masks exactly."""
    torch.manual_seed(0)
    model, tok = build_scrm(dict(TINY, set_dropout=0.0, lora={"enabled": True, "r": 4, "alpha": 8, "dropout": 0.5}), "cpu")
    model.train()
    r, b = _batch(synth, tok, 6)
    n_valid = max(1, int(b["pair_mask"].flatten(1).any(1).sum()))
    kw = dict(lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"), amp=False,
              n_valid=n_valid, renderer=r, seed=0, step=0, use_perm=False)
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    std = accumulate_step(model, [b], grad_cache=False, **kw)
    g_std = _grads(model)
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    gc = accumulate_step(model, [b], grad_cache=True, chunk_tokens=100000, **kw)
    g_gc = _grads(model)
    assert abs(std["loss"] - gc["loss"]) < 1e-5
    _assert_grads_close(g_std, g_gc)


def test_grad_cache_selective_checkpointing_matches_standard(synth):
    """Pass 3 with half the layers keeping activations (rest checkpointed) and LoRA dropout gives the standard
    path's grads, and every layer is checkpointed again afterwards (pass 1 / eval stay low-memory)."""
    torch.manual_seed(0)
    model, tok = build_scrm(dict(TINY, set_dropout=0.0, gradient_checkpointing=True,
                                 lora={"enabled": True, "r": 4, "alpha": 8, "dropout": 0.5}), "cpu")
    model.train()
    layers = model.ckpt_layers()
    assert len(layers) == 4
    r, b = _batch(synth, tok, 6)
    n_tok = b["n_tokens"]
    n_valid = max(1, int(b["pair_mask"].flatten(1).any(1).sum()))
    kw = dict(lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"), amp=False,
              n_valid=n_valid, renderer=r, seed=0, step=0, use_perm=False)
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    std = accumulate_step(model, [b], grad_cache=False, **kw)
    g_std = _grads(model)
    stored = []   # layers keeping activations during each pass-3 chunk's forward
    enc = model.embed_indices
    model.embed_indices = lambda pool, idx: (stored.append(sum(not m.gradient_checkpointing for m in layers)),
                                             enc(pool, idx))[1]
    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    # ceil half: `keep_activations(L * act_tokens // T)` then keeps exactly L/2 layers for odd T too (T == act_tokens
    # here, so the stored activation count is half the chunk's tokens either way)
    gc = accumulate_step(model, [b], grad_cache=True, chunk_tokens=10 ** 9, act_tokens=-(-n_tok // 2), **kw)
    g_gc = _grads(model)
    assert stored == [0, 2]   # pass 1 fully checkpointed, pass 3 keeps 2 of 4 layers
    assert all(m.gradient_checkpointing for m in layers)
    assert abs(std["loss"] - gc["loss"]) < 1e-5
    _assert_grads_close(g_std, g_gc)


def _ddp_worker(rank, world, synth, out_dir, mode, chunk_tokens, n_items, skew=False):
    dist.init_process_group("gloo", init_method=f"file://{out_dir}/pg", rank=rank, world_size=world)
    torch.manual_seed(0)
    model, tok = build_scrm(dict(TINY, set_dropout=0.0), "cpu")
    model.train()
    broadcast_params(model)
    r, _ = _batch(synth, tok, n_items)
    cfg = load_config(None, [f"data.local_dir={synth}", f"data.eval_max_rows={n_items}",
                             f"data.eval_max_rows_per_source={n_items}"])
    items = EvalSet.from_config(cfg, r).items[:n_items]
    # skew: rank 0 holds all but one set, so grad-cache balancing must move rank 0's sequences to rank 1
    b = collate((items[:-1], items[-1:])[rank] if skew else items[rank::world], r.pad_id)
    n_valid = reduce_ints(max(1, int(b["pair_mask"].flatten(1).any(1).sum())), torch.device("cpu"), dist.ReduceOp.SUM)
    accumulate_step(model, [b], lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"),
                    amp=False, n_valid=n_valid, grad_cache=(mode == "gc"), chunk_tokens=chunk_tokens, use_perm=False)
    all_reduce_grads([p for p in model.parameters() if p.requires_grad])
    torch.save({n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None},
               os.path.join(out_dir, f"grads_{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("mode,chunk,skew", [("std", 10 ** 9, False), ("gc", 64, False), ("gc", 64, True)])
def test_ddp_gradient_reduction_matches_single_process(synth, tmp_path, mode, chunk, skew):
    """2 ranks over disjoint shards, grads SUM-all-reduced, == single-process grads over the union (both modes;
    grad cache also with cross-rank sequence balancing of a skewed split)."""
    n_items = 6
    torch.manual_seed(0)
    model, tok = build_scrm(dict(TINY, set_dropout=0.0), "cpu")
    model.train()
    r, _ = _batch(synth, tok, n_items)
    cfg = load_config(None, [f"data.local_dir={synth}", f"data.eval_max_rows={n_items}",
                             f"data.eval_max_rows_per_source={n_items}"])
    b = collate(EvalSet.from_config(cfg, r).items[:n_items], r.pad_id)
    n_valid = max(1, int(b["pair_mask"].flatten(1).any(1).sum()))
    accumulate_step(model, [b], lcfg={}, d_batch={"max_tokens_per_batch": 10 ** 9}, device=torch.device("cpu"),
                    amp=False, n_valid=n_valid, grad_cache=(mode == "gc"), chunk_tokens=chunk, use_perm=False)
    ref = _grads(model)

    out = tmp_path / f"{mode}_{chunk}_{skew}"
    out.mkdir()
    # spawn, not fork: the parent already ran multithreaded torch ops, and a forked child deadlocks on the
    # inherited intra-op (OpenMP) thread pool at its first parallel kernel
    mp.start_processes(_ddp_worker, args=(2, synth, str(out), mode, chunk, n_items, skew), nprocs=2, join=True,
                       start_method="spawn")
    g0 = torch.load(out / "grads_0.pt", weights_only=True)
    g1 = torch.load(out / "grads_1.pt", weights_only=True)
    _assert_grads_close(ref, g0)
    for n in g0:   # identical after the all-reduce
        assert torch.equal(g0[n], g1[n]), n


def test_balance_plan_splits_tokens_evenly():
    from scrm.gradcache import balance_plan
    lens = [900, 50, 400, 400, 300, 120, 80, 10, 10, 700]
    plan = balance_plan(lens, owner=[0] * len(lens), world=3)
    assert sorted(i for p in plan for i in p) == list(range(len(lens)))
    loads = [sum(lens[i] for i in p) for p in plan]
    assert max(loads) - min(loads) <= max(lens), loads
    assert balance_plan([5, 5], owner=[1, 0], world=2) == [[1], [0]]   # equal-load ties keep sequences home


def test_train_grad_cache_end_to_end(synth, tmp_path):
    out = tmp_path / "run"
    cfg = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={out}",
                                                  "train.max_steps=2", "train.eval_every=0", "train.save_every=0",
                                                  "train.grad_cache=true", "train.grad_cache_chunk_tokens=64",
                                                  "data.render.max_graded=null"])
    res = train(cfg)
    assert res["step"] == 2
    assert os.path.exists(os.path.join(res["ckpt"], "scrm_head.pt"))
