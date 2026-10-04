"""Gradient caching for the SCRM training step (optional `train.grad_cache`).

A loader micro-batch is a "group" of whole sets; its backbone embeddings do not depend on the set head, so the group
is encoded once without grad (pass 1), the set head + loss are computed on the cached embeddings (pass 2), and each
sequence chunk is re-encoded with grad only to push the head's embedding gradients back into the backbone (pass 3).
That decouples the backbone activation memory from `data.batch.max_tokens_per_batch` (the group budget): pass 1/3 run
on chunks of at most `train.grad_cache_chunk_tokens` tokens. Dropout masks are replayed by restoring the RNG state
recorded per chunk before pass 1. See docs/TRAINING.md.
"""
from __future__ import annotations

import torch

from .packing import pack_bfd


def chunk_plan(seq_lens, chunk_tokens: int | None) -> list[list[int]]:
    """Indices of the pack sequences grouped into backbone chunks of <= chunk_tokens tokens (a longer sequence gets
    its own chunk). Sequences inside a chunk keep ascending original index order and chunks are ordered by their first
    index, so a budget >= all tokens yields exactly one chunk in the original pack order (== the standard path)."""
    lens = [max(int(x), 1) for x in seq_lens]
    if not chunk_tokens or chunk_tokens <= 0:
        return [list(range(len(lens)))]
    chunks = [sorted(b) for b in pack_bfd(lens, int(chunk_tokens))]
    chunks.sort(key=lambda c: c[0])
    return chunks


def _rng_state(device):
    st = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def _set_rng_state(st, device):
    torch.set_rng_state(st["cpu"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"], device)


def grad_cache_step(model, packs, chunk_tokens: int | None, *, amp_ctx, head_loss):
    """Three-pass gradient-cached forward/backward over one group.

    `packs`: list of (pack, candidate_mask) that need backbone embeddings (the main pack, and the second permutation
    pack when `loss.perm_detach=false`). `head_loss(leaves) -> (loss, extras)` runs the set head on the detached,
    grad-enabled embeddings and returns a scalar loss to backward. Returns `(loss.detach(), extras, embeds_detached)`.
    """
    device = packs[0][0]["input_ids"].device
    plans, embeds, states = [], [], []
    for pack, _ in packs:
        plan = chunk_plan(pack["seq_lens"], chunk_tokens)
        plans.append(plan)
        st = []
        e = None
        with torch.no_grad(), amp_ctx():
            for idx in plan:
                st.append(_rng_state(device))
                h = model.embed_indices(pack, idx)
                if e is None:
                    e = torch.empty(len(pack["seq_lens"]), h.size(-1), dtype=h.dtype, device=h.device)
                e[torch.as_tensor(idx, device=h.device)] = h
        states.append(st)
        embeds.append(e)
    leaves = [e.detach().requires_grad_(True) for e in embeds]
    loss, extras = head_loss(leaves)
    if loss.requires_grad:
        loss.backward()
    # RNG stream after the head (== after one full standard forward): pass 3 replays pass 1, then we restore it
    final_rng = _rng_state(device)
    if getattr(model, "backbone_trainable", False):
        for (pack, _), plan, st, leaf in zip(packs, plans, states, leaves):
            g = leaf.grad
            if g is None:
                continue
            for idx, s in zip(plan, st):
                _set_rng_state(s, device)
                with amp_ctx():
                    h = model.embed_indices(pack, idx)
                torch.autograd.backward(h, g[torch.as_tensor(idx, device=h.device)])
        _set_rng_state(final_rng, device)
    return loss.detach(), extras, [e.detach() for e in embeds]
