"""Gradient caching for the SCRM training step (optional `train.grad_cache`).

A loader micro-batch is a "group" of whole sets; its backbone embeddings do not depend on the set head, so the group
is encoded once without grad (pass 1), the set head + loss are computed on the cached embeddings (pass 2), and each
sequence chunk is re-encoded with grad only to push the head's embedding gradients back into the backbone (pass 3).
That decouples the backbone activation memory from `data.batch.max_tokens_per_batch` (the group budget): pass 1/3 run
on chunks of at most `train.grad_cache_chunk_tokens` tokens. Dropout masks are replayed by restoring the RNG state
recorded per chunk before pass 1.

Under DDP the backbone work is balanced across ranks at sequence granularity: every rank's sequences are pooled,
assigned to ranks by token count (LPT), encoded by their assignee, and the embeddings / embedding grads are exchanged
with SUM all-reduces. The set head and loss stay on the rank that owns the set, and the backbone grads land on the
encoding rank, so the step's SUM-all-reduced gradient is unchanged. See docs/TRAINING.md.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

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


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _all_gather_var(t: torch.Tensor, world: int) -> list[torch.Tensor]:
    """all_gather of a 1-D tensor whose length differs per rank."""
    n = torch.tensor([t.numel()], device=t.device, dtype=torch.long)
    ns = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(ns, n)
    ns = [int(x) for x in ns]
    buf = t.new_zeros(max(ns))
    buf[:t.numel()] = t
    out = [torch.empty_like(buf) for _ in range(world)]
    dist.all_gather(out, buf)
    return [o[:k] for o, k in zip(out, ns)]


def balance_plan(lens, owner, world: int) -> list[list[int]]:
    """Pool sequence indices per rank: longest-first onto the least-loaded rank (ties keep the owner, then lowest
    rank). Deterministic, so every rank computes the same plan from the same gathered lengths."""
    load = [0] * world
    out: list[list[int]] = [[] for _ in range(world)]
    for i in sorted(range(len(lens)), key=lambda i: (-lens[i], i)):
        r = min(range(world), key=lambda r: (load[r], r != owner[i], r))
        load[r] += max(int(lens[i]), 1)
        out[r].append(i)
    return [sorted(x) for x in out]


def _pool(packs, world: int, rank: int):
    """All ranks' pack sequences as one virtual pack (global order: pack, rank, sequence), each sequence's owner rank,
    and this rank's [start, end) rows of the pool for each of its packs."""
    ids, pos, lens, owner, own = [], [], [], [], []
    for pack, _ in packs:
        L = torch.as_tensor(pack["seq_lens"], dtype=torch.long, device=pack["input_ids"].device)
        if world == 1:
            parts = [(pack["input_ids"], pack["position_ids"], L)]
        else:
            parts = list(zip(_all_gather_var(pack["input_ids"], world), _all_gather_var(pack["position_ids"], world),
                             _all_gather_var(L, world)))
        for r, (i, p, l) in enumerate(parts):
            if r == rank:
                own.append((len(lens), len(lens) + l.numel()))
            ids.append(i); pos.append(p)
            lens += l.tolist()
            owner += [r] * l.numel()
    return {"input_ids": torch.cat(ids), "position_ids": torch.cat(pos), "seq_lens": lens}, owner, own


def grad_cache_step(model, packs, chunk_tokens: int | None, *, amp_ctx, head_loss):
    """Three-pass gradient-cached forward/backward over one group (balanced across DDP ranks, see module doc).

    `packs`: list of (pack, candidate_mask) that need backbone embeddings (the main pack, and the second permutation
    pack when `loss.perm_detach=false`). `head_loss(leaves) -> (loss, extras)` runs the set head on the detached,
    grad-enabled embeddings and returns a scalar loss to backward. Every rank must call this with the same number of
    packs. Returns `(loss.detach(), extras, embeds_detached)`.
    """
    device = packs[0][0]["input_ids"].device
    world, rank = _world()
    pool, owner, own = _pool(packs, world, rank)
    mine = balance_plan(pool["seq_lens"], owner, world)[rank] if world > 1 else list(range(len(owner)))
    # chunk this rank's assigned sequences; chunk_plan indexes into `mine`
    plan = [[mine[j] for j in c] for c in chunk_plan([pool["seq_lens"][i] for i in mine], chunk_tokens)] if mine else []
    d = model.set_encoder.proj.in_features
    # fp32 exchange buffer: a rank may encode nothing, and SUM over zeros is exact
    emb = torch.zeros(len(owner), d, dtype=torch.float32, device=device)
    states = []
    with torch.no_grad(), amp_ctx():
        for idx in plan:
            states.append(_rng_state(device))
            emb[torch.as_tensor(idx, device=device)] = model.embed_indices(pool, idx).float()
    if world > 1:
        dist.all_reduce(emb, op=dist.ReduceOp.SUM)
    leaves = [emb[a:b].clone().requires_grad_(True) for a, b in own]
    loss, extras = head_loss(leaves)
    if loss.requires_grad:
        loss.backward()
    # RNG stream after the head (== after one full standard forward): pass 3 replays pass 1, then we restore it
    final_rng = _rng_state(device)
    if getattr(model, "backbone_trainable", False):
        grad = torch.zeros_like(emb)
        for (a, b), leaf in zip(own, leaves):
            if leaf.grad is not None:
                grad[a:b] = leaf.grad
        if world > 1:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        for idx, s in zip(plan, states):
            g = grad[torch.as_tensor(idx, device=device)]
            if not g.any():
                continue
            _set_rng_state(s, device)
            with amp_ctx():
                h = model.embed_indices(pool, idx)
            torch.autograd.backward(h, g.to(h.dtype))
        _set_rng_state(final_rng, device)
    return loss.detach(), extras, [x.detach() for x in leaves]
