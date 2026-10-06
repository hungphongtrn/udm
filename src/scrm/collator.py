"""Collation of Items into padding-free packs of branch segments."""
from __future__ import annotations

import numpy as np
import torch


def collate(items, pad_id: int = 0) -> dict:
    """Items (decision sets) -> one padding-free pack of branch segments.

    Each set contributes ONE shared prefix segment (state, instruction, all options, "Grade this choice: ") and one
    suffix segment per graded option ("Option k: {text}" + end of user turn + assistant header). The prefix is encoded
    once and every suffix branch continues it (see scrm.model): the pack is, in set order,
    [prefix, suffix_0, ..., suffix_{n-1}], with position ids 0..P-1 for the prefix and P..P+s_k-1 for suffix k — the
    exact positions the full sequence `prefix + suffix_k` would see. Token cost per set is P + sum(suffix lengths).

    pack = {input_ids [T], position_ids [T], seg_lens (list[int], one per segment, prefix + suffixes),
            seg_prefix [S] (index of the set's prefix segment; a prefix segment points at itself),
            row_seg [M] (segment index of each graded row), read_idx [M] (last token of each graded row),
            row_set [M] / row_slot [M] (set and slot of each graded row)};
    candidate_mask / tiers / pair_mask are per set [B, N(, N)]; M = number of graded options."""
    B = len(items)
    N = max(len(i.suffixes) for i in items)
    seg_lens: list[int] = []
    seg_prefix: list[int] = []
    ids: list[np.ndarray] = []
    pos: list[np.ndarray] = []
    row_seg: list[int] = []
    row_set: list[int] = []
    row_slot: list[int] = []
    read: list[int] = []
    t = s = 0
    for b, it in enumerate(items):
        P = len(it.prefix)
        seg_lens.append(P)
        seg_prefix.append(s)
        ids.append(it.prefix)
        pos.append(np.arange(P))
        p_seg, s, t = s, s + 1, t + P
        for k, x in enumerate(it.suffixes):
            n = len(x)
            seg_lens.append(n)
            seg_prefix.append(p_seg)
            ids.append(x)
            pos.append(np.arange(P, P + n))
            row_seg.append(s)
            row_set.append(b)
            row_slot.append(k)
            read.append(t + n - 1)
            s += 1
            t += n
    pack = {"input_ids": torch.from_numpy(np.concatenate(ids)).long(),
            "position_ids": torch.from_numpy(np.concatenate(pos)).long(),
            "seg_lens": seg_lens,
            "seg_prefix": torch.tensor(seg_prefix, dtype=torch.long),
            "row_seg": torch.tensor(row_seg, dtype=torch.long),
            "read_idx": torch.tensor(read, dtype=torch.long),
            "row_set": torch.tensor(row_set, dtype=torch.long),
            "row_slot": torch.tensor(row_slot, dtype=torch.long)}
    cmask = torch.zeros((B, N), dtype=torch.bool)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.suffixes)
        cmask[b, :n] = True
        tiers[b, :n] = torch.from_numpy(it.tiers)
    from .losses import tier_pair_mask
    return {"pack": pack, "candidate_mask": cmask, "tiers": tiers, "pair_mask": tier_pair_mask(tiers),
            "n_tokens": t, "pad_id": pad_id,
            "family": [i.family for i in items], "source_id": [i.source_id for i in items],
            "decision_set_id": [i.decision_set_id for i in items], "choice_ids": [i.choice_ids for i in items],
            "items": items}


def set_rows(pack: dict) -> tuple[list[list[int]], list[int]]:
    """(graded rows of each set, packed token cost of each set) in pack order. A set costs len(prefix) +
    sum(len(suffix)) because the prefix is encoded once and shared by all of its branches."""
    seg_prefix = pack["seg_prefix"].tolist()
    row_seg = pack["row_seg"].tolist()
    seg_lens = [int(x) for x in pack["seg_lens"]]
    order, rows, cost = [], {}, {}
    for r, sg in enumerate(row_seg):
        p = seg_prefix[sg]
        if p not in rows:
            order.append(p)
            rows[p], cost[p] = [], seg_lens[p]
        rows[p].append(r)
        cost[p] += seg_lens[sg]
    return [rows[p] for p in order], [cost[p] for p in order]


def row_chunks(pack: dict, max_tokens: int | None = None) -> list[list[int]]:
    """Graded rows of the pack grouped into chunks of whole sets whose packed cost sums to <= max_tokens (a set
    bigger than the budget gets a chunk of its own). Chunks keep ascending row order, so a budget >= all tokens
    yields exactly one chunk with every row in pack order (== `embed_indices` on all rows)."""
    from .packing import chunk_plan
    rows, cost = set_rows(pack)
    return [[r for i in c for r in rows[i]] for c in chunk_plan(cost, max_tokens)]


def to_device(batch: dict, device) -> dict:
    def mv(v):
        if torch.is_tensor(v):
            return v.to(device, non_blocking=True)
        if isinstance(v, dict):
            return {k: mv(x) for k, x in v.items()}
        return v
    return {k: mv(v) for k, v in batch.items()}


def align_slots(order_from, order_to):
    """For each slot k of `order_from`, the slot index j in `order_to` holding the same candidate."""
    order_to = np.asarray(order_to)
    sorter = np.argsort(order_to)
    return sorter[np.searchsorted(order_to, np.asarray(order_from), sorter=sorter)]
