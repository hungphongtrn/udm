"""Collation of Items into padding-free packs."""
from __future__ import annotations

import numpy as np
import torch


def collate(items, pad_id: int = 0) -> dict:
    """Items (decision sets) -> one padding-free pack. Every graded option is a full sequence (shared prompt + its
    suffix); all sequences are concatenated in set order:
    pack = {input_ids [T], position_ids [T] (restart at 0 per sequence), seq_lens (list[int], one per sequence),
            read_idx [M] (last token of each sequence), row_set [M] / row_slot [M] (set and slot of each sequence)};
    candidate_mask / tiers / pair_mask are per set [B, N(, N)]."""
    B = len(items)
    N = max(len(i.suffixes) for i in items)
    seqs = [np.concatenate([it.prefix, x]) for it in items for x in it.suffixes]
    seq_lens = [len(x) for x in seqs]
    lens = torch.tensor(seq_lens)
    ends = lens.cumsum(0)
    ns = torch.tensor([len(it.suffixes) for it in items])
    pack = {"input_ids": torch.from_numpy(np.concatenate(seqs)).long(),
            "position_ids": torch.cat([torch.arange(n) for n in seq_lens]),
            "seq_lens": seq_lens, "read_idx": ends - 1,
            "row_set": torch.repeat_interleave(torch.arange(B), ns),
            "row_slot": torch.cat([torch.arange(int(n)) for n in ns])}
    cmask = torch.zeros((B, N), dtype=torch.bool)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.suffixes)
        cmask[b, :n] = True
        tiers[b, :n] = torch.from_numpy(it.tiers)
    from .losses import tier_pair_mask
    return {"pack": pack, "candidate_mask": cmask, "tiers": tiers, "pair_mask": tier_pair_mask(tiers),
            "n_tokens": int(ends[-1]), "pad_id": pad_id,
            "family": [i.family for i in items], "source_id": [i.source_id for i in items],
            "decision_set_id": [i.decision_set_id for i in items], "choice_ids": [i.choice_ids for i in items],
            "items": items}


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
