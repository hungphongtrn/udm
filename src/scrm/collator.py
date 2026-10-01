"""Collation of Items into padded tensors."""
from __future__ import annotations

import torch


def collate(items, pad_id: int = 0) -> dict:
    """Items (decision sets) -> per-set shared prefix + padded per-option suffixes.
    sets[b] = {prefix [P], suffix [n, S] (right padded), suffix_mask [n, S], read_pos [n] (last real suffix token)};
    candidate_mask / tiers / pair_mask are per set [B, N(, N)]."""
    B = len(items)
    N = max(len(i.suffixes) for i in items)
    sets, n_tok, n_pad = [], 0, 0
    for it in items:
        n, S = len(it.suffixes), max(len(x) for x in it.suffixes)
        suf = torch.full((n, S), pad_id, dtype=torch.long)
        sm = torch.zeros((n, S), dtype=torch.long)
        for k, x in enumerate(it.suffixes):
            suf[k, :len(x)] = torch.from_numpy(x)
            sm[k, :len(x)] = 1
        sets.append({"prefix": torch.from_numpy(it.prefix).long(), "suffix": suf, "suffix_mask": sm,
                     "read_pos": sm.sum(1) - 1})
        n_tok += len(it.prefix) + int(sm.sum())
        n_pad += len(it.prefix) + n * S
    cmask = torch.zeros((B, N), dtype=torch.bool)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.suffixes)
        cmask[b, :n] = True
        tiers[b, :n] = torch.from_numpy(it.tiers)
    from .losses import tier_pair_mask
    return {"sets": sets, "candidate_mask": cmask, "tiers": tiers, "pair_mask": tier_pair_mask(tiers),
            "n_tokens": n_tok, "n_padded_tokens": n_pad, "pad_id": pad_id,
            "family": [i.family for i in items], "source_id": [i.source_id for i in items],
            "decision_set_id": [i.decision_set_id for i in items], "choice_ids": [i.choice_ids for i in items],
            "items": items}


def to_device(batch: dict, device) -> dict:
    def mv(v):
        if torch.is_tensor(v):
            return v.to(device, non_blocking=True)
        if isinstance(v, dict):
            return {k: mv(x) for k, x in v.items()}
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [mv(x) for x in v]
        return v
    return {k: mv(v) for k, v in batch.items()}


def align_slots(order_from, order_to):
    """For each slot k of `order_from`, the slot index j in `order_to` holding the same candidate."""
    import numpy as np
    order_to = np.asarray(order_to)
    sorter = np.argsort(order_to)
    return sorter[np.searchsorted(order_to, np.asarray(order_from), sorter=sorter)]
