"""Collation of Items into padded tensors."""
from __future__ import annotations

import torch


def collate(items, pad_id: int = 0) -> dict:
    B = len(items)
    L = max(len(i.input_ids) for i in items)
    N = max(len(i.cand_pos) for i in items)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    cpos = torch.zeros((B, N), dtype=torch.long)
    cmask = torch.zeros((B, N), dtype=torch.bool)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    for b, it in enumerate(items):
        l, n = len(it.input_ids), len(it.cand_pos)
        input_ids[b, :l] = torch.from_numpy(it.input_ids)
        attn[b, :l] = 1
        cpos[b, :n] = torch.from_numpy(it.cand_pos)
        cmask[b, :n] = True
        tiers[b, :n] = torch.from_numpy(it.tiers)
    from .losses import tier_pair_mask
    return {"input_ids": input_ids, "attention_mask": attn, "candidate_positions": cpos, "candidate_mask": cmask,
            "tiers": tiers, "pair_mask": tier_pair_mask(tiers),
            "family": [i.family for i in items], "source_id": [i.source_id for i in items],
            "decision_set_id": [i.decision_set_id for i in items], "choice_ids": [i.choice_ids for i in items],
            "items": items}


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def align_slots(order_from, order_to):
    """For each slot k of `order_from`, the slot index j in `order_to` holding the same candidate."""
    import numpy as np
    order_to = np.asarray(order_to)
    sorter = np.argsort(order_to)
    return sorter[np.searchsorted(order_to, np.asarray(order_from), sorter=sorter)]
