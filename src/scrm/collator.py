"""Collation of Items into padded tensors."""
from __future__ import annotations

import torch


def collate(items, pad_id: int = 0) -> dict:
    """Items (decision sets) -> one row per graded option.
    input_ids/attention_mask [S, L]; read_pos [S] (last real token = end-of-turn); seq_b / seq_slot [S] map each row
    to its set and candidate slot; candidate_mask / tiers / pair_mask are per set [B, N(, N)]."""
    B = len(items)
    seqs = [(b, k, s) for b, it in enumerate(items) for k, s in enumerate(it.seqs)]
    S = len(seqs)
    L = max(len(s) for _, _, s in seqs)
    N = max(len(i.seqs) for i in items)
    input_ids = torch.full((S, L), pad_id, dtype=torch.long)
    attn = torch.zeros((S, L), dtype=torch.long)
    read_pos = torch.zeros(S, dtype=torch.long)
    seq_b = torch.zeros(S, dtype=torch.long)
    seq_slot = torch.zeros(S, dtype=torch.long)
    for r, (b, k, s) in enumerate(seqs):
        input_ids[r, :len(s)] = torch.from_numpy(s)
        attn[r, :len(s)] = 1
        read_pos[r], seq_b[r], seq_slot[r] = len(s) - 1, b, k
    cmask = torch.zeros((B, N), dtype=torch.bool)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.seqs)
        cmask[b, :n] = True
        tiers[b, :n] = torch.from_numpy(it.tiers)
    from .losses import tier_pair_mask
    return {"input_ids": input_ids, "attention_mask": attn, "read_pos": read_pos, "seq_b": seq_b, "seq_slot": seq_slot,
            "candidate_mask": cmask, "tiers": tiers, "pair_mask": tier_pair_mask(tiers),
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
