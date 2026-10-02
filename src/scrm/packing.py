"""Best-Fit-Decreasing bin packing of decision sets into token-budgeted micro-batches ("packs").

`SegmentTree` and the BFD loop are adapted from TRL (`trl/data_utils.py`: `_SegmentTree`, `_pack_bfd`; Apache-2.0,
Copyright The HuggingFace Team), after "Fewer Truncations Improve Language Modeling" (arXiv:2404.10830). TRL packs
token lists of a pyarrow table; here the items are whole decision sets (never split or truncated) and the result is a
list of index bins. A set longer than the capacity gets a bin of its own.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence


class SegmentTree:
    """`SegmentTree(maxval)`: efficiently finds the smallest stored value >= a query in [1, maxval]."""

    def __init__(self, maxval: int):
        self.maxval = maxval
        self.tree_size = 1 << (maxval - 1).bit_length()   # round up to a power of two
        self.tree = [0] * (2 * self.tree_size)

    def _update(self, i: int):
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
            self.tree[i] = left if left >= right else right

    def add(self, val: int):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        self._update(i)

    def remove(self, val: int):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        self._update(i)

    def search(self, val: int) -> int:
        assert 0 < val <= self.maxval
        i = 1
        while i < self.tree_size:
            i = i << 1 if self.tree[i << 1] >= val else (i << 1) + 1
        return self.tree[i]


def pack_bfd(lengths: Sequence[int], capacity: int, max_items: int | None = None) -> list[list[int]]:
    """Best-fit-decreasing: indices of `lengths` grouped into bins of total length <= capacity (and <= max_items
    members). Items longer than `capacity` are returned as singleton bins. Deterministic for a given input."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[dict] = []
    tree = SegmentTree(capacity)
    tree.add(capacity)                                   # a fresh, empty bin is always available
    space_to_bin: dict[int, deque] = defaultdict(deque)  # remaining space -> bins with exactly that space
    for idx in order:
        length = max(int(lengths[idx]), 1)
        if length > capacity:
            bins.append({"ids": [idx], "length": length})
            continue
        space = tree.search(length)
        if space < capacity:
            b = space_to_bin[space].popleft()            # existing bin with the tightest fit
            if not space_to_bin[space]:
                tree.remove(space)
        else:
            b = {"ids": [], "length": 0}
            bins.append(b)
        b["ids"].append(idx)
        b["length"] += length
        space -= length
        if space > 0 and (max_items is None or len(b["ids"]) < max_items):
            space_to_bin[space].append(b)
            tree.add(space)
    return [b["ids"] for b in bins]
