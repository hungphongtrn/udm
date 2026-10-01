"""Row parsing, text rendering, tokenisation, truncation / candidate subsampling, sequence assembly."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tokens import SPECIAL_TOKENS

# ----------------------------------------------------------------------------- text rendering


def _loads(s: Any) -> Any:
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", "replace")
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _dump(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def render_instruction(obj: Any) -> str:
    """JSON string -> as is. Object {type, instructions, criteria?} -> instructions + criteria lines."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        ins = obj.get("instructions", obj.get("instruction"))
        if ins is not None:
            parts.append(_dump(ins).strip())
        crit = obj.get("criteria")
        if crit:
            lines = []
            if isinstance(crit, dict):
                lines = [f"- {k}: {_dump(v)}" for k, v in crit.items()]
            elif isinstance(crit, (list, tuple)):
                lines = [f"- {_dump(v)}" for v in crit]
            else:
                lines = [f"- {_dump(crit)}"]
            parts.append("Criteria:\n" + "\n".join(lines))
        if not parts:
            parts.append(_dump(obj))
        return "\n".join(parts)
    return _dump(obj)


def render_state(obj: Any) -> str:
    """JSON string -> as is. Anything else -> compact readable JSON."""
    if obj is None:
        return ""
    return obj if isinstance(obj, str) else _dump(obj)


@dataclass
class Example:
    decision_set_id: str
    source_id: str
    family: str
    label_kind: str
    source_split: str
    instruction: str
    state: str
    choice_ids: list[str]
    texts: list[str]
    tiers: list[int]

    @classmethod
    def from_raw(cls, instruction, state, candidates: list[str]) -> "Example":
        """Unlabelled example for inference (tiers all 0)."""
        return cls("", "", "", "", "", render_instruction(instruction), render_state(state),
                   [str(i) for i in range(len(candidates))], [c if isinstance(c, str) else _dump(c) for c in candidates],
                   [0] * len(candidates))


def parse_row(row: dict, use_candidate_rows: bool = True) -> Example | None:
    """Contract row -> Example, or None if untrainable (<2 candidates / no pair)."""
    try:
        options = _loads(row["options_json"])
        tier_list = _loads(row["tier_json"])
        if not isinstance(options, dict) or not isinstance(tier_list, list):
            return None
        tier_of = {}
        for k, grp in enumerate(tier_list):
            for cid in grp:
                tier_of[cid] = k
        order: list[str] = []
        cr = row.get("candidate_rows") if use_candidate_rows else None
        if cr:
            order = [c["choice_id"] for c in sorted(cr, key=lambda c: c["candidate_order"])]
            order = [c for c in order if c in options]
            seen = set(order)
            order += [c for c in options if c not in seen]
        else:
            order = list(options.keys())
        order = [c for c in order if c in tier_of]
        if len(order) < 2 or len({tier_of[c] for c in order}) < 2:
            return None
        return Example(
            decision_set_id=str(row.get("decision_set_id") or ""), source_id=str(row.get("source_id") or ""),
            family=str(row.get("family") or ""), label_kind=str(row.get("label_kind") or ""),
            source_split=str(row.get("source_split") or ""),
            instruction=render_instruction(_loads(row.get("instruction_json"))),
            state=render_state(_loads(row.get("state_json"))),
            choice_ids=order, texts=[_dump(options[c]) for c in order], tiers=[tier_of[c] for c in order])
    except Exception:
        return None


# ----------------------------------------------------------------------------- tokenisation / assembly

DEFAULT_RENDER = {"max_len": 2048, "cand_max_tokens": 128, "state_max_tokens": 1024, "instr_max_tokens": 256,
                  "max_candidates": 64, "min_state_tokens": 64, "state_truncate": "middle"}


@dataclass
class TokExample:
    ex: Example
    instr_ids: list[int]
    state_ids: list[int]
    cand_ids: list[list[int]]
    tiers: np.ndarray


@dataclass
class Item:
    input_ids: np.ndarray
    cand_pos: np.ndarray
    tiers: np.ndarray
    order: np.ndarray          # tokex candidate index for each slot
    choice_ids: list[str]
    family: str
    source_id: str
    decision_set_id: str
    tok: TokExample = field(repr=False, default=None)
    cap: int | None = None
    kept: list = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def subsample_indices(tiers: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick <= max_n candidates keeping ALL tier-0 items (as far as budget allows) and at least one
    from every other tier; remaining slots filled randomly. Returned sorted (canonical order)."""
    n = len(tiers)
    if n <= max_n:
        return np.arange(n)
    levels = sorted(set(tiers.tolist()))
    top = np.where(tiers == levels[0])[0]
    n_other_levels = len(levels) - 1
    keep: list[int] = []
    # reserve room for one representative of other tiers (at most max_n - 1 of them keep a pair possible)
    reserve = min(n_other_levels, max(1, max_n // 2)) if n_other_levels else 0
    top_cap = max(1, max_n - max(reserve, 1))
    top_keep = rng.permutation(top)[:top_cap] if len(top) > top_cap else top
    keep.extend(top_keep.tolist())
    reps = []
    for lv in levels[1:]:
        reps.append(int(rng.choice(np.where(tiers == lv)[0])))
    rng.shuffle(reps)
    for r in reps:
        if len(keep) < max_n:
            keep.append(r)
    chosen = set(keep)
    rest = np.array([i for i in range(n) if i not in chosen])
    room = max_n - len(keep)
    if room > 0 and len(rest):
        keep.extend(rng.permutation(rest)[:room].tolist())
    return np.array(sorted(keep))


def _protected(tiers_kept: np.ndarray, idxs: list[int]) -> set[int]:
    """Candidates that must not be dropped by the budget loop: all tier-0 + one rep of each other tier."""
    tmin = tiers_kept[idxs].min()
    prot = {i for i in idxs if tiers_kept[i] == tmin}
    seen = set()
    for i in idxs:
        t = tiers_kept[i]
        if t != tmin and t not in seen:
            seen.add(t)
            prot.add(i)
    return prot


class Renderer:
    def __init__(self, tokenizer, rcfg: dict | None = None):
        self.tok = tokenizer
        self.cfg = {**DEFAULT_RENDER, **(rcfg or {})}
        self.sp = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.marker = tokenizer("\n...\n", add_special_tokens=False)["input_ids"]
        self.nl2 = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

    # --- tokenisation
    def tokenize(self, ex: Example) -> TokExample:
        c = self.cfg
        state = ex.state
        max_chars = 8 * max(c["state_max_tokens"] or c["max_len"], 64) + 64
        if len(state) > max_chars:   # cheap char-level pre-truncation (middle) before tokenising
            h = int(max_chars * 0.6)
            state = state[:h] + "\n...\n" + state[-(max_chars - h):]
        texts = [ex.instruction, state] + list(ex.texts)
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        instr = ids[0][: c["instr_max_tokens"]] + (self.nl2 if ex.instruction else [])
        cands = [x[: c["cand_max_tokens"]] or [self.nl2[0] if self.nl2 else 0] for x in ids[2:]]
        return TokExample(ex, instr, ids[1], cands, np.asarray(ex.tiers))

    def _truncate_state(self, ids: list[int], budget: int) -> list[int]:
        if budget <= 0:
            return []
        if len(ids) <= budget:
            return ids
        mode = self.cfg["state_truncate"]
        if mode == "right":
            return ids[:budget]
        if mode == "left":
            return ids[-budget:]
        m = self.marker
        if budget <= len(m) + 4:
            return ids[:budget]
        head = int((budget - len(m)) * 0.6)
        tail = budget - len(m) - head
        return ids[:head] + m + (ids[-tail:] if tail > 0 else [])

    # --- assembly
    def assemble(self, t: TokExample, rng: np.random.Generator | None, shuffle: bool, kept=None, cap=None,
                 relax: bool = False) -> Item | None:
        """Build an Item. `rng`=None & shuffle=False -> canonical order, deterministic subsampling.
        relax=True (inference): never drop the example for lack of trainable pairs."""
        c = self.cfg
        sub_rng = rng if rng is not None else np.random.default_rng(
            zlib.crc32(t.ex.decision_set_id.encode()) if t.ex.decision_set_id else 0)
        n = len(t.cand_ids)
        tiers = t.tiers
        if kept is None:
            idxs = subsample_indices(tiers, c["max_candidates"], sub_rng).tolist() if n > c["max_candidates"] \
                else list(range(n))
        else:
            idxs = list(kept)
        cand = t.cand_ids if cap is None else [x[:cap] for x in t.cand_ids]
        overhead = 1 + len(t.instr_ids) + 3
        min_state = min(len(t.state_ids), c["min_state_tokens"])

        def cost(ix):
            return sum(len(cand[i]) + 2 for i in ix)

        if kept is None:
            # budget loop 1: drop unprotected candidates (random) until minimal state fits
            while c["max_len"] - overhead - cost(idxs) < min_state:
                prot = _protected(tiers, idxs)
                drop = [i for i in idxs if i not in prot]
                if not drop:
                    break
                idxs.remove(int(sub_rng.choice(drop)))
            # loop 2: shrink per-candidate tokens
            avail = c["max_len"] - overhead - min_state
            if cost(idxs) > avail:
                cap = max(4, (avail // max(len(idxs), 1)) - 2)
                cand = [x[:cap] for x in t.cand_ids]
                # loop 3: still too long -> keep minimal pair set
                while cost(idxs) > avail and len(idxs) > 2:
                    prot = sorted(_protected(tiers, idxs))
                    drop = [i for i in idxs if i not in prot] or [i for i in prot if i != prot[0]]
                    tmin = tiers[idxs].min()
                    drop = [i for i in drop if not (tiers[i] == tmin and sum(tiers[j] == tmin for j in idxs) == 1)]
                    if not drop:
                        break
                    idxs.remove(int(sub_rng.choice(drop)))
                if cost(idxs) > avail:
                    return None
        if len(idxs) < 1 or (not relax and len(set(tiers[idxs].tolist())) < 2):
            return None
        kept_final = sorted(idxs)
        order = list(kept_final)
        if shuffle:
            (rng or sub_rng).shuffle(order)
        avail_state = c["max_len"] - overhead - cost(order)
        sb = min(avail_state, c["state_max_tokens"] or avail_state)
        state = self._truncate_state(t.state_ids, sb)
        sp = self.sp
        ids = [sp["<|state_start|>"]] + t.instr_ids + state + [sp["<|state_end|>"], sp["<|candidate_set_start|>"]]
        pos = []
        for i in order:
            ids.append(sp["<|candidate_start|>"])
            ids.extend(cand[i])
            ids.append(sp["<|candidate_end|>"])
            pos.append(len(ids) - 1)
        ids.append(sp["<|candidate_set_end|>"])
        order_a = np.asarray(order)
        it = Item(np.asarray(ids, dtype=np.int64), np.asarray(pos, dtype=np.int64), tiers[order_a].astype(np.int64),
                  order_a, [t.ex.choice_ids[i] for i in order], t.ex.family, t.ex.source_id, t.ex.decision_set_id,
                  tok=t, cap=cap)
        it.kept = kept_final
        return it

    def reshuffle(self, item: Item, rng: np.random.Generator) -> Item:
        return self.assemble(item.tok, rng, True, kept=item.kept, cap=item.cap)
