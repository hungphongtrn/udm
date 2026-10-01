"""Dataset loading, filtering, source mixing, token-budget batching, deterministic eval sets."""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .collator import collate
from .render import Renderer, parse_row

META_COLS = ["decision_set_id", "source_id", "family", "label_kind", "source_split"]
BASE_COLS = META_COLS + ["state_json", "instruction_json", "options_json", "tier_json"]
FILTER_COLS = ("source_id", "family", "label_kind", "source_split")


# ----------------------------------------------------------------------------- loading
def _files(dcfg: dict, split: str, globs: list[str] | None = None):
    """List of parquet paths/URLs for a split (or the given globs relative to the data root)."""
    local = dcfg.get("local_dir")
    pats = globs or [f"data/{split}-*.parquet"]
    if local:
        out = []
        for p in pats:
            out += sorted(glob.glob(os.path.join(local, p)))
            if not globs:  # flat layout fallback
                out += sorted(glob.glob(os.path.join(local, f"{split}-*.parquet")))
        out = sorted(set(out))
        if not out:
            raise FileNotFoundError(f"no parquet files for split={split} globs={pats} under {local}")
        return out
    return [f"hf://datasets/{dcfg['repo']}/{p}" for p in pats]


def load_base(dcfg: dict, split: str, streaming: bool, candidate_rows: bool = False, globs=None):
    from datasets import load_dataset
    cols = BASE_COLS + (["candidate_rows"] if candidate_rows else [])
    return load_dataset("parquet", data_files=_files(dcfg, split, globs), split="train", streaming=streaming,
                        columns=cols)


# ----------------------------------------------------------------------------- filters / routing
def _as_list(v):
    return [v] if isinstance(v, str) else list(v)


class Pred:
    """include/exclude on string columns."""

    def __init__(self, include: dict | None = None, exclude: dict | None = None):
        self.include = {k: set(_as_list(v)) for k, v in (include or {}).items()}
        self.exclude = {k: set(_as_list(v)) for k, v in (exclude or {}).items()}

    def row(self, r: dict) -> bool:
        return all(r.get(k) in v for k, v in self.include.items()) and \
            not any(r.get(k) in v for k, v in self.exclude.items())

    def mask(self, table) -> np.ndarray:
        import pyarrow as pa
        import pyarrow.compute as pc
        m = np.ones(table.num_rows, dtype=bool)
        for k, v in self.include.items():
            col = table.column(k)
            m &= np.asarray(pc.is_in(col, value_set=pa.array(sorted(v), type=col.type)).fill_null(False))
        for k, v in self.exclude.items():
            col = table.column(k)
            m &= ~np.asarray(pc.is_in(col, value_set=pa.array(sorted(v), type=col.type)).fill_null(False))
        return m


class Router:
    """Row -> group index (first matching group wins); G = implicit 'other'; -1 = filtered out."""

    def __init__(self, filters: dict, groups: list[dict], drop_unmatched: bool):
        self.filt = Pred(filters.get("include"), filters.get("exclude"))
        self.matches = [Pred(g.get("match"), g.get("exclude")) for g in groups]
        self.G = len(groups)
        self.drop_unmatched = drop_unmatched

    def assign_row(self, r: dict) -> int:
        if not self.filt.row(r):
            return -1
        for i, m in enumerate(self.matches):
            if m.row(r):
                return i
        return -1 if self.drop_unmatched else self.G

    def assign_table(self, table) -> np.ndarray:
        out = np.full(table.num_rows, -1 if self.drop_unmatched else self.G, dtype=np.int64)
        done = np.zeros(table.num_rows, dtype=bool)
        for i, m in enumerate(self.matches):
            mk = m.mask(table) & ~done
            out[mk] = i
            done |= mk
        out[~self.filt.mask(table)] = -1
        return out


# ----------------------------------------------------------------------------- group sources / mixer
class GroupSource:
    def __init__(self, name: str, weight: float, max_rows: int | None):
        self.name, self.weight, self.max_rows = name, float(weight), max_rows
        self.size: int | None = None

    def iter_rows(self, seed: int, wid: int = 0, nw: int = 1) -> Iterator[dict]:
        raise NotImplementedError


class MapGroup(GroupSource):
    def __init__(self, name, weight, max_rows, ds, indices: np.ndarray, seed: int):
        super().__init__(name, weight, max_rows)
        if max_rows and len(indices) > max_rows:
            indices = np.sort(np.random.default_rng(seed + 31).permutation(indices)[:max_rows])
        self.ds, self.indices, self.size = ds, indices, len(indices)

    def iter_rows(self, seed, wid=0, nw=1):
        perm = np.random.default_rng(seed).permutation(self.indices)[wid::nw]
        for s in range(0, len(perm), 64):
            chunk = self.ds[perm[s:s + 64].tolist()]
            keys = list(chunk.keys())
            for j in range(len(chunk[keys[0]])):
                yield {k: chunk[k][j] for k in keys}


class StreamGroup(GroupSource):
    def __init__(self, name, weight, max_rows, ds, buffer: int, seed: int):
        super().__init__(name, weight, max_rows)
        self.ds, self.buffer, self.seed = ds, buffer, seed

    def iter_rows(self, seed, wid=0, nw=1):
        ds = self.ds.shuffle(seed=seed, buffer_size=self.buffer)
        if self.max_rows:
            ds = ds.take(self.max_rows)
        yield from ds   # HF IterableDataset shards across DataLoader workers itself


def build_groups(dcfg: dict, split: str, filters: dict, streaming: bool, seed: int) -> list[GroupSource]:
    groups_cfg = dcfg.get("groups") or []
    router = Router(filters, groups_cfg, dcfg.get("drop_unmatched", False))
    names = [g.get("name", f"group{i}") for i, g in enumerate(groups_cfg)] + ["other"]
    weights = [g.get("weight", 1.0) for g in groups_cfg] + [dcfg.get("other_weight", 1.0)]
    caps = [g.get("max_rows") for g in groups_cfg] + [dcfg.get("other_max_rows")]
    out: list[GroupSource] = []
    if not streaming:
        ds = load_base(dcfg, split, False)
        assign = router.assign_table(ds.data.table if hasattr(ds.data, "table") else ds.data)
        for g in range(router.G + 1):
            idx = np.where(assign == g)[0]
            if len(idx):
                out.append(MapGroup(names[g], weights[g], caps[g], ds, idx, seed))
    else:
        for g in range(router.G + 1):
            globs = groups_cfg[g].get("files") if g < router.G else None
            if g == router.G and drop_all_other(dcfg):
                continue
            ds = load_base(dcfg, split, True, globs=globs)
            ds = ds.filter(lambda r, g=g: router.assign_row(r) == g)
            out.append(StreamGroup(names[g], weights[g], caps[g], ds, dcfg.get("shuffle_buffer", 10000), seed))
    out = [o for o in out if o.weight > 0]
    if not out:
        raise RuntimeError("no training rows matched filters/groups")
    return out


def drop_all_other(dcfg) -> bool:
    return bool(dcfg.get("drop_unmatched", False)) or dcfg.get("other_weight", 1.0) <= 0


class Mixer:
    """Infinite (or `max_examples`-bounded) weighted interleaving of group row iterators."""

    def __init__(self, groups: list[GroupSource], seed: int, wid=0, nw=1, max_examples=None, epoch0=0):
        self.groups, self.seed, self.wid, self.nw, self.max_examples, self.epoch0 = groups, seed, wid, nw, max_examples, epoch0

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.wid])
        G = len(self.groups)
        w = np.array([g.weight for g in self.groups], dtype=float)
        alive = np.ones(G, dtype=bool)
        epochs = [self.epoch0] * G
        iters = [g.iter_rows(self.seed + 1000 * epochs[i] + i, self.wid, self.nw) for i, g in enumerate(self.groups)]
        n = 0
        while alive.any():
            p = w * alive
            g = int(rng.choice(G, p=p / p.sum()))
            try:
                row = next(iters[g])
            except StopIteration:
                epochs[g] += 1
                iters[g] = self.groups[g].iter_rows(self.seed + 1000 * epochs[g] + g, self.wid, self.nw)
                try:
                    row = next(iters[g])
                except StopIteration:
                    alive[g] = False
                    continue
            yield self.groups[g].name, row
            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                return


# ----------------------------------------------------------------------------- batching
def make_batches(items, max_tokens: int, max_bs: int):
    items = sorted(items, key=lambda i: i.n_tokens)
    batches, cur = [], []
    for it in items:
        if cur and ((len(cur) + 1) * it.n_tokens > max_tokens or len(cur) >= max_bs):
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


class TrainStream(IterableDataset):
    """Yields collated, token-budget batches (tokenisation + collation happen inside DataLoader workers)."""

    def __init__(self, groups, renderer: Renderer, dcfg: dict, seed: int, max_examples=None, keep_items=False,
                 epoch0: int = 0):
        self.groups, self.r, self.seed = groups, renderer, seed
        self.bcfg = dcfg["batch"]
        self.max_examples, self.keep_items, self.epoch0 = max_examples, keep_items, epoch0
        self.total_rows = sum(g.size for g in groups) if all(g.size is not None for g in groups) else None
        self.stats = {"dropped": 0}

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng([self.seed, wid, 99])
        me = None if self.max_examples is None else max(1, self.max_examples // nw)
        pool = []
        for _, row in Mixer(self.groups, self.seed, wid, nw, me, self.epoch0):
            ex = parse_row(row, use_candidate_rows=False)
            if ex is None:
                continue
            it = self.r.assemble(self.r.tokenize(ex), rng, shuffle=True)
            if it is None:
                continue
            pool.append(it)
            if len(pool) >= self.bcfg["bucket_size"]:
                yield from self._flush(pool, rng)
                pool = []
        yield from self._flush(pool, rng)

    def _flush(self, pool, rng):
        bs = make_batches(pool, self.bcfg["max_tokens_per_batch"], self.bcfg["max_batch_size"])
        for i in rng.permutation(len(bs)):
            b = collate(bs[i], self.r.pad_id)
            if not self.keep_items:
                b.pop("items")
            yield b


def make_train_loader(cfg: dict, renderer: Renderer, seed: int, max_examples=None, keep_items=False, epoch0=0):
    d = cfg["data"]
    groups = build_groups(d, d["train_split"], d["filters"], d["streaming"], seed)
    stream = TrainStream(groups, renderer, d, seed, max_examples, keep_items, epoch0)
    nw = d["num_workers"]
    kw = dict(num_workers=nw, prefetch_factor=d["prefetch_factor"], persistent_workers=True) if nw > 0 else {}
    return DataLoader(stream, batch_size=None, **kw), stream


# ----------------------------------------------------------------------------- eval
def select_eval_rows(dcfg: dict, split: str, filters: dict, max_rows, per_source, seed: int = 0, streaming=None) -> list[dict]:
    """Deterministic capped sample of rows (includes candidate_rows for canonical order)."""
    streaming = dcfg["streaming"] if streaming is None else streaming
    pred = Pred(filters.get("include"), filters.get("exclude"))
    rows: list[dict] = []
    if not streaming:
        ds = load_base(dcfg, split, False, candidate_rows=True)
        table = ds.data.table if hasattr(ds.data, "table") else ds.data
        keep = pred.mask(table)
        src = np.asarray(table.column("source_id").to_numpy(zero_copy_only=False))
        srcs = sorted(set(src[keep].tolist()))
        per = per_source
        idx = []
        rng = np.random.default_rng(seed)
        for s in srcs:
            ii = np.where(keep & (src == s))[0]
            if per and len(ii) > per:
                ii = rng.permutation(ii)[:per]
            idx.extend(ii.tolist())
        idx = sorted(idx)
        if max_rows and len(idx) > max_rows:
            idx = sorted(rng.permutation(idx)[:max_rows].tolist())
        for s in range(0, len(idx), 256):
            ch = ds[idx[s:s + 256]]
            ks = list(ch.keys())
            rows += [{k: ch[k][j] for k in ks} for j in range(len(ch[ks[0]]))]
    else:
        ds = load_base(dcfg, split, True, candidate_rows=True)
        counts: dict[str, int] = {}
        scanned = 0
        for r in ds:
            scanned += 1
            if pred.row(r) and (not per_source or counts.get(r["source_id"], 0) < per_source):
                counts[r["source_id"]] = counts.get(r["source_id"], 0) + 1
                rows.append(r)
            if (max_rows and len(rows) >= max_rows) or scanned >= dcfg.get("eval_max_scan_rows", 200000):
                break
    return rows


class EvalSet:
    """Tokenised, canonical-order eval items (cached) split into token-budget batches."""

    def __init__(self, rows: list[dict], renderer: Renderer, dcfg: dict):
        self.r = renderer
        self.items = []
        self.n_rows, self.n_dropped = len(rows), 0
        for row in rows:
            ex = parse_row(row, use_candidate_rows=True)
            it = renderer.assemble(renderer.tokenize(ex), None, False) if ex is not None else None
            if it is None:
                self.n_dropped += 1
                continue
            self.items.append(it)
        b = dcfg["batch"]
        self._batches = [collate(x, renderer.pad_id) for x in make_batches(self.items, b["max_tokens_per_batch"], b["max_batch_size"])]

    def batches(self):
        return self._batches

    @classmethod
    def from_config(cls, cfg: dict, renderer: Renderer, split: str | None = None, filters: dict | None = None,
                    max_rows="cfg", per_source="cfg"):
        d = cfg["data"]
        rows = select_eval_rows(d, split or d["eval_split"], filters if filters is not None else d["eval_filters"],
                                d["eval_max_rows"] if max_rows == "cfg" else max_rows,
                                d["eval_max_rows_per_source"] if per_source == "cfg" else per_source, cfg["seed"])
        return cls(rows, renderer, d)
