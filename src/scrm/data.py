"""Dataset loading, filtering, source mixing, token-budget batching, deterministic eval sets."""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .collator import collate
from .packing import pack_bfd
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
    """Whole sets best-fit-decreasing packed into micro-batches of <= max_tokens tokens and <= max_bs sets."""
    return [[items[i] for i in b] for b in pack_bfd([it.n_tokens for it in items], max_tokens, max_bs)]


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
EVAL_DRAW_FACTOR = 10   # per source, read at most this many x the quota while replacing sets that do not render


def _open(path: str):
    if path.startswith("hf://"):
        import fsspec
        return fsspec.open(path, "rb").open()
    return path


def _concrete(paths: list[str]) -> list[str]:
    """Expand hf:// glob patterns (local paths from `_files` are already concrete)."""
    out = []
    for p in paths:
        if p.startswith("hf://"):
            import fsspec
            out += sorted(f"hf://{x}" for x in fsspec.filesystem("hf").glob(p[len("hf://"):]))
        else:
            out.append(p)
    return out


def _eval_candidates(dcfg: dict, split: str, filters: dict, seed: int):
    """(files, {source_id: [(file_idx, row_idx), ...] in a seeded random order}) over every file of the split.
    Only the filter columns are read here, so every source of the split is reachable regardless of file order."""
    import pyarrow.parquet as pq
    pred = Pred(filters.get("include"), filters.get("exclude"))
    files = _concrete(_files(dcfg, split))
    by_src: dict[str, list] = {}
    for fi, f in enumerate(files):
        t = pq.read_table(_open(f), columns=list(FILTER_COLS))
        keep = np.where(pred.mask(t))[0]
        src = np.asarray(t.column("source_id").to_numpy(zero_copy_only=False))[keep]
        for s in np.unique(src):
            by_src.setdefault(str(s), []).append(np.stack([np.full((src == s).sum(), fi), keep[src == s]], 1))
    rng = np.random.default_rng(seed)
    return files, {s: [tuple(x) for x in rng.permutation(np.concatenate(by_src[s]))] for s in sorted(by_src)}


def _read_rows(files: list[str], locs: list[tuple[int, int]]) -> list[dict]:
    """Rows at (file_idx, row_idx) locations (BASE_COLS + candidate_rows), in the order given; reads only the row
    groups that contain them."""
    import pyarrow.parquet as pq
    cols = BASE_COLS + ["candidate_rows"]
    out: dict[tuple[int, int], dict] = {}
    by_file: dict[int, list[int]] = {}
    for fi, ri in locs:
        by_file.setdefault(int(fi), []).append(int(ri))
    for fi, rows in by_file.items():
        pf = pq.ParquetFile(_open(files[fi]))
        starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        rg = np.searchsorted(starts, rows, side="right") - 1
        for g in np.unique(rg):
            local = [r - int(starts[g]) for r, gg in zip(rows, rg) if gg == g]
            for r, rec in zip(local, pf.read_row_group(int(g), columns=cols).take(local).to_pylist()):
                out[(fi, r + int(starts[g]))] = rec
    return [out[(int(fi), int(ri))] for fi, ri in locs]


def _cap_total(per_src: dict[str, list], max_rows, seed: int) -> list:
    """Flatten per-source picks (sorted by source); if more than max_rows, keep a seeded random subset."""
    flat = [x for s in sorted(per_src) for x in per_src[s]]
    if max_rows and len(flat) > max_rows:
        keep = np.sort(np.random.default_rng(seed + 1).permutation(len(flat))[:max_rows])
        flat = [flat[i] for i in keep]
    return flat


def select_eval_rows(dcfg: dict, split: str, filters: dict, max_rows, per_source, seed: int = 0) -> list[dict]:
    """Deterministic sample: `per_source` random rows of every source in the split (None = all), then at most
    `max_rows` in total (includes candidate_rows for canonical order)."""
    files, cands = _eval_candidates(dcfg, split, filters, seed)
    picks = {s: c[:per_source] if per_source else c for s, c in cands.items()}
    return _read_rows(files, _cap_total(picks, max_rows, seed))


def _eval_renderer(renderer: Renderer) -> Renderer:
    # eval grades every shown option (render.eval_max_graded, default None = all), whatever train uses
    return Renderer(renderer.tok, {**renderer.cfg, "max_graded": renderer.cfg.get("eval_max_graded")})


def _eval_item(renderer: Renderer, row: dict):
    ex = parse_row(row, use_candidate_rows=True)
    return renderer.assemble(renderer.tokenize(ex), None, False) if ex is not None else None


class EvalSet:
    """Tokenised, canonical-order eval items (cached) split into token-budget batches."""

    def __init__(self, rows: list[dict], renderer: Renderer, dcfg: dict):
        self.r = _eval_renderer(renderer)
        items = [_eval_item(self.r, row) for row in rows]
        self._init(dcfg, [i for i in items if i is not None], len(rows), {})
        self.n_dropped = sum(i is None for i in items)

    def _init(self, dcfg, items, n_rows, dropped_by_source):
        self.items, self.n_rows = items, n_rows
        self.dropped_by_source = dropped_by_source
        self.n_dropped = sum(dropped_by_source.values())
        b = dcfg["batch"]
        self._batches = [collate(x, self.r.pad_id) for x in make_batches(self.items, b["max_tokens_per_batch"], b["max_batch_size"])]

    def batches(self):
        return self._batches

    def per_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.items:
            out[i.source_id] = out.get(i.source_id, 0) + 1
        return dict(sorted(out.items()))

    def describe(self) -> str:
        return (f"{len(self.items)} items from {len(self.per_source())} sources {self.per_source()}; "
                f"{self.n_dropped} drawn rows did not render (over max_len / untrainable) {self.dropped_by_source}")

    @classmethod
    def from_config(cls, cfg: dict, renderer: Renderer, split: str | None = None, filters: dict | None = None,
                    max_rows="cfg", per_source="cfg"):
        """`per_source` rendered sets from every source of the split: rows are drawn in a seeded random order and a
        row that does not render is replaced by the next one (at most EVAL_DRAW_FACTOR x quota rows per source),
        then at most `max_rows` sets in total."""
        d = cfg["data"]
        max_rows = d["eval_max_rows"] if max_rows == "cfg" else max_rows
        per_source = d["eval_max_rows_per_source"] if per_source == "cfg" else per_source
        files, cands = _eval_candidates(d, split or d["eval_split"], filters if filters is not None else d["eval_filters"],
                                        cfg["seed"])
        self = cls.__new__(cls)
        self.r = _eval_renderer(renderer)
        kept, dropped, n_rows = {}, {}, 0
        for s, locs in cands.items():
            quota = per_source or len(locs)
            limit = min(len(locs), quota * EVAL_DRAW_FACTOR)
            got, pos = [], 0
            while len(got) < quota and pos < limit:
                chunk = locs[pos:min(limit, pos + 2 * (quota - len(got)))]
                pos += len(chunk)
                for row in _read_rows(files, chunk):
                    it = _eval_item(self.r, row)
                    if it is None:
                        dropped[s] = dropped.get(s, 0) + 1
                    elif len(got) < quota:
                        got.append(it)
                n_rows += len(chunk)
            kept[s] = got
        self._init(d, _cap_total(kept, max_rows, cfg["seed"]), n_rows, dropped)
        return self
