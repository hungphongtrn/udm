"""Build CLI: convert source datasets into udm-massive-typed v2 parquet shards.

    python -m scrm_data.build --source tasksource samatv samatv_clean50k openjev --out /data/scrm_out \
        --cache-dir /data/hf_cache --workers 32

Stages (so leakage filtering sees the held-out keys): 1. eval units (validation/test/ood/calibration),
2. train units, 3. dependent train units (samatv_clean50k, deduplicated against samatv).
Everything is resumable: a finished unit leaves <out>/_work/<slug>/<unit>/DONE.json.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import RECIPE_VERSION, SCHEMA_VERSION, TARGET_REPO
from . import keys as K
from .hfio import open_parquet
from .schema import SCHEMA, rows_to_table
from .sources import openjev, samatv, tasksource
from .sources.base import ConvResult, Unit

ALL_SOURCES = ["tasksource", "samatv", "samatv_clean50k", "openjev"]
DEPENDS = {"samatv_clean50k": ["samatv"]}      # dedupe against these sources' keys/ids
STAGE_ORDER = ["eval", "train", "train_dep"]


# ---------------------------------------------------------------------------- source specs
def _prefilter_samatv_default(batch: pa.RecordBatch):
    mask = pc.fill_null(pc.struct_field(batch.column("training"), "choice_eligible"), False)
    n_bad = batch.num_rows - (pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)
    return batch.filter(mask), {"not_choice_eligible": n_bad} if n_bad else {}


SPEC = {
    "tasksource": dict(
        columns=tasksource.RAW_COLUMNS, batch_size=4000, prefilter=None,
        convert=lambda raw, u, rev: tasksource.convert(raw, u.split, rev, u.config)),
    "samatv": dict(
        columns=samatv.DEFAULT_COLUMNS, batch_size=96, prefilter=_prefilter_samatv_default,
        convert=lambda raw, u, rev: samatv.convert_default(raw, u.split, rev, u.config)),
    "samatv_clean50k": dict(
        columns=samatv.CLEAN_COLUMNS, batch_size=256, prefilter=None,
        convert=lambda raw, u, rev: samatv.convert_clean(raw, "train", rev, u.config)),
    "openjev": dict(
        columns=None, batch_size=4000, prefilter=None,
        convert=lambda raw, u, rev: openjev.convert(raw, u.split, u.config, rev)),
}
SOURCE_ID = {"tasksource": tasksource.REPO_ID, "samatv": samatv.REPO_ID, "samatv_clean50k": samatv.REPO_ID,
             "openjev": openjev.REPO_ID}
REVISIONS = {"tasksource": tasksource.REVISION, "samatv": samatv.REVISION, "samatv_clean50k": samatv.REVISION,
             "openjev": openjev.REVISION}


# ---------------------------------------------------------------------------- writer
class PartWriter:
    """Writes rows of one split into rolling parquet parts (~target_bytes each, zstd)."""

    def __init__(self, directory: str, split: str, target_bytes: int, level: int = 3):
        self.dir, self.split, self.target, self.level = directory, split, target_bytes, level
        self.idx = 0
        self.fh = None
        self.w = None
        self.cur = None
        self.cur_rows = 0
        self.parts: list = []

    def _open(self):
        name = f"{self.split}-p{self.idx:04d}.parquet"
        self.cur = os.path.join(self.dir, name)
        self.fh = open(self.cur, "wb")
        self.w = pq.ParquetWriter(self.fh, SCHEMA, compression="zstd", compression_level=self.level)
        self.cur_rows = 0
        self.idx += 1

    def _close(self):
        if self.w is not None:
            self.w.close()
            size = self.fh.tell()
            self.fh.close()
            self.parts.append({"split": self.split, "file": os.path.basename(self.cur), "rows": self.cur_rows, "bytes": size})
            self.w = self.fh = None

    def write(self, rows: list):
        if not rows:
            return
        if self.w is None:
            self._open()
        self.w.write_table(rows_to_table(rows), row_group_size=2000)
        self.cur_rows += len(rows)
        if self.fh.tell() >= self.target:
            self._close()

    def close(self):
        self._close()
        return self.parts


# ---------------------------------------------------------------------------- unit execution
def _work_dir(out: str, unit: Unit) -> str:
    return os.path.join(out, "_work", unit.source, unit.unit_id)


def run_unit(unit: Unit, ctx: dict) -> dict:
    out = ctx["out"]
    work = _work_dir(out, unit)
    done = os.path.join(work, "DONE.json")
    if os.path.exists(done) and not ctx.get("force"):
        with open(done) as f:
            st = json.load(f)
        st["resumed"] = True
        return st
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    spec = SPEC[unit.source]
    rev = unit.revision
    pf = open_parquet(unit.repo_id, unit.revision, unit.path, ctx.get("cache_dir"), stream=ctx.get("stream", False),
                      local_root=ctx.get("local_root"))
    rgs = list(range(pf.num_row_groups)) if unit.row_groups is None else list(range(*unit.row_groups))

    # leakage / duplicate control inputs
    heldout = None
    if unit.stage in ("train", "train_dep"):
        hp = os.path.join(out, "keys", "_heldout.npy")
        heldout = np.load(hp, mmap_mode="r") if os.path.exists(hp) else np.empty(0, dtype=np.uint64)
    dd_keys = dd_ids = None
    if unit.source in DEPENDS:
        parts_k, parts_i = [], []
        for dep in DEPENDS[unit.source]:
            parts_k.append(K.load_split_keys(out, ["train", "validation", "test"], only_slugs=[dep], col=0))
            parts_i.append(K.load_split_keys(out, ["train", "validation", "test"], only_slugs=[dep], col=1))
        dd_keys, dd_ids = np.unique(np.concatenate(parts_k)), np.unique(np.concatenate(parts_i))

    writers = {sp: PartWriter(work, sp, ctx["shard_bytes"]) for sp in ("train", "validation", "test")}
    keys_out = {sp: [] for sp in writers}
    seen = {sp: set() for sp in writers}
    drops = collections.Counter()
    fam = collections.defaultdict(collections.Counter)      # split -> family -> rows
    kinds = collections.defaultdict(collections.Counter)    # split -> label_kind -> rows
    cand_rows = collections.Counter()
    written = collections.Counter()
    read = 0
    limit = ctx.get("limit_rows")
    pending = {sp: [] for sp in writers}

    kw = {"batch_size": spec["batch_size"], "row_groups": rgs}
    if spec["columns"]:
        kw["columns"] = spec["columns"]
    for batch in pf.iter_batches(**kw):
        if limit is not None and read >= limit:
            break
        if limit is not None and read + batch.num_rows > limit:
            batch = batch.slice(0, limit - read)
        read += batch.num_rows
        if spec["prefilter"]:
            batch, pre = spec["prefilter"](batch)
            drops.update(pre)
        if batch.num_rows == 0:
            continue
        results = []
        for raw in batch.to_pylist():
            res: ConvResult = spec["convert"](raw, unit, rev)
            if res.row is None:
                drops[res.reason] += 1
                continue
            results.append(res)
        if not results:
            continue
        karr = np.fromiter((r.key for r in results), dtype=np.uint64, count=len(results))
        idarr = np.fromiter((K.id64(r.row["source_row_id"]) for r in results), dtype=np.uint64, count=len(results))
        bad = np.zeros(len(results), dtype=bool)
        reasons = [None] * len(results)
        if dd_keys is not None:
            m = K.isin_sorted(karr, dd_keys) | K.isin_sorted(idarr, dd_ids)
            for i in np.nonzero(m)[0]:
                reasons[i] = "duplicate_of_" + DEPENDS[unit.source][0]
            bad |= m
        if heldout is not None and len(heldout):
            m = K.isin_sorted(karr, np.asarray(heldout)) & np.array([r.split == "train" for r in results])
            for i in np.nonzero(m & ~bad)[0]:
                reasons[i] = "leak_train_content_in_heldout"
            bad |= m
        for i, res in enumerate(results):
            if bad[i]:
                drops[reasons[i]] += 1
                continue
            k = int(karr[i])
            if k in seen[res.split]:
                drops["duplicate_content_in_split"] += 1
                continue
            seen[res.split].add(k)
            keys_out[res.split].append((k, int(idarr[i])))
            pending[res.split].append(res.row)
            written[res.split] += 1
            fam[res.split][res.row["family"]] += 1
            kinds[res.split][res.row["label_kind"]] += 1
            cand_rows[res.split] += res.row["candidate_count"]
        for sp, rows in pending.items():
            if len(rows) >= 1000:
                writers[sp].write(rows)
                pending[sp] = []
    parts = []
    for sp, w in writers.items():
        w.write(pending[sp])
        parts.extend(w.close())
        if keys_out[sp]:
            K.save_keys(os.path.join(out, "keys", unit.source, f"{unit.unit_id}.{sp}.npy"), keys_out[sp])
    stats = {
        "unit": unit.unit_id, "source": unit.source, "config": unit.config, "path": unit.path,
        "row_groups": unit.row_groups, "stage": unit.stage, "read_rows": read, "written": dict(written),
        "candidate_rows": dict(cand_rows), "drops": dict(drops),
        "families": {sp: dict(c) for sp, c in fam.items()}, "label_kinds": {sp: dict(c) for sp, c in kinds.items()},
        "parts": parts, "seconds": round(time.time() - t0, 1),
    }
    tmp = done + ".tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f)
    os.replace(tmp, done)
    return stats


# ---------------------------------------------------------------------------- planning
def plan_units(sources: list, args) -> tuple:
    units, skipped = [], {}
    for s in sources:
        rev = REVISIONS[s] if not args.revision else args.revision
        if s == "tasksource":
            us = tasksource.plan(rev)
        elif s == "samatv":
            us = samatv.plan(rev, "default")
        elif s == "samatv_clean50k":
            us = samatv.plan(rev, "general-clean-50k")
            for u in us:
                u.stage = "train_dep"
        elif s == "openjev":
            us, sk = openjev.plan(rev, args.openjev_configs)
            skipped.update(sk)
        units.extend(us)
    if args.unit_filter:
        units = [u for u in units if re.search(args.unit_filter, u.unit_id)]
    if args.limit_rows is None and args.unit_bytes:
        units = [x for u in units for x in split_unit(u, args)]
    return units, skipped


def split_unit(unit: Unit, args) -> list:
    """Split a big file into row-group ranges (~unit_bytes compressed each) for parallelism."""
    if unit.source not in ("samatv",):
        return [unit]
    try:
        pf = open_parquet(unit.repo_id, unit.revision, unit.path, args.cache_dir, stream=True)
        md = pf.metadata
    except Exception as e:  # footer unreachable: process whole file
        print(f"[plan] footer read failed for {unit.path}: {e}", file=sys.stderr)
        return [unit]
    out, start, acc = [], 0, 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        acc += sum(rg.column(j).total_compressed_size for j in range(rg.num_columns))
        if acc >= args.unit_bytes or i == md.num_row_groups - 1:
            u = dataclasses.replace(unit, unit_id=f"{unit.unit_id}.rg{start}-{i + 1}", row_groups=(start, i + 1))
            out.append(u)
            start, acc = i + 1, 0
    return out


# ---------------------------------------------------------------------------- finalize
def merge_parts(paths: list, dst: str, level: int = 3) -> None:
    """Concatenate parquet parts (same schema) into one file, streaming row groups."""
    tmp = dst + ".tmp"
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd", compression_level=level) as w:
        for p in paths:
            pf = pq.ParquetFile(p)
            for b in pf.iter_batches(batch_size=2000):
                w.write_table(pa.Table.from_batches([b], schema=SCHEMA), row_group_size=2000)
    os.replace(tmp, dst)


def refresh_heldout(out: str) -> int:
    arr = K.load_split_keys(out, ["validation", "test"])
    os.makedirs(os.path.join(out, "keys"), exist_ok=True)
    np.save(os.path.join(out, "keys", "_heldout.npy"), arr)
    return int(arr.size)


def finalize(out: str, slug: str, unit_stats: list, skipped: dict, args, selected: list) -> dict:
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    pat = re.compile(rf"^(train|validation|test)-{re.escape(slug)}-\d{{5}}-of-\d{{5}}\.parquet$")
    for f in os.listdir(data_dir):
        if pat.match(f):
            os.remove(os.path.join(data_dir, f))
    by_split = collections.defaultdict(list)
    for st in sorted(unit_stats, key=lambda s: s["unit"]):
        for p in sorted(st["parts"], key=lambda p: p["file"]):
            by_split[p["split"]].append((os.path.join(out, "_work", slug, st["unit"], p["file"]), p))
    shards = []
    for sp, lst in by_split.items():
        groups, cur, acc = [], [], 0
        for path, p in lst:                      # pack small parts into ~shard_bytes output shards
            cur.append(path)
            acc += p["bytes"]
            if acc >= args.shard_bytes:
                groups.append(cur)
                cur, acc = [], 0
        if cur:
            groups.append(cur)
        m = len(groups)
        for i, g in enumerate(groups):
            name = f"{sp}-{slug}-{i:05d}-of-{m:05d}.parquet"
            dst = os.path.join(data_dir, name)
            if len(g) == 1:
                try:
                    os.link(g[0], dst)
                except OSError:
                    shutil.copy2(g[0], dst)
            else:
                merge_parts(g, dst)
            md = pq.ParquetFile(dst).metadata
            shards.append({"path": f"data/{name}", "split": sp, "rows": md.num_rows, "bytes": os.path.getsize(dst)})
    drops = collections.Counter()
    written = collections.Counter()
    cands = collections.Counter()
    fam = collections.defaultdict(collections.Counter)
    kinds = collections.defaultdict(collections.Counter)
    read = 0
    for s in unit_stats:
        drops.update(s["drops"])
        written.update(s["written"])
        cands.update(s["candidate_rows"])
        read += s["read_rows"]
        for sp, c in s["families"].items():
            fam[sp].update(c)
        for sp, c in s["label_kinds"].items():
            kinds[sp].update(c)
    from . import report as R

    receipt = {
        "schema_version": SCHEMA_VERSION, "recipe_version": RECIPE_VERSION, "target_repo": TARGET_REPO,
        "source": slug, "source_repo": SOURCE_ID[slug], "source_revision": REVISIONS[slug] if not args.revision else args.revision,
        "partial_build": args.limit_rows is not None, "limit_rows_per_unit": args.limit_rows,
        "raw_rows_read": read, "rows_written": dict(written), "candidate_rows_written": dict(cands),
        "families": {sp: dict(c) for sp, c in fam.items()}, "label_kinds": {sp: dict(c) for sp, c in kinds.items()},
        "drops": dict(drops), "skipped_configs_exact_subset": skipped, "units": len(unit_stats),
        "shards": shards, "shard_bytes_total": sum(s["bytes"] for s in shards),
        "leakage_audit": R.leakage_audit(out, slug), "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.join(out, "receipts"), exist_ok=True)
    with open(os.path.join(out, "receipts", f"{slug}.json"), "w") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
    return receipt


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", default=["all"], help=f"{ALL_SOURCES} or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="huggingface_hub cache dir for downloaded source files")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit-rows", type=int, default=None, help="max raw rows read per unit (smoke tests); implies --stream")
    ap.add_argument("--stream", action="store_true", help="read parquet via HTTPS range requests instead of downloading")
    ap.add_argument("--local-root", default=None, help="read source files from <local-root>/<repo path> instead of the Hub")
    ap.add_argument("--shard-bytes", type=int, default=400 * 1024 * 1024, help="target compressed size of output parts")
    ap.add_argument("--unit-bytes", type=int, default=256 * 1024 * 1024, help="split large source files into row-group ranges of this size (0=off)")
    ap.add_argument("--openjev-configs", nargs="+", default=None)
    ap.add_argument("--unit-filter", default=None, help="regex on unit ids (debugging)")
    ap.add_argument("--revision", default=None, help="override pinned source revision")
    ap.add_argument("--existing-keys", action="store_true", help="export content keys of existing repo val/test rows first")
    ap.add_argument("--delete-source", action="store_true", help="delete downloaded source files after their units finish")
    ap.add_argument("--force", action="store_true", help="ignore DONE markers")
    ap.add_argument("--clean-work", action="store_true", help="remove _work after finalizing (keeps data/, keys/, receipts/)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(argv)
    if args.limit_rows is not None:
        args.stream = True
    sources = ALL_SOURCES if "all" in args.source else args.source
    for s in sources:
        if s not in SPEC:
            ap.error(f"unknown source {s}")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for dep_src in list(sources):
        for dep in DEPENDS.get(dep_src, []):
            if dep not in sources and not os.path.isdir(os.path.join(out, "keys", dep)):
                ap.error(f"{dep_src} is deduplicated against {dep}: build {dep} first (or together)")
    if args.existing_keys:
        print("exporting existing keys:", K.export_existing(out))
    units, skipped = plan_units(sources, args)
    print(f"[plan] {len(units)} units: " + ", ".join(f"{s}={sum(u.source == s for u in units)}" for s in sources))
    if args.plan_only:
        for u in units:
            print(u.stage, u.source, u.unit_id, u.row_groups)
        return 0
    ctx = {"out": out, "cache_dir": args.cache_dir, "limit_rows": args.limit_rows, "stream": args.stream,
           "local_root": args.local_root, "shard_bytes": args.shard_bytes, "force": args.force}
    all_stats = collections.defaultdict(list)
    file_left = collections.Counter((u.repo_id, u.path) for u in units)
    t0 = time.time()
    for stage in STAGE_ORDER:
        stage_units = [u for u in units if u.stage == stage]
        if not stage_units:
            continue
        if stage != "eval":
            n = refresh_heldout(out)
            print(f"[stage {stage}] held-out keys: {n}")
        print(f"[stage {stage}] {len(stage_units)} units, workers={args.workers}")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_unit, u, ctx): u for u in stage_units}
            for i, fut in enumerate(as_completed(futs), 1):
                u = futs[fut]
                st = fut.result()
                all_stats[u.source].append(st)
                print(f"  [{i}/{len(stage_units)}] {u.source}/{u.unit_id}: read={st['read_rows']} written={st['written']} "
                      f"drops={sum(st['drops'].values())} {st['seconds']}s{' (resumed)' if st.get('resumed') else ''}", flush=True)
                if args.delete_source:
                    file_left[(u.repo_id, u.path)] -= 1
                    if file_left[(u.repo_id, u.path)] == 0:
                        _delete_cached(u, args.cache_dir)
        if stage == "eval":
            pass
    for s in sources:
        # include already-finished units of this source that were not part of this run's filter
        r = finalize(out, s, all_stats[s], skipped if s == "openjev" else {}, args, sources)
        print(f"[receipt] {s}: rows={r['rows_written']} drops={r['drops']} leakage={r['leakage_audit']['train_rows_in_heldout']}")
    if args.clean_work:
        shutil.rmtree(os.path.join(out, "_work"), ignore_errors=True)
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


def _delete_cached(u: Unit, cache_dir: Optional[str]):
    from .hfio import local_if_cached

    p = local_if_cached(u.repo_id, u.revision, u.path, cache_dir)
    if p and os.path.exists(p):
        real = os.path.realpath(p)
        os.remove(p)
        if os.path.exists(real) and os.path.isfile(real):
            os.remove(real)


if __name__ == "__main__":
    sys.exit(main())
