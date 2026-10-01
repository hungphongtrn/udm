"""Content-key files used for leakage control, plus export of keys for the EXISTING repo rows.

Layout: <out>/keys/<slug>/<unit_id>.<split>.npy   uint64 array of shape (n, 2): [content_key64, source_row_id64]
        <out>/keys/_existing/<split>.npy          uint64 array of shape (n,) (existing repo rows)
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Optional

import numpy as np

from . import TARGET_REPO
from . import canonical as C


def save_keys(path: str, pairs: list) -> None:
    arr = np.asarray(pairs, dtype=np.uint64).reshape(-1, 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def _load(path: str, col: int = 0) -> np.ndarray:
    a = np.load(path)
    return a if a.ndim == 1 else a[:, col]


def load_split_keys(out: str, splits: Iterable[str], exclude_slug: Optional[str] = None, col: int = 0,
                    only_slugs: Optional[Iterable[str]] = None) -> np.ndarray:
    """Sorted unique uint64 keys of all key files for the given splits (all sources + existing rows)."""
    parts = []
    for sp in splits:
        for p in glob.glob(os.path.join(out, "keys", "*", f"*.{sp}.npy")):
            slug = os.path.basename(os.path.dirname(p))
            if exclude_slug and slug == exclude_slug:
                continue
            if only_slugs is not None and slug not in only_slugs:
                continue
            parts.append(_load(p, col))
        if only_slugs is None:
            ex = os.path.join(out, "keys", "_existing", f"{sp}.npy")
            if os.path.exists(ex) and col == 0:
                parts.append(_load(ex))
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


def isin_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if sorted_set.size == 0:
        return np.zeros(values.shape, dtype=bool)
    idx = np.searchsorted(sorted_set, values)
    idx[idx == sorted_set.size] = 0
    return sorted_set[idx] == values


def id64(source_row_id: str) -> int:
    return C.key64(C.sha256_hex(source_row_id))


# ---------------------------------------------------------------------------- existing rows export
EXISTING_SHARDS = {
    "train": [f"data/train-{i:05d}-of-00016.parquet" for i in range(16)],
    "validation": [f"data/validation-{i:05d}-of-00008.parquet" for i in range(8)],
    "test": [f"data/test-{i:05d}-of-00008.parquet" for i in range(8)],
}


def export_existing(out: str, splits=("validation", "test"), include_massive: bool = False, repo: str = TARGET_REPO,
                    limit_shards: Optional[int] = None, last_shards: Optional[int] = None) -> dict:
    """Stream state/instruction/options of the existing remote rows (duckdb over HTTPS, nothing is
    downloaded to disk) and store their content keys. By default MASSIVE rows are skipped (intent
    classification over fixed labels cannot collide with the new sources)."""
    import duckdb

    con = duckdb.connect()
    con.execute("install httpfs; load httpfs;")
    stats = {}
    for sp in splits:
        keys = []
        for rel in (EXISTING_SHARDS[sp][-last_shards:] if last_shards else EXISTING_SHARDS[sp][:limit_shards]):
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"
            where = "" if include_massive else "where family != 'massive'"
            cur = con.execute(f"select state_json, instruction_json, options_json from read_parquet('{url}') {where}")
            while True:
                rows = cur.fetchmany(5000)
                if not rows:
                    break
                keys.extend(C.key64(C.content_key_from_columns(*r)) for r in rows)
        d = os.path.join(out, "keys", "_existing")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{sp}.npy"), np.unique(np.asarray(keys, dtype=np.uint64)))
        stats[sp] = len(keys)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export content keys of existing udm-massive-typed rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--include-massive", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=None, help="only the first N shards per split (testing)")
    ap.add_argument("--last-shards", type=int, default=None, help="only the last N shards per split (typed-decisions rows live there)")
    a = ap.parse_args(argv)
    print(export_existing(a.out, a.splits, a.include_massive, limit_shards=a.limit_shards, last_shards=a.last_shards))


if __name__ == "__main__":
    main()
