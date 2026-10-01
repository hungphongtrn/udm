"""Validate built shards.

    python -m scrm_data.validate --out <out> [--workers N] [--remote-schema | --offline]

Checks per shard: Arrow schema == schema of an existing remote shard (footer only), JSON columns parse and
are canonical, tiers partition options exactly (>= 2 non-empty tiers), candidate_count == len(options) ==
len(candidate_rows), hash recipes (choice_id, record_id, decision_set_id), partition_role vs split, label
kind vocabulary; globally: unique decision_set_id across all shards.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from . import canonical as C
from .schema import PARTITION_ROLE, SCHEMA, read_remote_schema, schemas_equivalent
from .tiers import LABEL_KINDS

MAX_ERRORS = 20
NAME_RE = re.compile(r"^(train|validation|test)-([A-Za-z0-9_]+)-(\d{5})-of-(\d{5})\.parquet$")


def validate_rows(rows: list, split: str, canonical_every: int = 20) -> tuple:
    """Validate a list of row dicts; returns (errors, n_ok). Pure function (used by tests)."""
    errs = []

    def err(i, msg):
        if len(errs) < MAX_ERRORS:
            errs.append(f"row {i} ({rows[i].get('decision_set_id', '?')[:12]}): {msg}")

    for i, r in enumerate(rows):
        try:
            options = json.loads(r["options_json"])
            tiers = json.loads(r["tier_json"])
            probs = json.loads(r["probabilities_json"])
            state = json.loads(r["state_json"])
            instr = json.loads(r["instruction_json"])
            sl = json.loads(r["source_label_or_null"]) if r["source_label_or_null"] is not None else None
        except Exception as e:  # noqa
            err(i, f"json parse: {e}")
            continue
        if not isinstance(options, dict) or not options:
            err(i, "options_json not a non-empty object")
            continue
        n = len(options)
        cr = r["candidate_rows"]
        if not (r["candidate_count"] == n == len(cr)):
            err(i, f"candidate_count {r['candidate_count']} / options {n} / candidate_rows {len(cr)} differ")
        if n < 2:
            err(i, "fewer than 2 candidates")
        ids = list(options)
        flat = [c for t in tiers for c in t] if isinstance(tiers, list) and all(isinstance(t, list) for t in tiers) else None
        if flat is None:
            err(i, "tier_json not a list of lists")
        else:
            if sorted(flat) != sorted(ids):
                err(i, "tiers do not partition options exactly")
            if any(len(t) == 0 for t in tiers):
                err(i, "empty tier")
            if len(tiers) < 2:
                err(i, "no trainable pair (single tier)")
        if not isinstance(probs, dict) or any(k not in options for k in probs):
            err(i, "probabilities keys not in options")
        if sl is None or "kind" not in sl:
            err(i, "source_label_or_null missing kind")
        if r["label_kind"] not in LABEL_KINDS:
            err(i, f"unknown label_kind {r['label_kind']}")
        for k, (cid, text) in enumerate(options.items()):
            if C.sha256_hex(text) != cid:
                err(i, "choice_id != sha256(option text)")
                break
        order = [c["candidate_order"] for c in cr]
        if order != list(range(len(cr))):
            err(i, "candidate_order not 0..n-1")
        if [c["choice_id"] for c in cr] and set(c["choice_id"] for c in cr) != set(ids):
            err(i, "candidate_rows choice_ids != options keys")
        ds = r["decision_set_id"]
        for c in cr:
            if c["decision_set_id"] != ds or c["normalized_text_sha256"] != c["choice_id"] \
                    or c["record_id"] != C.record_id_of(ds, c["choice_id"]):
                err(i, "candidate_rows hash/ids inconsistent")
                break
        if ds != C.decision_set_id_of(r["source_id"], r["source_config"], r["source_split"], r["source_row_id"],
                                      r["raw_record_sha256"]):
            err(i, "decision_set_id does not match recipe")
        if r["partition_role"] != PARTITION_ROLE[split]:
            err(i, f"partition_role {r['partition_role']} != {PARTITION_ROLE[split]}")
        if not r["family"]:
            err(i, "empty family")
        if i % canonical_every == 0:
            for col in ("options_json", "tier_json", "probabilities_json", "state_json", "instruction_json", "source_label_or_null"):
                if r[col] is not None and C.dumps(json.loads(r[col])) != r[col]:
                    err(i, f"{col} not canonical JSON")
    return errs, len(rows) - len(errs)


def validate_file(path: str, expected_schema=None, canonical_every: int = 20) -> dict:
    m = NAME_RE.match(os.path.basename(path))
    out = {"path": path, "errors": [], "rows": 0}
    if not m:
        out["errors"].append("file name does not match {split}-{slug}-{NNNNN}-of-{MMMMM}.parquet")
        return out
    split = m.group(1)
    pf = pq.ParquetFile(path)
    if not schemas_equivalent(pf.schema_arrow, expected_schema or SCHEMA):
        out["errors"].append(f"schema mismatch vs expected:\n{pf.schema_arrow}")
        return out
    ids = []
    for b in pf.iter_batches(batch_size=1000):
        rows = b.to_pylist()
        errs, _ = validate_rows(rows, split, canonical_every)
        out["errors"].extend(errs[: max(0, MAX_ERRORS - len(out["errors"]))])
        out["rows"] += len(rows)
        ids.extend(C.key64(r["decision_set_id"]) for r in rows)
    out["ids"] = np.asarray(ids, dtype=np.uint64)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="build output dir (contains data/)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--offline", action="store_true", help="do not fetch the remote schema; use the built-in constant")
    ap.add_argument("--canonical-every", type=int, default=20, help="check canonical JSON on every Nth row")
    ap.add_argument("--glob", default="*.parquet")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.out, "data", a.glob)))
    if not files:
        print("no shards found", file=sys.stderr)
        return 2
    expected = SCHEMA
    if not a.offline:
        try:
            expected = read_remote_schema()
            print("[validate] compared against the schema of an existing remote shard (footer fetched)")
        except Exception as e:
            print(f"[validate] WARNING could not fetch remote schema ({e}); using built-in schema constant", file=sys.stderr)
    bad = 0
    total = 0
    all_ids = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(validate_file, files, [expected] * len(files), [a.canonical_every] * len(files)):
            total += res["rows"]
            if res["errors"]:
                bad += 1
                print(f"FAIL {os.path.basename(res['path'])}")
                for e in res["errors"][:MAX_ERRORS]:
                    print("   ", e)
            else:
                print(f"ok   {os.path.basename(res['path'])} rows={res['rows']}")
            if "ids" in res:
                all_ids.append(res["ids"])
    ids = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.uint64)
    dup = int(ids.size - np.unique(ids).size)
    if dup:
        print(f"FAIL duplicate decision_set_id (64-bit prefix) count: {dup}")
        bad += 1
    print(f"[validate] files={len(files)} rows={total} failed_files={bad} duplicate_ids={dup}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
