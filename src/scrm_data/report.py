"""Build reports: leakage audit over key files and a summary over receipts.

    python -m scrm_data.report --out <out>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np

from . import keys as K


def _slug_keys(out: str, slug: str, split: str) -> np.ndarray:
    parts = [K._load(p, 0) for p in glob.glob(os.path.join(out, "keys", slug, f"*.{split}.npy"))]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)


def leakage_audit(out: str, slug: str) -> dict:
    """Leakage statistics for one source against ALL key files present in <out>/keys (other sources and
    the existing repo rows if exported):
      train_rows_in_heldout        rows of `slug` train whose content key occurs in any validation/test set
      eval_rows_in_other_split     validation rows of slug that also occur in test sets (and vice versa)
      duplicate_keys_in_split      content-key duplicates inside each split of this source (cross-unit)
    """
    tr = _slug_keys(out, slug, "train")
    va = _slug_keys(out, slug, "validation")
    te = _slug_keys(out, slug, "test")
    held = K.load_split_keys(out, ["validation", "test"])
    held_va = K.load_split_keys(out, ["validation"])
    held_te = K.load_split_keys(out, ["test"])
    res = {
        "train_rows": int(tr.size), "validation_rows": int(va.size), "test_rows": int(te.size),
        "train_rows_in_heldout": int(K.isin_sorted(tr, held).sum()) if tr.size else 0,
        "validation_rows_in_test_sets": int(K.isin_sorted(va, held_te).sum()) if va.size else 0,
        "test_rows_in_validation_sets": int(K.isin_sorted(te, held_va).sum()) if te.size else 0,
        "duplicate_keys_in_split": {
            "train": int(tr.size - np.unique(tr).size), "validation": int(va.size - np.unique(va).size),
            "test": int(te.size - np.unique(te).size)},
        "existing_repo_keys_loaded": os.path.exists(os.path.join(out, "keys", "_existing", "test.npy")),
    }
    return res


def summarize(out: str) -> dict:
    summ = {"sources": {}, "total_rows": {"train": 0, "validation": 0, "test": 0}}
    for p in sorted(glob.glob(os.path.join(out, "receipts", "*.json"))):
        if p.endswith("summary.json"):
            continue
        r = json.load(open(p))
        summ["sources"][r["source"]] = {"rows": r["rows_written"], "drops": r["drops"],
                                        "shards": len(r["shards"]), "bytes": r["shard_bytes_total"],
                                        "partial_build": r["partial_build"], "leakage": r["leakage_audit"]}
        for sp, n in r["rows_written"].items():
            summ["total_rows"][sp] += n
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarise build receipts and audit leakage")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = summarize(a.out)
    with open(os.path.join(a.out, "receipts", "summary.json"), "w") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    print(json.dumps(s, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
