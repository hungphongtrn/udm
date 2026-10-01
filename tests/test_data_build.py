"""End-to-end build + validate on a tiny local fixture (no network)."""
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from scrm_data import build, validate
from scrm_data.schema import SCHEMA, schemas_equivalent

TS_SCHEMA = pa.schema([
    ("state", pa.string()), ("kind", pa.string()), ("id", pa.string()), ("options", pa.list_(pa.string())),
    ("target", pa.list_(pa.float64())), ("question", pa.string()), ("source", pa.string()), ("variant", pa.string()),
    ("split", pa.string()), ("group_id", pa.string()), ("question_id", pa.string()), ("license", pa.string()),
    ("license_use", pa.string())])


def mk(i, split, state=None, kind="choice"):
    opts = ["alpha", "beta", "gamma"] if kind != "noul" else []
    tgt = [0.0, 1.0, 0.0] if kind == "choice" else ([0.0, 0.2, 0.8] if kind == "score" else [0.9])
    return {"state": state or f"state {i}", "kind": kind, "id": f"id{i}", "options": opts, "target": tgt,
            "question": "q?", "source": "demo/src", "variant": "direct", "split": split, "group_id": f"g{i}",
            "question_id": "decision", "license": "mit", "license_use": "commercial"}


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TS_SCHEMA), path)


def test_build_end_to_end(tmp_path):
    root = tmp_path / "src"
    out = tmp_path / "out"
    write(str(root / "data/validation-00000-of-00001.parquet"), [mk(1, "dev", "SHARED"), mk(2, "dev")])
    write(str(root / "data/test-00000-of-00001.parquet"), [mk(3, "test", kind="score"), mk(4, "test", kind="noul")])
    train = [mk(10, "train", "SHARED"), mk(11, "train"), mk(12, "train"), mk(13, "train", "dup"), mk(14, "train", "dup"),
             mk(15, "train", kind="noul")]
    write(str(root / "data/train-00000-of-00013.parquet"), train)
    rc = build.main(["--source", "tasksource", "--out", str(out), "--workers", "2", "--local-root", str(root),
                     "--unit-filter", "^(train-00000-of-00013|validation-00000-of-00001|test-00000-of-00001)$",
                     "--shard-bytes", "1000000000"])
    assert rc == 0
    receipt = json.load(open(out / "receipts" / "tasksource.json"))
    assert receipt["rows_written"] == {"validation": 2, "test": 2, "train": 4}
    assert receipt["drops"]["leak_train_content_in_heldout"] == 1       # train row identical to a validation row
    assert receipt["drops"]["duplicate_content_in_split"] == 1          # the two "dup" rows
    assert receipt["leakage_audit"]["train_rows_in_heldout"] == 0
    names = sorted(os.path.basename(p) for p in glob.glob(str(out / "data" / "*.parquet")))
    assert names == ["test-tasksource-00000-of-00001.parquet", "train-tasksource-00000-of-00001.parquet",
                     "validation-tasksource-00000-of-00001.parquet"]
    for p in glob.glob(str(out / "data" / "*.parquet")):
        assert schemas_equivalent(pq.ParquetFile(p).schema_arrow, SCHEMA)
    assert validate.main(["--out", str(out), "--offline", "--workers", "1"]) == 0
    # resume: a second run uses the DONE markers and produces the same receipt
    assert build.main(["--source", "tasksource", "--out", str(out), "--workers", "1", "--local-root", str(root),
                       "--unit-filter", "^(train-00000-of-00013|validation-00000-of-00001|test-00000-of-00001)$"]) == 0
    assert json.load(open(out / "receipts" / "tasksource.json"))["rows_written"] == receipt["rows_written"]


def test_push_readme_update_is_idempotent():
    from scrm_data import push

    old = """---
dataset_info:
  splits:
  - name: train
    num_bytes: 100
    num_examples: 10
  - name: validation
    num_bytes: 50
    num_examples: 5
  - name: test
    num_bytes: 60
    num_examples: 6
  download_size: 1000
  dataset_size: 210
configs:
- config_name: default
---
body
"""
    stats = {"train": {"n": 3, "b": 30, "dl": 10}, "validation": {"n": 1, "b": 5, "dl": 2}, "test": {"n": 2, "b": 7, "dl": 3}}
    rec = {"x": {"source_repo": "a/b", "source_revision": "0123456789", "rows_written": {"train": 3},
                 "label_kinds": {"train": {"source_dataset_label": 3}}}}
    new = push.update_readme_text(old, stats, rec)
    assert "num_examples: 13" in new and "num_bytes: 130" in new and "download_size: 1015" in new and "dataset_size: 252" in new
    assert "- config_name: default" in new
    again = push.update_readme_text(new, stats, rec)       # re-running must not double count
    assert again == new
