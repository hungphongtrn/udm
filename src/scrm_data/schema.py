"""Arrow schema of the udm-massive-typed shards (schema_version udm.unified-huggingface-dataset/v2)."""
from __future__ import annotations

import json
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from . import TARGET_REPO

CANDIDATE_STRUCT = pa.struct(
    [
        pa.field("record_id", pa.string()),
        pa.field("decision_set_id", pa.string()),
        pa.field("choice_id", pa.string()),
        pa.field("candidate_order", pa.int32()),
        pa.field("normalized_text_sha256", pa.string()),
    ]
)

SCHEMA = pa.schema(
    [
        pa.field("decision_set_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("source_revision", pa.string()),
        pa.field("source_config", pa.string()),
        pa.field("source_split", pa.string()),
        pa.field("source_row_id", pa.string()),
        pa.field("source_parent_id", pa.string()),
        pa.field("family", pa.string()),
        pa.field("partition_role", pa.string()),
        pa.field("lineage_key", pa.string()),
        pa.field("state_json", pa.string()),
        pa.field("instruction_json", pa.string()),
        pa.field("options_json", pa.string()),
        pa.field("tier_json", pa.string()),
        pa.field("probabilities_json", pa.string()),
        pa.field("candidate_count", pa.int32()),
        pa.field("label_kind", pa.string()),
        pa.field("source_label_or_null", pa.string()),
        pa.field("instruction_template_id", pa.string()),
        pa.field("option_permutation_seed_or_null", pa.string()),
        pa.field("raw_record_sha256", pa.string()),
        pa.field("license", pa.string()),
        pa.field("known_pretraining_overlap", pa.string()),
        pa.field("contamination_status", pa.string()),
        pa.field("quarantine_reason_or_null", pa.string()),
        pa.field("candidate_rows", pa.list_(CANDIDATE_STRUCT)),
    ]
)
COLUMNS = SCHEMA.names
assert len(COLUMNS) == 26

PARTITION_ROLE = {"train": "train", "validation": "dev", "test": "test"}
SPLITS = ("train", "validation", "test")


def rows_to_table(rows: list[dict]) -> pa.Table:
    """Build an Arrow table with exactly SCHEMA from output row dicts."""
    cols = {name: [r[name] for r in rows] for name in COLUMNS}
    return pa.Table.from_pydict(cols, schema=SCHEMA)


def schemas_equivalent(a: pa.Schema, b: pa.Schema) -> bool:
    """Equality ignoring schema metadata (and the name of list child fields, which differs
    between 'item' and 'element' depending on the writer)."""
    return _norm(a) == _norm(b)


def _norm(s: pa.Schema) -> list:
    return [(f.name, _norm_type(f.type)) for f in s]


def _norm_type(t: pa.DataType):
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return ("list", _norm_type(t.value_type))
    if pa.types.is_struct(t):
        return ("struct", [(t.field(i).name, _norm_type(t.field(i).type)) for i in range(t.num_fields)])
    return str(t)


def read_remote_schema(url: str | None = None) -> pa.Schema:
    """Fetch only the parquet footer of an existing remote shard of the target repo."""
    import fsspec  # provided by huggingface_hub dependencies

    url = url or f"https://huggingface.co/datasets/{TARGET_REPO}/resolve/main/data/test-00007-of-00008.parquet"
    with fsspec.open(url, "rb") as f:
        return pq.ParquetFile(f).schema_arrow


def validate_table_schema(table: pa.Table) -> list[str]:
    errs = []
    if not schemas_equivalent(table.schema, SCHEMA):
        errs.append(f"schema mismatch: {table.schema} != expected")
    return errs
