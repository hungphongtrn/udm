"""Canonical JSON, hashing and text normalisation helpers.

Verified against existing udm-massive-typed rows (see docs/DATA_SPEC.md section 3):
  * every JSON column == json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
  * choice_id == normalized_text_sha256 == sha256(utf8(option text))
Defined by this project (the original recipes for decision_set_id / record_id / raw_record_sha256 /
text normalisation could not be recovered): see the functions below.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable, Sequence

from . import RECIPE_VERSION


def dumps(obj: Any) -> str:
    """Canonical JSON (verified recipe). Raises ValueError on NaN/Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """NFC, CRLF/CR -> LF, strip leading/trailing whitespace. Inner whitespace is preserved so code,
    tables and indentation keep their meaning. The stored option text IS the normalised text, so
    sha256(options[choice_id]) == choice_id as in the existing data."""
    t = unicodedata.normalize("NFC", text)
    if "\r" in t:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


def choice_id_of(normalized_text: str) -> str:
    return sha256_hex(normalized_text)


def decision_set_id_of(source_id: str, source_config: str, source_split: str, source_row_id: str,
                       raw_record_sha256: str) -> str:
    return sha256_hex(
        dumps(
            {
                "recipe": RECIPE_VERSION,
                "source_id": source_id,
                "source_config": source_config,
                "source_split": source_split,
                "source_row_id": source_row_id,
                "raw_record_sha256": raw_record_sha256,
            }
        )
    )


def record_id_of(decision_set_id: str, choice_id: str) -> str:
    return sha256_hex(dumps({"decision_set_id": decision_set_id, "choice_id": choice_id, "recipe": RECIPE_VERSION}))


def instruction_text_of(instruction_json: str) -> str:
    """Instruction text of an instruction_json value (string, or object with 'instructions')."""
    obj = json.loads(instruction_json)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        v = obj.get("instructions", "")
        return v if isinstance(v, str) else dumps(v)
    return dumps(obj)


def instruction_key_of(instruction_json: str) -> str:
    """Instruction part of the content key: '<type>\x1d<instruction text>' (type is '' for plain strings), so the
    same state/options asked as `choice` and as `score` stay distinct decisions."""
    obj = json.loads(instruction_json)
    typ = obj.get("type", "") if isinstance(obj, dict) else ""
    return f"{typ}\x1d{instruction_text_of(instruction_json)}"


def content_key_hex(state_json: str, instruction_key: str, option_texts: Iterable[str]) -> str:
    """Content key used for leakage / duplicate control: sha256 over state_json, the instruction key
    (type + text) and the SORTED candidate texts (candidate order and labels do not matter)."""
    payload = "\x1f".join([state_json, instruction_key, "\x1e".join(sorted(option_texts))])
    return sha256_hex(payload)


def key64(hex_digest: str) -> int:
    return int(hex_digest[:16], 16)


def content_key_from_columns(state_json: str, instruction_json: str, options_json: str) -> str:
    """Recompute the content key from stored columns (used for the export of existing rows)."""
    return content_key_hex(state_json, instruction_key_of(instruction_json), json.loads(options_json).values())


def hash_fraction(lineage_key: str) -> float:
    return int(sha256_hex(lineage_key)[:12], 16) / float(1 << 48)


def hash_split(lineage_key: str, fractions: Sequence[float] = (0.96, 0.02, 0.02)) -> str:
    """Deterministic train/validation/test assignment by lineage key."""
    x = hash_fraction("scrm-split:" + lineage_key)
    if x < fractions[0]:
        return "train"
    if x < fractions[0] + fractions[1]:
        return "validation"
    return "test"
