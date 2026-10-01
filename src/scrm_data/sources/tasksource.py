"""tasksource/tasksource-jev-typed-decisions -> decision sets.

Raw columns: state, kind (choice|score|noul), id, options (list[str]), target (list[float] aligned with
options; [p] for noul), question, source, variant, split, group_id, question_id, license, license_use.
"""
from __future__ import annotations

from .. import canonical as C
from ..rows import RowMeta, build_row
from ..tiers import DropRow, choice_labeling, collapse_duplicates, noul_binary_labeling, noul_independent_labeling, score_labeling
from .base import ConvResult, Unit, safe

SLUG = "tasksource"
REPO_ID = "tasksource/tasksource-jev-typed-decisions"
REVISION = "8173a06c7bb640b6158c6a93519cb535196bef15"
DEFAULT_LICENSE = "other"
RAW_COLUMNS = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split",
               "group_id", "question_id", "license", "license_use"]
NOUL_TEXTS = ["no", "yes"]


def plan(revision: str = REVISION, files=None) -> list:
    files = files or ([f"data/train-{i:05d}-of-00013.parquet" for i in range(13)]
                      + ["data/validation-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"])
    units = []
    for f in files:
        name = f.rsplit("/", 1)[-1]
        split = name.split("-")[0]
        units.append(Unit(SLUG, name.replace(".parquet", ""), REPO_ID, revision, f, split, "default",
                          stage="train" if split == "train" else "eval"))
    return units


@safe
def convert(raw: dict, split: str, revision: str = REVISION, config: str = "default") -> ConvResult:
    kind = raw["kind"]
    options = list(raw.get("options") or [])
    target = list(raw.get("target") or [])
    raw_sha = C.sha256_hex(C.dumps({k: raw.get(k) for k in RAW_COLUMNS}))
    state = raw["state"] if raw["state"] is not None else ""   # empty state is legitimate (options carry the content)
    meta = RowMeta(
        source_id=REPO_ID, source_revision=revision, source_config=config, source_split=split,
        source_row_id=str(raw["id"]), source_parent_id=str(raw["group_id"]),
        family=f"tasksource:{raw['source']}", lineage_key=str(raw["group_id"]),
        instruction_template_id=f"tasksource.{kind}.{raw.get('variant') or 'direct'}",
        license=raw.get("license") or DEFAULT_LICENSE, raw_record_sha256=raw_sha, split=split,
    )
    instruction = {"type": kind, "instructions": raw["question"] or ""}
    extra = {"source_task": raw["source"], "variant": raw.get("variant"), "question_id": raw.get("question_id"),
             "license_use": raw.get("license_use"), "source_split_label": raw.get("split")}
    if kind == "noul":
        if options:
            if len(options) != len(target):
                raise DropRow("target_length_mismatch")
            texts = [C.normalize_text(o) for o in options]
            keep, _ = collapse_duplicates(texts, strict=True)
            labeling = noul_independent_labeling(target)
            row, key = build_row(meta, state, instruction, texts, labeling, extra_label=extra)
        else:
            if len(target) != 1:
                raise DropRow("noul_target_not_single_probability")
            labeling = noul_binary_labeling(target[0], target, extra={**extra, "options": NOUL_TEXTS, "p_yes": target[0]})
            row, key = build_row(meta, state, instruction, NOUL_TEXTS, labeling)
    elif kind in ("choice", "score"):
        if len(options) != len(target):
            raise DropRow("target_length_mismatch")
        texts = [C.normalize_text(o) for o in options]
        if kind == "score":
            collapse_duplicates(texts, strict=True)
            labeling = score_labeling(target)
            row, key = build_row(meta, state, instruction, texts, labeling, score_levels=True, extra_label=extra)
        else:
            keep, collapsed = collapse_duplicates(texts, target)
            kt = [target[i] for i in keep]
            labeling = choice_labeling(kt, raw_target=target, collapsed=collapsed)
            row, key = build_row(meta, state, instruction, [texts[i] for i in keep], labeling, extra_label=extra)
    else:
        raise DropRow("unknown_kind")
    return ConvResult(split, row, key, None)
