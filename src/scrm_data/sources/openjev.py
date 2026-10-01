"""ZefanCai/Open-Jev (12 configs x 5 splits) -> decision sets.

Raw columns: id, group_id, split, source, kind, question, options (list[str]), target (list[float]),
state_json, metadata_json (privileged, NOT copied), record_json (exact original record), original_line_number.
Open-Jev `noul` rows are yes/no distributions over options ['no','yes'] -> 2-candidate decisions.
"""
from __future__ import annotations

import json

from .. import canonical as C
from ..rows import RowMeta, build_row
from ..tiers import DropRow, choice_labeling, collapse_duplicates, noul_binary_labeling, score_labeling
from .base import ConvResult, Unit, safe

SLUG = "openjev"
REPO_ID = "ZefanCai/Open-Jev"
REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
LICENSE = "cc0-1.0"
CONFIGS = [
    "release-v2-redistributable", "browser-drone-expansion-v1-redistributable", "citation-control-v1",
    "entity-alignment-control-v1", "amount-extraction-control-v1", "email-selection-control-v1",
    "phone-extraction-control-v1", "context-retention-control-v1", "sponsor-segment-control-v1",
    "silent-failure-control-v1", "ir-control-v1", "mailroom-control-v1",
]
RAW_SPLITS = ["train", "calibration", "validation", "test", "ood"]
SPLIT_MAP = {"train": "train", "calibration": "validation", "validation": "validation", "test": "test", "ood": "test"}
# release-v2-redistributable is an exact subset (same ids, same records, same splits) of
# browser-drone-expansion-v1-redistributable (verified); it is skipped when its superset is converted.
SUBSET_OF = {"release-v2-redistributable": "browser-drone-expansion-v1-redistributable"}


def plan(revision: str = REVISION, configs=None) -> tuple:
    configs = list(configs or CONFIGS)
    skipped = {c: SUBSET_OF[c] for c in configs if c in SUBSET_OF and SUBSET_OF[c] in configs}
    units = []
    for cfg in configs:
        if cfg in skipped:
            continue
        for sp in RAW_SPLITS:
            units.append(Unit(SLUG, f"{cfg}__{sp}", REPO_ID, revision, f"data/{cfg}/{sp}-00000-of-00001.parquet", sp, cfg,
                              stage="train" if SPLIT_MAP[sp] == "train" else "eval"))
    return units, skipped


@safe
def convert(raw: dict, raw_split: str, config: str, revision: str = REVISION) -> ConvResult:
    split = SPLIT_MAP[raw_split]
    kind = raw["kind"]
    options = list(raw.get("options") or [])
    target = list(raw.get("target") or [])
    if len(options) != len(target):
        raise DropRow("target_length_mismatch")
    raw_sha = C.sha256_hex(raw["record_json"] if raw.get("record_json") else C.dumps({k: raw[k] for k in ("id", "group_id", "kind", "question", "options", "target", "state_json")}))
    state = json.loads(raw["state_json"])
    meta = RowMeta(
        source_id=REPO_ID, source_revision=revision, source_config=config, source_split=raw_split,
        source_row_id=str(raw["id"]), source_parent_id=str(raw["group_id"]),
        family=f"openjev:{raw['source']}", lineage_key=str(raw["group_id"]),
        instruction_template_id=f"openjev.{kind}", license=LICENSE, raw_record_sha256=raw_sha, split=split,
    )
    instruction = {"type": kind, "instructions": raw["question"] or ""}
    extra = {"task": raw["source"], "config": config, "original_line_number": raw.get("original_line_number")}
    texts = [C.normalize_text(o) for o in options]
    if kind == "noul":
        if sorted(texts) != ["no", "yes"]:
            raise DropRow("noul_unexpected_options")
        p_yes = target[texts.index("yes")]
        labeling = noul_binary_labeling(p_yes, target, extra={**extra, "options": options, "p_yes": p_yes})
        row, key = build_row(meta, state, instruction, ["no", "yes"], labeling)
    elif kind == "choice":
        keep, collapsed = collapse_duplicates(texts, target)
        labeling = choice_labeling([target[i] for i in keep], raw_target=target, collapsed=collapsed)
        row, key = build_row(meta, state, instruction, [texts[i] for i in keep], labeling, extra_label=extra)
    elif kind == "score":
        collapse_duplicates(texts, strict=True)
        labeling = score_labeling(target)
        row, key = build_row(meta, state, instruction, texts, labeling, score_levels=True, extra_label=extra)
    else:
        raise DropRow("unknown_kind")
    return ConvResult(split, row, key, None)
