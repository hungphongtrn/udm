"""samatv256/jev-decisions-v1 (configs `default` and `general-clean-50k`) -> agent choice decision sets.

default rows: state{system,user_goal,history[{role,payload_json}],environment_json}, candidates[...],
target{candidate_id,...}, training{choice_eligible,...}, training_split in {train,val,test}.
general-clean-50k rows: state{system,user_goal,environment,history[{role,content}]}, question,
answer_options[{type,label,description,schema_json}], target_index (train only).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from .. import canonical as C
from ..rows import RowMeta, build_row
from ..schema import PARTITION_ROLE
from ..tiers import DropRow, agent_labeling, collapse_duplicates
from .base import ConvResult, Unit, safe

SLUG = "samatv"
SLUG_CLEAN = "samatv_clean50k"
REPO_ID = "samatv256/jev-decisions-v1"
REVISION = "c12aadf1f01c72616bfab0b02480e21806397669"
DEFAULT_LICENSE = "cc-by-4.0"
QUESTION = "Given the current state and available options,\nwhich option should be selected?"
SPLIT_MAP = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}

DEFAULT_COLUMNS = ["id", "source", "decision_type", "status", "content_hash", "state", "candidates", "target",
                   "labels", "provenance", "training", "training_split"]
CLEAN_COLUMNS = ["id", "source", "state", "question", "question_type", "answer_options", "target_index"]


def plan(revision: str = REVISION, config: str = "default", files=None) -> list:
    units = []
    if config == "default":
        files = files or ([f"data/train/train-{i:05d}-of-00011.parquet" for i in range(11)]
                          + ["data/validation/validation-00000-of-00001.parquet", "data/test/test-00000-of-00001.parquet"])
        for f in files:
            name = f.rsplit("/", 1)[-1].replace(".parquet", "")
            split = name.split("-")[0]
            units.append(Unit(SLUG, name, REPO_ID, revision, f, split, "default", stage="train" if split == "train" else "eval"))
    else:
        files = files or [f"clean50k/data/train-{i:05d}-of-00005.parquet" for i in range(5)]
        for f in files:
            name = f.rsplit("/", 1)[-1].replace(".parquet", "")
            # train-only source: hash-split by lineage key, so every unit may emit rows of all splits;
            # it must run after the eval stage so leakage filtering sees the held-out keys.
            units.append(Unit(SLUG_CLEAN, name, REPO_ID, revision, f, None, "general-clean-50k", stage="train"))
    return units


def _parse(s: Optional[str]) -> Any:
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


@lru_cache(maxsize=8192)
def _candidate_text(name: str, description: Optional[str], params_json: Optional[str], typ: Optional[str] = None) -> str:
    obj: dict = {"name": name if name is not None else ""}
    if description:
        obj["description"] = description
    p = _parse(params_json)
    if p not in (None, {}, []):
        obj["parameters"] = p
    if typ and typ != "action":
        obj["type"] = typ
    return C.normalize_text(C.dumps(obj))


def _state(system, user_goal, history, environment) -> dict:
    st: dict = {}
    if system:
        st["system"] = system
    if user_goal:
        st["user_goal"] = user_goal
    if history:
        st["history"] = [
            {"role": h.get("role"), "payload": _parse(h.get("payload_json", h.get("content")))}
            for h in history
        ]
    env = _parse(environment)
    if env not in (None, {}, [], ""):
        st["environment"] = env
    return st


def _family(source: str, prov: dict) -> str:
    name = source.split("/")[-1]
    raw_split = (prov or {}).get("raw_split") or ""
    cfg = (prov or {}).get("source_config") or ""
    suffix = raw_split if raw_split not in ("", "train") else (cfg if cfg not in ("", "default") else "")
    return f"jev_agent:{name}" + (f"/{suffix}" if suffix else "")


def _license(prov: dict) -> str:
    lic = _parse((prov or {}).get("license_metadata_json"))
    if isinstance(lic, dict) and lic.get("license"):
        return str(lic["license"])
    if isinstance(lic, str) and lic:
        return lic
    return DEFAULT_LICENSE


@safe
def convert_default(raw: dict, unit_split: str, revision: str = REVISION, config: str = "default") -> ConvResult:
    tr = raw.get("training") or {}
    if not tr.get("choice_eligible"):
        raise DropRow("not_choice_eligible")
    tgt = raw.get("target")
    if not tgt or tgt.get("candidate_id") is None:
        raise DropRow("no_target_candidate")
    cands = raw.get("candidates") or []
    if len(cands) < 2:
        raise DropRow("fewer_than_2_candidates")
    cand_ids = [c["id"] for c in cands]
    try:
        t_idx = cand_ids.index(tgt["candidate_id"])
    except ValueError:
        raise DropRow("target_not_in_candidates")
    texts = [C.normalize_text(_candidate_text(c["name"], c.get("description"), c.get("parameters_json"))) for c in cands]
    keep, collapsed = collapse_duplicates(texts)
    if len(keep) < 2:
        raise DropRow("fewer_than_2_candidates")
    # map the target to its kept representative
    if t_idx not in keep:
        for dropped_i, kept_i in collapsed:
            if dropped_i == t_idx:
                t_idx = kept_i
                break
    new_t = keep.index(t_idx)
    prov = raw.get("provenance") or {}
    split = SPLIT_MAP.get(raw.get("training_split") or "", unit_split)
    traj = str(prov.get("trajectory_id") or raw["id"])
    raw_sha = C.sha256_hex(C.dumps({"id": raw["id"], "content_hash": raw.get("content_hash")}))
    meta = RowMeta(
        source_id=REPO_ID, source_revision=revision, source_config=config, source_split=unit_split,
        source_row_id=str(raw["id"]), source_parent_id=traj, family=_family(raw["source"], prov),
        lineage_key=f"{prov.get('dataset_id') or raw['source']}:{traj}",
        instruction_template_id="jev_agent.choice_v1", license=_license(prov), raw_record_sha256=raw_sha, split=split,
    )
    st = raw.get("state") or {}
    state = _state(st.get("system"), st.get("user_goal"), st.get("history"), st.get("environment_json"))
    instruction = {"type": "choice", "instructions": QUESTION}
    labels = raw.get("labels")
    ordered = raw.get("ordered_targets") or []
    sl_extra = {
        "kind": "agent_choice", "target": tgt, "labels": labels, "decision_type": raw.get("decision_type"),
        "candidate_ids": [cand_ids[i] for i in keep],
        "ordered_target_candidate_ids": [o.get("candidate_id") for o in ordered] if ordered else [],
        "provenance": {k: prov.get(k) for k in ("dataset_id", "dataset_revision", "source_config", "raw_split",
                                                 "trajectory_id", "step_index", "decision_ordinal")},
        "supervision_evidence": tr.get("supervision_evidence"), "quality_weight": tr.get("quality_weight"),
    }
    if collapsed:
        sl_extra["collapsed_duplicate_indices"] = collapsed
    labeling = agent_labeling(new_t, len(keep), lambda ids, e=sl_extra: dict(e))
    row, key = build_row(meta, state, instruction, [texts[i] for i in keep], labeling)
    return ConvResult(split, row, key, None)


@safe
def convert_clean(raw: dict, unit_split: str, revision: str = REVISION, config: str = "general-clean-50k") -> ConvResult:
    opts = raw.get("answer_options") or []
    if len(opts) < 2:
        raise DropRow("fewer_than_2_candidates")
    ti = raw.get("target_index")
    if ti is None or not (0 <= ti < len(opts)):
        raise DropRow("target_not_in_candidates")
    texts = [C.normalize_text(_candidate_text(o["label"], o.get("description"), o.get("schema_json"), o.get("type")))
             for o in opts]
    keep, collapsed = collapse_duplicates(texts)
    if len(keep) < 2:
        raise DropRow("fewer_than_2_candidates")
    t_idx = ti
    if t_idx not in keep:
        for dropped_i, kept_i in collapsed:
            if dropped_i == t_idx:
                t_idx = kept_i
    new_t = keep.index(t_idx)
    lineage = str(raw["id"])
    split = "train"   # provisional; the real split is a hash of the content key (see below)
    raw_sha = C.sha256_hex(C.dumps({"id": raw["id"], "config": config}))
    src = raw["source"]
    meta = RowMeta(
        source_id=REPO_ID, source_revision=revision, source_config=config, source_split="train",
        source_row_id=lineage, source_parent_id=lineage, family=_family(src, {}), lineage_key=lineage,
        instruction_template_id="jev_agent.choice_v1", license=DEFAULT_LICENSE, raw_record_sha256=raw_sha, split=split,
    )
    st = raw.get("state") or {}
    state = _state(st.get("system"), st.get("user_goal"), st.get("history"), st.get("environment"))
    instruction = {"type": "choice", "instructions": QUESTION}
    sl_extra = {"kind": "agent_choice", "target": {"target_index": ti}, "labels": None, "decision_type": "tool_choice",
                "candidate_ids": [None] * len(keep), "provenance": {"dataset_id": src, "source_config": config,
                                                                     "raw_split": "train"}}
    if collapsed:
        sl_extra["collapsed_duplicate_indices"] = collapsed
    labeling = agent_labeling(new_t, len(keep), lambda ids, e=sl_extra: dict(e))
    row, key = build_row(meta, state, instruction, [texts[i] for i in keep], labeling)
    # general-clean-50k has no trajectory id: split deterministically (96/2/2) by a hash of the CONTENT key so
    # identical content can never straddle splits.
    split = C.hash_split(f"content:{key:016x}")
    row["partition_role"] = PARTITION_ROLE[split]
    return ConvResult(split, row, key, None)
