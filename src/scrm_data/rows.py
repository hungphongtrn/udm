"""Assemble a full 26-column output row from converter components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from . import canonical as C
from .schema import PARTITION_ROLE
from .tiers import DropRow, Labeling


@dataclass
class RowMeta:
    source_id: str
    source_revision: str
    source_config: str
    source_split: str          # raw split name (e.g. train / validation / test / calibration / ood)
    source_row_id: str
    source_parent_id: str
    family: str
    lineage_key: str
    instruction_template_id: str
    license: Optional[str]
    raw_record_sha256: str
    split: str                 # output split: train | validation | test


def build_row(meta: RowMeta, state: Any, instruction: Any, norm_texts: Sequence[str], labeling: Labeling,
              score_levels: bool = False, extra_label: Optional[dict] = None):
    """Returns (row_dict, key64). `norm_texts` must already be normalised and duplicate free."""
    n = len(norm_texts)
    if n < 2:
        raise DropRow("fewer_than_2_candidates")
    ids = [C.choice_id_of(t) for t in norm_texts]
    if len(set(ids)) != n:
        raise DropRow("duplicate_choice_ids")
    state_json = C.dumps(state)
    instruction_json = C.dumps(instruction)
    options = dict(zip(ids, norm_texts))
    options_json = C.dumps(options)
    tier_ids = [sorted(ids[i] for i in tier) for tier in labeling.tiers if tier]
    if len(tier_ids) < 2:
        raise DropRow("no_trainable_pair")
    if sum(len(t) for t in tier_ids) != n:
        raise DropRow("tiers_do_not_partition")
    tier_json = C.dumps(tier_ids)
    probabilities_json = C.dumps(dict(zip(ids, labeling.probs))) if labeling.probs is not None else "{}"
    sl = labeling.source_label(ids)
    if score_levels:
        sl["levels"] = list(norm_texts)
    if extra_label:
        sl.update(extra_label)
    source_label_json = C.dumps(sl)
    ds_id = C.decision_set_id_of(meta.source_id, meta.source_config, meta.source_split,
                                 meta.source_row_id, meta.raw_record_sha256)
    cand_rows = [
        {
            "record_id": C.record_id_of(ds_id, cid),
            "decision_set_id": ds_id,
            "choice_id": cid,
            "candidate_order": i,
            "normalized_text_sha256": cid,
        }
        for i, cid in enumerate(ids)
    ]
    row = {
        "decision_set_id": ds_id,
        "source_id": meta.source_id,
        "source_revision": meta.source_revision,
        "source_config": meta.source_config,
        "source_split": meta.source_split,
        "source_row_id": meta.source_row_id,
        "source_parent_id": meta.source_parent_id,
        "family": meta.family,
        "partition_role": PARTITION_ROLE[meta.split],
        "lineage_key": meta.lineage_key,
        "state_json": state_json,
        "instruction_json": instruction_json,
        "options_json": options_json,
        "tier_json": tier_json,
        "probabilities_json": probabilities_json,
        "candidate_count": n,
        "label_kind": labeling.label_kind,
        "source_label_or_null": source_label_json,
        "instruction_template_id": meta.instruction_template_id,
        "option_permutation_seed_or_null": None,
        "raw_record_sha256": meta.raw_record_sha256,
        "license": meta.license,
        "known_pretraining_overlap": None,
        "contamination_status": "not_assessed",
        "quarantine_reason_or_null": None,
        "candidate_rows": cand_rows,
    }
    key = C.key64(C.content_key_hex(state_json, C.instruction_text_of(instruction_json), norm_texts))
    return row, key
