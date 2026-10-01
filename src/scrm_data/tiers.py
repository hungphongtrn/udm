"""Label -> tier conversion (see docs/CONTRACT.md section 3 and docs/DATA_SPEC.md section 5).

All functions work on candidate INDICES (after duplicate collapsing). A `Labeling` carries
  tiers        list of lists of candidate indices, tier 0 = best
  probs        per-candidate soft target or None  (-> probabilities_json)
  label_kind   one of LABEL_KINDS
  source_label callable(choice_ids) -> JSON-able dict  (-> source_label_or_null)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

LABEL_KINDS = (
    "source_dataset_label",
    "soft_choice_distribution",
    "ordinal_score_distance_tiers",
    "independent_labels_binary_tiers",
    "agent_choice_target",
)
NOUL_THRESHOLD = 0.5
EPS = 1e-9


class DropRow(Exception):
    """Row cannot be converted; `reason` is counted in the build receipt."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Labeling:
    tiers: list
    probs: Optional[list]
    label_kind: str
    source_label: Callable[[list], dict]


def _q(p: float) -> float:
    return round(float(p), 9)


def _check_probs(target: Sequence[float], n: int, what: str = "target") -> list:
    if target is None or len(target) != n:
        raise DropRow(f"{what}_length_mismatch")
    out = []
    for p in target:
        if p is None or not math.isfinite(float(p)) or float(p) < -EPS:
            raise DropRow(f"invalid_{what}")
        out.append(max(0.0, float(p)))
    return out


def collapse_duplicates(norm_texts: Sequence[str], values: Optional[Sequence[float]] = None, strict: bool = False):
    """Collapse candidates whose normalised text is identical (same choice_id).

    Returns (keep_indices, collapsed) where collapsed is a list of [dropped_idx, kept_idx].
    If `values` (per-candidate target) is given, duplicates must agree on it, else the row is dropped
    ('duplicate_options_conflict'). With strict=True any duplicate drops the row ('duplicate_levels').
    """
    first: dict = {}
    keep, collapsed = [], []
    for i, t in enumerate(norm_texts):
        if t == "":
            raise DropRow("empty_option_text")
        j = first.get(t)
        if j is None:
            first[t] = i
            keep.append(i)
            continue
        if strict:
            raise DropRow("duplicate_levels")
        if values is not None and abs(float(values[i]) - float(values[j])) > EPS:
            raise DropRow("duplicate_options_conflict")
        collapsed.append([i, j])
    return keep, collapsed


def _group_desc(values: Sequence[float]) -> list:
    """Indices grouped by distinct value, descending."""
    groups: dict = {}
    for i, v in enumerate(values):
        groups.setdefault(_q(v), []).append(i)
    return [groups[k] for k in sorted(groups, reverse=True)]


def _require_pair(tiers: list, reason: str = "no_trainable_pair"):
    if len([t for t in tiers if t]) < 2:
        raise DropRow(reason)


def choice_labeling(target: Sequence[float], raw_target: Optional[Sequence[float]] = None,
                    collapsed: Optional[list] = None) -> Labeling:
    """Choice distribution. One-hot -> source_dataset_label [[winner],[rest]];
    otherwise soft_choice_distribution grouped by distinct probability (descending)."""
    p = _check_probs(target, len(target))
    raw = list(raw_target) if raw_target is not None else list(target)
    n = len(p)
    if n < 2:
        raise DropRow("fewer_than_2_candidates")
    tiers = _group_desc(p)
    _require_pair(tiers)
    one_hot = (
        sum(1 for x in p if abs(x - 1.0) < EPS) == 1 and sum(1 for x in p if x > EPS) == 1
    )
    extra = {"collapsed_duplicate_indices": collapsed} if collapsed else {}
    if one_hot:
        w = max(range(n), key=lambda i: p[i])
        tiers = [[w], [i for i in range(n) if i != w]]

        def sl(ids, w=w, raw=raw, extra=extra):
            return {"kind": "choice", "target": raw, "winner": [ids[w]], **extra}

        return Labeling(tiers, None, "source_dataset_label", sl)

    def sl2(ids, raw=raw, extra=extra):
        return {"kind": "choice_soft", "target": raw, **extra}

    return Labeling(tiers, [round(x, 6) for x in p], "soft_choice_distribution", sl2)


def score_labeling(target: Sequence[float], raw_target: Optional[Sequence[float]] = None) -> Labeling:
    """Ordinal levels (candidate order == level order, lowest first). The true level is the unique argmax;
    tier k holds the levels at ordinal distance k. If the argmax is not unique, tiers are formed by
    |level - expected_level| (true_level_index = null)."""
    p = _check_probs(target, len(target))
    n = len(p)
    if n < 2:
        raise DropRow("fewer_than_2_candidates")
    raw = list(raw_target) if raw_target is not None else list(target)
    total = sum(p)
    if total <= EPS:
        raise DropRow("zero_target_mass")
    expected = sum(i * x for i, x in enumerate(p)) / total
    mx = max(p)
    argmaxes = [i for i, x in enumerate(p) if abs(x - mx) < EPS]
    if len(argmaxes) == 1:
        t = argmaxes[0]
        dist = [abs(i - t) for i in range(n)]
    else:
        t = None
        dist = [round(abs(i - expected), 6) for i in range(n)]
    groups: dict = {}
    for i, d in enumerate(dist):
        groups.setdefault(d, []).append(i)
    tiers = [groups[k] for k in sorted(groups)]
    _require_pair(tiers)

    def sl(ids, t=t, expected=expected, raw=raw, n=n):
        return {
            "kind": "score",
            "levels": None,  # filled by the converter (needs texts); see rows.build_row
            "level_choice_ids": list(ids),
            "target": raw,
            "true_level_index": t,
            "expected_level": round(expected, 9),
        }

    return Labeling(tiers, [round(x, 6) for x in p], "ordinal_score_distance_tiers", sl)


def noul_binary_labeling(p_yes: float, raw_target: Sequence[float], no_idx: int = 0, yes_idx: int = 1,
                         extra: Optional[dict] = None) -> Labeling:
    """Yes/no probability as a 2-candidate decision [no, yes]. tiers [[yes],[no]] if p_yes > 0.5,
    [[no],[yes]] if p_yes < 0.5; p_yes == 0.5 is a tie -> dropped ('noul_tie')."""
    if p_yes is None or not math.isfinite(float(p_yes)) or not (-EPS <= float(p_yes) <= 1 + EPS):
        raise DropRow("invalid_target")
    p = min(1.0, max(0.0, float(p_yes)))
    if abs(p - NOUL_THRESHOLD) < EPS:
        raise DropRow("noul_tie")
    tiers = [[yes_idx], [no_idx]] if p > NOUL_THRESHOLD else [[no_idx], [yes_idx]]
    probs = [0.0, 0.0]
    probs[yes_idx] = round(p, 6)
    probs[no_idx] = round(1.0 - p, 6)
    raw = list(raw_target)
    extra = dict(extra or {})

    def sl(ids, raw=raw, extra=extra):
        return {"kind": "noul", "target": raw, "threshold": NOUL_THRESHOLD, **extra}

    return Labeling(tiers, probs, "independent_labels_binary_tiers", sl)


def noul_independent_labeling(probs: Sequence[float], raw_target: Optional[Sequence[float]] = None) -> Labeling:
    """Independent per-candidate probabilities (NOT normalised): [[p >= 0.5], [p < 0.5]]. If one side is
    empty the row is kept only when the probabilities differ (then tiers by probability), else dropped."""
    p = _check_probs(probs, len(probs))
    n = len(p)
    if n < 2:
        raise DropRow("fewer_than_2_candidates")
    raw = list(raw_target) if raw_target is not None else list(probs)
    hi = [i for i in range(n) if p[i] >= NOUL_THRESHOLD - EPS]
    lo = [i for i in range(n) if p[i] < NOUL_THRESHOLD - EPS]
    if hi and lo:
        tiers = [hi, lo]
    else:
        tiers = _group_desc(p)
        _require_pair(tiers, "noul_no_distinct_probabilities")

    def sl(ids, raw=raw):
        return {"kind": "noul", "target": raw, "threshold": NOUL_THRESHOLD}

    return Labeling(tiers, [round(x, 6) for x in p], "independent_labels_binary_tiers", sl)


def agent_labeling(target_idx: int, n: int, source_label: Callable[[list], dict]) -> Labeling:
    if n < 2:
        raise DropRow("fewer_than_2_candidates")
    tiers = [[target_idx], [i for i in range(n) if i != target_idx]]
    return Labeling(tiers, None, "agent_choice_target", source_label)
