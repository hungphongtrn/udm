import pytest

from scrm_data import canonical as C
from scrm_data.tiers import (DropRow, agent_labeling, choice_labeling, collapse_duplicates, noul_binary_labeling,
                             noul_independent_labeling, score_labeling)


def test_choice_one_hot():
    lab = choice_labeling([0, 1.0, 0, 0])
    assert lab.label_kind == "source_dataset_label"
    assert lab.tiers == [[1], [0, 2, 3]]
    assert lab.probs is None
    sl = lab.source_label(["a", "b", "c", "d"])
    assert sl == {"kind": "choice", "target": [0, 1.0, 0, 0], "winner": ["b"]}


def test_choice_soft_groups_by_distinct_probability():
    lab = choice_labeling([0.1, 0.5, 0.1, 0.3])
    assert lab.label_kind == "soft_choice_distribution"
    assert lab.tiers == [[1], [3], [0, 2]]
    assert lab.probs == [0.1, 0.5, 0.1, 0.3]
    assert lab.source_label(list("abcd")) == {"kind": "choice_soft", "target": [0.1, 0.5, 0.1, 0.3]}


def test_choice_multi_winner_and_uniform():
    lab = choice_labeling([0.5, 0.5, 0.0])
    assert lab.label_kind == "soft_choice_distribution" and lab.tiers == [[0, 1], [2]]
    with pytest.raises(DropRow) as e:
        choice_labeling([1 / 9] * 9)
    assert e.value.reason == "no_trainable_pair"


def test_choice_invalid():
    with pytest.raises(DropRow):
        choice_labeling([float("nan"), 1.0])
    with pytest.raises(DropRow):
        choice_labeling([1.0])


def test_score_distance_tiers_contract_example():
    # true = 2 of 0..4 -> [[2],[1,3],[0,4]]
    lab = score_labeling([0, 0, 1.0, 0, 0])
    assert lab.tiers == [[2], [1, 3], [0, 4]]
    assert lab.label_kind == "ordinal_score_distance_tiers"
    sl = lab.source_label(list("abcde"))
    assert sl["kind"] == "score" and sl["true_level_index"] == 2 and sl["expected_level"] == 2.0
    assert sl["level_choice_ids"] == list("abcde") and sl["target"] == [0, 0, 1.0, 0, 0]


def test_score_soft_uses_argmax_and_keeps_raw_target():
    lab = score_labeling([0.0, 0.0625, 0.625, 0.3125])
    assert lab.tiers == [[2], [1, 3], [0]]
    sl = lab.source_label(list("abcd"))
    assert sl["target"] == [0.0, 0.0625, 0.625, 0.3125]
    assert abs(sl["expected_level"] - (0.0625 + 2 * 0.625 + 3 * 0.3125)) < 1e-9
    assert lab.probs == [0.0, 0.0625, 0.625, 0.3125]


def test_score_tie_falls_back_to_expected_level_distance():
    lab = score_labeling([0.5, 0.5, 0.0])
    assert lab.tiers == [[0, 1], [2]]
    assert lab.source_label(list("abc"))["true_level_index"] is None


def test_score_edge_levels():
    assert score_labeling([1, 0, 0]).tiers == [[0], [1], [2]]
    assert score_labeling([0, 0, 1]).tiers == [[2], [1], [0]]


def test_noul_binary_threshold_and_roundtrip():
    lab = noul_binary_labeling(0.7, [0.7])
    assert lab.tiers == [[1], [0]] and lab.label_kind == "independent_labels_binary_tiers"
    assert lab.probs == [0.3, 0.7]
    sl = lab.source_label(["no_id", "yes_id"])
    assert sl["kind"] == "noul" and sl["target"] == [0.7] and sl["threshold"] == 0.5
    lab2 = noul_binary_labeling(0.2, [0.2])
    assert lab2.tiers == [[0], [1]]
    with pytest.raises(DropRow) as e:
        noul_binary_labeling(0.5, [0.5])
    assert e.value.reason == "noul_tie"


def test_noul_independent_multilabel():
    lab = noul_independent_labeling([0.9, 0.2, 0.5, 0.1])
    assert lab.tiers == [[0, 2], [1, 3]]
    assert lab.probs == [0.9, 0.2, 0.5, 0.1]          # NOT normalised
    # all above threshold but different -> tiers by probability
    lab = noul_independent_labeling([0.9, 0.7, 0.9])
    assert lab.tiers == [[0, 2], [1]]
    with pytest.raises(DropRow):
        noul_independent_labeling([0.9, 0.9])


def test_agent_tiers():
    lab = agent_labeling(2, 4, lambda ids: {"kind": "agent_choice"})
    assert lab.tiers == [[2], [0, 1, 3]] and lab.label_kind == "agent_choice_target"
    with pytest.raises(DropRow):
        agent_labeling(0, 1, lambda ids: {})


def test_collapse_duplicates():
    texts = ["a", "b", "a"]
    keep, collapsed = collapse_duplicates(texts, [0.5, 0.5, 0.5])
    assert keep == [0, 1] and collapsed == [[2, 0]]
    with pytest.raises(DropRow) as e:
        collapse_duplicates(texts, [0.5, 0.2, 0.1])
    assert e.value.reason == "duplicate_options_conflict"
    with pytest.raises(DropRow) as e:
        collapse_duplicates(texts, strict=True)
    assert e.value.reason == "duplicate_levels"
    with pytest.raises(DropRow):
        collapse_duplicates(["a", ""])
