"""Focused checks for split integrity, masking, and from-scratch bagging."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from shared.bagging import (
    ScratchBaggingClassifier,
    average_pairwise_disagreement,
    mean_vote_margin,
    tree_leaf_statistics,
)
from shared.data import (
    create_identity_split,
    get_or_create_split,
    load_split,
    mask_central_block,
    seed_from_roll_number,
    split_digest,
)


def test_seed_uses_all_numeric_characters_in_order() -> None:
    assert seed_from_roll_number("AB-01/204-X") == 1204
    with pytest.raises(ValueError, match="no numeric characters"):
        seed_from_roll_number("ABC")


def test_identity_split_has_required_counts_and_is_repeatable(tmp_path) -> None:
    target = np.repeat(np.arange(40), 10)
    first = create_identity_split(target, 123456)
    second = create_identity_split(target, 123456)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    assert first.train.size == 240
    assert first.validation.size == 80
    assert first.test.size == 80
    assert all(np.sum(target[first.train] == identity) == 6 for identity in range(40))
    assert all(
        np.sum(target[first.validation] == identity) == 2 for identity in range(40)
    )
    assert all(np.sum(target[first.test] == identity) == 2 for identity in range(40))
    assert np.unique(np.concatenate([first.train, first.validation, first.test])).size == 400

    destination = tmp_path / "split.npz"
    created, was_created = get_or_create_split(target, 123456, destination)
    loaded = load_split(destination, target)
    assert was_created
    assert split_digest(created) == split_digest(loaded)
    with pytest.raises(ValueError, match="different roll-number seed"):
        get_or_create_split(target, 654321, destination)


def test_mask_zeros_only_required_center_block() -> None:
    values = np.ones((2, 4096), dtype=float)
    masked = mask_central_block(values).reshape(-1, 64, 64)
    assert np.all(masked[:, 24:40, 24:40] == 0.0)
    assert np.sum(masked[0] == 0.0) == 16 * 16
    assert masked[0, 23, 24] == 1.0
    assert masked[0, 40, 39] == 1.0
    assert np.all(values == 1.0), "The source array must not be modified."


def test_scratch_bagging_votes_oob_and_prefixes() -> None:
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
            [0.0, 1.1],
        ]
    )
    y = np.repeat(np.arange(3), 3)
    model = ScratchBaggingClassifier(
        7, max_depth=2, min_samples_leaf=1, random_state=99
    ).fit(X, y)

    prefix_votes = model.vote_counts(X, 5)
    assert prefix_votes.shape == (9, 3)
    assert np.all(prefix_votes.sum(axis=1) == 5)
    assert np.allclose(model.predict_proba(X, 5).sum(axis=1), 1.0)
    assert model.tree_predictions(X, 5).shape == (5, 9)
    assert len(model.bootstrap_indices_) == 7
    assert all(indices.size == X.shape[0] for indices in model.bootstrap_indices_)

    oob = model.oob_metrics(X, y, 7)
    assert 0.0 <= oob["oob_accuracy"] <= 1.0
    assert 0.0 < oob["oob_coverage"] <= 1.0
    assert 0 < oob["oob_covered_samples"] <= X.shape[0]


def test_disagreement_margin_and_leaf_statistics() -> None:
    predictions = np.asarray([[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]])
    # Pair disagreements are 2/4, 3/4, and 1/4, whose average is 1/2.
    assert average_pairwise_disagreement(predictions) == pytest.approx(0.5)
    assert np.isnan(average_pairwise_disagreement(predictions[:1]))

    votes = np.asarray([[4, 1, 0], [2, 2, 1]])
    assert mean_vote_margin(votes) == pytest.approx(((4 - 1) / 5 + 0 / 5) / 2)

    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.asarray([0] * 6 + [1] * 6)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    stats = tree_leaf_statistics(tree)
    assert stats["number_of_leaves"] == 2
    assert stats["mean_samples_per_leaf"] == pytest.approx(6.0)
    assert stats["singleton_leaf_fraction"] == 0.0
