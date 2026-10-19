"""Numerical and behavioral tests for the from-scratch SAMME implementation."""

from __future__ import annotations

import numpy as np
import pytest

from shared.boosting import (
    ScratchSAMMEClassifier,
    samme_learner_weight,
    samme_weight_update,
)


def test_three_class_one_round_matches_manual_update() -> None:
    weights = np.full(9, 1.0 / 9.0)
    incorrect = np.asarray([False] * 6 + [True] * 3)
    error = float(np.dot(weights, incorrect))
    alpha, status = samme_learner_weight(error, 3)
    assert status == "accepted"
    assert alpha == pytest.approx(np.log(4.0))

    unnormalized, normalized = samme_weight_update(weights, incorrect, alpha)
    assert np.allclose(unnormalized[:6], 1.0 / 9.0)
    assert np.allclose(unnormalized[6:], 4.0 / 9.0)
    assert np.allclose(normalized[:6], 1.0 / 18.0)
    assert np.allclose(normalized[6:], 2.0 / 9.0)
    assert normalized.sum() == pytest.approx(1.0)


def test_not_better_than_random_is_rejected_explicitly() -> None:
    alpha, status = samme_learner_weight(2.0 / 3.0, 3)
    assert alpha is None
    assert status == "not_better_than_random"

    X = np.zeros((9, 2))
    y = np.repeat(np.arange(3), 3)
    model = ScratchSAMMEClassifier(
        5, max_depth=1, random_state=10
    ).fit(X, y)
    assert model.achieved_rounds_ == 0
    assert model.termination_reason_ == "weak_learner_not_better_than_random"
    assert model.rejected_round_["attempted_round"] == 1


def test_zero_error_is_finite_and_stops_cleanly() -> None:
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.repeat(np.arange(3), 3)
    model = ScratchSAMMEClassifier(
        10, max_depth=2, random_state=11
    ).fit(X, y)
    assert model.achieved_rounds_ == 1
    assert model.termination_reason_ == "perfect_learner"
    assert np.isfinite(model.estimator_weights_[0])
    assert model.score(X, y) == 1.0


def test_weighted_votes_normalize_and_prefixes_are_deterministic() -> None:
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
    first = ScratchSAMMEClassifier(
        5, max_depth=1, random_state=22
    ).fit(X, y)
    second = ScratchSAMMEClassifier(
        5, max_depth=1, random_state=22
    ).fit(X, y)
    assert first.achieved_rounds_ == 5
    assert np.allclose(first.estimator_weights_, second.estimator_weights_)
    assert np.array_equal(first.learner_predictions(X), second.learner_predictions(X))
    q = first.predict_proba(X, 3)
    assert q.shape == (9, 3)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.all(q >= 0.0)
    assert first.sample_weight_history_.shape == (6, 9)
