"""Tests for Question 3 resampling and measured bias-variance utilities."""

from __future__ import annotations

import numpy as np
import pytest

from question_3.bias_variance import (
    align_tree_probabilities,
    average_pairwise_disagreement_from_counts,
    create_within_identity_resamples,
    measured_bias_variance,
    resample_digest,
)


def test_within_identity_resamples_are_valid_and_repeatable() -> None:
    targets = np.repeat(np.arange(40), 10)
    original_train = np.concatenate(
        [np.arange(identity * 10, identity * 10 + 6) for identity in range(40)]
    )
    first = create_within_identity_resamples(original_train, targets, 30, 1234)
    second = create_within_identity_resamples(original_train, targets, 30, 1234)
    assert first.shape == (30, 240)
    assert np.array_equal(first, second)
    assert resample_digest(first, 1234) == resample_digest(second, 1234)
    for repetition in first:
        _, counts = np.unique(targets[repetition], return_counts=True)
        assert np.all(counts == 6)
        assert set(repetition).issubset(set(original_train))


def test_measured_bias_variance_matches_manual_binary_example() -> None:
    # Two repetitions, two samples, two classes.
    predictions = np.asarray(
        [
            [[1.0, 0.0], [0.4, 0.6]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    targets = np.asarray([0, 1])
    result = measured_bias_variance(predictions, targets, np.asarray([0, 1]))
    mean_prediction = predictions.mean(axis=0)
    one_hot = np.eye(2)[targets]
    expected_bias = np.mean(np.sum((mean_prediction - one_hot) ** 2, axis=1))
    expected_variance = np.mean(
        np.sum((predictions - mean_prediction[None, :, :]) ** 2, axis=2)
    )
    assert result.bias_squared == pytest.approx(expected_bias)
    assert result.variance == pytest.approx(expected_variance)
    assert result.mean_accuracy == pytest.approx(0.75)
    assert result.accuracy_std == pytest.approx(np.std([0.5, 1.0], ddof=1))


def test_probability_alignment_and_disagreement() -> None:
    probabilities = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    aligned = align_tree_probabilities(
        probabilities, np.asarray([1, 3]), np.arange(4)
    )
    assert aligned.shape == (2, 4)
    assert np.allclose(aligned[:, [1, 3]], probabilities)
    assert np.allclose(aligned.sum(axis=1), 1.0)

    counts = np.asarray([[2, 1], [3, 0]])
    # First sample: 2 of 3 tree pairs disagree. Second sample: 0 of 3.
    assert average_pairwise_disagreement_from_counts(counts) == pytest.approx(1.0 / 3.0)
