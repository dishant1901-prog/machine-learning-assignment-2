"""Resampling and measured bias-variance utilities for Question 3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiasVarianceResult:
    bias_squared: float
    variance: float
    mean_accuracy: float
    accuracy_std: float
    repetition_accuracies: np.ndarray


def create_within_identity_resamples(
    original_training_indices: np.ndarray,
    all_targets: np.ndarray,
    n_repetitions: int,
    seed: int,
) -> np.ndarray:
    """Sample six indices with replacement from each identity's original six."""
    training_indices = np.asarray(original_training_indices, dtype=np.int64)
    targets = np.asarray(all_targets)
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be positive.")
    training_targets = targets[training_indices]
    identities = np.unique(training_targets)
    if identities.size != 40:
        raise ValueError("The original training split must contain all 40 identities.")
    members_by_identity: dict[int, np.ndarray] = {}
    for identity in identities:
        members = training_indices[training_targets == identity]
        if members.size != 6:
            raise ValueError(
                f"Identity {int(identity)} has {members.size} original training faces, not six."
            )
        members_by_identity[int(identity)] = members

    rng = np.random.default_rng(seed)
    resamples = np.empty((n_repetitions, training_indices.size), dtype=np.int64)
    for repetition in range(n_repetitions):
        offset = 0
        for identity in identities:
            members = members_by_identity[int(identity)]
            resamples[repetition, offset : offset + 6] = rng.choice(
                members, size=6, replace=True
            )
            offset += 6
    validate_resamples(resamples, training_indices, targets)
    return resamples


def validate_resamples(
    resamples: np.ndarray,
    original_training_indices: np.ndarray,
    all_targets: np.ndarray,
) -> None:
    values = np.asarray(resamples, dtype=np.int64)
    original = np.asarray(original_training_indices, dtype=np.int64)
    targets = np.asarray(all_targets)
    if values.ndim != 2 or values.shape[1] != 240:
        raise ValueError(f"Expected resample shape (R, 240), received {values.shape}.")
    allowed = set(original.tolist())
    for repetition, indices in enumerate(values):
        if not set(indices.tolist()).issubset(allowed):
            raise ValueError(f"Repetition {repetition} uses an index outside the original train split.")
        labels, counts = np.unique(targets[indices], return_counts=True)
        if labels.size != 40 or not np.all(counts == 6):
            raise ValueError(
                f"Repetition {repetition} does not contain six draws per identity."
            )


def resample_digest(resamples: np.ndarray, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(resamples, dtype="<i8").tobytes())
    digest.update(str(seed).encode("ascii"))
    return digest.hexdigest()


def measured_bias_variance(
    predictions: np.ndarray,
    true_targets: np.ndarray,
    classes: np.ndarray,
) -> BiasVarianceResult:
    """Apply the assignment's vector bias-squared and variance definitions."""
    values = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(true_targets)
    class_order = np.asarray(classes)
    if values.ndim != 3:
        raise ValueError("predictions must have shape (repetitions, samples, classes).")
    repetitions, n_samples, n_classes = values.shape
    if targets.shape != (n_samples,) or class_order.shape != (n_classes,):
        raise ValueError("Targets/classes do not align with prediction vectors.")
    if repetitions < 2:
        raise ValueError("At least two repetitions are required to estimate variability.")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("Prediction vectors must be finite and nonnegative.")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-10):
        raise ValueError("Every class vector must sum to one.")

    target_positions = np.searchsorted(class_order, targets)
    if not np.array_equal(class_order[target_positions], targets):
        raise ValueError("A target is absent from the global class order.")
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), target_positions] = 1.0
    mean_prediction = values.mean(axis=0)
    bias_squared = float(np.mean(np.sum((mean_prediction - one_hot) ** 2, axis=1)))
    variance = float(
        np.mean(np.sum((values - mean_prediction[None, :, :]) ** 2, axis=2))
    )
    predicted_positions = np.argmax(values, axis=2)
    predicted_classes = class_order[predicted_positions]
    accuracies = np.mean(predicted_classes == targets[None, :], axis=1)
    return BiasVarianceResult(
        bias_squared=bias_squared,
        variance=variance,
        mean_accuracy=float(accuracies.mean()),
        accuracy_std=float(accuracies.std(ddof=1)),
        repetition_accuracies=accuracies,
    )


def align_tree_probabilities(
    probabilities: np.ndarray,
    observed_classes: np.ndarray,
    global_classes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    observed = np.asarray(observed_classes)
    global_order = np.asarray(global_classes)
    aligned = np.zeros((values.shape[0], global_order.size), dtype=np.float64)
    positions = np.searchsorted(global_order, observed)
    if not np.array_equal(global_order[positions], observed):
        raise ValueError("Observed tree class is absent from global class order.")
    aligned[:, positions] = values
    if not np.allclose(aligned.sum(axis=1), 1.0):
        raise RuntimeError("Aligned tree probabilities do not sum to one.")
    return aligned


def average_pairwise_disagreement_from_counts(vote_counts: np.ndarray) -> float:
    """Compute average tree-pair disagreement from per-sample class counts."""
    counts = np.asarray(vote_counts, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError("vote_counts must have shape (samples, classes).")
    totals = counts.sum(axis=1)
    if np.any(totals != totals[0]) or totals[0] < 2:
        raise ValueError("Every row must contain the same number of at least two votes.")
    total_pairs = int(totals[0] * (totals[0] - 1) // 2)
    agreeing_pairs = np.sum(counts * (counts - 1) // 2, axis=1)
    return float(np.mean(1.0 - agreeing_pairs / total_pairs))
