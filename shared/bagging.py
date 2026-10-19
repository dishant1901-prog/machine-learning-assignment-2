"""A small, explicit bagging implementation used throughout the assignment."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class ScratchBaggingClassifier:
    """Bootstrap aggregation of decision trees with hard-vote predictions.

    The class intentionally implements sampling, OOB accounting, and aggregation
    directly instead of wrapping a scikit-learn ensemble estimator.
    """

    def __init__(
        self,
        n_estimators: int,
        *,
        max_depth: int | None,
        min_samples_leaf: int,
        random_state: int,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchBaggingClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError("X and y have incompatible shapes.")

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_training_samples_ = X.shape[0]
        self.estimators_: list[DecisionTreeClassifier] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self.distinct_counts_: list[int] = []
        self.fit_times_: list[float] = []

        rng = np.random.default_rng(self.random_state)
        max_tree_seed = np.iinfo(np.int32).max
        for _ in range(self.n_estimators):
            sample_indices = rng.integers(
                0, self.n_training_samples_, size=self.n_training_samples_, dtype=np.int64
            )
            tree_seed = int(rng.integers(0, max_tree_seed))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            start = perf_counter()
            tree.fit(X[sample_indices], y[sample_indices])
            elapsed = perf_counter() - start

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sample_indices)
            self.distinct_counts_.append(int(np.unique(sample_indices).size))
            self.fit_times_.append(elapsed)
        return self

    def _check_prefix(self, n_estimators: int | None) -> int:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("The ensemble has not been fitted.")
        count = self.n_estimators if n_estimators is None else int(n_estimators)
        if not 1 <= count <= len(self.estimators_):
            raise ValueError(f"n_estimators must be in [1, {len(self.estimators_)}].")
        return count

    def tree_predictions(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        count = self._check_prefix(n_estimators)
        values = np.asarray(X)
        return np.vstack([tree.predict(values) for tree in self.estimators_[:count]])

    def vote_counts(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        predictions = self.tree_predictions(X, n_estimators)
        counts = np.zeros((predictions.shape[1], self.classes_.size), dtype=np.int64)
        rows = np.arange(predictions.shape[1])
        for tree_predictions in predictions:
            class_positions = np.searchsorted(self.classes_, tree_predictions)
            counts[rows, class_positions] += 1
        return counts

    def predict_proba(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        count = self._check_prefix(n_estimators)
        return self.vote_counts(X, count) / count

    def predict(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        counts = self.vote_counts(X, n_estimators)
        return self.classes_[np.argmax(counts, axis=1)]

    def score(
        self, X: np.ndarray, y: np.ndarray, n_estimators: int | None = None
    ) -> float:
        return float(accuracy_score(y, self.predict(X, n_estimators)))

    def oob_metrics(
        self, X: np.ndarray, y: np.ndarray, n_estimators: int | None = None
    ) -> dict[str, float | int]:
        """Return prefix OOB accuracy and coverage using only available OOB votes."""
        count = self._check_prefix(n_estimators)
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != self.n_training_samples_ or y.size != self.n_training_samples_:
            raise ValueError("OOB evaluation requires the original training rows in order.")

        votes = np.zeros((y.size, self.classes_.size), dtype=np.int64)
        vote_totals = np.zeros(y.size, dtype=np.int64)
        all_rows = np.arange(y.size)
        for tree, bootstrap in zip(
            self.estimators_[:count], self.bootstrap_indices_[:count]
        ):
            in_bag = np.zeros(y.size, dtype=bool)
            in_bag[np.unique(bootstrap)] = True
            oob_rows = all_rows[~in_bag]
            if oob_rows.size == 0:
                continue
            predictions = tree.predict(X[oob_rows])
            class_positions = np.searchsorted(self.classes_, predictions)
            votes[oob_rows, class_positions] += 1
            vote_totals[oob_rows] += 1

        covered = vote_totals > 0
        covered_count = int(covered.sum())
        accuracy = float("nan")
        if covered_count:
            predictions = self.classes_[np.argmax(votes[covered], axis=1)]
            accuracy = float(accuracy_score(y[covered], predictions))
        return {
            "oob_accuracy": accuracy,
            "oob_covered_samples": covered_count,
            "oob_coverage": float(covered.mean()),
        }

    def cumulative_fit_time(self, n_estimators: int | None = None) -> float:
        count = self._check_prefix(n_estimators)
        return float(np.sum(self.fit_times_[:count]))


def average_pairwise_disagreement(tree_predictions: np.ndarray) -> float:
    """Average fraction of samples on which two distinct trees disagree."""
    predictions = np.asarray(tree_predictions)
    if predictions.ndim != 2:
        raise ValueError("Expected predictions with shape (n_trees, n_samples).")
    n_trees = predictions.shape[0]
    if n_trees < 2:
        return float("nan")
    disagreement_sum = 0.0
    pair_count = 0
    for first in range(n_trees - 1):
        for second in range(first + 1, n_trees):
            disagreement_sum += float(np.mean(predictions[first] != predictions[second]))
            pair_count += 1
    return disagreement_sum / pair_count


def mean_vote_margin(vote_counts: np.ndarray) -> float:
    """Mean winning-vote fraction minus runner-up-vote fraction."""
    counts = np.asarray(vote_counts)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("Expected a two-dimensional vote table with at least two classes.")
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every sample must have at least one vote.")
    top_two = np.partition(counts, kth=counts.shape[1] - 2, axis=1)[:, -2:]
    margins = (top_two[:, 1] - top_two[:, 0]) / totals
    return float(np.mean(margins))


def tree_leaf_statistics(tree: DecisionTreeClassifier) -> dict[str, float | int]:
    """Summarize unweighted training occupancy of terminal nodes."""
    structure = tree.tree_
    leaf_mask = structure.children_left == structure.children_right
    samples = structure.n_node_samples[leaf_mask]
    return {
        "number_of_leaves": int(samples.size),
        "mean_samples_per_leaf": float(np.mean(samples)),
        "singleton_leaf_fraction": float(np.mean(samples == 1)),
    }
