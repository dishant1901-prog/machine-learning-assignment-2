"""From-scratch multiclass SAMME boosting built from individual trees."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def samme_learner_weight(
    weighted_error: float,
    n_classes: int,
    *,
    zero_error_floor: float = 1e-12,
) -> tuple[float | None, str]:
    """Return (alpha, status) under the multiclass SAMME acceptance rule."""
    error = float(weighted_error)
    if n_classes < 2:
        raise ValueError("SAMME requires at least two classes.")
    if not 0.0 <= error <= 1.0:
        raise ValueError("weighted_error must lie in [0, 1].")
    random_prediction_error = 1.0 - 1.0 / n_classes
    if error >= random_prediction_error - 1e-12:
        return None, "not_better_than_random"
    if error <= zero_error_floor:
        clipped = zero_error_floor
        alpha = np.log((1.0 - clipped) / clipped) + np.log(n_classes - 1.0)
        return float(alpha), "perfect_learner"
    alpha = np.log((1.0 - error) / error) + np.log(n_classes - 1.0)
    return float(alpha), "accepted"


def samme_weight_update(
    sample_weights: np.ndarray, incorrect: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply and normalize w_i <- w_i exp(alpha * I[incorrect])."""
    weights = np.asarray(sample_weights, dtype=np.float64)
    mistakes = np.asarray(incorrect, dtype=bool)
    if weights.ndim != 1 or mistakes.shape != weights.shape:
        raise ValueError("sample_weights and incorrect must be matching vectors.")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("sample_weights must be nonnegative and sum to one.")
    unnormalized = weights * np.exp(float(alpha) * mistakes.astype(np.float64))
    normalizer = float(unnormalized.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("SAMME weight normalization is not finite and positive.")
    return unnormalized, unnormalized / normalizer


class ScratchSAMMEClassifier:
    """Explicit SAMME implementation using weighted decision trees as learners."""

    def __init__(
        self,
        n_estimators: int,
        *,
        max_depth: int,
        random_state: int,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if max_depth < 1:
            raise ValueError("max_depth must be positive.")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchSAMMEClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
            raise ValueError("X and y have incompatible shapes.")
        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("SAMME requires at least two observed classes.")
        self.n_features_in_ = X.shape[1]
        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: list[float] = []
        self.weighted_errors_: list[float] = []
        self.fit_times_: list[float] = []
        self.training_predictions_: list[np.ndarray] = []
        self.sample_weight_history_: list[np.ndarray] = [
            np.full(y.size, 1.0 / y.size, dtype=np.float64)
        ]
        self.round_records_: list[dict[str, float | int | str]] = []
        self.rejected_round_: dict[str, float | int | str] | None = None
        self.termination_reason_ = "requested_rounds_completed"

        rng = np.random.default_rng(self.random_state)
        max_tree_seed = np.iinfo(np.int32).max
        for round_index in range(self.n_estimators):
            weights_before = self.sample_weight_history_[-1]
            tree_seed = int(rng.integers(0, max_tree_seed))
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=tree_seed,
            )
            start = perf_counter()
            tree.fit(X, y, sample_weight=weights_before)
            elapsed = perf_counter() - start
            predictions = tree.predict(X)
            incorrect = predictions != y
            weighted_error = float(np.dot(weights_before, incorrect))
            alpha, status = samme_learner_weight(
                weighted_error, int(self.classes_.size)
            )

            if alpha is None:
                self.termination_reason_ = "weak_learner_not_better_than_random"
                self.rejected_round_ = {
                    "attempted_round": round_index + 1,
                    "weighted_error": weighted_error,
                    "random_error_threshold": 1.0 - 1.0 / self.classes_.size,
                    "status": status,
                    "fit_time_seconds": elapsed,
                }
                break

            _, weights_after = samme_weight_update(weights_before, incorrect, alpha)
            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.weighted_errors_.append(weighted_error)
            self.fit_times_.append(elapsed)
            self.training_predictions_.append(np.asarray(predictions))
            self.sample_weight_history_.append(weights_after)
            heaviest_count = min(10, weights_after.size)
            top_weight_mass = float(
                np.partition(weights_after, weights_after.size - heaviest_count)[
                    -heaviest_count:
                ].sum()
            )
            self.round_records_.append(
                {
                    "round": round_index + 1,
                    "weighted_training_error": weighted_error,
                    "learner_weight": alpha,
                    "maximum_sample_weight": float(weights_after.max()),
                    "top_10_weight_mass": top_weight_mass,
                    "fit_time_seconds": elapsed,
                    "status": status,
                }
            )
            if status == "perfect_learner":
                self.termination_reason_ = "perfect_learner"
                break

        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=np.float64)
        self.weighted_errors_ = np.asarray(self.weighted_errors_, dtype=np.float64)
        self.fit_times_ = np.asarray(self.fit_times_, dtype=np.float64)
        self.sample_weight_history_ = np.asarray(
            self.sample_weight_history_, dtype=np.float64
        )
        self.training_predictions_ = np.asarray(self.training_predictions_)
        return self

    @property
    def achieved_rounds_(self) -> int:
        return len(self.estimators_)

    def _check_prefix(self, n_estimators: int | None) -> int:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("The boosting model has not been fitted.")
        count = self.achieved_rounds_ if n_estimators is None else int(n_estimators)
        if not 1 <= count <= self.achieved_rounds_:
            raise ValueError(
                f"n_estimators must be in [1, {self.achieved_rounds_}]; "
                f"termination={self.termination_reason_}."
            )
        return count

    def learner_predictions(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        count = self._check_prefix(n_estimators)
        values = np.asarray(X)
        return np.vstack([tree.predict(values) for tree in self.estimators_[:count]])

    def weighted_vote_scores(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        """Return S_k(x), the unnormalized weighted class-vote scores."""
        count = self._check_prefix(n_estimators)
        predictions = self.learner_predictions(X, count)
        scores = np.zeros((predictions.shape[1], self.classes_.size), dtype=np.float64)
        rows = np.arange(predictions.shape[1])
        for round_predictions, alpha in zip(
            predictions, self.estimator_weights_[:count]
        ):
            class_positions = np.searchsorted(self.classes_, round_predictions)
            scores[rows, class_positions] += alpha
        return scores

    def predict_proba(
        self, X: np.ndarray, n_estimators: int | None = None
    ) -> np.ndarray:
        """Return the assignment's normalized weighted vote q_k(x)."""
        count = self._check_prefix(n_estimators)
        total_weight = float(self.estimator_weights_[:count].sum())
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            raise FloatingPointError("Total accepted learner weight is invalid.")
        probabilities = self.weighted_vote_scores(X, count) / total_weight
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise RuntimeError("Normalized SAMME votes do not sum to one.")
        return probabilities

    def predict(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        scores = self.weighted_vote_scores(X, n_estimators)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(
        self, X: np.ndarray, y: np.ndarray, n_estimators: int | None = None
    ) -> float:
        return float(accuracy_score(y, self.predict(X, n_estimators)))

    def cumulative_fit_time(self, n_estimators: int | None = None) -> float:
        count = self._check_prefix(n_estimators)
        return float(self.fit_times_[:count].sum())
