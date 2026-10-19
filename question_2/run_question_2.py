"""Run all SAMME experiments and generate evidence for Question 2."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from shared.bagging import ScratchBaggingClassifier  # noqa: E402
from shared.boosting import (  # noqa: E402
    ScratchSAMMEClassifier,
    samme_weight_update,
)
from shared.data import (  # noqa: E402
    derive_seed,
    load_config,
    load_olivetti,
    load_split,
    mask_central_block,
    project_path,
    seed_from_roll_number,
    split_digest,
)


LOGGER = logging.getLogger("question_2")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSIGNMENT_ROOT / "shared" / "config.json",
        help="Path to the assignment JSON configuration.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_toy_example(
    random_state: int, tables_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the required <=10 sample, three-class example for one full round."""
    X = np.arange(9, dtype=np.float64).reshape(-1, 1)
    y = np.repeat(np.arange(3), 3)
    model = ScratchSAMMEClassifier(
        1, max_depth=1, random_state=random_state
    ).fit(X, y)
    if model.achieved_rounds_ != 1:
        raise RuntimeError("The toy example did not produce one accepted SAMME round.")

    initial_weights = model.sample_weight_history_[0]
    updated_weights = model.sample_weight_history_[1]
    predictions = model.training_predictions_[0]
    incorrect = predictions != y
    alpha = float(model.estimator_weights_[0])
    unnormalized, independently_normalized = samme_weight_update(
        initial_weights, incorrect, alpha
    )
    if not np.allclose(updated_weights, independently_normalized):
        raise RuntimeError("Toy update does not match the shared SAMME update rule.")

    table = pd.DataFrame(
        {
            "sample": np.arange(y.size),
            "x": X[:, 0],
            "true_class": y,
            "initial_weight": initial_weights,
            "weak_prediction": predictions,
            "incorrect": incorrect.astype(int),
            "unnormalized_updated_weight": unnormalized,
            "normalized_updated_weight": updated_weights,
        }
    )
    summary: dict[str, object] = {
        "n_samples": int(y.size),
        "n_classes": 3,
        "weak_learner_depth": 1,
        "weighted_error": float(model.weighted_errors_[0]),
        "random_error_threshold": 2.0 / 3.0,
        "learner_weight_alpha": alpha,
        "status": model.round_records_[0]["status"],
        "update_rule": "w_i <- w_i * exp(alpha * I[prediction_i != y_i]), then normalize",
        "normalization_constant": float(unnormalized.sum()),
        "initial_weight_sum": float(initial_weights.sum()),
        "updated_weight_sum": float(updated_weights.sum()),
        "random_state": random_state,
    }
    table.to_csv(tables_dir / "q2a_toy_round.csv", index=False)
    write_json(output_dir / "q2a_toy_round_summary.json", summary)
    LOGGER.info(
        "Toy SAMME round: error=%.6f alpha=%.6f normalization=%.6f",
        summary["weighted_error"],
        summary["learner_weight_alpha"],
        summary["normalization_constant"],
    )
    return table, summary


def fit_depth_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    depths: list[int],
    round_counts: list[int],
    master_seed: int,
) -> tuple[pd.DataFrame, dict[int, ScratchSAMMEClassifier], dict[int, int]]:
    models: dict[int, ScratchSAMMEClassifier] = {}
    seeds: dict[int, int] = {}
    rows: list[dict[str, object]] = []
    maximum_rounds = max(round_counts)
    for depth in depths:
        seed = derive_seed(master_seed, f"q2_boosting_depth_{depth}")
        seeds[depth] = seed
        LOGGER.info("Fitting SAMME depth=%d for up to %d rounds", depth, maximum_rounds)
        model = ScratchSAMMEClassifier(
            maximum_rounds, max_depth=depth, random_state=seed
        ).fit(X_train, y_train)
        models[depth] = model
        LOGGER.info(
            "Depth=%d achieved %d rounds; termination=%s",
            depth,
            model.achieved_rounds_,
            model.termination_reason_,
        )
        for rounds in round_counts:
            available = model.achieved_rounds_ >= rounds
            row: dict[str, object] = {
                "weak_learner_depth": depth,
                "rounds": rounds,
                "available": available,
                "achieved_rounds": model.achieved_rounds_,
                "termination_reason": model.termination_reason_,
                "train_accuracy": np.nan,
                "validation_accuracy": np.nan,
                "cumulative_learner_fit_time_seconds": np.nan,
            }
            if available:
                row.update(
                    {
                        "train_accuracy": model.score(X_train, y_train, rounds),
                        "validation_accuracy": model.score(
                            X_validation, y_validation, rounds
                        ),
                        "cumulative_learner_fit_time_seconds": model.cumulative_fit_time(
                            rounds
                        ),
                    }
                )
                LOGGER.info(
                    "Depth=%d rounds=%d: train=%.4f validation=%.4f",
                    depth,
                    rounds,
                    row["train_accuracy"],
                    row["validation_accuracy"],
                )
            rows.append(row)
    return pd.DataFrame(rows), models, seeds


def select_boosting_configuration(grid: pd.DataFrame) -> dict[str, object]:
    available = grid[grid["available"]].copy()
    if available.empty:
        raise RuntimeError("No requested boosting configuration completed.")
    selected = available.sort_values(
        ["validation_accuracy", "rounds", "weak_learner_depth"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    return {
        "weak_learner_depth": int(selected["weak_learner_depth"]),
        "n_estimators": int(selected["rounds"]),
        "selection_metric": "clean_validation_accuracy",
        "validation_accuracy": float(selected["validation_accuracy"]),
        "tie_break": "fewer rounds, then shallower weak learner",
    }


def plot_validation_grid(grid: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharex=True, sharey=True)
    for depth, subset in grid[grid["available"]].groupby("weak_learner_depth"):
        ordered = subset.sort_values("rounds")
        axes[0].plot(
            ordered["rounds"], ordered["train_accuracy"], marker="o", label=f"depth {depth}"
        )
        axes[1].plot(
            ordered["rounds"],
            ordered["validation_accuracy"],
            marker="o",
            label=f"depth {depth}",
        )
    axes[0].set_title("Training")
    axes[1].set_title("Clean validation")
    axes[0].set_ylabel("Accuracy")
    for axis in axes:
        axis.set_xlabel("Boosting rounds")
        axis.set_xticks(sorted(grid["rounds"].unique()))
        axis.set_ylim(0.0, 1.03)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def diagnostics_frame(
    model: ScratchSAMMEClassifier, n_estimators: int
) -> pd.DataFrame:
    return pd.DataFrame(model.round_records_[:n_estimators])


def plot_selected_diagnostics(
    diagnostics: pd.DataFrame, n_classes: int, destination: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True)
    axes[0, 0].plot(
        diagnostics["round"], diagnostics["weighted_training_error"], color="#31688e"
    )
    axes[0, 0].axhline(
        1.0 - 1.0 / n_classes,
        color="black",
        linestyle="--",
        linewidth=1,
        label="random-prediction threshold",
    )
    axes[0, 0].set_ylabel("Weighted training error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(
        diagnostics["round"], diagnostics["learner_weight"], color="#d95f02"
    )
    axes[0, 1].set_ylabel("Learner weight, alpha")
    axes[1, 0].plot(
        diagnostics["round"], diagnostics["maximum_sample_weight"], color="#1b9e77"
    )
    axes[1, 0].set_ylabel("Maximum sample weight")
    axes[1, 1].plot(
        diagnostics["round"], diagnostics["top_10_weight_mass"], color="#7570b3"
    )
    axes[1, 1].set_ylabel("Weight mass of 10 heaviest images")
    for axis in axes.flat:
        axis.set_xlabel("Boosting round")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_comparison_table(
    selected_boosting_model: ScratchSAMMEClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    q1_bagging_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for count in (25, 50, 100):
        bag_row = q1_bagging_results.loc[q1_bagging_results["B"] == count].iloc[0]
        rows.append(
            {
                "model": "bagging",
                "learners": count,
                "train_accuracy": float(bag_row["train_accuracy"]),
                "validation_accuracy": float(bag_row["validation_accuracy"]),
                "test_accuracy": float(bag_row["test_accuracy"]),
                "training_time_seconds": float(
                    bag_row["cumulative_tree_fit_time_seconds"]
                ),
                "timing_scope": "cumulative base-learner fit calls",
            }
        )
        rows.append(
            {
                "model": "boosting",
                "learners": count,
                "train_accuracy": selected_boosting_model.score(X_train, y_train, count),
                "validation_accuracy": selected_boosting_model.score(
                    X_validation, y_validation, count
                ),
                "test_accuracy": selected_boosting_model.score(X_test, y_test, count),
                "training_time_seconds": selected_boosting_model.cumulative_fit_time(count),
                "timing_scope": "cumulative base-learner fit calls",
            }
        )
    return pd.DataFrame(rows)


def save_heaviest_training_case(
    model: ScratchSAMMEClassifier,
    n_estimators: int,
    training_images: np.ndarray,
    training_targets: np.ndarray,
    original_training_indices: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    final_weights = model.sample_weight_history_[n_estimators]
    position = int(np.argmax(final_weights))
    rows: list[dict[str, object]] = []
    for round_index in range(n_estimators):
        prediction = int(model.training_predictions_[round_index, position])
        rows.append(
            {
                "round": round_index + 1,
                "sample_weight_before": float(
                    model.sample_weight_history_[round_index, position]
                ),
                "weak_prediction": prediction,
                "true_identity": int(training_targets[position]),
                "correct": prediction == int(training_targets[position]),
                "learner_weight": float(model.estimator_weights_[round_index]),
                "weighted_training_error": float(model.weighted_errors_[round_index]),
                "sample_weight_after": float(
                    model.sample_weight_history_[round_index + 1, position]
                ),
            }
        )
    history = pd.DataFrame(rows)
    history.to_csv(tables_dir / "q2c_heaviest_sample_history.csv", index=False)
    rank_order = np.argsort(-final_weights, kind="stable")
    rank = int(np.flatnonzero(rank_order == position)[0] + 1)
    summary: dict[str, object] = {
        "training_position": position,
        "original_dataset_index": int(original_training_indices[position]),
        "true_identity": int(training_targets[position]),
        "final_sample_weight": float(final_weights[position]),
        "maximum_weight_over_selected_rounds": float(
            model.sample_weight_history_[: n_estimators + 1, position].max()
        ),
        "final_weight_rank": rank,
        "misclassified_rounds": int((~history["correct"]).sum()),
        "correct_rounds": int(history["correct"].sum()),
        "rounds_where_weight_increased": int(
            (history["sample_weight_after"] > history["sample_weight_before"]).sum()
        ),
        "selected_rounds": n_estimators,
    }
    write_json(output_dir / "q2c_heaviest_sample_summary.json", summary)

    fig = plt.figure(figsize=(9.0, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.1, 2.1])
    image_axis = fig.add_subplot(grid[0, 0])
    weight_axis = fig.add_subplot(grid[0, 1])
    prediction_axis = fig.add_subplot(grid[0, 2])
    image_axis.imshow(training_images[position], cmap="gray", vmin=0.0, vmax=1.0)
    image_axis.set_title(
        f"Training face\nidentity {int(training_targets[position])}, index {int(original_training_indices[position])}"
    )
    image_axis.axis("off")
    weight_axis.semilogy(
        np.arange(n_estimators + 1),
        model.sample_weight_history_[: n_estimators + 1, position],
        color="#1b9e77",
    )
    weight_axis.set_xlabel("Boosting round")
    weight_axis.set_ylabel("Sample weight (log scale)")
    colors = np.where(history["correct"], "#1b9e77", "#d95f02")
    prediction_axis.scatter(
        history["round"], history["weak_prediction"], c=colors, s=15
    )
    prediction_axis.axhline(
        training_targets[position], color="black", linestyle="--", label="true identity"
    )
    prediction_axis.set_xlabel("Boosting round")
    prediction_axis.set_ylabel("Weak-learner prediction")
    prediction_axis.legend(frameon=False, fontsize=8)
    for axis in (weight_axis, prediction_axis):
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "q2c_heaviest_training_face.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return history, summary


def model_set_metrics(
    tree: DecisionTreeClassifier,
    bagging: ScratchBaggingClassifier,
    bagging_size: int,
    boosting: ScratchSAMMEClassifier,
    boosting_size: int,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict[str, object]]:
    tree_q = tree.predict_proba(X)
    bag_q = bagging.predict_proba(X, bagging_size)
    boost_q = boosting.predict_proba(X, boosting_size)
    return [
        {
            "model": "single_tree",
            "accuracy": float(accuracy_score(y, tree.predict(X))),
            "mean_winning_confidence": float(tree_q.max(axis=1).mean()),
            "confidence_definition": "maximum leaf class probability",
        },
        {
            "model": "bagging",
            "accuracy": float(accuracy_score(y, bagging.predict(X, bagging_size))),
            "mean_winning_confidence": float(bag_q.max(axis=1).mean()),
            "confidence_definition": "winning hard-vote fraction",
        },
        {
            "model": "boosting",
            "accuracy": float(accuracy_score(y, boosting.predict(X, boosting_size))),
            "mean_winning_confidence": float(boost_q.max(axis=1).mean()),
            "confidence_definition": "winning normalized weighted-vote fraction",
        },
    ]


def pairwise_disagreement_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    n_votes = int(counts.sum())
    if n_votes < 2:
        return float("nan")
    agreeing_pairs = int(np.sum(counts * (counts - 1) // 2))
    total_pairs = n_votes * (n_votes - 1) // 2
    return 1.0 - agreeing_pairs / total_pairs


def vote_margin_from_vector(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    top_two = np.partition(vector, vector.size - 2)[-2:]
    return float(top_two[1] - top_two[0])


def top_vote_dictionary(classes: np.ndarray, values: np.ndarray) -> dict[str, float]:
    positions = np.flatnonzero(values > 0.0)
    return {
        str(int(classes[position])): float(values[position]) for position in positions
    }


def choose_mask_response_case(
    y_true: np.ndarray,
    bag_clean: np.ndarray,
    bag_masked: np.ndarray,
    boost_clean: np.ndarray,
    boost_masked: np.ndarray,
) -> tuple[int, str]:
    bag_clean_correct = bag_clean == y_true
    boost_clean_correct = boost_clean == y_true
    mask_correctness_differs = (bag_masked == y_true) != (boost_masked == y_true)
    priority = np.flatnonzero(
        bag_clean_correct & boost_clean_correct & mask_correctness_differs
    )
    if priority.size:
        return int(priority[0]), "both correct when clean; only one remains correct when masked"
    secondary = np.flatnonzero(mask_correctness_differs)
    if secondary.size:
        return int(secondary[0]), "models differ in correctness on the masked face"
    response_differs = (bag_clean != bag_masked) != (boost_clean != boost_masked)
    fallback = np.flatnonzero(response_differs)
    if fallback.size:
        return int(fallback[0]), "one model changes class under masking and the other does not"
    raise RuntimeError("No test face shows a different masking response between the models.")


def save_mask_response_case(
    images: np.ndarray,
    original_test_indices: np.ndarray,
    y_test: np.ndarray,
    X_clean: np.ndarray,
    X_masked: np.ndarray,
    bagging: ScratchBaggingClassifier,
    bagging_size: int,
    boosting: ScratchSAMMEClassifier,
    boosting_size: int,
    output_dir: Path,
    figures_dir: Path,
) -> dict[str, object]:
    bag_clean_predictions = bagging.predict(X_clean, bagging_size)
    bag_masked_predictions = bagging.predict(X_masked, bagging_size)
    boost_clean_predictions = boosting.predict(X_clean, boosting_size)
    boost_masked_predictions = boosting.predict(X_masked, boosting_size)
    position, selection_reason = choose_mask_response_case(
        y_test,
        bag_clean_predictions,
        bag_masked_predictions,
        boost_clean_predictions,
        boost_masked_predictions,
    )

    bag_counts_clean = bagging.vote_counts(X_clean[position : position + 1], bagging_size)[0]
    bag_counts_masked = bagging.vote_counts(X_masked[position : position + 1], bagging_size)[0]
    bag_q_clean = bag_counts_clean / bagging_size
    bag_q_masked = bag_counts_masked / bagging_size
    bag_tree_clean = bagging.tree_predictions(
        X_clean[position : position + 1], bagging_size
    )[:, 0]
    bag_tree_masked = bagging.tree_predictions(
        X_masked[position : position + 1], bagging_size
    )[:, 0]

    boost_q_clean = boosting.predict_proba(
        X_clean[position : position + 1], boosting_size
    )[0]
    boost_q_masked = boosting.predict_proba(
        X_masked[position : position + 1], boosting_size
    )[0]
    boost_learner_clean = boosting.learner_predictions(
        X_clean[position : position + 1], boosting_size
    )[:, 0]
    boost_learner_masked = boosting.learner_predictions(
        X_masked[position : position + 1], boosting_size
    )[:, 0]
    alphas = boosting.estimator_weights_[:boosting_size]

    record: dict[str, object] = {
        "selection_reason": selection_reason,
        "test_position": position,
        "original_dataset_index": int(original_test_indices[position]),
        "true_identity": int(y_test[position]),
        "bagging_clean_prediction": int(bag_clean_predictions[position]),
        "bagging_masked_prediction": int(bag_masked_predictions[position]),
        "bagging_clean_winning_fraction": float(bag_q_clean.max()),
        "bagging_masked_winning_fraction": float(bag_q_masked.max()),
        "bagging_clean_vote_margin": vote_margin_from_vector(bag_q_clean),
        "bagging_masked_vote_margin": vote_margin_from_vector(bag_q_masked),
        "bagging_clean_pairwise_tree_disagreement": pairwise_disagreement_from_counts(
            bag_counts_clean
        ),
        "bagging_masked_pairwise_tree_disagreement": pairwise_disagreement_from_counts(
            bag_counts_masked
        ),
        "bagging_fraction_trees_changing_prediction": float(
            np.mean(bag_tree_clean != bag_tree_masked)
        ),
        "bagging_clean_nonzero_vote_counts": top_vote_dictionary(
            bagging.classes_, bag_counts_clean
        ),
        "bagging_masked_nonzero_vote_counts": top_vote_dictionary(
            bagging.classes_, bag_counts_masked
        ),
        "boosting_clean_prediction": int(boost_clean_predictions[position]),
        "boosting_masked_prediction": int(boost_masked_predictions[position]),
        "boosting_clean_winning_fraction": float(boost_q_clean.max()),
        "boosting_masked_winning_fraction": float(boost_q_masked.max()),
        "boosting_clean_vote_margin": vote_margin_from_vector(boost_q_clean),
        "boosting_masked_vote_margin": vote_margin_from_vector(boost_q_masked),
        "boosting_fraction_learners_changing_prediction": float(
            np.mean(boost_learner_clean != boost_learner_masked)
        ),
        "boosting_alpha_weighted_fraction_changing_prediction": float(
            alphas[boost_learner_clean != boost_learner_masked].sum() / alphas.sum()
        ),
        "boosting_clean_nonzero_weighted_votes": top_vote_dictionary(
            boosting.classes_, boost_q_clean
        ),
        "boosting_masked_nonzero_weighted_votes": top_vote_dictionary(
            boosting.classes_, boost_q_masked
        ),
    }
    write_json(output_dir / "q2d_mask_response_case.json", record)

    masked_image = X_masked[position].reshape(64, 64)
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.0))
    plot_rows = [
        ("Clean", images[position], bag_q_clean, boost_q_clean),
        ("Masked", masked_image, bag_q_masked, boost_q_masked),
    ]
    for row_index, (set_name, image, bag_values, boost_values) in enumerate(plot_rows):
        axes[row_index, 0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(
            f"{set_name} face\ntrue identity {int(y_test[position])}"
        )
        axes[row_index, 0].axis("off")
        for axis, values, title in (
            (axes[row_index, 1], bag_values, "Bagging vote fractions"),
            (axes[row_index, 2], boost_values, "Boosting weighted votes"),
        ):
            top = np.argsort(values)[-6:][::-1]
            axis.bar(
                [str(int(bagging.classes_[item])) for item in top],
                values[top],
                color="#4472a8" if "Bagging" in title else "#d95f02",
            )
            axis.set_title(f"{set_name}: {title}")
            axis.set_xlabel("Identity (top six)")
            axis.set_ylabel("Vote fraction")
            axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "q2d_mask_response_case.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return record


def write_measured_summary(
    destination: Path,
    toy_table: pd.DataFrame,
    toy_summary: dict[str, object],
    grid: pd.DataFrame,
    selected: dict[str, object],
    comparison: pd.DataFrame,
    robustness: pd.DataFrame,
    heaviest: dict[str, object],
    mask_case: dict[str, object],
) -> None:
    text = f"""# Question 2 measured results

This is a generated evidence summary. Full-precision source tables and figures are stored beside it.

## Q2(a): one complete three-class SAMME round

Weighted error: {toy_summary['weighted_error']:.6f}; learner weight: {toy_summary['learner_weight_alpha']:.6f}; normalization constant: {toy_summary['normalization_constant']:.6f}.

{toy_table.to_markdown(index=False, floatfmt='.6f')}

## Q2(b): clean-validation grid

Selected depth `{selected['weak_learner_depth']}` and `{selected['n_estimators']}` rounds using clean validation accuracy only.

{grid.to_markdown(index=False, floatfmt='.6f')}

## Q2(c): bagging versus boosting

{comparison.to_markdown(index=False, floatfmt='.6f')}

Heaviest selected-model training image: position {heaviest['training_position']}, original dataset index {heaviest['original_dataset_index']}, identity {heaviest['true_identity']}, final weight {heaviest['final_sample_weight']:.8f}, misclassified in {heaviest['misclassified_rounds']} of {heaviest['selected_rounds']} rounds.

## Q2(d): clean and masked comparison

{robustness.to_markdown(index=False, floatfmt='.6f')}

Mask-response case: dataset index {mask_case['original_dataset_index']}, true identity {mask_case['true_identity']}. See `q2d_mask_response_case.json` for complete votes and diagnostics.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    master_seed = seed_from_roll_number(config["student"]["roll_number"])
    q2_config = config["question_2"]
    output_dir = project_path(q2_config["output_dir"])
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    started = datetime.now(timezone.utc)
    wall_start = perf_counter()

    dataset = load_olivetti(config["data"]["data_home"])
    split = load_split(config["data"]["split_file"], dataset.target)
    if split.seed != master_seed:
        raise RuntimeError("Configured roll-number seed does not match the saved split.")
    X_train, y_train = dataset.data[split.train], dataset.target[split.train]
    X_validation, y_validation = (
        dataset.data[split.validation],
        dataset.target[split.validation],
    )
    X_test, y_test = dataset.data[split.test], dataset.target[split.test]
    X_test_masked = mask_central_block(X_test)
    LOGGER.info("Loaded locked split %s", split_digest(split))

    toy_seed = derive_seed(master_seed, "q2_toy")
    toy_table, toy_summary = run_toy_example(
        toy_seed, tables_dir, output_dir
    )

    depths = [int(value) for value in q2_config["weak_learner_depths"]]
    round_counts = [int(value) for value in q2_config["round_counts"]]
    grid, boosting_models, depth_seeds = fit_depth_grid(
        X_train,
        y_train,
        X_validation,
        y_validation,
        depths,
        round_counts,
        master_seed,
    )
    grid.to_csv(tables_dir / "q2b_validation_grid.csv", index=False)
    plot_validation_grid(grid, figures_dir / "q2b_validation_grid.png")
    selected = select_boosting_configuration(grid)
    selected_depth = int(selected["weak_learner_depth"])
    selected_rounds = int(selected["n_estimators"])
    selected_model = boosting_models[selected_depth]
    selected.update(
        {
            "random_state": depth_seeds[selected_depth],
            "master_seed": master_seed,
            "split_sha256": split_digest(split),
            "achieved_rounds": selected_model.achieved_rounds_,
            "termination_reason": selected_model.termination_reason_,
            "frozen_before_test_evaluation": True,
        }
    )
    write_json(output_dir / "frozen_boosting_config.json", selected)
    LOGGER.info(
        "Frozen boosting selection before test: depth=%d rounds=%d validation=%.4f",
        selected_depth,
        selected_rounds,
        selected["validation_accuracy"],
    )

    diagnostics = diagnostics_frame(selected_model, selected_rounds)
    diagnostics.to_csv(tables_dir / "q2b_selected_round_diagnostics.csv", index=False)
    plot_selected_diagnostics(
        diagnostics,
        n_classes=np.unique(y_train).size,
        destination=figures_dir / "q2b_selected_round_diagnostics.png",
    )

    q1_bagging_results = pd.read_csv(
        ASSIGNMENT_ROOT / "question_1" / "outputs" / "tables" / "q1bc_bagging_results.csv"
    )
    comparison = create_comparison_table(
        selected_model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        q1_bagging_results,
    )
    comparison.to_csv(tables_dir / "q2c_bagging_boosting_comparison.csv", index=False)

    _, heaviest_summary = save_heaviest_training_case(
        selected_model,
        selected_rounds,
        dataset.images[split.train],
        y_train,
        split.train,
        tables_dir,
        figures_dir,
        output_dir,
    )

    q1_output = ASSIGNMENT_ROOT / "question_1" / "outputs"
    frozen_tree = read_json(q1_output / "frozen_baseline_config.json")
    frozen_bagging = read_json(q1_output / "frozen_bagging_config.json")
    tree = DecisionTreeClassifier(
        max_depth=frozen_tree["max_depth"],
        min_samples_leaf=int(frozen_tree["min_samples_leaf"]),
        random_state=int(frozen_tree["random_state"]),
    ).fit(X_train, y_train)
    bagging_size = int(frozen_bagging["n_estimators"])
    LOGGER.info("Refitting frozen Q1 bagging model for Q2(d) face-level comparison")
    bagging = ScratchBaggingClassifier(
        bagging_size,
        max_depth=frozen_bagging["max_depth"],
        min_samples_leaf=int(frozen_bagging["min_samples_leaf"]),
        random_state=int(frozen_bagging["random_state"]),
    ).fit(X_train, y_train)
    q1_final_row = q1_bagging_results.loc[
        q1_bagging_results["B"] == bagging_size
    ].iloc[0]
    if not np.isclose(bagging.score(X_test, y_test), q1_final_row["test_accuracy"]):
        raise RuntimeError("Refitted bagging model does not reproduce the frozen Q1 result.")

    robustness_rows: list[dict[str, object]] = []
    for set_name, values in (("clean", X_test), ("masked", X_test_masked)):
        for row in model_set_metrics(
            tree,
            bagging,
            bagging_size,
            selected_model,
            selected_rounds,
            values,
            y_test,
        ):
            robustness_rows.append({"test_set": set_name, **row})
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(tables_dir / "q2d_clean_masked_comparison.csv", index=False)

    mask_case = save_mask_response_case(
        dataset.images[split.test],
        split.test,
        y_test,
        X_test,
        X_test_masked,
        bagging,
        bagging_size,
        selected_model,
        selected_rounds,
        output_dir,
        figures_dir,
    )

    write_measured_summary(
        output_dir / "MEASURED_RESULTS.md",
        toy_table,
        toy_summary,
        grid,
        selected,
        comparison,
        robustness,
        heaviest_summary,
        mask_case,
    )
    ended = datetime.now(timezone.utc)
    metadata = {
        "run_started_utc": started.isoformat(),
        "run_ended_utc": ended.isoformat(),
        "wall_time_seconds": perf_counter() - wall_start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "master_seed": master_seed,
        "derived_seeds": {"toy": toy_seed, **{str(k): v for k, v in depth_seeds.items()}},
        "split_sha256": split_digest(split),
        "selected_depth": selected_depth,
        "selected_rounds": selected_rounds,
        "output_directory": output_dir,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    LOGGER.info(
        "Q2 run complete: selected depth=%d rounds=%d wall time=%.2fs",
        selected_depth,
        selected_rounds,
        metadata["wall_time_seconds"],
    )


if __name__ == "__main__":
    main()
