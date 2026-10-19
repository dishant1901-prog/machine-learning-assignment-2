"""Run every experiment and generate every evidence artifact for Question 1."""

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
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from shared.bagging import (  # noqa: E402
    ScratchBaggingClassifier,
    average_pairwise_disagreement,
    mean_vote_margin,
    tree_leaf_statistics,
)
from shared.data import (  # noqa: E402
    derive_seed,
    get_or_create_split,
    load_config,
    load_olivetti,
    mask_central_block,
    project_path,
    seed_from_roll_number,
    split_digest,
)


LOGGER = logging.getLogger("question_1")


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
    if isinstance(value, (np.integer,)):
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


def depth_label(depth: int | None) -> str:
    return "None" if depth is None else str(depth)


def fit_baseline_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    depths: list[int | None],
    leaf_sizes: list[int],
    random_state: int,
) -> tuple[pd.DataFrame, DecisionTreeClassifier, dict[str, object]]:
    rows: list[dict[str, object]] = []
    fitted: dict[tuple[int | None, int], DecisionTreeClassifier] = {}
    for depth in depths:
        for leaf_size in leaf_sizes:
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=random_state,
            )
            start = perf_counter()
            tree.fit(X_train, y_train)
            elapsed = perf_counter() - start
            stats = tree_leaf_statistics(tree)
            row: dict[str, object] = {
                "max_depth": depth_label(depth),
                "min_samples_leaf": leaf_size,
                "actual_depth": int(tree.get_depth()),
                "train_accuracy": float(tree.score(X_train, y_train)),
                "validation_accuracy": float(tree.score(X_validation, y_validation)),
                "fit_time_seconds": elapsed,
                **stats,
            }
            rows.append(row)
            fitted[(depth, leaf_size)] = tree
            LOGGER.info(
                "Baseline depth=%s leaf=%d: train=%.4f validation=%.4f leaves=%d",
                depth_label(depth),
                leaf_size,
                row["train_accuracy"],
                row["validation_accuracy"],
                row["number_of_leaves"],
            )

    def selection_key(row: dict[str, object]) -> tuple[float, float, int]:
        configured_depth = row["max_depth"]
        depth_complexity = np.inf if configured_depth == "None" else int(configured_depth)
        return (
            float(row["validation_accuracy"]),
            -float(depth_complexity),
            int(row["min_samples_leaf"]),
        )

    selected_row = max(rows, key=selection_key)
    selected_depth = (
        None if selected_row["max_depth"] == "None" else int(selected_row["max_depth"])
    )
    selected_leaf = int(selected_row["min_samples_leaf"])
    selected_tree = fitted[(selected_depth, selected_leaf)]
    selected = {
        "max_depth": selected_depth,
        "min_samples_leaf": selected_leaf,
        "selection_metric": "validation_accuracy",
        "validation_accuracy": float(selected_row["validation_accuracy"]),
        "tie_break": "shallower max_depth, then larger min_samples_leaf",
        "random_state": random_state,
    }
    return pd.DataFrame(rows), selected_tree, selected


def plot_baseline_grid(results: pd.DataFrame, destination: Path) -> None:
    frame = results.copy()
    frame["depth_order"] = frame["max_depth"].replace("None", "unlimited")
    depth_order = ["2", "4", "8", "16", "unlimited"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    colors = {1: "#31688e", 2: "#35b779", 4: "#fde725"}
    for leaf_size, subset in frame.groupby("min_samples_leaf"):
        ordered = subset.set_index("depth_order").loc[depth_order]
        label = f"min leaf = {leaf_size}"
        axes[0].plot(
            depth_order,
            ordered["train_accuracy"],
            marker="o",
            color=colors[int(leaf_size)],
            linestyle="--",
            label=label,
        )
        axes[0].plot(
            depth_order,
            ordered["validation_accuracy"],
            marker="s",
            color=colors[int(leaf_size)],
            label=f"{label} (validation)",
        )
        axes[1].plot(
            depth_order,
            ordered["mean_samples_per_leaf"],
            marker="o",
            color=colors[int(leaf_size)],
            label=label,
        )
        axes[2].plot(
            depth_order,
            ordered["singleton_leaf_fraction"],
            marker="o",
            color=colors[int(leaf_size)],
            label=label,
        )
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].legend(
        handles=[
            Line2D(
                [0], [0], color="black", linestyle="--", marker="o", label="Train"
            ),
            Line2D(
                [0], [0], color="black", linestyle="-", marker="s", label="Validation"
            ),
        ],
        loc="upper left",
        frameon=False,
    )
    axes[1].set_ylabel("Mean training samples per leaf")
    axes[2].set_ylabel("Fraction of singleton leaves")
    axes[2].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xlabel("Configured maximum depth")
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_bagging_prefixes_before_test(
    ensemble: ScratchBaggingClassifier,
    ensemble_sizes: list[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> pd.DataFrame:
    n_train = X_train.shape[0]
    theoretical_count = n_train * (1.0 - (1.0 - 1.0 / n_train) ** n_train)
    rows: list[dict[str, object]] = []
    for count in ensemble_sizes:
        validation_tree_predictions = ensemble.tree_predictions(X_validation, count)
        validation_votes = ensemble.vote_counts(X_validation, count)
        oob = ensemble.oob_metrics(X_train, y_train, count)
        row: dict[str, object] = {
            "B": count,
            "train_accuracy": ensemble.score(X_train, y_train, count),
            "validation_accuracy": ensemble.score(X_validation, y_validation, count),
            "average_distinct_examples": float(
                np.mean(ensemble.distinct_counts_[:count])
            ),
            "theoretical_distinct_examples": theoretical_count,
            "theoretical_distinct_fraction": theoretical_count / n_train,
            "average_pairwise_disagreement_validation": average_pairwise_disagreement(
                validation_tree_predictions
            ),
            "mean_vote_margin_validation": mean_vote_margin(validation_votes),
            "cumulative_tree_fit_time_seconds": ensemble.cumulative_fit_time(count),
            **oob,
        }
        rows.append(row)
        LOGGER.info(
            "Bagging B=%d: train=%.4f validation=%.4f OOB=%.4f coverage=%.3f",
            count,
            row["train_accuracy"],
            row["validation_accuracy"],
            row["oob_accuracy"],
            row["oob_coverage"],
        )
    return pd.DataFrame(rows)


def select_bagging_size(results: pd.DataFrame) -> dict[str, object]:
    selected_row = results.sort_values(
        ["validation_accuracy", "B"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return {
        "n_estimators": int(selected_row["B"]),
        "selection_metric": "validation_accuracy",
        "validation_accuracy": float(selected_row["validation_accuracy"]),
        "tie_break": "smaller ensemble",
    }


def add_test_accuracies(
    results: pd.DataFrame,
    ensemble: ScratchBaggingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    completed = results.copy()
    completed["test_accuracy"] = [
        ensemble.score(X_test, y_test, int(count)) for count in completed["B"]
    ]
    return completed


def plot_bagging_diagnostics(results: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), sharex=True)
    axes[0].plot(results["B"], results["oob_accuracy"], marker="o")
    axes[0].set_ylabel("OOB accuracy")
    axes[1].plot(
        results["B"],
        results["average_pairwise_disagreement_validation"],
        marker="o",
        color="#d95f02",
    )
    axes[1].set_ylabel("Pairwise disagreement (validation)")
    axes[2].plot(
        results["B"],
        results["mean_vote_margin_validation"],
        marker="o",
        color="#1b9e77",
    )
    axes[2].set_ylabel("Mean vote margin (validation)")
    for axis in axes:
        axis.set_xlabel("Number of trees, B")
        axis.set_xscale("log")
        axis.set_xticks(results["B"])
        axis.set_xticklabels(results["B"].astype(str), rotation=45)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bagging_accuracy_and_sampling(results: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    for column, label, marker in (
        ("train_accuracy", "Train", "o"),
        ("validation_accuracy", "Validation", "s"),
        ("test_accuracy", "Clean test", "^"),
    ):
        axes[0].plot(results["B"], results[column], marker=marker, label=label)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].legend(frameon=False)
    axes[1].plot(
        results["B"], results["average_distinct_examples"], marker="o", label="Measured"
    )
    axes[1].axhline(
        results["theoretical_distinct_examples"].iloc[0],
        linestyle="--",
        color="black",
        label="Theoretical expectation",
    )
    axes[1].set_ylabel("Distinct training examples per bootstrap")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Number of trees, B")
        axis.set_xscale("log")
        axis.set_xticks(results["B"])
        axis.set_xticklabels(results["B"].astype(str), rotation=45)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def model_set_metrics(
    tree: DecisionTreeClassifier,
    ensemble: ScratchBaggingClassifier,
    ensemble_size: int,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict[str, object]]:
    tree_probabilities = tree.predict_proba(X)
    bag_probabilities = ensemble.predict_proba(X, ensemble_size)
    return [
        {
            "model": "single_tree",
            "accuracy": float(accuracy_score(y, tree.predict(X))),
            "mean_winning_confidence": float(np.mean(np.max(tree_probabilities, axis=1))),
            "confidence_definition": "maximum leaf class probability",
        },
        {
            "model": "bagging",
            "accuracy": float(accuracy_score(y, ensemble.predict(X, ensemble_size))),
            "mean_winning_confidence": float(np.mean(np.max(bag_probabilities, axis=1))),
            "confidence_definition": "winning hard-vote fraction",
        },
    ]


def select_case_positions(
    y_true: np.ndarray, tree_predictions: np.ndarray, bag_predictions: np.ndarray
) -> list[tuple[str, int]]:
    corrected = np.flatnonzero((tree_predictions != y_true) & (bag_predictions == y_true))
    both_wrong = np.flatnonzero((tree_predictions != y_true) & (bag_predictions != y_true))
    if corrected.size < 2 or both_wrong.size < 2:
        raise RuntimeError(
            "The fixed clean test set does not contain the required two examples in each "
            f"category (corrected={corrected.size}, both_wrong={both_wrong.size})."
        )
    return [
        ("bagging_corrects_tree", int(corrected[0])),
        ("bagging_corrects_tree", int(corrected[1])),
        ("both_wrong", int(both_wrong[0])),
        ("both_wrong", int(both_wrong[1])),
    ]


def save_case_evidence(
    destination_dir: Path,
    images: np.ndarray,
    original_test_indices: np.ndarray,
    y_true: np.ndarray,
    tree: DecisionTreeClassifier,
    ensemble: ScratchBaggingClassifier,
    ensemble_size: int,
) -> pd.DataFrame:
    tree_predictions = tree.predict(images.reshape(-1, 4096))
    tree_probabilities = tree.predict_proba(images.reshape(-1, 4096))
    bag_predictions = ensemble.predict(images.reshape(-1, 4096), ensemble_size)
    votes = ensemble.vote_counts(images.reshape(-1, 4096), ensemble_size)
    cases = select_case_positions(y_true, tree_predictions, bag_predictions)
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(len(cases), 2, figsize=(8.0, 9.0))
    for row_number, (category, position) in enumerate(cases):
        nonzero_positions = np.flatnonzero(votes[position] > 0)
        vote_dictionary = {
            str(int(ensemble.classes_[class_position])): int(votes[position, class_position])
            for class_position in nonzero_positions
        }
        tree_class_position = int(np.searchsorted(tree.classes_, tree_predictions[position]))
        rows.append(
            {
                "category": category,
                "test_position": position,
                "original_dataset_index": int(original_test_indices[position]),
                "true_identity": int(y_true[position]),
                "tree_prediction": int(tree_predictions[position]),
                "tree_confidence": float(tree_probabilities[position, tree_class_position]),
                "bagging_prediction": int(bag_predictions[position]),
                "bagging_confidence": float(votes[position].max() / ensemble_size),
                "bagging_nonzero_vote_counts": json.dumps(vote_dictionary, sort_keys=True),
            }
        )

        image_axis, vote_axis = axes[row_number]
        image_axis.imshow(images[position], cmap="gray", vmin=0.0, vmax=1.0)
        image_axis.set_title(
            f"{category.replace('_', ' ')}\ntrue={int(y_true[position])}, "
            f"tree={int(tree_predictions[position])}, bag={int(bag_predictions[position])}"
        )
        image_axis.axis("off")
        vote_axis.bar(
            [str(int(ensemble.classes_[item])) for item in nonzero_positions],
            votes[position, nonzero_positions],
            color="#4472a8",
        )
        vote_axis.axhline(ensemble_size / 2, color="black", linestyle=":", linewidth=0.8)
        vote_axis.set_ylabel("Votes")
        vote_axis.set_xlabel("Predicted identity (nonzero votes)")
        vote_axis.tick_params(axis="x", rotation=45)
        vote_axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination_dir / "q1d_case_evidence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def write_factual_summary(
    destination: Path,
    baseline_results: pd.DataFrame,
    selected_baseline: dict[str, object],
    selected_tree_metrics: dict[str, object],
    bagging_results: pd.DataFrame,
    selected_bagging: dict[str, object],
    robustness: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    baseline_table = baseline_results.to_markdown(index=False, floatfmt=".4f")
    bagging_table = bagging_results.to_markdown(index=False, floatfmt=".4f")
    robustness_table = robustness.to_markdown(index=False, floatfmt=".4f")
    cases_table = cases.to_markdown(index=False, floatfmt=".4f")
    text = f"""# Question 1 measured results

This file is generated from the saved experiment results. It is an evidence summary, not final report prose.

## Q1(a): baseline grid

Selected configuration: `max_depth={selected_baseline['max_depth']}`, `min_samples_leaf={selected_baseline['min_samples_leaf']}` using validation accuracy only.

Selected-model clean test accuracy: {selected_tree_metrics['test_accuracy']:.4f}.

{baseline_table}

## Q1(b-c): bagging prefixes

Selected ensemble size: `B={selected_bagging['n_estimators']}` using validation accuracy only.

Pairwise disagreement is undefined for B=1 and is therefore stored as a missing value. OOB accuracy uses only samples with at least one OOB prediction; OOB coverage is reported separately.

{bagging_table}

## Q1(d): clean and masked evaluation

{robustness_table}

## Face-level evidence

The corresponding face images and vote charts are in `q1d_case_evidence.png`.

{cases_table}
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    student = config.get("student", {})
    master_seed = seed_from_roll_number(student.get("roll_number"))
    q1_config = config["question_1"]
    output_dir = project_path(q1_config["output_dir"])
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    run_started = datetime.now(timezone.utc)
    wall_start = perf_counter()

    LOGGER.info("Starting official Q1 run with configured roll-number seed.")
    dataset = load_olivetti(config["data"]["data_home"])
    split, split_created = get_or_create_split(
        dataset.target, master_seed, config["data"]["split_file"]
    )
    LOGGER.info(
        "Split %s; digest=%s; train=%d validation=%d test=%d",
        "created" if split_created else "loaded",
        split_digest(split),
        split.train.size,
        split.validation.size,
        split.test.size,
    )

    X_train, y_train = dataset.data[split.train], dataset.target[split.train]
    X_validation, y_validation = (
        dataset.data[split.validation],
        dataset.target[split.validation],
    )
    X_test, y_test = dataset.data[split.test], dataset.target[split.test]
    X_test_masked = mask_central_block(X_test)

    depths = [None if value is None else int(value) for value in q1_config["max_depths"]]
    leaf_sizes = [int(value) for value in q1_config["min_samples_leaf"]]
    ensemble_sizes = [int(value) for value in q1_config["ensemble_sizes"]]
    baseline_seed = derive_seed(master_seed, "q1_baseline")
    bagging_seed = derive_seed(master_seed, "q1_bagging")

    baseline_results, selected_tree, selected_baseline = fit_baseline_grid(
        X_train,
        y_train,
        X_validation,
        y_validation,
        depths,
        leaf_sizes,
        baseline_seed,
    )
    selected_baseline.update(
        {
            "master_seed": master_seed,
            "split_sha256": split_digest(split),
            "frozen_before_test_evaluation": True,
        }
    )
    write_json(output_dir / "frozen_baseline_config.json", selected_baseline)
    baseline_results.to_csv(tables_dir / "q1a_baseline_grid.csv", index=False)
    plot_baseline_grid(baseline_results, figures_dir / "q1a_depth_diagnostics.png")

    selected_tree_metrics = {
        "train_accuracy": float(selected_tree.score(X_train, y_train)),
        "validation_accuracy": float(selected_tree.score(X_validation, y_validation)),
        "test_accuracy": float(selected_tree.score(X_test, y_test)),
        **tree_leaf_statistics(selected_tree),
    }
    write_json(output_dir / "selected_tree_metrics.json", selected_tree_metrics)

    max_ensemble_size = max(ensemble_sizes)
    ensemble = ScratchBaggingClassifier(
        max_ensemble_size,
        max_depth=selected_baseline["max_depth"],
        min_samples_leaf=int(selected_baseline["min_samples_leaf"]),
        random_state=bagging_seed,
    ).fit(X_train, y_train)
    bagging_results = evaluate_bagging_prefixes_before_test(
        ensemble,
        ensemble_sizes,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    selected_bagging = select_bagging_size(bagging_results)
    selected_bagging.update(
        {
            "max_depth": selected_baseline["max_depth"],
            "min_samples_leaf": selected_baseline["min_samples_leaf"],
            "random_state": bagging_seed,
            "master_seed": master_seed,
            "split_sha256": split_digest(split),
            "prefix_of_single_200_tree_fit": True,
            "frozen_before_test_evaluation": True,
        }
    )
    write_json(output_dir / "frozen_bagging_config.json", selected_bagging)

    bagging_results = add_test_accuracies(bagging_results, ensemble, X_test, y_test)
    bagging_results.to_csv(tables_dir / "q1bc_bagging_results.csv", index=False)
    plot_bagging_diagnostics(
        bagging_results, figures_dir / "q1c_oob_disagreement_margin.png"
    )
    plot_bagging_accuracy_and_sampling(
        bagging_results, figures_dir / "q1b_accuracy_and_sampling.png"
    )

    selected_B = int(selected_bagging["n_estimators"])
    robustness_rows: list[dict[str, object]] = []
    for set_name, values in (("clean", X_test), ("masked", X_test_masked)):
        for row in model_set_metrics(
            selected_tree, ensemble, selected_B, values, y_test
        ):
            robustness_rows.append({"test_set": set_name, **row})
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(tables_dir / "q1d_clean_masked_comparison.csv", index=False)

    cases = save_case_evidence(
        figures_dir,
        dataset.images[split.test],
        split.test,
        y_test,
        selected_tree,
        ensemble,
        selected_B,
    )
    cases.to_csv(tables_dir / "q1d_case_evidence.csv", index=False)

    write_factual_summary(
        output_dir / "MEASURED_RESULTS.md",
        baseline_results,
        selected_baseline,
        selected_tree_metrics,
        bagging_results,
        selected_bagging,
        robustness,
        cases,
    )

    run_ended = datetime.now(timezone.utc)
    metadata = {
        "run_started_utc": run_started.isoformat(),
        "run_ended_utc": run_ended.isoformat(),
        "wall_time_seconds": perf_counter() - wall_start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "master_seed": master_seed,
        "derived_seeds": {"baseline": baseline_seed, "bagging": bagging_seed},
        "split_created": split_created,
        "split_sha256": split_digest(split),
        "output_directory": output_dir,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    LOGGER.info(
        "Q1 complete. Selected tree depth=%s leaf=%d; final B=%d; wall time=%.2fs",
        depth_label(selected_baseline["max_depth"]),
        selected_baseline["min_samples_leaf"],
        selected_B,
        metadata["wall_time_seconds"],
    )


if __name__ == "__main__":
    main()
