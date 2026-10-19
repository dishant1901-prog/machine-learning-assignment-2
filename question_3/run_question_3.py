"""Run the 30-refit measured bias-variance experiments for Question 3."""

from __future__ import annotations

import argparse
import hashlib
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
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from question_3.bias_variance import (  # noqa: E402
    align_tree_probabilities,
    average_pairwise_disagreement_from_counts,
    create_within_identity_resamples,
    measured_bias_variance,
    resample_digest,
    validate_resamples,
)
from shared.bagging import ScratchBaggingClassifier, tree_leaf_statistics  # noqa: E402
from shared.boosting import ScratchSAMMEClassifier  # noqa: E402
from shared.data import (  # noqa: E402
    derive_seed,
    load_config,
    load_olivetti,
    load_split,
    project_path,
    seed_from_roll_number,
    split_digest,
)


LOGGER = logging.getLogger("question_3")
PREDICTION_KEYS = (
    "selected_tree",
    "selected_bagging_200",
    "selected_boosting",
    "tree_depth_2",
    "tree_depth_16",
    "bagging_100_depth_2",
    "bagging_100_depth_16",
    "bagging_100_selected",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ASSIGNMENT_ROOT / "shared" / "config.json",
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


def protocol_hash(protocol: dict[str, object]) -> str:
    encoded = json.dumps(json_safe(protocol), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def get_or_create_resamples(
    path: Path,
    original_training_indices: np.ndarray,
    all_targets: np.ndarray,
    n_repetitions: int,
    seed: int,
) -> tuple[np.ndarray, bool]:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            values = np.asarray(saved["indices"], dtype=np.int64)
            saved_seed = int(str(saved["seed"].item()))
            saved_digest = str(saved["sha256"].item())
        validate_resamples(values, original_training_indices, all_targets)
        if saved_seed != seed or values.shape[0] != n_repetitions:
            raise ValueError("Saved Q3 resamples do not match the configured protocol.")
        if resample_digest(values, seed) != saved_digest:
            raise ValueError("Saved Q3 resample digest does not match its contents.")
        return values, False

    values = create_within_identity_resamples(
        original_training_indices, all_targets, n_repetitions, seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=values,
        seed=np.asarray(str(seed)),
        sha256=np.asarray(resample_digest(values, seed)),
    )
    return values, True


def ensemble_leaf_statistics(
    ensemble: ScratchBaggingClassifier, n_estimators: int
) -> dict[str, float]:
    rows = [tree_leaf_statistics(tree) for tree in ensemble.estimators_[:n_estimators]]
    return {
        "mean_number_of_leaves": float(np.mean([row["number_of_leaves"] for row in rows])),
        "mean_samples_per_leaf": float(
            np.mean([row["mean_samples_per_leaf"] for row in rows])
        ),
        "mean_singleton_leaf_fraction": float(
            np.mean([row["singleton_leaf_fraction"] for row in rows])
        ),
    }


def save_repetition_checkpoint(
    path: Path,
    predictions: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    arrays = {key: np.asarray(predictions[key], dtype=np.float64) for key in PREDICTION_KEYS}
    arrays["metadata_json"] = np.asarray(json.dumps(json_safe(metadata), sort_keys=True))
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_repetition_checkpoint(
    path: Path, expected_protocol_hash: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        predictions = {key: np.asarray(saved[key], dtype=np.float64) for key in PREDICTION_KEYS}
        metadata = json.loads(str(saved["metadata_json"].item()))
    if metadata["protocol_hash"] != expected_protocol_hash:
        raise ValueError(f"Checkpoint {path.name} belongs to a different Q3 protocol.")
    for key, values in predictions.items():
        if values.shape != (80, 40) or not np.allclose(values.sum(axis=1), 1.0):
            raise ValueError(f"Checkpoint {path.name} has invalid vectors for {key}.")
    return predictions, metadata


def fit_tree_vector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    ).fit(X_train, y_train)
    vector = align_tree_probabilities(tree.predict_proba(X_test), tree.classes_, classes)
    statistics = {
        **tree_leaf_statistics(tree),
        "actual_depth": int(tree.get_depth()),
    }
    return vector, statistics


def run_repetition(
    repetition: int,
    resampled_indices: np.ndarray,
    all_data: np.ndarray,
    all_targets: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    master_seed: int,
    tree_config: dict[str, object],
    bagging_config: dict[str, object],
    boosting_config: dict[str, object],
    expected_protocol_hash: str,
    checkpoint_path: str,
) -> int:
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        load_repetition_checkpoint(checkpoint, expected_protocol_hash)
        return repetition

    started = perf_counter()
    X_train = np.asarray(all_data[resampled_indices])
    y_train = np.asarray(all_targets[resampled_indices])
    predictions: dict[str, np.ndarray] = {}
    seeds: dict[str, int] = {}
    metadata: dict[str, object] = {
        "repetition": repetition,
        "protocol_hash": expected_protocol_hash,
        "leaf_statistics": {},
        "bagging_leaf_statistics": {},
        "bagging_disagreement": {},
    }

    seeds["selected_tree"] = derive_seed(master_seed, f"q3_r{repetition:02d}_selected_tree")
    predictions["selected_tree"], selected_tree_stats = fit_tree_vector(
        X_train,
        y_train,
        X_test,
        classes,
        max_depth=tree_config["max_depth"],
        min_samples_leaf=int(tree_config["min_samples_leaf"]),
        random_state=seeds["selected_tree"],
    )
    metadata["leaf_statistics"]["selected"] = selected_tree_stats

    seeds["selected_bagging"] = derive_seed(
        master_seed, f"q3_r{repetition:02d}_selected_bagging"
    )
    selected_bagging = ScratchBaggingClassifier(
        int(bagging_config["n_estimators"]),
        max_depth=bagging_config["max_depth"],
        min_samples_leaf=int(bagging_config["min_samples_leaf"]),
        random_state=seeds["selected_bagging"],
    ).fit(X_train, y_train)
    predictions["selected_bagging_200"] = selected_bagging.predict_proba(X_test, 200)
    predictions["bagging_100_selected"] = selected_bagging.predict_proba(X_test, 100)
    selected_counts_100 = selected_bagging.vote_counts(X_test, 100)
    metadata["bagging_disagreement"]["selected"] = average_pairwise_disagreement_from_counts(
        selected_counts_100
    )
    metadata["bagging_leaf_statistics"]["selected"] = ensemble_leaf_statistics(
        selected_bagging, 100
    )
    del selected_bagging

    seeds["selected_boosting"] = derive_seed(
        master_seed, f"q3_r{repetition:02d}_selected_boosting"
    )
    selected_boosting = ScratchSAMMEClassifier(
        int(boosting_config["n_estimators"]),
        max_depth=int(boosting_config["weak_learner_depth"]),
        random_state=seeds["selected_boosting"],
    ).fit(X_train, y_train)
    if selected_boosting.achieved_rounds_ < 1:
        raise RuntimeError(f"Boosting repetition {repetition} accepted no weak learner.")
    predictions["selected_boosting"] = selected_boosting.predict_proba(X_test)
    metadata["boosting_achieved_rounds"] = selected_boosting.achieved_rounds_
    metadata["boosting_termination_reason"] = selected_boosting.termination_reason_
    del selected_boosting

    for depth in (2, 16):
        depth_key = str(depth)
        seeds[f"tree_depth_{depth}"] = derive_seed(
            master_seed, f"q3_r{repetition:02d}_tree_depth_{depth}"
        )
        predictions[f"tree_depth_{depth}"], depth_tree_stats = fit_tree_vector(
            X_train,
            y_train,
            X_test,
            classes,
            max_depth=depth,
            min_samples_leaf=int(tree_config["min_samples_leaf"]),
            random_state=seeds[f"tree_depth_{depth}"],
        )
        metadata["leaf_statistics"][depth_key] = depth_tree_stats

        seeds[f"bagging_100_depth_{depth}"] = derive_seed(
            master_seed, f"q3_r{repetition:02d}_bagging_100_depth_{depth}"
        )
        ensemble = ScratchBaggingClassifier(
            100,
            max_depth=depth,
            min_samples_leaf=int(tree_config["min_samples_leaf"]),
            random_state=seeds[f"bagging_100_depth_{depth}"],
        ).fit(X_train, y_train)
        prediction_key = f"bagging_100_depth_{depth}"
        predictions[prediction_key] = ensemble.predict_proba(X_test, 100)
        counts = ensemble.vote_counts(X_test, 100)
        metadata["bagging_disagreement"][depth_key] = (
            average_pairwise_disagreement_from_counts(counts)
        )
        metadata["bagging_leaf_statistics"][depth_key] = ensemble_leaf_statistics(
            ensemble, 100
        )
        del ensemble

    metadata["seeds"] = seeds
    metadata["elapsed_seconds"] = perf_counter() - started
    save_repetition_checkpoint(checkpoint, predictions, metadata)
    return repetition


def metrics_to_row(model: str, result: object) -> dict[str, object]:
    return {
        "model": model,
        "bias_squared": result.bias_squared,
        "variance": result.variance,
        "mean_test_accuracy": result.mean_accuracy,
        "test_accuracy_std_ddof1": result.accuracy_std,
    }


def plot_selected_models(summary: pd.DataFrame, destination: Path) -> None:
    labels = ["Tree", "Bagging", "Boosting"]
    colors = ["#31688e", "#35b779", "#d95f02"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    positions = np.arange(len(summary))
    width = 0.36
    axes[0].bar(
        positions - width / 2, summary["bias_squared"], width, label="Bias squared", color="#4472a8"
    )
    axes[0].bar(
        positions + width / 2, summary["variance"], width, label="Variance", color="#ed7d31"
    )
    axes[0].set_ylabel("Measured value")
    axes[0].legend(frameon=False)
    axes[1].bar(
        positions,
        summary["mean_test_accuracy"],
        yerr=summary["test_accuracy_std_ddof1"],
        capsize=4,
        color=colors,
    )
    axes[1].set_ylabel("Clean-test accuracy across refits")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_depth_bias_variance(depth_summary: pd.DataFrame, destination: Path) -> None:
    labels = ["Depth 2", "Depth 16", "Unrestricted"]
    if len(depth_summary) != len(labels):
        raise ValueError("The depth-study plot expects exactly three declared settings.")
    positions = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    for axis, tree_column, bag_column, ylabel in (
        (axes[0], "tree_bias_squared", "bagging_bias_squared", "Bias squared"),
        (axes[1], "tree_variance", "bagging_variance", "Variance"),
        (axes[2], "tree_mean_accuracy", "bagging_mean_accuracy", "Mean clean-test accuracy"),
    ):
        axis.bar(
            positions - width / 2,
            depth_summary[tree_column],
            width,
            label="Single tree",
            color="#4472a8",
        )
        axis.bar(
            positions + width / 2,
            depth_summary[bag_column],
            width,
            label="Bagging (100)",
            color="#70ad47",
        )
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_depth_evidence(depth_summary: pd.DataFrame, destination: Path) -> None:
    labels = ["Depth 2", "Depth 16", "Unrestricted"]
    if len(depth_summary) != len(labels):
        raise ValueError("The depth-study plot expects exactly three declared settings.")
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    axes[0].plot(
        positions,
        depth_summary["tree_mean_samples_per_leaf"],
        marker="o",
        color="#31688e",
    )
    axes[0].set_ylabel("Tree mean samples per leaf")
    axes[1].plot(
        positions,
        depth_summary["bagging_mean_disagreement"],
        marker="o",
        color="#d95f02",
    )
    axes[1].set_ylabel("Bagging pairwise disagreement")
    axes[2].bar(
        positions,
        100.0 * depth_summary["accuracy_improvement"],
        color="#35b779",
    )
    axes[2].set_ylabel("Bagging accuracy gain (pp)")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Maximum depth")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_measured_summary(
    destination: Path,
    selected_summary: pd.DataFrame,
    depth_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> None:
    text = f"""# Question 3 measured results

All quantities use the fixed clean test set and the same 30 saved within-identity training resamples. Accuracy standard deviations use `ddof=1`; the assignment's bias/variance formulas use their stated population denominators.

## Q3(a): selected models

{selected_summary.to_markdown(index=False, floatfmt='.6f')}

## Q3(b-c): depth study

{depth_summary.to_markdown(index=False, floatfmt='.6f')}

## Q3(d): final summary

{final_summary.to_markdown(index=False, floatfmt='.6f')}
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    master_seed = seed_from_roll_number(config["student"]["roll_number"])
    q3_config = config["question_3"]
    output_dir = project_path(q3_config["output_dir"])
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    for directory in (tables_dir, figures_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    started = datetime.now(timezone.utc)
    wall_start = perf_counter()

    dataset = load_olivetti(config["data"]["data_home"])
    split = load_split(config["data"]["split_file"], dataset.target)
    if split.seed != master_seed:
        raise RuntimeError("Saved split and configured roll number do not agree.")
    classes = np.arange(40, dtype=np.int64)
    X_test, y_test = dataset.data[split.test], dataset.target[split.test]

    q1_output = ASSIGNMENT_ROOT / "question_1" / "outputs"
    q2_output = ASSIGNMENT_ROOT / "question_2" / "outputs"
    tree_config = read_json(q1_output / "frozen_baseline_config.json")
    bagging_config = read_json(q1_output / "frozen_bagging_config.json")
    boosting_config = read_json(q2_output / "frozen_boosting_config.json")
    n_repetitions = int(q3_config["n_repetitions"])
    resampling_seed = derive_seed(master_seed, "q3_within_identity_resampling")
    resample_path = project_path(q3_config["resample_file"])
    resamples, resamples_created = get_or_create_resamples(
        resample_path,
        split.train,
        dataset.target,
        n_repetitions,
        resampling_seed,
    )
    resamples_sha = resample_digest(resamples, resampling_seed)

    protocol: dict[str, object] = {
        "master_seed": master_seed,
        "split_sha256": split_digest(split),
        "resampling_seed": resampling_seed,
        "resamples_sha256": resamples_sha,
        "n_repetitions": n_repetitions,
        "global_class_order": classes,
        "selected_tree": {
            "max_depth": tree_config["max_depth"],
            "min_samples_leaf": tree_config["min_samples_leaf"],
        },
        "selected_bagging": {
            "n_estimators": bagging_config["n_estimators"],
            "max_depth": bagging_config["max_depth"],
            "min_samples_leaf": bagging_config["min_samples_leaf"],
        },
        "selected_boosting": {
            "n_estimators": boosting_config["n_estimators"],
            "weak_learner_depth": boosting_config["weak_learner_depth"],
        },
        "depth_study": {
            "shallow": 2,
            "deep_capped": 16,
            "selected_and_deepest": None,
            "min_samples_leaf": tree_config["min_samples_leaf"],
            "bagging_estimators": 100,
        },
        "randomness_policy": (
            "Independent deterministic child seeds per repetition and model; "
            "all selected hyperparameters remain frozen."
        ),
        "accuracy_std_ddof": 1,
    }
    run_protocol_hash = protocol_hash(protocol)
    protocol["protocol_sha256"] = run_protocol_hash
    write_json(output_dir / "q3_protocol.json", protocol)
    LOGGER.info(
        "Q3 protocol frozen: repetitions=%d, resamples=%s, protocol=%s",
        n_repetitions,
        resamples_sha,
        run_protocol_hash,
    )

    checkpoint_paths = [
        checkpoints_dir / f"repetition_{repetition:02d}.npz"
        for repetition in range(n_repetitions)
    ]
    missing = []
    for repetition, path in enumerate(checkpoint_paths):
        if path.exists():
            load_repetition_checkpoint(path, run_protocol_hash)
        else:
            missing.append(repetition)
    LOGGER.info(
        "Checkpoint scan: %d complete, %d remaining", n_repetitions - len(missing), len(missing)
    )
    if missing:
        n_jobs = min(int(q3_config["n_jobs"]), len(missing))
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size=1)(
            delayed(run_repetition)(
                repetition,
                resamples[repetition],
                dataset.data,
                dataset.target,
                X_test,
                classes,
                master_seed,
                tree_config,
                bagging_config,
                boosting_config,
                run_protocol_hash,
                str(checkpoint_paths[repetition]),
            )
            for repetition in missing
        )

    all_predictions: dict[str, list[np.ndarray]] = {key: [] for key in PREDICTION_KEYS}
    repetition_metadata: list[dict[str, object]] = []
    for path in checkpoint_paths:
        predictions, metadata = load_repetition_checkpoint(path, run_protocol_hash)
        for key in PREDICTION_KEYS:
            all_predictions[key].append(predictions[key])
        repetition_metadata.append(metadata)
    stacked = {key: np.stack(values, axis=0) for key, values in all_predictions.items()}
    np.savez_compressed(output_dir / "q3_prediction_vectors.npz", **stacked)

    selected_key_labels = [
        ("selected_tree", "single_tree"),
        ("selected_bagging_200", "bagging_200"),
        ("selected_boosting", "boosting_100"),
    ]
    selected_results: dict[str, object] = {}
    selected_rows: list[dict[str, object]] = []
    accuracy_rows = {"repetition": np.arange(n_repetitions)}
    for key, label in selected_key_labels:
        result = measured_bias_variance(stacked[key], y_test, classes)
        selected_results[label] = result
        selected_rows.append(metrics_to_row(label, result))
        accuracy_rows[label] = result.repetition_accuracies
    selected_summary = pd.DataFrame(selected_rows)
    selected_summary.to_csv(tables_dir / "q3a_selected_models.csv", index=False)
    pd.DataFrame(accuracy_rows).to_csv(
        tables_dir / "q3a_repetition_accuracies.csv", index=False
    )
    plot_selected_models(selected_summary, figures_dir / "q3a_selected_models.png")

    depth_specs = [
        ("2", "2 (shallow)", "tree_depth_2", "bagging_100_depth_2"),
        ("16", "16 (deep capped)", "tree_depth_16", "bagging_100_depth_16"),
        ("selected", "None (selected)", "selected_tree", "bagging_100_selected"),
    ]
    depth_rows: list[dict[str, object]] = []
    for metadata_key, label, tree_key, bag_key in depth_specs:
        tree_result = measured_bias_variance(stacked[tree_key], y_test, classes)
        bag_result = measured_bias_variance(stacked[bag_key], y_test, classes)
        tree_leaf_rows = [item["leaf_statistics"][metadata_key] for item in repetition_metadata]
        bag_leaf_rows = [
            item["bagging_leaf_statistics"][metadata_key] for item in repetition_metadata
        ]
        disagreements = np.asarray(
            [item["bagging_disagreement"][metadata_key] for item in repetition_metadata],
            dtype=np.float64,
        )
        variance_removed = tree_result.variance - bag_result.variance
        depth_rows.append(
            {
                "depth_label": label,
                "tree_bias_squared": tree_result.bias_squared,
                "tree_variance": tree_result.variance,
                "tree_mean_accuracy": tree_result.mean_accuracy,
                "tree_accuracy_std_ddof1": tree_result.accuracy_std,
                "tree_mean_actual_depth": float(
                    np.mean([row["actual_depth"] for row in tree_leaf_rows])
                ),
                "tree_mean_number_of_leaves": float(
                    np.mean([row["number_of_leaves"] for row in tree_leaf_rows])
                ),
                "tree_mean_samples_per_leaf": float(
                    np.mean([row["mean_samples_per_leaf"] for row in tree_leaf_rows])
                ),
                "tree_mean_singleton_leaf_fraction": float(
                    np.mean([row["singleton_leaf_fraction"] for row in tree_leaf_rows])
                ),
                "bagging_bias_squared": bag_result.bias_squared,
                "bagging_variance": bag_result.variance,
                "bagging_mean_accuracy": bag_result.mean_accuracy,
                "bagging_accuracy_std_ddof1": bag_result.accuracy_std,
                "bagging_mean_tree_leaves": float(
                    np.mean([row["mean_number_of_leaves"] for row in bag_leaf_rows])
                ),
                "bagging_mean_samples_per_leaf": float(
                    np.mean([row["mean_samples_per_leaf"] for row in bag_leaf_rows])
                ),
                "bagging_mean_singleton_leaf_fraction": float(
                    np.mean([row["mean_singleton_leaf_fraction"] for row in bag_leaf_rows])
                ),
                "bagging_mean_disagreement": float(disagreements.mean()),
                "bagging_disagreement_std_ddof1": float(disagreements.std(ddof=1)),
                "variance_removed_absolute": variance_removed,
                "variance_removed_percent": 100.0 * variance_removed / tree_result.variance,
                "accuracy_improvement": bag_result.mean_accuracy - tree_result.mean_accuracy,
            }
        )
    depth_summary = pd.DataFrame(depth_rows)
    depth_summary.to_csv(tables_dir / "q3bc_depth_study.csv", index=False)
    plot_depth_bias_variance(
        depth_summary, figures_dir / "q3b_depth_bias_variance_accuracy.png"
    )
    plot_depth_evidence(depth_summary, figures_dir / "q3c_depth_evidence.png")

    original_robustness = pd.read_csv(
        q2_output / "tables" / "q2d_clean_masked_comparison.csv"
    )
    selected_lookup = {
        "single_tree": selected_results["single_tree"],
        "bagging": selected_results["bagging_200"],
        "boosting": selected_results["boosting_100"],
    }
    final_rows: list[dict[str, object]] = []
    for model in ("single_tree", "bagging", "boosting"):
        clean_accuracy = float(
            original_robustness.loc[
                (original_robustness["model"] == model)
                & (original_robustness["test_set"] == "clean"),
                "accuracy",
            ].iloc[0]
        )
        masked_accuracy = float(
            original_robustness.loc[
                (original_robustness["model"] == model)
                & (original_robustness["test_set"] == "masked"),
                "accuracy",
            ].iloc[0]
        )
        result = selected_lookup[model]
        final_rows.append(
            {
                "model": model,
                "clean_test_accuracy": clean_accuracy,
                "masked_test_accuracy": masked_accuracy,
                "masked_minus_clean_accuracy": masked_accuracy - clean_accuracy,
                "bias_squared": result.bias_squared,
                "variance": result.variance,
            }
        )
    final_summary = pd.DataFrame(final_rows)
    final_summary.to_csv(tables_dir / "q3d_final_summary.csv", index=False)

    metadata_rows: list[dict[str, object]] = []
    for item in repetition_metadata:
        metadata_rows.append(
            {
                "repetition": item["repetition"],
                "elapsed_seconds": item["elapsed_seconds"],
                "boosting_achieved_rounds": item["boosting_achieved_rounds"],
                "boosting_termination_reason": item["boosting_termination_reason"],
            }
        )
    pd.DataFrame(metadata_rows).sort_values("repetition").to_csv(
        tables_dir / "q3_repetition_metadata.csv", index=False
    )
    write_measured_summary(
        output_dir / "MEASURED_RESULTS.md",
        selected_summary,
        depth_summary,
        final_summary,
    )

    ended = datetime.now(timezone.utc)
    run_metadata = {
        "run_started_utc": started.isoformat(),
        "run_ended_utc": ended.isoformat(),
        "wall_time_seconds": perf_counter() - wall_start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib_n_jobs": int(q3_config["n_jobs"]),
        "master_seed": master_seed,
        "resampling_seed": resampling_seed,
        "resamples_created": resamples_created,
        "resamples_sha256": resamples_sha,
        "protocol_sha256": run_protocol_hash,
        "split_sha256": split_digest(split),
        "n_repetitions": n_repetitions,
        "output_directory": output_dir,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    LOGGER.info("Q3 run complete in %.2f seconds", run_metadata["wall_time_seconds"])


if __name__ == "__main__":
    main()
