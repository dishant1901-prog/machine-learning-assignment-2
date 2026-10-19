"""Verify the saved Question 3 evidence without refitting any model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from question_3.bias_variance import (  # noqa: E402
    measured_bias_variance,
    resample_digest,
    validate_resamples,
)
from question_3.run_question_3 import (  # noqa: E402
    PREDICTION_KEYS,
    load_repetition_checkpoint,
    protocol_hash,
)
from shared.data import (  # noqa: E402
    load_config,
    load_olivetti,
    load_split,
    project_path,
    seed_from_roll_number,
    split_digest,
)


def require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    config = load_config(ASSIGNMENT_ROOT / "shared" / "config.json")
    q3_config = config["question_3"]
    output_dir = project_path(q3_config["output_dir"])
    tables_dir = output_dir / "tables"

    master_seed = seed_from_roll_number(config["student"]["roll_number"])
    dataset = load_olivetti(config["data"]["data_home"])
    split = load_split(config["data"]["split_file"], dataset.target)
    y_test = dataset.target[split.test]
    classes = np.arange(40, dtype=np.int64)

    with (output_dir / "q3_protocol.json").open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    recorded_protocol_hash = str(protocol["protocol_sha256"])
    hash_payload = dict(protocol)
    hash_payload.pop("protocol_sha256")
    if protocol_hash(hash_payload) != recorded_protocol_hash:
        raise AssertionError("The Q3 protocol digest does not match its contents.")
    if int(protocol["master_seed"]) != master_seed:
        raise AssertionError("The Q3 protocol uses the wrong roll-number seed.")
    if str(protocol["split_sha256"]) != split_digest(split):
        raise AssertionError("The Q3 protocol does not use the saved official split.")

    resample_path = project_path(q3_config["resample_file"])
    with np.load(resample_path, allow_pickle=False) as saved:
        resamples = np.asarray(saved["indices"], dtype=np.int64)
        resampling_seed = int(str(saved["seed"].item()))
        recorded_resample_hash = str(saved["sha256"].item())
    validate_resamples(resamples, split.train, dataset.target)
    if resamples.shape != (30, 240):
        raise AssertionError(f"Unexpected resample shape: {resamples.shape}")
    computed_resample_hash = resample_digest(resamples, resampling_seed)
    if computed_resample_hash != recorded_resample_hash:
        raise AssertionError("The saved resample digest does not match its contents.")
    if recorded_resample_hash != str(protocol["resamples_sha256"]):
        raise AssertionError("The resamples do not match the frozen Q3 protocol.")

    with np.load(output_dir / "q3_prediction_vectors.npz", allow_pickle=False) as saved:
        predictions = {
            key: np.asarray(saved[key], dtype=np.float64) for key in PREDICTION_KEYS
        }
    for key, values in predictions.items():
        if values.shape != (30, 80, 40):
            raise AssertionError(f"{key} has unexpected shape {values.shape}.")
        if np.any(values < -1e-12) or not np.allclose(values.sum(axis=2), 1.0):
            raise AssertionError(f"{key} does not contain valid class vectors.")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_paths = sorted(checkpoint_dir.glob("repetition_*.npz"))
    if len(checkpoint_paths) != 30:
        raise AssertionError(f"Expected 30 checkpoints, found {len(checkpoint_paths)}.")
    for repetition, path in enumerate(checkpoint_paths):
        saved_predictions, metadata = load_repetition_checkpoint(
            path, recorded_protocol_hash
        )
        if int(metadata["repetition"]) != repetition:
            raise AssertionError(f"Checkpoint ordering failed at {path.name}.")
        if int(metadata["boosting_achieved_rounds"]) != 100:
            raise AssertionError(f"Boosting stopped early in {path.name}.")
        if metadata["boosting_termination_reason"] != "requested_rounds_completed":
            raise AssertionError(f"Unexpected boosting termination in {path.name}.")
        for key in PREDICTION_KEYS:
            if not np.array_equal(saved_predictions[key], predictions[key][repetition]):
                raise AssertionError(f"Combined vectors differ from {path.name}: {key}.")

    q3a = pd.read_csv(tables_dir / "q3a_selected_models.csv").set_index("model")
    selected_keys = {
        "single_tree": "selected_tree",
        "bagging_200": "selected_bagging_200",
        "boosting_100": "selected_boosting",
    }
    results = {}
    target_positions = np.searchsorted(classes, y_test)
    one_hot = np.zeros((y_test.size, classes.size), dtype=np.float64)
    one_hot[np.arange(y_test.size), target_positions] = 1.0
    for model, key in selected_keys.items():
        result = measured_bias_variance(predictions[key], y_test, classes)
        results[model] = result
        for column, actual in (
            ("bias_squared", result.bias_squared),
            ("variance", result.variance),
            ("mean_test_accuracy", result.mean_accuracy),
            ("test_accuracy_std_ddof1", result.accuracy_std),
        ):
            require_close(actual, float(q3a.loc[model, column]), f"{model} {column}")
        mean_squared_error = float(
            np.mean(np.sum((predictions[key] - one_hot[None, :, :]) ** 2, axis=2))
        )
        require_close(
            mean_squared_error,
            result.bias_squared + result.variance,
            f"{model} squared-error decomposition",
        )

    depth_table = pd.read_csv(tables_dir / "q3bc_depth_study.csv").set_index(
        "depth_label"
    )
    depth_keys = {
        "2 (shallow)": ("tree_depth_2", "bagging_100_depth_2"),
        "16 (deep capped)": ("tree_depth_16", "bagging_100_depth_16"),
        "None (selected)": ("selected_tree", "bagging_100_selected"),
    }
    for label, (tree_key, bag_key) in depth_keys.items():
        tree_result = measured_bias_variance(predictions[tree_key], y_test, classes)
        bag_result = measured_bias_variance(predictions[bag_key], y_test, classes)
        row = depth_table.loc[label]
        checks = {
            "tree_bias_squared": tree_result.bias_squared,
            "tree_variance": tree_result.variance,
            "tree_mean_accuracy": tree_result.mean_accuracy,
            "tree_accuracy_std_ddof1": tree_result.accuracy_std,
            "bagging_bias_squared": bag_result.bias_squared,
            "bagging_variance": bag_result.variance,
            "bagging_mean_accuracy": bag_result.mean_accuracy,
            "bagging_accuracy_std_ddof1": bag_result.accuracy_std,
            "variance_removed_absolute": tree_result.variance - bag_result.variance,
            "variance_removed_percent": 100.0
            * (tree_result.variance - bag_result.variance)
            / tree_result.variance,
            "accuracy_improvement": bag_result.mean_accuracy - tree_result.mean_accuracy,
        }
        for column, actual in checks.items():
            require_close(actual, float(row[column]), f"{label} {column}")

    q2_robustness = pd.read_csv(
        project_path(config["question_2"]["output_dir"])
        / "tables"
        / "q2d_clean_masked_comparison.csv"
    )
    q3d = pd.read_csv(tables_dir / "q3d_final_summary.csv").set_index("model")
    for model in ("single_tree", "bagging", "boosting"):
        clean = float(
            q2_robustness.loc[
                (q2_robustness["model"] == model)
                & (q2_robustness["test_set"] == "clean"),
                "accuracy",
            ].iloc[0]
        )
        masked = float(
            q2_robustness.loc[
                (q2_robustness["model"] == model)
                & (q2_robustness["test_set"] == "masked"),
                "accuracy",
            ].iloc[0]
        )
        require_close(clean, float(q3d.loc[model, "clean_test_accuracy"]), f"{model} clean")
        require_close(masked, float(q3d.loc[model, "masked_test_accuracy"]), f"{model} masked")
        require_close(
            masked - clean,
            float(q3d.loc[model, "masked_minus_clean_accuracy"]),
            f"{model} mask delta",
        )

    for name in (
        "q3a_selected_models.png",
        "q3b_depth_bias_variance_accuracy.png",
        "q3c_depth_evidence.png",
    ):
        path = output_dir / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing figure: {path}")

    print("Question 3 artifact verification passed.")
    print(f"  protocol SHA-256: {recorded_protocol_hash}")
    print(f"  resample SHA-256: {recorded_resample_hash}")
    print("  30 checkpoints and all eight (30, 80, 40) vector arrays agree")
    print("  bias/variance, depth study, accuracies, and robustness table agree")


if __name__ == "__main__":
    main()
