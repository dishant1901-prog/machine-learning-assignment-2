"""Independently verify the saved Question 2 artifacts and selected SAMME refit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from shared.boosting import ScratchSAMMEClassifier  # noqa: E402
from shared.data import (  # noqa: E402
    load_config,
    load_olivetti,
    load_split,
    mask_central_block,
    project_path,
    seed_from_roll_number,
    split_digest,
)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pairwise_disagreement(counts: np.ndarray) -> float:
    total = int(counts.sum())
    agreeing = int(np.sum(counts * (counts - 1) // 2))
    return 1.0 - agreeing / (total * (total - 1) // 2)


def main() -> None:
    config = load_config(ASSIGNMENT_ROOT / "shared" / "config.json")
    seed = seed_from_roll_number(config["student"]["roll_number"])
    dataset = load_olivetti(config["data"]["data_home"])
    split = load_split(config["data"]["split_file"], dataset.target)
    assert split.seed == seed == 242003

    output = project_path(config["question_2"]["output_dir"])
    tables = output / "tables"
    figures = output / "figures"
    toy = pd.read_csv(tables / "q2a_toy_round.csv")
    toy_summary = read_json(output / "q2a_toy_round_summary.json")
    grid = pd.read_csv(tables / "q2b_validation_grid.csv")
    frozen = read_json(output / "frozen_boosting_config.json")
    diagnostics = pd.read_csv(tables / "q2b_selected_round_diagnostics.csv")
    comparison = pd.read_csv(tables / "q2c_bagging_boosting_comparison.csv")
    history = pd.read_csv(tables / "q2c_heaviest_sample_history.csv")
    heaviest = read_json(output / "q2c_heaviest_sample_summary.json")
    robustness = pd.read_csv(tables / "q2d_clean_masked_comparison.csv")
    mask_case = read_json(output / "q2d_mask_response_case.json")

    assert len(toy) == 9 and toy["true_class"].nunique() == 3
    assert np.isclose(toy["initial_weight"].sum(), 1.0)
    assert np.isclose(toy["normalized_updated_weight"].sum(), 1.0)
    error = float(
        np.dot(toy["initial_weight"], toy["incorrect"].astype(bool))
    )
    alpha = np.log((1.0 - error) / error) + np.log(2.0)
    assert np.isclose(error, toy_summary["weighted_error"])
    assert np.isclose(alpha, toy_summary["learner_weight_alpha"])
    expected_unnormalized = toy["initial_weight"] * np.exp(
        alpha * toy["incorrect"]
    )
    assert np.allclose(expected_unnormalized, toy["unnormalized_updated_weight"])

    expected_grid = {(depth, rounds) for depth in (1, 2, 3, 4) for rounds in (10, 25, 50, 100)}
    assert set(zip(grid["weak_learner_depth"], grid["rounds"])) == expected_grid
    available = grid[grid["available"]]
    selected = available.sort_values(
        ["validation_accuracy", "rounds", "weak_learner_depth"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    assert int(selected["weak_learner_depth"]) == frozen["weak_learner_depth"]
    assert int(selected["rounds"]) == frozen["n_estimators"]
    assert frozen["frozen_before_test_evaluation"] is True

    expected_alphas = np.log(
        (1.0 - diagnostics["weighted_training_error"])
        / diagnostics["weighted_training_error"]
    ) + np.log(39.0)
    assert np.allclose(expected_alphas, diagnostics["learner_weight"])
    assert (diagnostics["weighted_training_error"] < 39.0 / 40.0).all()
    assert len(diagnostics) == frozen["n_estimators"]
    assert set(zip(comparison["model"], comparison["learners"])) == {
        (model, learners)
        for model in ("bagging", "boosting")
        for learners in (25, 50, 100)
    }
    assert set(zip(robustness["test_set"], robustness["model"])) == {
        (test_set, model)
        for test_set in ("clean", "masked")
        for model in ("single_tree", "bagging", "boosting")
    }

    X_train, y_train = dataset.data[split.train], dataset.target[split.train]
    X_validation, y_validation = dataset.data[split.validation], dataset.target[split.validation]
    X_test, y_test = dataset.data[split.test], dataset.target[split.test]
    X_masked = mask_central_block(X_test)
    model = ScratchSAMMEClassifier(
        100,
        max_depth=int(frozen["weak_learner_depth"]),
        random_state=int(frozen["random_state"]),
    ).fit(X_train, y_train)
    rounds = int(frozen["n_estimators"])
    assert model.achieved_rounds_ == 100
    assert np.isclose(model.score(X_train, y_train, rounds), 1.0)
    assert np.isclose(model.score(X_validation, y_validation, rounds), frozen["validation_accuracy"])
    boost_100 = comparison[
        (comparison["model"] == "boosting") & (comparison["learners"] == 100)
    ].iloc[0]
    assert np.isclose(model.score(X_test, y_test, rounds), boost_100["test_accuracy"])
    masked_boost = robustness[
        (robustness["test_set"] == "masked") & (robustness["model"] == "boosting")
    ].iloc[0]
    assert np.isclose(model.score(X_masked, y_test, rounds), masked_boost["accuracy"])

    final_weights = model.sample_weight_history_[rounds]
    heaviest_position = int(np.argmax(final_weights))
    assert heaviest_position == heaviest["training_position"]
    assert np.isclose(final_weights[heaviest_position], heaviest["final_sample_weight"])
    assert np.array_equal(
        model.training_predictions_[:rounds, heaviest_position],
        history["weak_prediction"].to_numpy(),
    )
    assert int((history["correct"] == False).sum()) == heaviest["misclassified_rounds"]  # noqa: E712

    case_position = int(mask_case["test_position"])
    clean_q = model.predict_proba(X_test[case_position : case_position + 1], rounds)[0]
    masked_q = model.predict_proba(X_masked[case_position : case_position + 1], rounds)[0]
    for class_id, value in mask_case["boosting_clean_nonzero_weighted_votes"].items():
        assert np.isclose(clean_q[int(class_id)], value)
    for class_id, value in mask_case["boosting_masked_nonzero_weighted_votes"].items():
        assert np.isclose(masked_q[int(class_id)], value)
    assert np.isclose(clean_q.sum(), 1.0) and np.isclose(masked_q.sum(), 1.0)

    for set_name in ("clean", "masked"):
        counts_dict = mask_case[f"bagging_{set_name}_nonzero_vote_counts"]
        counts = np.asarray(list(counts_dict.values()), dtype=np.int64)
        assert counts.sum() == 200
        assert np.isclose(
            pairwise_disagreement(counts),
            mask_case[f"bagging_{set_name}_pairwise_tree_disagreement"],
        )

    required_figures = {
        "q2b_validation_grid.png",
        "q2b_selected_round_diagnostics.png",
        "q2c_heaviest_training_face.png",
        "q2d_mask_response_case.png",
    }
    assert required_figures == {path.name for path in figures.glob("*.png")}
    assert all((figures / name).stat().st_size > 0 for name in required_figures)

    final_class_mass = {
        int(identity): float(final_weights[y_train == identity].sum())
        for identity in np.unique(y_train)
    }
    case_true = int(mask_case["true_identity"])
    bag_masked_class = int(mask_case["bagging_masked_prediction"])
    print("Question 2 artifact and selected-model verification passed.")
    print(f"split_sha256={split_digest(split)}")
    print(
        f"selected_boosting=depth:{frozen['weak_learner_depth']},"
        f"rounds:{frozen['n_estimators']}"
    )
    print(
        f"final_training_weight_mass_identity_{case_true}="
        f"{final_class_mass[case_true]:.8f}"
    )
    print(
        f"final_training_weight_mass_identity_{bag_masked_class}="
        f"{final_class_mass[bag_masked_class]:.8f}"
    )


if __name__ == "__main__":
    main()
