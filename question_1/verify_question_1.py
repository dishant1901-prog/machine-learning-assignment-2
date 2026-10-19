"""Verify saved Question 1 artifacts without refitting the 200-tree ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
if str(ASSIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_ROOT))

from shared.data import (  # noqa: E402
    create_identity_split,
    load_config,
    load_olivetti,
    load_split,
    project_path,
    seed_from_roll_number,
    split_digest,
)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    config = load_config(ASSIGNMENT_ROOT / "shared" / "config.json")
    seed = seed_from_roll_number(config["student"]["roll_number"])
    assert seed == 242003

    dataset = load_olivetti(config["data"]["data_home"])
    split = load_split(config["data"]["split_file"], dataset.target)
    recreated = create_identity_split(dataset.target, seed)
    assert split_digest(split) == split_digest(recreated)
    assert split.seed == seed

    output = project_path(config["question_1"]["output_dir"])
    tables = output / "tables"
    figures = output / "figures"
    baseline = pd.read_csv(tables / "q1a_baseline_grid.csv", keep_default_na=False)
    bagging = pd.read_csv(tables / "q1bc_bagging_results.csv")
    robustness = pd.read_csv(tables / "q1d_clean_masked_comparison.csv")
    cases = pd.read_csv(tables / "q1d_case_evidence.csv")
    frozen_tree = read_json(output / "frozen_baseline_config.json")
    frozen_bag = read_json(output / "frozen_bagging_config.json")

    def baseline_key(row: pd.Series) -> tuple[float, float, int]:
        depth = np.inf if row["max_depth"] == "None" else int(row["max_depth"])
        return (
            float(row["validation_accuracy"]),
            -float(depth),
            int(row["min_samples_leaf"]),
        )

    selected_baseline = max(
        (row for _, row in baseline.iterrows()), key=baseline_key
    )
    selected_depth = (
        None
        if selected_baseline["max_depth"] == "None"
        else int(selected_baseline["max_depth"])
    )
    assert frozen_tree["max_depth"] == selected_depth
    assert frozen_tree["min_samples_leaf"] == int(
        selected_baseline["min_samples_leaf"]
    )

    selected_bag = bagging.sort_values(
        ["validation_accuracy", "B"], ascending=[False, True], kind="stable"
    ).iloc[0]
    assert frozen_bag["n_estimators"] == int(selected_bag["B"])
    assert set(bagging["B"]) == {1, 5, 10, 25, 50, 100, 200}

    n_train = split.train.size
    theoretical = n_train * (1.0 - (1.0 - 1.0 / n_train) ** n_train)
    assert np.allclose(bagging["theoretical_distinct_examples"], theoretical)
    assert bagging.loc[bagging["B"] >= 25, "oob_coverage"].eq(1.0).all()
    assert np.isnan(bagging.loc[bagging["B"] == 1, "average_pairwise_disagreement_validation"]).all()

    assert set(zip(robustness["test_set"], robustness["model"])) == {
        ("clean", "single_tree"),
        ("clean", "bagging"),
        ("masked", "single_tree"),
        ("masked", "bagging"),
    }
    assert (cases["category"] == "bagging_corrects_tree").sum() == 2
    assert (cases["category"] == "both_wrong").sum() == 2
    for encoded_votes in cases["bagging_nonzero_vote_counts"]:
        votes = json.loads(encoded_votes)
        assert sum(votes.values()) == frozen_bag["n_estimators"]

    required_figures = {
        "q1a_depth_diagnostics.png",
        "q1b_accuracy_and_sampling.png",
        "q1c_oob_disagreement_margin.png",
        "q1d_case_evidence.png",
    }
    assert required_figures == {path.name for path in figures.glob("*.png")}
    assert all((figures / name).stat().st_size > 0 for name in required_figures)

    print("Question 1 artifact verification passed.")
    print(f"seed={seed}")
    print(f"split_sha256={split_digest(split)}")
    print(
        f"selected_tree=max_depth:{frozen_tree['max_depth']},"
        f"min_samples_leaf:{frozen_tree['min_samples_leaf']}"
    )
    print(f"selected_bagging_B={frozen_bag['n_estimators']}")


if __name__ == "__main__":
    main()
