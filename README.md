# Machine Learning Assignment 2

This directory contains the reproducible implementation and report material for Assignment 2.

## Setup

Python 3.13 is used in the current environment. Install the recorded dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The submitted configuration records Dishant Kaushik, roll number `M24MA2003`. Its numeric characters give the experiment seed `242003`.

## Question 1

Run the verification tests:

```powershell
python -m pytest -q
```

Run the complete Question 1 experiment from this directory:

```powershell
python question_1/run_question_1.py --config shared/config.json
```

The first run may download Olivetti Faces into `shared/data/`. All tabular results, frozen configurations, logs, and figures are written beneath `question_1/outputs/`.

Verify the saved split, selections, tables, case votes, and figure artifacts without refitting:

```powershell
python question_1/verify_question_1.py
```

## Question 2

Run the complete SAMME experiment:

```powershell
python question_2/run_question_2.py --config shared/config.json
```

Independently refit the selected boosting model and verify the saved Q2 artifacts:

```powershell
python question_2/verify_question_2.py
```

## Question 3

Run the complete 30-resample bias-variance experiment (the current recorded run took about 29 minutes with 10 parallel jobs):

```powershell
python question_3/run_question_3.py --config shared/config.json
```

The runner writes a checkpoint after every repetition and safely reuses matching checkpoints if interrupted. Verify all saved resamples, checkpoints, prediction vectors, formulas, result tables, and figures without refitting:

```powershell
python question_3/verify_question_3.py
```

## Exact verification sequence

From the `assignment_2` directory, the following commands check the complete saved submission without rerunning the long model-selection experiments:

```powershell
python -m pytest -q
python question_1/verify_question_1.py
python question_2/verify_question_2.py
python question_3/verify_question_3.py
```

The Q1 and Q3 verifiers operate on the saved evidence. The Q2 verifier independently refits the selected boosting model as an additional consistency check.

## Report

The editable two-column source and build instructions are in `report/`. After the repository is finalized, render the report with the repository commit hash:

```powershell
powershell -ExecutionPolicy Bypass -File report/build_report.ps1
```

The generated `report/final_report.pdf` is intentionally not versioned because it embeds the hash of the final repository commit. It is the PDF submitted separately to the assignment portal.

## Saved evidence

- `shared/split_indices.npz` is the immutable 240/80/80 split.
- `question_1/outputs/`, `question_2/outputs/`, and `question_3/outputs/` contain full-precision tables, frozen configurations, logs, and figures.
- `question_3/outputs/checkpoints/` contains the 30 verified refit checkpoints, allowing the long Q3 runner to resume without repeating completed work.
- The Olivetti download cache under `shared/data/` is not versioned; scikit-learn downloads it automatically on first use.

Do not treat a run as official unless the generated split passes all recorded integrity checks.
