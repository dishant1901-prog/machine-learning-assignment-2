# Assignment 2 Working Rules

This file is the standing checklist for all work on **Machine Learning Assignment 2: Bagging, Boosting and Bias-Variance on Face Recognition**. Read it before starting a new question and before accepting any result as final.

## 1. Source of truth and submission limits

- The authoritative brief is `../ML_Assignment_2_new.pdf` (2 pages).
- Total: 100 marks. This is an individual assignment.
- Due: Wednesday, 21 October 2026.
- Final report: one CVPR-format, two-column PDF, at most 8 pages excluding references.
- Page 1 must contain the student's name, complete roll number, repository link, and final commit hash.
- The repository must include complete code, the saved split, configurations, final logs, figure-generation scripts, an environment file, and exact reproduction commands.
- Everything submitted must be understandable and rerunnable by the student. Never invent measurements, timings, plots, examples, logs, citations, or conclusions.

## 2. Workspace and file organization

- Keep Assignment 2 work inside this `assignment_2/` directory.
- Keep shared data/setup/code and immutable artifacts in `shared/`.
- Keep question-specific code, outputs, notes, and figures in `question_1/`, `question_2/`, and `question_3/`.
- Keep report sources and the final PDF in `report/`.
- Do not overwrite the supplied PDF.
- Use clear filenames and relative paths. Generated outputs must be traceable to a script and configuration.
- Update `STATUS.md` at every meaningful checkpoint: setup, split creation, completion of each subpart, model selection/freeze, verification, report integration, and final reproduction run.

## 3. Reproducibility and leakage prevention

- Before running experiments, obtain the student's **complete roll number**. Concatenate all of its numeric characters in their original order and use the resulting non-negative integer as the seed for `numpy.random.default_rng`.
- Derive all additional randomness deterministically from the roll-number seed. Record seeds/configurations in machine-readable artifacts and logs.
- Use `fetch_olivetti_faces`: 40 identities, 10 images per identity, image size 64 x 64.
- Within every identity, randomly split its 10 images into exactly 6 train, 2 validation, and 2 test images. Expected totals: 240 train, 80 validation, and 80 test.
- Save split indices once under `shared/`; verify disjointness, full coverage, and per-class counts; reuse the exact split everywhere.
- Use raw flattened pixel values only. Do not use PCA, feature selection, embeddings, or any other dimensionality reduction.
- Train every model only on clean training images.
- Construct the masked test set from the fixed clean test set by setting rows 24 through 39 and columns 24 through 39 (`24:40` in Python slicing) to zero, then flattening consistently.
- Use validation accuracy only for hyperparameter/model-size selection. Never use clean test, masked test, or Question 3 refits to tune a model.
- Once a setting is selected, freeze it and save it before evaluating test performance or beginning dependent questions.
- Establish deterministic tie-breaks before inspecting test results. Unless a question requires otherwise, among equal validation accuracies prefer the less complex option: smaller ensemble/round count, then shallower depth, then larger `min_samples_leaf`.

## 4. General implementation and evidence standards

- `DecisionTreeClassifier` may be used. Bagging and SAMME boosting must be implemented in project code, not through scikit-learn bagging or boosting estimators.
- Add checks for shapes, class order, probability/vote normalization, split integrity, and deterministic reruns.
- Preserve the same global class order (0 through 39) in every probability or vote vector.
- Save raw result tables before formatting report tables or plots.
- Every figure must have labeled axes, units/definitions where needed, a readable legend, and a script that regenerates it.
- Report enough precision to distinguish configurations without implying false accuracy; retain full precision in saved raw outputs.
- Conclusions must refer to measured results and specific plots/tables, not generic textbook claims.
- Keep prose natural, specific, and consistent with the actual experiment. Avoid inflated claims, canned filler, and unsupported causal language.
- Code should be straightforward, commented where reasoning is non-obvious, and easy to explain during a rerun or viva.

## 5. Question 1: tree baseline and bagging (35 marks)

### Q1(a), 9 marks

- Train the full grid of `max_depth in {2, 4, 8, 16, None}` and `min_samples_leaf in {1, 2, 4}`.
- Select exactly one setting using validation accuracy only and freeze it.
- For the selected baseline, report train, validation, and clean test accuracy; number of leaves; mean training samples per leaf; and fraction of leaves containing exactly one training sample.
- Explain depth effects using the measured grid and leaf occupancy, not only theory.

### Q1(b), 9 marks

- Implement bootstrap sampling and hard-vote aggregation directly.
- Each tree receives a bootstrap sample of the original training-set size (240 after the required split) and uses the frozen Q1(a) tree settings.
- Evaluate `B in {1, 5, 10, 25, 50, 100, 200}` and report train, validation, and clean test accuracy plus the average distinct examples per bootstrap.
- Compare measured distinct counts with `n * (1 - (1 - 1/n)^n)` and the asymptotic fraction `1 - exp(-1)`.
- Prefer one reproducible 200-tree run whose prefixes define smaller B values, so comparisons are paired and do not add avoidable random variation.

### Q1(c), 8 marks

- For every B, compute out-of-bag (OOB) accuracy, average pairwise tree disagreement, and mean vote margin (winning vote fraction minus runner-up vote fraction).
- For OOB accuracy, evaluate only training samples with at least one OOB prediction and also record OOB coverage, especially for small B.
- Compute disagreement and margin on the clean validation set for model analysis/selection; label that evaluation set explicitly.
- Plot all three quantities against B.
- Select and freeze one final ensemble size using clean validation accuracy only. Explain whether OOB accuracy, disagreement, and margin support the same interpretation.

### Q1(d), 9 marks

- Compare the frozen tree and final bagged ensemble on both clean and masked test sets.
- Use tree leaf class probability (`predict_proba`) as tree confidence and winning vote fraction as bagging confidence; report the mean winning confidence over each evaluated set.
- Find two fixed-test faces where bagging corrects the tree and two where both models are wrong.
- For each example, show the image, true label, tree prediction, and complete/nonzero bagging vote counts needed to support the explanation.
- Distinguish errors caused by unstable individual-tree decisions from errors shared across the ensemble or caused by missing information.

## 6. Question 2: multiclass SAMME boosting (35 marks)

- Implement SAMME directly with
  `alpha_t = log((1 - error_t) / error_t) + log(K - 1)` and
  `S_k(x) = sum_t alpha_t * 1[h_t(x) = k]`.
- Update sample weights with the same explicit rule in the toy and face experiments, normalize after each round, and test the update numerically.
- Use normalized weighted votes `q_k(x) = S_k(x) / sum_t(alpha_t)` for confidence and class-vector outputs.
- A learner is useful only when `error_t < 1 - 1/K`. If this condition fails, do not add a non-positive/invalid learner; stop cleanly (or use a predeclared deterministic retry policy), and log the reason and achieved round count. Handle zero error safely and explicitly.

### Q2(a), 8 marks

- Create a three-class example with no more than 10 samples.
- Log one complete round: initial weights, weak learner predictions, weighted error, learner weight, unnormalized updated weights, and normalized updated weights.

### Q2(b), 9 marks

- On Olivetti Faces evaluate weak-learner depths `{1, 2, 3, 4}` and rounds `{10, 25, 50, 100}`.
- Select depth and rounds using clean validation accuracy only, apply the declared tie-break, and freeze the result.
- For the selected model, plot per-round weighted training error, learner weight, maximum sample weight, and total weight of the 10 heaviest training images.

### Q2(c), 9 marks

- Compare boosting and bagging at 25, 50, and 100 learners.
- Report train, validation, and clean test accuracy plus training time, using the same machine, consistent timing boundaries, and the same execution conditions.
- Inspect one training image receiving a very large boosting weight. Preserve its round-by-round true label, predictions, correctness, and weight history, and explain the repeated attention using those measurements.

### Q2(d), 9 marks

- Freeze the selected boosting model before this comparison.
- Evaluate the frozen baseline tree, final bagging model, and selected boosting model on clean and masked test sets.
- Report accuracy and mean winning vote/probability for each model and set.
- Find one fixed-test face where bagging and boosting respond differently to masking. Explain it using measured tree disagreement and boosting/sample-weight behavior.

## 7. Question 3: measured bias and variance (30 marks)

- Keep the final clean test set fixed.
- Create exactly 30 training sets. Within each identity independently, sample six faces with replacement from that identity's original six training faces.
- Refit, without any retuning, the selected baseline tree, final bagging model, and selected boosting model on every resampled training set.
- For the single tree use its full 40-class probability vector; for bagging use the vote-fraction vector; for boosting use the normalized weighted-vote vector.
- With repetitions `r`, compute the mean prediction vector per fixed test face and implement the brief's exact squared-bias and variance formulas. Add numerical checks that vectors align with the same class order and normalize as intended.

### Q3(a), 7 marks

- Report measured bias squared, variance, and mean plus standard deviation of clean-test accuracy over the 30 refits for all three selected models.

### Q3(b), 8 marks

- Predeclare one shallow depth, the selected depth, and one much deeper depth without retuning on test results.
- At each depth evaluate both a single tree and a 100-tree bagged ensemble across the same 30 resamples.
- Quantify depth effects on bias/variance and how much variance bagging removes at each depth.

### Q3(c), 7 marks

- Determine from measurements whether the deepest tree is the depth at which bagging improves test accuracy most.
- The explanation must combine measured bias, variance, leaf occupancy, and disagreement.

### Q3(d), 8 marks

- End with one summary table for the selected tree, bagging, and boosting models containing clean-test accuracy, masked-test accuracy, bias squared, and variance.
- Use that table and earlier plots to identify stability, sensitivity to masking, and whether lowest variance coincides with best test accuracy.

## 8. Checkpoint and final-quality gate

Before marking any subpart complete:

1. Run the relevant code from a clean entry point with the recorded seed/configuration.
2. Confirm expected files were regenerated and no test information entered selection.
3. Check formulas and at least one value manually or with an independent sanity check.
4. Review plots/tables against raw logs.
5. Update `STATUS.md` with what is complete, evidence paths, frozen decisions, failures, and the exact next action.
6. Do not call the assignment complete until an exact end-to-end reproduction command succeeds and the report satisfies the CVPR/page/submission checklist.
