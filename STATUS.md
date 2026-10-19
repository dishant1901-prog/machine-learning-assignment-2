# Assignment 2 Status

**Last updated:** 2026-10-19 (Asia/Calcutta)
**Current checkpoint:** CP9 — final report and complete submission verification passed
**Overall state:** Questions 1–3, the seven-page report source, saved artifacts, and reproducibility checks are complete; the frozen repository commit is ready to upload and render into the submission PDF

## Completed

- Located and read the complete supplied brief: `../ML_Assignment_2_new.pdf` (2 pages).
- Confirmed the assignment contains three dependent questions totaling 100 marks:
  - Q1: decision-tree baseline and bagging from scratch (35)
  - Q2: multiclass SAMME boosting from scratch (35)
  - Q3: measured bias and variance (30)
- Recorded the submission requirements, model grids, metrics, masking procedure, bias/variance formulas, and anti-leakage constraints in `RULES.md`.
- Created separate work areas for `question_1/`, `question_2/`, and `question_3/`, with shared artifacts in `shared/` and report material in `report/`.
- Implemented reusable dataset loading, roll-number seed extraction, deterministic per-identity splitting, split validation/digesting, and central masking in `shared/data.py`.
- Implemented bootstrap sampling, tree fitting, prefix voting, OOB evaluation, pairwise disagreement, vote margin, and leaf statistics without using a library bagging estimator in `shared/bagging.py`.
- Implemented the complete Q1 experiment runner, including validation-only freezing, requested tables, diagnostic plots, robustness evaluation, and face-level case evidence in `question_1/run_question_1.py`.
- Added five focused tests; all five pass under Python 3.13.7.
- Downloaded and verified the official Olivetti Faces data: 400 flattened 4096-pixel samples, 400 images of shape 64 x 64, and identities 0 through 39. Cached artifact: `shared/data/olivetti_py3.pkz`.
- Recorded roll number `M24MA2003`; its numeric characters give the mandatory seed `242003`.
- Generated and saved the official 240/80/80 identity-stratified split in `shared/split_indices.npz`. Its SHA-256 split digest is `5f1c58f75abcc0d7df5cc9f24217731e04d1df3a34240723407d96781990c6c2`.
- Completed Q1(a). Validation selected `max_depth=None`, `min_samples_leaf=2`; the frozen tree has 45 leaves, 5.33 mean training samples per leaf, no singleton leaves, and train/validation/clean-test accuracies of 97.92%/56.25%/57.50%.
- Completed Q1(b-c). Validation selected the 200-tree ensemble; train/validation/clean-test accuracies are 100.00%/86.25%/87.50%. At B=200, OOB accuracy is 79.17%, validation disagreement is 0.7897, and mean validation vote margin is 0.2718.
- Verified the bootstrap calculation: the exact expectation is 151.893 distinct examples from 240 draws; the measured 200-tree average is 151.85.
- Completed Q1(d). On the masked test set, tree accuracy is 33.75% and bagging accuracy is 70.00%. Saved two bagging-corrects-tree cases and two both-wrong cases with complete nonzero vote counts.
- Saved all requested raw tables, frozen configuration files, run metadata/logs, four figures, and a measured summary under `question_1/outputs/`.
- Added `question_1/verify_question_1.py`; artifact verification passes for the split, validation selections, formulas, OOB conventions, case vote totals, and figures.
- Wrote concise, measurement-based Q1 prose in `question_1/REPORT_DRAFT.md`.
- Implemented the SAMME acceptance rule, alpha calculation, shared weight update, normalized weighted voting, learner histories, and explicit random-threshold/zero-error handling in `shared/boosting.py`.
- Added Q2 numerical tests for the manual three-class update, random-quality rejection, perfect-learner stopping, deterministic prefixes, and vote normalization. The full project test suite currently has 9 passing tests.
- Implemented and ran the official Q2(a-d) evidence pipeline in `question_2/run_question_2.py`.
- Completed Q2(a). The nine-sample, three-class round gives weighted error 1/3, alpha `log(4)=1.386294`, normalization constant 2, and the exact expected final weights. The face implementation uses the same update function.
- Completed Q2(b). All depth/round combinations completed. Clean validation selected and froze depth 4 with 100 rounds (51.25% validation accuracy) before Q2 test evaluation. Saved all four requested per-round diagnostics.
- Completed Q2(c). At 25/50/100 learners, bagging clean-test accuracy is 76.25%/81.25%/83.75%, versus 26.25%/43.75%/52.50% for boosting. Corresponding cumulative learner-fit times are 23.04/45.33/90.93 s and 7.12/14.44/28.80 s.
- Inspected the final heaviest training face (identity 36, dataset index 362): final weight 0.08428, 87 misclassified rounds, and 87 weight increases. Saved its complete prediction/weight history and figure.
- Completed Q2(d). Clean/masked accuracies are tree 57.50%/33.75%, bagging 87.50%/70.00%, and boosting 52.50%/53.75%. Saved a measured identity-9 masking case with complete bagging and weighted-boosting votes.
- Added `question_2/verify_question_2.py`. It independently refits the selected SAMME model and verifies the toy calculation, selection, alpha formula, histories, clean/masked results, face case, and figures.
- Wrote compact, measurement-based Q2 prose in `question_2/REPORT_DRAFT.md`.
- Predeclared the Q3 depth study as depth 2 (shallow), depth 16 (deep capped), and the selected unrestricted depth, all with the frozen `min_samples_leaf=2`.
- Implemented deterministic within-identity resampling, exact vector bias-squared/variance calculations, probability alignment, checkpointed parallel refits, leaf occupancy, and ensemble disagreement measurements for Q3.
- Added Q3 tests; the complete project suite has 12 passing tests.
- Completed the official 30-refit Q3 run using deterministic within-identity resampling. The run took 1740.85 seconds, and all 30 boosting refits completed the requested 100 rounds.
- Saved all eight full prediction-vector arrays with shape `(30, 80, 40)`, 30 matching per-repetition checkpoints, and the frozen Q3 protocol. The resample digest is `956e02031c331505858446b1c382315762ceb295ddd8327e103f6d726c3deed0`; the protocol digest is `1de73ca964c6dbef118e8062803de386a5cd67669f035aab739a33d787dc5f97`.
- Completed Q3(a). Measured bias squared/variance/mean clean-test accuracy are tree 0.40394/0.64272/47.67%, bagging 0.49887/0.02062/78.29%, and boosting 0.88015/0.01587/33.63%.
- Completed Q3(b-c). Bagging removes 77.05%, 91.76%, and 96.19% of measured variance at depths 2, 16, and unrestricted, respectively. The largest accuracy gain is at depth 16 (37.96 percentage points), not at the deepest unrestricted setting (29.46 points); saved leaf occupancy and disagreement evidence supports the explanation.
- Completed Q3(d). The final table combines clean/masked accuracies with measured bias and variance. Boosting has the lowest variance but bagging has the highest clean and masked accuracy, while the single tree has the largest masking loss.
- Added `question_3/verify_question_3.py`; independent artifact verification passes for resamples, protocol/split digests, checkpoints, prediction-vector normalization, bias/variance calculations, squared-error decomposition, depth results, robustness values, and figures.
- Wrote the measured Q3 narrative in `question_3/REPORT_DRAFT.md` and reviewed all three Q3 figures for readable labels.
- Integrated the verified Q1–Q3 evidence into a seven-page, two-column report source under `report/`; the provisional PDF passed page-count, text-extraction, and visual layout inspection.
- Added exact saved-artifact verification and report-build instructions to `README.md`.
- Passed the final quality gate: 12/12 unit tests and all three question-specific verifiers succeed from the documented entry points.

## Frozen decisions already recorded

- The saved identity-stratified 6/2/2 split is immutable and will be reused for Q2 and Q3.
- All model selection will use clean validation accuracy only.
- Clean and masked test sets will not influence tuning.
- Bagging and SAMME will be project implementations; only individual decision trees may use scikit-learn.
- Smaller/simpler configurations will break exact validation-accuracy ties as specified in `RULES.md`.
- Results, timings, selected examples, explanations, and figures must come from saved experiment evidence; none may be fabricated.

## Remaining submission metadata

- Student name recorded for report page 1: `Dishant Kaushik`.
- Repository URL verified and recorded: `https://github.com/dishant1901-prog/machine-learning-assignment-2`.
- Final commit hash can only be added after the repository is finalized.

## Next checkpoint

**Submission handoff**

1. Commit and push the frozen, verified repository contents.
2. Render `report/final_report.pdf` with that repository commit hash on page 1.
3. Confirm the remote commit and the final PDF's page count and metadata before portal submission.

## Checkpoint history

| Checkpoint | Date | State | Evidence |
|---|---|---|---|
| CP0: instruction review and rules | 2026-10-19 | Complete | `RULES.md`, directory structure, supplied PDF |
| CP1: common setup and saved split | 2026-10-19 | Complete | `shared/split_indices.npz`, digest recorded above, 5 passing tests |
| CP2: Q1(a) baseline selected/frozen | 2026-10-19 | Complete | `question_1/outputs/frozen_baseline_config.json`, baseline table/figure |
| CP3: Q1(b-c) bagging evaluated/selected | 2026-10-19 | Complete | `question_1/outputs/frozen_bagging_config.json`, bagging table/figures |
| CP4: Q1(d) robustness and cases | 2026-10-19 | Complete | clean/masked table, case CSV/figure, `REPORT_DRAFT.md`, passing artifact verification |
| CP5: Q2(a-b) SAMME validated/selected | 2026-10-19 | Complete | toy CSV/JSON, validation grid, frozen boosting config, diagnostics table/figure |
| CP6: Q2(c-d) comparisons/robustness | 2026-10-19 | Complete | comparison/robustness tables, hard-face and mask-response evidence, `REPORT_DRAFT.md`, passing verifier |
| CP7: Q3(a-b) refit measurements | 2026-10-19 | Complete | 30 checkpoints, prediction vectors, Q3(a-b) tables/figures, verified digests |
| CP8: Q3(c-d) synthesis | 2026-10-19 | Complete | Q3(c-d) tables/figures, `REPORT_DRAFT.md`, passing artifact verification |
| CP9: report and end-to-end reproduction | 2026-10-19 | Complete | Seven-page report source; 12 passing tests; passing Q1, Q2, and Q3 verifiers; PDF layout inspected |
