# Question 1 report draft

The numbers below come from the official split generated with seed 242003. This draft is deliberately compact so it can be adapted to the final eight-page CVPR report. Percentages are rounded to two decimal places; full-precision values are preserved in `outputs/tables/`.

## (a) Single-tree baseline

The validation grid selected an unrestricted tree with `min_samples_leaf=2`. The unrestricted settings with leaf sizes 1 and 2 both obtained 56.25% validation accuracy, so the predeclared tie-break selected leaf size 2 because it avoids singleton leaves. Test accuracy was evaluated only after this choice was frozen.

| Model | Train acc. (%) | Val. acc. (%) | Test acc. (%) | Leaves | Mean samples/leaf | Singleton leaves (%) |
|---|---:|---:|---:|---:|---:|---:|
| Selected tree (`depth=None`, `min_leaf=2`) | 97.92 | 56.25 | 57.50 | 45 | 5.33 | 0.00 |

The depth sweep shows why the unrestricted tree was selected, but also shows the cost of depth. For `min_samples_leaf=2`, increasing the depth limit from 2 to 4, 8, 16, and unrestricted increased the number of leaves from 3 to 5, 9, 17, and 45. The mean leaf occupancy fell from 80.00 to 48.00, 26.67, 14.12, and 5.33 samples. Training accuracy rose from 7.50% to 97.92%, while validation accuracy rose from 7.50% to 56.25%. The widening train-validation gap indicates that the later splits increasingly specialize to the 240 training faces. With `min_samples_leaf=1`, the unrestricted tree reached 100% training accuracy and 11.11% of its leaves contained a single training face, but validation accuracy remained 56.25%. Thus, isolating the last few training examples did not improve validation performance; the selected leaf size of 2 removed these singleton leaves without losing validation accuracy. Figure `q1a_depth_diagnostics.png` shows the complete grid.

## (b) Bagging from scratch

Each tree was trained on 240 draws with replacement and used the frozen tree settings. Smaller ensembles are prefixes of the same 200-tree run, so changes with B are not confounded by separate bootstrap sequences.

| B | Train acc. (%) | Val. acc. (%) | Test acc. (%) | Mean distinct examples |
|---:|---:|---:|---:|---:|
| 1 | 72.08 | 32.50 | 35.00 | 151.00 |
| 5 | 95.42 | 50.00 | 62.50 | 153.40 |
| 10 | 99.17 | 70.00 | 68.75 | 151.50 |
| 25 | 100.00 | 76.25 | 76.25 | 151.16 |
| 50 | 100.00 | 82.50 | 81.25 | 150.76 |
| 100 | 100.00 | 81.25 | 83.75 | 151.05 |
| 200 | 100.00 | 86.25 | 87.50 | 151.85 |

For a bootstrap of size n, the expected number of represented training examples is

`n[1 - (1 - 1/n)^n]`.

At n=240 this is 151.893 examples, or 63.289% of the training set, close to the limiting fraction `1-exp(-1)`. The measured average at B=200 was 151.85 (63.271%), only 0.043 examples below the exact expectation. The greater variation at small B is expected because it averages only a few bootstrap samples. Validation accuracy was not perfectly monotone (82.50% at B=50 and 81.25% at B=100), but B=200 had the highest validation accuracy, 86.25%, and was frozen as the final ensemble before examining its test result.

## (c) OOB accuracy, disagreement, and vote margin

| B | OOB acc. (%) | OOB coverage (%) | Pairwise disagreement | Mean vote margin |
|---:|---:|---:|---:|---:|
| 1 | 34.83 | 37.08 | undefined | 1.0000 |
| 5 | 30.95 | 87.50 | 0.8163 | 0.2275 |
| 10 | 35.02 | 98.75 | 0.8064 | 0.2313 |
| 25 | 59.58 | 100.00 | 0.8014 | 0.2535 |
| 50 | 70.00 | 100.00 | 0.7944 | 0.2670 |
| 100 | 76.67 | 100.00 | 0.7919 | 0.2641 |
| 200 | 79.17 | 100.00 | 0.7897 | 0.2718 |

The three diagnostics are related but do not tell exactly the same story. OOB coverage is only 37.08% for B=1, making its OOB accuracy unstable; coverage reaches 100% by B=25, after which OOB accuracy climbs from 59.58% to 79.17%. Validation vote margin generally increases after B=5, from 0.2275 to 0.2718, although it dips slightly at B=100, like validation accuracy. Pairwise disagreement instead decreases slightly from 0.8163 at B=5 to 0.7897 at B=200. The ensemble therefore does not improve because its trees become more diverse as B grows. It improves because many already diverse, noisy decisions are averaged, giving a more stable winning class and a somewhat larger margin. The B=1 margin of 1 is a hard-voting artifact: a single vote always has a winning fraction of one and says nothing about consensus. These qualifications are visible in `q1c_oob_disagreement_margin.png`.

## (d) Clean and masked test sets

| Model | Test set | Accuracy (%) | Mean winning confidence |
|---|---|---:|---:|
| Single tree | Clean | 57.50 | 0.9813 |
| Single tree | Masked | 33.75 | 1.0000 |
| Bagging (B=200) | Clean | 87.50 | 0.4079 |
| Bagging (B=200) | Masked | 70.00 | 0.2974 |

Masking the central 16 x 16 region reduced single-tree accuracy by 23.75 percentage points and bagging accuracy by 17.50 points. Bagging therefore retained more accuracy both absolutely and relative to its clean result. The confidence values must be interpreted according to their definitions. The tree uses the empirical class proportion in its reached leaf; masking can route an unfamiliar image into a pure leaf, so its mean confidence rose to 1.0 even while accuracy collapsed. This is overconfidence rather than robustness. Bagging uses the winning vote fraction, which fell from 0.4079 to 0.2974 under masking and therefore reflected the increased disagreement caused by the missing pixels.

Two clean-test cases illustrate errors that aggregation can remove. For identity 0 (dataset index 6), the single tree predicted identity 15 with confidence 1.0, but the ensemble gave 65 votes to the correct identity 0 and 45 to identity 15. For identity 1 (index 10), the tree predicted 16, whereas 74 trees voted for the correct identity 1; the next largest vote count was only 10. These are unstable partition errors: alternative bootstrap trees create different boundaries, and their plurality recovers the correct label even though no class receives an absolute majority.

The two shared errors have a different structure. For identity 4 (index 41), both models predicted 37; bagging split almost evenly between wrong identity 37 (58 votes) and the true identity 4 (53 votes). Here aggregation exposes strong ambiguity but cannot resolve a correlated error. For identity 12 (index 127), votes were diffuse: identity 20 received 20, identity 0 received 19, identity 39 received 17, identity 4 received 16, and the true identity received only 8. When few base trees support the correct class, averaging cannot reconstruct the missing evidence. The faces and all nonzero vote counts are recorded in `q1d_case_evidence.png` and `outputs/tables/q1d_case_evidence.csv`.
