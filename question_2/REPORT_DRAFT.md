# Question 2 report draft

All results use the split fixed in Question 1 and seed 242003. The tables below are rounded for presentation; full-precision values and complete histories are in `outputs/`.

## (a) One complete SAMME round

I used nine one-dimensional samples, with three consecutive samples from each of classes 0, 1, and 2. A depth-1 tree predicted

`[0, 0, 0, 1, 1, 1, 1, 1, 1]`.

All initial weights were 1/9. The final three samples were wrong, so the weighted error was 3/9 = 1/3, below the three-class random-prediction threshold 2/3. Therefore,

`alpha = log((1 - 1/3)/(1/3)) + log(3 - 1) = log(4) = 1.386294`.

Using `w_i <- w_i exp(alpha I[h(x_i) != y_i])`, each correct sample retained unnormalized weight 1/9 and each error received 4/9. The normalization constant was 2, giving final weights 1/18 for a correct sample and 2/9 for an error.

| Samples | True class | Prediction | Initial weight | Wrong? | Unnormalized weight | Normalized weight |
|---|---:|---:|---:|---:|---:|---:|
| 0, 1, 2 | 0 | 0 | 0.111111 | 0 | 0.111111 | 0.055556 |
| 3, 4, 5 | 1 | 1 | 0.111111 | 0 | 0.111111 | 0.055556 |
| 6, 7, 8 | 2 | 1 | 0.111111 | 1 | 0.444444 | 0.222222 |

The face experiments call the same update function. Before accepting a learner, the implementation checks `error < 1 - 1/K`; a learner at or above this threshold is logged and rejected, and boosting stops without adding it. A zero-error learner uses a documented numerical floor to obtain a finite alpha and then stops because the training data are already separated. Both edge cases are covered by numerical tests. No learner was rejected in the official face run.

## (b) Depth and round selection

The grid below was evaluated using only the fixed clean validation set.

| Weak depth | Rounds | Train acc. (%) | Validation acc. (%) |
|---:|---:|---:|---:|
| 1 | 10 | 5.00 | 5.00 |
| 1 | 25 | 10.00 | 10.00 |
| 1 | 50 | 12.50 | 10.00 |
| 1 | 100 | 15.83 | 11.25 |
| 2 | 10 | 12.08 | 7.50 |
| 2 | 25 | 15.42 | 8.75 |
| 2 | 50 | 27.50 | 16.25 |
| 2 | 100 | 40.42 | 20.00 |
| 3 | 10 | 27.08 | 16.25 |
| 3 | 25 | 52.50 | 25.00 |
| 3 | 50 | 48.75 | 22.50 |
| 3 | 100 | 74.58 | 32.50 |
| 4 | 10 | 37.08 | 17.50 |
| 4 | 25 | 62.50 | 26.25 |
| 4 | 50 | 98.75 | 43.75 |
| 4 | 100 | 100.00 | 51.25 |

Depth 4 with 100 rounds had the highest validation accuracy and was frozen before any Q2 test evaluation. The shallow learners underfit this 40-class problem: even 100 depth-1 trees reached only 15.83% training accuracy. Depth 4 supplies enough class separation for boosting to fit all training faces by round 100, although its validation accuracy remains much lower than its training accuracy.

For the selected model, per-round weighted errors ranged from 0.3714 to 0.8750, always below the 39/40 = 0.975 random threshold. Their corresponding learner weights ranged from 1.7177 to 4.1897. Weight concentration increased markedly: the maximum individual weight reached 0.0910 at round 41, and by round 100 the ten heaviest faces carried 0.4344 of the total weight. These four histories are shown in `q2b_selected_round_diagnostics.png`.

## (c) Bagging versus boosting

Times measure cumulative calls that fit the base learners; bootstrap generation and metric evaluation are excluded for both methods.

| Learners | Model | Train acc. (%) | Validation acc. (%) | Test acc. (%) | Training time (s) |
|---:|---|---:|---:|---:|---:|
| 25 | Bagging | 100.00 | 76.25 | 76.25 | 23.04 |
| 25 | Boosting | 62.50 | 26.25 | 26.25 | 7.12 |
| 50 | Bagging | 100.00 | 82.50 | 81.25 | 45.33 |
| 50 | Boosting | 98.75 | 43.75 | 43.75 | 14.44 |
| 100 | Bagging | 100.00 | 81.25 | 83.75 | 90.93 |
| 100 | Boosting | 100.00 | 51.25 | 52.50 | 28.80 |

Boosting is about three times faster here because its weak learners have depth 4, whereas bagging uses the unrestricted frozen tree. The speed difference does not compensate for the accuracy gap: at 100 learners, bagging is 31.25 percentage points more accurate on the clean test set. Boosting eventually fits the training set, but its focus on repeatedly difficult samples does not generalize as effectively as averaging the much deeper bootstrap trees on this split.

The strongest example of this repeated focus is training identity 36 at original dataset index 362. It has final weight 0.08428, the largest of all 240 training weights, and is misclassified by 87 of the 100 weak learners. Every misclassification increases its weight. For example, consecutive mistakes in rounds 89-92 raise its weight from 0.01170 to 0.06846. A correct prediction at round 93 drops it to 0.00458, but seven further mistakes raise it to 0.08428 by round 100. Its prediction history spans many incorrect identities rather than one persistent alternative, suggesting that successive shallow partitions rarely isolate this face. The complete history and the face are shown in `q2c_heaviest_training_face.png`.

## (d) Clean and masked evaluation

| Model | Clean acc. (%) | Clean confidence | Masked acc. (%) | Masked confidence |
|---|---:|---:|---:|---:|
| Single tree | 57.50 | 0.9813 | 33.75 | 1.0000 |
| Bagging (200 trees) | 87.50 | 0.4079 | 70.00 | 0.2974 |
| Boosting (depth 4, 100 rounds) | 52.50 | 0.1207 | 53.75 | 0.1143 |

Bagging remains the most accurate model on both sets. Boosting gains one correct prediction after masking (a 1.25-point increase on an 80-image test set), but its mean winning weighted vote falls from 0.1207 to 0.1143. I therefore treat the accuracy increase as a split-specific set of label flips, not evidence that removing pixels improves the model. As in Q1, the tree's confidence of 1.0 after masking reflects pure reached leaves rather than reliable uncertainty.

One test face makes the bagging/boosting difference concrete. The face is identity 9 at original dataset index 97, and both ensembles predict 9 when it is clean. Bagging gives identity 9 46/200 votes, followed by identity 39 with 23 and identity 13 with 21. Masking changes 44.5% of its tree predictions; per-face pairwise disagreement rises from 0.9022 to 0.9132, and the result flips to identity 13 with 42 votes versus 26 for the true identity 9. Its winning margin falls from 0.115 to 0.080.

Boosting also becomes less certain, but does not flip. The clean weighted vote for identity 9 is 0.11363. After masking it falls to 0.08207, only just above identity 13 at 0.08152, leaving a margin of 0.00055. Only 15% of the weak learners change prediction, carrying 15.64% of total alpha weight. Unlike the independently bootstrapped deep trees, these learners were trained sequentially after the sample distribution had concentrated on a small set of hard training faces: the final ten heaviest images carry 43.44% of training weight. For this test face, the resulting shallow learners respond more consistently to the mask than the bagged trees, although the nearly tied weighted vote shows that this particular correct result is fragile rather than confident. The clean/masked faces and their leading votes are in `q2d_mask_response_case.png`; the JSON artifact preserves every nonzero vote.
