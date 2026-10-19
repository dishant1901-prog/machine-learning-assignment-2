# Question 3 report draft

I kept the clean test set fixed and generated 30 new training sets from the original training split. For every identity, six faces were drawn with replacement from that identity's six original training faces. Thus each refit still used 240 observations and all 40 identities, while duplicated and omitted faces changed between repetitions. No hyperparameter was selected again in this question.

For each fixed test face I saved a 40-dimensional prediction vector: leaf probabilities for the single tree, hard-vote fractions for bagging, and normalized alpha-weighted votes for boosting. Bias squared was computed between the mean of the 30 prediction vectors and the one-hot target. Variance was the mean squared deviation of each prediction vector from that mean. Accuracy standard deviations below are sample standard deviations (`ddof=1`); the bias and variance calculations use the population denominators specified in the question.

## (a) Selected models across 30 refits

| Model | Bias squared | Variance | Mean test acc. (%) | Acc. SD (pp) |
|---|---:|---:|---:|---:|
| Single tree | 0.40394 | 0.64272 | 47.67 | 4.59 |
| Bagging (200 trees) | 0.49887 | 0.02062 | 78.29 | 3.26 |
| Boosting (depth 4, 100 rounds) | 0.88015 | 0.01587 | 33.63 | 7.47 |

The unrestricted tree has the smallest squared bias, but it also has by far the largest variance. Its fitted class vector changes substantially when duplicate faces in the training set change. Averaging 200 bootstrap trees reduces variance from 0.64272 to 0.02062. Its squared bias is somewhat higher than that of a single tree, yet it has the best mean accuracy and the smallest accuracy spread of the three models.

Boosting has the lowest measured variance, 0.01587, but this does not make it the best classifier. Its squared bias is 0.88015 and its mean accuracy is only 33.63%. The depth-4 weak learners give repeatable weighted-vote vectors across the resampled training sets, but those vectors remain systematically far from the one-hot targets. This distinction is important because the reported bias is based on the complete probability or vote vector rather than only the winning class.

## (b) Effect of tree depth and bagging

Before running the refits, I fixed depth 2 as the shallow setting and depth 16 as a much deeper capped setting. The selected baseline has no maximum-depth limit, so it is also the deepest setting in this experiment. All trees use the already selected `min_samples_leaf=2`. Each bagged result in this depth comparison uses 100 trees and the same 30 training resamples as its corresponding single-tree result.

| Maximum depth | Model | Bias squared | Variance | Mean acc. (%) | Variance removed by bagging |
|---|---|---:|---:|---:|---:|
| 2 | Single tree | 0.90476 | 0.04633 | 6.71 | - |
| 2 | Bagging (100) | 0.92765 | 0.01063 | 14.21 | 77.05% |
| 16 | Single tree | 0.63420 | 0.24113 | 29.17 | - |
| 16 | Bagging (100) | 0.65462 | 0.01987 | 67.12 | 91.76% |
| Unrestricted | Single tree | 0.40394 | 0.64272 | 47.67 | - |
| Unrestricted | Bagging (100) | 0.49893 | 0.02450 | 77.12 | 96.19% |

Increasing depth lowers single-tree bias but raises its variance sharply. From depth 2 to the unrestricted setting, squared bias falls from 0.90476 to 0.40394 while variance rises from 0.04633 to 0.64272. Bagging suppresses most of this variability at every depth. Its absolute variance reductions are 0.03570, 0.22126, and 0.61822 for depths 2, 16, and unrestricted respectively. There is a modest increase in bagged bias at every setting, but the reduction in variance is much larger for the deeper trees.

## (c) Where bagging improves accuracy most

The deepest tree is not the point at which bagging produces the largest accuracy increase. Bagging improves mean accuracy by 7.50 percentage points at depth 2, 37.96 points at depth 16, and 29.46 points for the unrestricted trees.

The leaf measurements explain why variance reduction alone does not determine the accuracy gain. A depth-2 single tree has only 3.03 leaves on average, with about 79.33 training observations per leaf. Its high bias of 0.90476 shows severe underfitting. Although the corresponding ensembles have very high tree disagreement (0.9604), averaging many crude partitions cannot recover enough identity detail, so accuracy remains 14.21%.

At depth 16, a single tree has 18.27 leaves and 13.23 observations per leaf on average. Its bias is lower than at depth 2, while its variance of 0.24113 leaves substantial instability for bagging to remove. The ensemble retains high disagreement (0.8844) and reaches 67.12% accuracy. This is the largest gain because these trees are expressive enough to separate many identities but still diverse enough for averaging to correct many individual errors.

The unrestricted single trees have 40 leaves, six observations per leaf, and an average fitted depth of 34.9. They have the lowest single-tree bias but the highest variance. Bagging removes 96.19% of that variance, the largest proportional and absolute reduction, but the accuracy gain is smaller than at depth 16 because the starting single-tree accuracy is already 47.67%, leaving less room for improvement. The bagged bias also rises from 0.40394 to 0.49893, and mean disagreement is lower at 0.8094. Therefore the setting with the greatest variance reduction need not be the setting with the greatest increase in classification accuracy.

## (d) Accuracy, robustness, bias, and variance

The accuracy values below are from the single frozen models fitted to the original training set in Questions 1 and 2. Bias and variance are measured across the 30 Question 3 refits.

| Model | Clean acc. (%) | Masked acc. (%) | Change (pp) | Bias squared | Variance |
|---|---:|---:|---:|---:|---:|
| Single tree | 57.50 | 33.75 | -23.75 | 0.40394 | 0.64272 |
| Bagging (200 trees) | 87.50 | 70.00 | -17.50 | 0.49887 | 0.02062 |
| Boosting (depth 4, 100 rounds) | 52.50 | 53.75 | +1.25 | 0.88015 | 0.01587 |

Bagging gives the strongest overall result: it is the most accurate model on both clean and masked faces and is much more stable across training resamples than the single tree. The single tree is the most sensitive to the missing central region, losing 23.75 percentage points, compared with 17.50 points for bagging.

Boosting has the smallest mask accuracy change and the lowest variance, but neither observation means that it is the best model. Its clean and masked accuracies are both well below bagging, and its bias is the largest. The 1.25-point masked increase is only one additional correct prediction among 80 test faces; Question 2 also showed that its mean winning weighted-vote fraction decreased after masking. I therefore interpret the increase as a split-specific set of prediction flips, not a benefit from removing pixels. In this experiment, the lowest variance belongs to boosting, whereas the best clean-test accuracy belongs to bagging.
