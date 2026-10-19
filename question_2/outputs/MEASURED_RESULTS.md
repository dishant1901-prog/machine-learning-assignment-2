# Question 2 measured results

This is a generated evidence summary. Full-precision source tables and figures are stored beside it.

## Q2(a): one complete three-class SAMME round

Weighted error: 0.333333; learner weight: 1.386294; normalization constant: 2.000000.

|   sample |        x |   true_class |   initial_weight |   weak_prediction |   incorrect |   unnormalized_updated_weight |   normalized_updated_weight |
|---------:|---------:|-------------:|-----------------:|------------------:|------------:|------------------------------:|----------------------------:|
| 0.000000 | 0.000000 |     0.000000 |         0.111111 |          0.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 1.000000 | 1.000000 |     0.000000 |         0.111111 |          0.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 2.000000 | 2.000000 |     0.000000 |         0.111111 |          0.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 3.000000 | 3.000000 |     1.000000 |         0.111111 |          1.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 4.000000 | 4.000000 |     1.000000 |         0.111111 |          1.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 5.000000 | 5.000000 |     1.000000 |         0.111111 |          1.000000 |    0.000000 |                      0.111111 |                    0.055556 |
| 6.000000 | 6.000000 |     2.000000 |         0.111111 |          1.000000 |    1.000000 |                      0.444444 |                    0.222222 |
| 7.000000 | 7.000000 |     2.000000 |         0.111111 |          1.000000 |    1.000000 |                      0.444444 |                    0.222222 |
| 8.000000 | 8.000000 |     2.000000 |         0.111111 |          1.000000 |    1.000000 |                      0.444444 |                    0.222222 |

## Q2(b): clean-validation grid

Selected depth `4` and `100` rounds using clean validation accuracy only.

|   weak_learner_depth |   rounds | available   |   achieved_rounds | termination_reason         |   train_accuracy |   validation_accuracy |   cumulative_learner_fit_time_seconds |
|---------------------:|---------:|:------------|------------------:|:---------------------------|-----------------:|----------------------:|--------------------------------------:|
|                    1 |       10 | True        |               100 | requested_rounds_completed |         0.050000 |              0.050000 |                              0.764215 |
|                    1 |       25 | True        |               100 | requested_rounds_completed |         0.100000 |              0.100000 |                              1.912928 |
|                    1 |       50 | True        |               100 | requested_rounds_completed |         0.125000 |              0.100000 |                              3.785971 |
|                    1 |      100 | True        |               100 | requested_rounds_completed |         0.158333 |              0.112500 |                              7.455081 |
|                    2 |       10 | True        |               100 | requested_rounds_completed |         0.120833 |              0.075000 |                              1.439105 |
|                    2 |       25 | True        |               100 | requested_rounds_completed |         0.154167 |              0.087500 |                              3.575879 |
|                    2 |       50 | True        |               100 | requested_rounds_completed |         0.275000 |              0.162500 |                              7.282262 |
|                    2 |      100 | True        |               100 | requested_rounds_completed |         0.404167 |              0.200000 |                             15.243715 |
|                    3 |       10 | True        |               100 | requested_rounds_completed |         0.270833 |              0.162500 |                              2.306610 |
|                    3 |       25 | True        |               100 | requested_rounds_completed |         0.525000 |              0.250000 |                              5.670658 |
|                    3 |       50 | True        |               100 | requested_rounds_completed |         0.487500 |              0.225000 |                             11.076343 |
|                    3 |      100 | True        |               100 | requested_rounds_completed |         0.745833 |              0.325000 |                             21.828220 |
|                    4 |       10 | True        |               100 | requested_rounds_completed |         0.370833 |              0.175000 |                              2.868042 |
|                    4 |       25 | True        |               100 | requested_rounds_completed |         0.625000 |              0.262500 |                              7.123007 |
|                    4 |       50 | True        |               100 | requested_rounds_completed |         0.987500 |              0.437500 |                             14.443565 |
|                    4 |      100 | True        |               100 | requested_rounds_completed |         1.000000 |              0.512500 |                             28.804961 |

## Q2(c): bagging versus boosting

| model    |   learners |   train_accuracy |   validation_accuracy |   test_accuracy |   training_time_seconds | timing_scope                      |
|:---------|-----------:|-----------------:|----------------------:|----------------:|------------------------:|:----------------------------------|
| bagging  |         25 |         1.000000 |              0.762500 |        0.762500 |               23.036255 | cumulative base-learner fit calls |
| boosting |         25 |         0.625000 |              0.262500 |        0.262500 |                7.123007 | cumulative base-learner fit calls |
| bagging  |         50 |         1.000000 |              0.825000 |        0.812500 |               45.331081 | cumulative base-learner fit calls |
| boosting |         50 |         0.987500 |              0.437500 |        0.437500 |               14.443565 | cumulative base-learner fit calls |
| bagging  |        100 |         1.000000 |              0.812500 |        0.837500 |               90.926833 | cumulative base-learner fit calls |
| boosting |        100 |         1.000000 |              0.512500 |        0.525000 |               28.804961 | cumulative base-learner fit calls |

Heaviest selected-model training image: position 221, original dataset index 362, identity 36, final weight 0.08428330, misclassified in 87 of 100 rounds.

## Q2(d): clean and masked comparison

| test_set   | model       |   accuracy |   mean_winning_confidence | confidence_definition                     |
|:-----------|:------------|-----------:|--------------------------:|:------------------------------------------|
| clean      | single_tree |   0.575000 |                  0.981250 | maximum leaf class probability            |
| clean      | bagging     |   0.875000 |                  0.407937 | winning hard-vote fraction                |
| clean      | boosting    |   0.525000 |                  0.120703 | winning normalized weighted-vote fraction |
| masked     | single_tree |   0.337500 |                  1.000000 | maximum leaf class probability            |
| masked     | bagging     |   0.700000 |                  0.297375 | winning hard-vote fraction                |
| masked     | boosting    |   0.537500 |                  0.114318 | winning normalized weighted-vote fraction |

Mask-response case: dataset index 97, true identity 9. See `q2d_mask_response_case.json` for complete votes and diagnostics.
