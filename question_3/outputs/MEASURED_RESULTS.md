# Question 3 measured results

All quantities use the fixed clean test set and the same 30 saved within-identity training resamples. Accuracy standard deviations use `ddof=1`; the assignment's bias/variance formulas use their stated population denominators.

## Q3(a): selected models

| model        |   bias_squared |   variance |   mean_test_accuracy |   test_accuracy_std_ddof1 |
|:-------------|---------------:|-----------:|---------------------:|--------------------------:|
| single_tree  |       0.403944 |   0.642722 |             0.476667 |                  0.045926 |
| bagging_200  |       0.498874 |   0.020616 |             0.782917 |                  0.032577 |
| boosting_100 |       0.880149 |   0.015870 |             0.336250 |                  0.074665 |

## Q3(b-c): depth study

| depth_label      |   tree_bias_squared |   tree_variance |   tree_mean_accuracy |   tree_accuracy_std_ddof1 |   tree_mean_actual_depth |   tree_mean_number_of_leaves |   tree_mean_samples_per_leaf |   tree_mean_singleton_leaf_fraction |   bagging_bias_squared |   bagging_variance |   bagging_mean_accuracy |   bagging_accuracy_std_ddof1 |   bagging_mean_tree_leaves |   bagging_mean_samples_per_leaf |   bagging_mean_singleton_leaf_fraction |   bagging_mean_disagreement |   bagging_disagreement_std_ddof1 |   variance_removed_absolute |   variance_removed_percent |   accuracy_improvement |
|:-----------------|--------------------:|----------------:|---------------------:|--------------------------:|-------------------------:|-----------------------------:|-----------------------------:|------------------------------------:|-----------------------:|-------------------:|------------------------:|-----------------------------:|---------------------------:|--------------------------------:|---------------------------------------:|----------------------------:|---------------------------------:|----------------------------:|---------------------------:|-----------------------:|
| 2 (shallow)      |            0.904759 |        0.046329 |             0.067083 |                  0.008980 |                 2.000000 |                     3.033333 |                    79.333333 |                            0.000000 |               0.927654 |           0.010631 |                0.142083 |                     0.054088 |                   3.155333 |                       76.893333 |                               0.000000 |                    0.960368 |                         0.004102 |                    0.035697 |                  77.051988 |               0.075000 |
| 16 (deep capped) |            0.634196 |        0.241134 |             0.291667 |                  0.034947 |                16.000000 |                    18.266667 |                    13.225276 |                            0.000000 |               0.654618 |           0.019874 |                0.671250 |                     0.055140 |                  21.769000 |                       11.146221 |                               0.000000 |                    0.884445 |                         0.005674 |                    0.221260 |                  91.758216 |               0.379583 |
| None (selected)  |            0.403944 |        0.642722 |             0.476667 |                  0.045926 |                34.900000 |                    40.000000 |                     6.000000 |                            0.000000 |               0.498929 |           0.024499 |                0.771250 |                     0.037016 |                  44.384333 |                        5.416555 |                               0.000000 |                    0.809449 |                         0.009834 |                    0.618223 |                  96.188182 |               0.294583 |

## Q3(d): final summary

| model       |   clean_test_accuracy |   masked_test_accuracy |   masked_minus_clean_accuracy |   bias_squared |   variance |
|:------------|----------------------:|-----------------------:|------------------------------:|---------------:|-----------:|
| single_tree |              0.575000 |               0.337500 |                     -0.237500 |       0.403944 |   0.642722 |
| bagging     |              0.875000 |               0.700000 |                     -0.175000 |       0.498874 |   0.020616 |
| boosting    |              0.525000 |               0.537500 |                      0.012500 |       0.880149 |   0.015870 |
