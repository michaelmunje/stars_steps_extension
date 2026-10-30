# Policies in the training crowd (stars_reactive), 9 kinds x 30 test scenes

The evaluation the policies were selected by (decision 99): the config objective and its terms.

| Policy | objective | success | collision | robot_influence | intimate_violation_rate | time_efficiency | normalized_time | path_efficiency | command_smoothness | loops | min_ped_clearance | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_seeking | -2.074 | 0.385 | 0.615 | 0.243 | 0.116 | 0.343 | 0.743 | 0.381 | 0.292 | 0.000 | 0.225 | 0.992 |
| social_default | 0.094 | 0.904 | 0.037 | 0.248 | 0.186 | 0.779 | 0.408 | 0.875 | 0.540 | 0.003 | 0.310 | 0.762 |
| v4_cv | 0.421 | 0.989 | 0.011 | 0.243 | 0.027 | 0.822 | 0.378 | 0.885 | 0.360 | 0.055 | 0.565 | 0.823 |
| v4 | 0.434 | 0.989 | 0.007 | 0.240 | 0.027 | 0.822 | 0.377 | 0.884 | 0.337 | 0.053 | 0.566 | 0.837 |
| ppo_params | 0.419 | 0.985 | 0.007 | 0.242 | 0.027 | 0.820 | 0.378 | 0.878 | 0.346 | 0.060 | 0.566 | 0.847 |
| subgoal_ppo | 0.369 | 0.974 | 0.019 | 0.238 | 0.027 | 0.804 | 0.389 | 0.864 | 0.349 | 0.073 | 0.569 | 0.847 |
