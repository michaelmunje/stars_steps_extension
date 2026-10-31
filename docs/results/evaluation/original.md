# Policies in the training crowd (stars_reactive), 9 kinds x 30 test scenes

The evaluation the policies were selected by (decision 99): the config objective and its terms.

| Policy | objective | success | collision | robot_influence | intimate_violation_rate | time_efficiency | normalized_time | path_efficiency | command_smoothness | loops | min_ped_clearance | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_seeking | -2.074 | 0.385 | 0.615 | 0.243 | 0.116 | 0.343 | 0.743 | 0.381 | 0.292 | 0.000 | 0.225 | 0.991 |
| social_default | 0.078 | 0.900 | 0.041 | 0.248 | 0.186 | 0.776 | 0.410 | 0.872 | 0.540 | 0.006 | 0.310 | 0.753 |
| v4_cv | 0.448 | 0.996 | 0.004 | 0.243 | 0.027 | 0.825 | 0.376 | 0.889 | 0.359 | 0.060 | 0.565 | 0.814 |
| v4 | 0.421 | 0.985 | 0.011 | 0.240 | 0.027 | 0.820 | 0.378 | 0.882 | 0.336 | 0.051 | 0.566 | 0.834 |
| ppo_params | 0.424 | 0.989 | 0.007 | 0.242 | 0.027 | 0.821 | 0.377 | 0.881 | 0.347 | 0.067 | 0.565 | 0.836 |
| subgoal_ppo | 0.369 | 0.974 | 0.019 | 0.238 | 0.027 | 0.804 | 0.389 | 0.863 | 0.350 | 0.072 | 0.569 | 0.840 |
