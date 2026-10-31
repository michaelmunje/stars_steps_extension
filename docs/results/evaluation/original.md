# Policies in the training crowd (stars_reactive), 9 scene kinds

The evaluation the policies were selected by: the config objective and its terms.

| Policy | objective | success | collision | robot_influence | intimate_violation_rate | time_efficiency | normalized_time | path_efficiency | command_smoothness | loops | min_ped_clearance | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_seeking | -2.074 | 0.385 | 0.615 | 0.243 | 0.116 | 0.343 | 0.743 | 0.381 | 0.292 | 0.000 | 0.225 | 0.995 |
| random_weights | -0.723 | 0.607 | 0.115 | 0.280 | 0.155 | 0.420 | 0.671 | 0.581 | 0.413 | 0.020 | 0.666 | 0.901 |
| social_default | 0.084 | 0.900 | 0.037 | 0.248 | 0.187 | 0.776 | 0.410 | 0.872 | 0.541 | 0.006 | 0.310 | 0.750 |
| v4_cv | 0.448 | 0.996 | 0.004 | 0.243 | 0.027 | 0.825 | 0.376 | 0.889 | 0.359 | 0.060 | 0.565 | 0.819 |
| v4 | 0.434 | 0.989 | 0.007 | 0.240 | 0.027 | 0.822 | 0.377 | 0.884 | 0.337 | 0.053 | 0.566 | 0.827 |
| ppo_params | 0.418 | 0.985 | 0.007 | 0.242 | 0.027 | 0.820 | 0.378 | 0.878 | 0.346 | 0.065 | 0.566 | 0.836 |
| subgoal_ppo | 0.362 | 0.970 | 0.019 | 0.239 | 0.028 | 0.802 | 0.390 | 0.861 | 0.349 | 0.074 | 0.568 | 0.845 |
