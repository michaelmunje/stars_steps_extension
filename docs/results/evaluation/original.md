# Policies in the training crowd (stars_reactive), 9 scene kinds

The evaluation the policies were selected by: the config objective and its terms.

| Policy | objective | success | collision | robot_influence | intimate_violation_rate | time_efficiency | normalized_time | path_efficiency | command_smoothness | loops | min_ped_clearance | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_seeking | -2.046 | 0.393 | 0.607 | 0.244 | 0.116 | 0.350 | 0.738 | 0.388 | 0.293 | 0.000 | 0.226 | 0.973 |
| random_weights | -0.718 | 0.607 | 0.111 | 0.283 | 0.155 | 0.422 | 0.669 | 0.581 | 0.414 | 0.020 | 0.654 | 0.909 |
| social_default | 0.084 | 0.900 | 0.041 | 0.245 | 0.184 | 0.778 | 0.409 | 0.871 | 0.529 | 0.001 | 0.309 | 0.752 |
| v4_cv | 0.433 | 0.993 | 0.007 | 0.244 | 0.028 | 0.828 | 0.373 | 0.884 | 0.354 | 0.047 | 0.559 | 0.790 |
| v4 | 0.420 | 0.985 | 0.011 | 0.240 | 0.027 | 0.825 | 0.375 | 0.880 | 0.337 | 0.046 | 0.561 | 0.784 |
| ppo_params | 0.431 | 0.989 | 0.007 | 0.242 | 0.029 | 0.829 | 0.372 | 0.882 | 0.339 | 0.047 | 0.560 | 0.796 |
| subgoal_ppo | 0.380 | 0.974 | 0.015 | 0.239 | 0.028 | 0.811 | 0.383 | 0.864 | 0.337 | 0.067 | 0.573 | 0.809 |
