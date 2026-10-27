# Pedestrian conditions × policies

Scenes: crossing, group_toward, dense_flow (test split). Means over all scenes; the fixed-scene outcomes are those in the GIF grids.

| Condition | Policy | success | collision | intimate_violation_rate | min_ped_clearance | time_efficiency | path_efficiency | loops | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay | goal_seeking | 0.356 | 0.644 | 0.066 | 0.167 | 0.318 | 0.351 | 0.000 | 0.000 |
| replay | social_default | 0.911 | 0.089 | 0.084 | 0.248 | 0.799 | 0.884 | 0.000 | 0.000 |
| replay | v4_cv | 0.989 | 0.011 | 0.019 | 0.531 | 0.854 | 0.899 | 0.037 | 0.000 |
| replay | v4_stars | 1.000 | 0.000 | 0.021 | 0.529 | 0.857 | 0.905 | 0.045 | 0.000 |
| social_force | goal_seeking | 0.689 | 0.311 | 0.078 | 0.367 | 0.617 | 0.682 | 0.000 | 0.518 |
| social_force | social_default | 1.000 | 0.000 | 0.064 | 0.473 | 0.881 | 0.978 | 0.000 | 0.309 |
| social_force | v4_cv | 0.989 | 0.000 | 0.007 | 0.702 | 0.829 | 0.880 | 0.068 | 0.261 |
| social_force | v4_stars | 0.989 | 0.000 | 0.011 | 0.702 | 0.827 | 0.873 | 0.102 | 0.251 |
| reactive_replay | goal_seeking | 0.589 | 0.411 | 0.073 | 0.267 | 0.527 | 0.583 | 0.000 | 0.377 |
| reactive_replay | social_default | 0.989 | 0.011 | 0.069 | 0.404 | 0.839 | 0.951 | 0.046 | 0.197 |
| reactive_replay | v4_cv | 1.000 | 0.000 | 0.009 | 0.633 | 0.837 | 0.884 | 0.117 | 0.125 |
| reactive_replay | v4_stars | 1.000 | 0.000 | 0.009 | 0.652 | 0.846 | 0.894 | 0.071 | 0.119 |
| stars_blind | goal_seeking | 0.389 | 0.611 | 0.068 | 0.181 | 0.347 | 0.384 | 0.000 | 0.495 |
| stars_blind | social_default | 0.967 | 0.033 | 0.084 | 0.259 | 0.843 | 0.934 | 0.000 | 0.032 |
| stars_blind | v4_cv | 1.000 | 0.000 | 0.019 | 0.540 | 0.861 | 0.907 | 0.062 | 0.018 |
| stars_blind | v4_stars | 1.000 | 0.000 | 0.022 | 0.540 | 0.858 | 0.905 | 0.055 | 0.024 |
| stars_reactive | goal_seeking | 0.389 | 0.611 | 0.070 | 0.183 | 0.348 | 0.385 | 0.000 | 1.050 |
| stars_reactive | social_default | 0.933 | 0.067 | 0.093 | 0.260 | 0.814 | 0.903 | 0.001 | 0.552 |
| stars_reactive | v4_cv | 1.000 | 0.000 | 0.026 | 0.538 | 0.854 | 0.901 | 0.059 | 0.429 |
| stars_reactive | v4_stars | 1.000 | 0.000 | 0.028 | 0.552 | 0.849 | 0.897 | 0.065 | 0.441 |
| stars_social | goal_seeking | 0.522 | 0.478 | 0.070 | 0.273 | 0.467 | 0.517 | 0.000 | 1.074 |
| stars_social | social_default | 0.989 | 0.011 | 0.071 | 0.379 | 0.859 | 0.962 | 0.008 | 0.655 |
| stars_social | v4_cv | 1.000 | 0.000 | 0.011 | 0.630 | 0.840 | 0.888 | 0.065 | 0.463 |
| stars_social | v4_stars | 1.000 | 0.000 | 0.011 | 0.638 | 0.830 | 0.876 | 0.100 | 0.455 |

## Fixed scenes (v4 + STARS predictor)

| Condition | crossing-2000000 | group_toward-2000000 | dense_flow-2000000 |
| --- | --- | --- | --- |
| replay | reached 10.2 s, gap 0.52 m | reached 10.7 s, gap 0.41 m | reached 8.1 s, gap 0.37 m |
| social_force | reached 10.0 s, gap 0.71 m | reached 12.3 s, gap 0.67 m | reached 8.3 s, gap 0.33 m |
| reactive_replay | reached 10.2 s, gap 0.72 m | reached 11.2 s, gap 0.46 m | reached 8.1 s, gap 0.39 m |
| stars_blind | reached 10.2 s, gap 0.51 m | reached 10.8 s, gap 0.45 m | reached 8.1 s, gap 0.37 m |
| stars_reactive | reached 10.2 s, gap 0.13 m | reached 11.0 s, gap 0.44 m | reached 8.1 s, gap 0.37 m |
| stars_social | reached 10.2 s, gap 0.77 m | reached 11.9 s, gap 0.30 m | reached 8.2 s, gap 0.39 m |

## Crowd realism (recorded STEPS test scenes, robot absent)

| Condition | position_error | error_at_5s | error_at_10s | speed_sim | speed_real | overlap_rate |
| --- | --- | --- | --- | --- | --- | --- |
| replay | 0.000 | 0.000 | 0.000 | 0.158 | 0.158 | 0.011 |
| social_force | 0.429 | 0.656 | 0.657 | 0.177 | 0.160 | 0.001 |
| reactive_replay | 0.320 | 0.368 | 0.429 | 0.177 | 0.158 | 0.001 |
| stars_blind | 0.364 | 0.189 | 0.548 | 0.084 | 0.158 | 0.010 |
| stars_reactive | 0.363 | 0.188 | 0.545 | 0.084 | 0.158 | 0.010 |
| stars_social | 0.848 | 0.578 | 1.042 | 0.102 | 0.158 | 0.000 |
