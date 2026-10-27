# Policies, 90 test scenes (crossing, group_toward, dense_flow)

| Policy | success | collision | intimate_violation_rate | min_ped_clearance | time_efficiency | path_efficiency | loops | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4_cmaes | 1.000 | 0.000 | 0.028 | 0.552 | 0.849 | 0.897 | 0.065 | 0.445 |
| ppo_params | 1.000 | 0.000 | 0.030 | 0.546 | 0.852 | 0.900 | 0.062 | 0.437 |
| subgoal_ppo | 0.989 | 0.000 | 0.027 | 0.544 | 0.829 | 0.873 | 0.094 | 0.428 |
| arc_ppo | 0.900 | 0.089 | 0.080 | 0.441 | 0.777 | 0.814 | 0.063 | 0.489 |
