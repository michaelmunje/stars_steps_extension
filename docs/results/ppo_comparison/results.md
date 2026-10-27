# Policies, 90 test scenes (crossing, group_toward, dense_flow)

| Policy | success | collision | intimate_violation_rate | min_ped_clearance | time_efficiency | path_efficiency | loops | disruption_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4_cmaes | 1.000 | 0.000 | 0.027 | 0.551 | 0.849 | 0.896 | 0.066 | 0.440 |
| ppo | 1.000 | 0.000 | 0.030 | 0.546 | 0.852 | 0.899 | 0.068 | 0.451 |
| subgoal_ppo | 0.989 | 0.000 | 0.026 | 0.544 | 0.826 | 0.871 | 0.104 | 0.427 |
| arc_ppo | 0.911 | 0.089 | 0.080 | 0.441 | 0.781 | 0.819 | 0.054 | 0.458 |
