# Multi-agent STARS: steps test split

12500 windows (all); CVAE rows are best-of-K over K = 20 samples, deterministic rows K = 1. Distances in metres. Moving agents: ground-truth future mean speed > 0.2 m/s.

| run | agent ADE | agent FDE | moving ADE | moving FDE | stationary ADE | target ADE | target FDE | joint ADE | pred. collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant velocity | 0.276 | 0.518 | 0.894 | 1.771 | 0.178 | 0.279 | 0.527 | 0.279 | 0.003 |
| cvae_finetune | 0.110 | 0.195 | 0.532 | 1.049 | 0.043 | 0.115 | 0.209 | 0.116 | 0.001 |
| cvae_frozen | 0.113 | 0.197 | 0.545 | 1.062 | 0.044 | 0.118 | 0.210 | 0.120 | 0.001 |
| cvae_scratch | 0.126 | 0.223 | 0.625 | 1.228 | 0.046 | 0.133 | 0.240 | 0.132 | 0.001 |
| det_finetune | 0.115 | 0.205 | 0.559 | 1.107 | 0.045 | 0.121 | 0.221 | 0.119 | 0.001 |
| det_frozen | 0.123 | 0.219 | 0.604 | 1.192 | 0.046 | 0.128 | 0.234 | 0.127 | 0.001 |

## Closed loop (re-encoding the scene from the model's own predictions)

| run | agent ADE, exec 1 | agent ADE, exec 4 | agent ADE, exec 12 |
| --- | ---: | ---: | ---: |
| cvae_finetune | 0.109 | 0.111 | 0.110 |
| cvae_frozen | 0.113 | 0.112 | 0.113 |
| cvae_scratch | 0.132 | 0.123 | 0.126 |
| det_finetune | 0.126 | 0.117 | 0.115 |
| det_frozen | 0.127 | 0.123 | 0.123 |

## 21 s simulation realism

| run | sim_implausible_agent_rate | sim_collision_rate | sim_speed_wasserstein | sim_acceleration_wasserstein | sim_speed_mean | real_speed_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cvae_finetune | 0.000 | 0.001 | 0.070 | 0.300 | 0.082 | 0.141 |
| cvae_frozen | 0.000 | 0.001 | 0.084 | 0.294 | 0.061 | 0.141 |
| cvae_scratch | 0.000 | 0.001 | 0.087 | 0.301 | 0.058 | 0.141 |
| det_finetune | 0.000 | 0.001 | 0.068 | 0.302 | 0.077 | 0.141 |
| det_frozen | 0.000 | 0.001 | 0.089 | 0.301 | 0.057 | 0.141 |
