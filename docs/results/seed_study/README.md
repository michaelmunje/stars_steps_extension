# Random vs CMA-ES-tuned controller weights (STARS crowd, test split)

10 random and 0 CMA-ES seeds; each seed is evaluated on the same 270 test scenes (9 kinds x 30). Cells: mean over seeds ± standard error. v4 is one policy (no error). Metrics from Table I of Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation Algorithms* (2023); distances are measured edge to edge; T averages successful episodes only; CD averages scenes with walls only. Aggregated time (AT) is omitted: our pedestrians have no cooperative goals.

| Metric | Unit | Better | Random weights | CMA-ES | v4 |
| --- | --- | --- | --- | --- | --- |
| Success rate (`S`) |  | high | 0.583 ± 0.089 | — | 0.978 |
| Collision rate (`C`) |  | low | 0.088 ± 0.014 | — | 0.015 |
| Human collisions (`HC`) |  | low | 0.088 ± 0.014 | — | 0.015 |
| Wall collisions (`WC`) |  | low | 0.000 ± 0.000 | — | 0.000 |
| Timeouts (`TO`) |  | low | 0.330 ± 0.095 | — | 0.007 |
| Failures to progress (5 s, 0.1 m) (`FP`) | per episode | low | 1.651 ± 0.598 | — | 0.063 |
| Stalled time (< 0.05 m/s) (`ST`) | s | low | 8.503 ± 2.839 | — | 0.259 |
| Time to reach goal (successes) (`T`) | s | low | 15.822 ± 1.455 | — | 10.863 |
| Path length (`PL`) | m |  | 6.986 ± 0.892 | — | 10.095 |
| Success weighted by path length (`SPL`) |  | high | 0.548 ± 0.086 | — | 0.875 |
| Average speed (`Vavg`) | m/s |  | 0.499 ± 0.084 | — | 0.922 |
| Maximum speed (`Vmax`) | m/s |  | 0.763 ± 0.113 | — | 0.999 |
| Average acceleration magnitude (`Aavg`) | m/s^2 | low | 0.147 ± 0.025 | — | 0.128 |
| Minimum acceleration (`Amin`) | m/s^2 |  | -0.792 ± 0.108 | — | -0.830 |
| Maximum acceleration (`Amax`) | m/s^2 |  | 0.844 ± 0.103 | — | 1.000 |
| Average jerk magnitude (`Javg`) | m/s^3 | low | 0.790 ± 0.142 | — | 0.387 |
| Minimum jerk (`Jmin`) | m/s^3 |  | -11.256 ± 1.536 | — | -10.333 |
| Maximum jerk (`Jmax`) | m/s^3 |  | 10.648 ± 1.545 | — | 5.000 |
| Minimum clearance to walls (scenes with walls) (`CDmin`) | m | high | 1.506 ± 0.115 | — | 1.129 |
| Average clearance to walls (scenes with walls) (`CDavg`) | m |  | 2.403 ± 0.049 | — | 2.291 |
| Minimum distance to a human (`DHmin`) | m | high | 0.928 ± 0.292 | — | 0.561 |
| Personal space compliance (>= 0.5 m) (`SC`) |  | high | 0.875 ± 0.027 | — | 0.943 |
| Minimum time to collision (constant velocity, capped at 10 s) (`TTC`) | s | high | 4.529 ± 0.490 | — | 4.181 |
| Training objective J (not Table I) (`objective`) |  | high | -0.685 ± 0.155 | — | 0.402 |

## Success / collision rate by scene kind (mean over seeds)

| Kind | Random weights | CMA-ES | v4 |
| --- | --- | --- | --- |
| Crossing | 0.75 / 0.05 | — | 1.00 / 0.00 |
| Head-on | 0.76 / 0.05 | — | 0.93 / 0.07 |
| Overtaking a slow walker | 0.44 / 0.03 | — | 0.97 / 0.00 |
| Following | 0.80 / 0.00 | — | 1.00 / 0.00 |
| Oncoming group | 0.60 / 0.18 | — | 1.00 / 0.00 |
| Narrow passage | 0.67 / 0.13 | — | 0.97 / 0.03 |
| Standing group | 0.25 / 0.00 | — | 1.00 / 0.00 |
| Two-way corridor | 0.60 / 0.30 | — | 0.97 / 0.03 |
| Recorded STEPS crowd | 0.38 / 0.05 | — | 0.97 / 0.00 |
