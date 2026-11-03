# Random vs CMA-ES-tuned controller weights (STARS crowd, test split)

Random seed s: one weight set drawn uniformly from the CMA-ES search range. CMA-ES seed s: `train_cmaes.py --config cmaes_crowd_principles.yaml --seed s` from the default weights (200 generations, population 32, about 2 h on one GPU), objective J = SPL - 2 C + SC - 0.3 time/30 s. v4 was tuned for the earlier objective. Narrow passages are 2.0-2.8 m wide.

10 random and 10 CMA-ES seeds; each seed is evaluated on the same 270 test scenes (9 kinds x 30). Cells: mean over seeds ± standard error. v4 is one policy (no error). Metrics from Table I of Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation Algorithms* (2023); distances are measured edge to edge; T averages successful episodes only; CD averages scenes with walls only. Aggregated time (AT) is omitted: our pedestrians have no cooperative goals.

| Metric | Unit | Better | Random weights | CMA-ES | v4 |
| --- | --- | --- | --- | --- | --- |
| Success rate (`S`) |  | high | 0.584 ± 0.089 | 0.988 ± 0.002 | 0.981 |
| Collision rate (`C`) |  | low | 0.090 ± 0.014 | 0.004 ± 0.001 | 0.015 |
| Human collisions (`HC`) |  | low | 0.090 ± 0.014 | 0.004 ± 0.001 | 0.015 |
| Wall collisions (`WC`) |  | low | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 |
| Timeouts (`TO`) |  | low | 0.327 ± 0.096 | 0.007 ± 0.001 | 0.004 |
| Failures to progress (5 s, 0.1 m) (`FP`) | per episode | low | 1.635 ± 0.597 | 0.059 ± 0.006 | 0.056 |
| Stalled time (< 0.05 m/s) (`ST`) | s | low | 8.463 ± 2.837 | 0.170 ± 0.018 | 0.223 |
| Time to reach goal (successes) (`T`) | s | low | 15.799 ± 1.443 | 11.208 ± 0.093 | 10.927 |
| Path length (`PL`) | m |  | 7.001 ± 0.891 | 10.184 ± 0.058 | 10.127 |
| Success weighted by path length (`SPL`) |  | high | 0.549 ± 0.086 | 0.888 ± 0.003 | 0.878 |
| Average speed (`Vavg`) | m/s |  | 0.500 ± 0.084 | 0.899 ± 0.008 | 0.923 |
| Maximum speed (`Vmax`) | m/s |  | 0.763 ± 0.113 | 0.999 ± 0.001 | 1.000 |
| Average acceleration magnitude (`Aavg`) | m/s^2 | low | 0.147 ± 0.025 | 0.155 ± 0.005 | 0.129 |
| Minimum acceleration (`Amin`) | m/s^2 |  | -0.794 ± 0.107 | -0.944 ± 0.051 | -0.830 |
| Maximum acceleration (`Amax`) | m/s^2 |  | 0.846 ± 0.103 | 1.000 ± 0.000 | 1.000 |
| Average jerk magnitude (`Javg`) | m/s^3 | low | 0.791 ± 0.142 | 0.742 ± 0.061 | 0.386 |
| Minimum jerk (`Jmin`) | m/s^3 |  | -11.307 ± 1.537 | -12.037 ± 0.395 | -10.333 |
| Maximum jerk (`Jmax`) | m/s^3 |  | 10.693 ± 1.541 | 12.348 ± 0.739 | 4.963 |
| Minimum clearance to walls (scenes with walls) (`CDmin`) | m | high | 1.508 ± 0.116 | 1.055 ± 0.013 | 1.126 |
| Average clearance to walls (scenes with walls) (`CDavg`) | m |  | 2.403 ± 0.050 | 2.235 ± 0.006 | 2.293 |
| Minimum distance to a human (`DHmin`) | m | high | 0.927 ± 0.292 | 0.626 ± 0.009 | 0.561 |
| Personal space compliance (>= 0.5 m) (`SC`) |  | high | 0.875 ± 0.026 | 0.975 ± 0.003 | 0.944 |
| Minimum time to collision (constant velocity, capped at 10 s) (`TTC`) | s | high | 4.525 ± 0.490 | 4.269 ± 0.031 | 4.181 |
| Training objective J (not Table I) (`objective`) |  | high | 1.035 ± 0.076 | 1.741 ± 0.006 | 1.679 |

## Success / collision rate by scene kind (mean over seeds)

| Kind | Random weights | CMA-ES | v4 |
| --- | --- | --- | --- |
| Crossing | 0.75 / 0.05 | 1.00 / 0.00 | 1.00 / 0.00 |
| Head-on | 0.76 / 0.05 | 0.98 / 0.02 | 0.93 / 0.07 |
| Overtaking a slow walker | 0.44 / 0.03 | 1.00 / 0.00 | 0.97 / 0.00 |
| Following | 0.80 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Oncoming group | 0.61 / 0.17 | 1.00 / 0.00 | 1.00 / 0.00 |
| Narrow passage | 0.67 / 0.13 | 0.95 / 0.01 | 0.97 / 0.03 |
| Standing group | 0.25 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Two-way corridor | 0.60 / 0.31 | 0.99 / 0.00 | 0.97 / 0.03 |
| Recorded STEPS crowd | 0.38 / 0.06 | 0.97 / 0.01 | 1.00 / 0.00 |
