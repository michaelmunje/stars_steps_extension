# Random vs CMA-ES-tuned controller weights (STARS crowd, test split)

Random seed s: one weight set drawn uniformly from the CMA-ES search range. CMA-ES seed s: `train_cmaes.py --config cmaes_crowd_principles.yaml --seed s --random-init s`, i.e. CMA-ES started from random seed s's weights (200 generations, population 32, about 2 h on one GPU), objective J = SPL - 2 C + SC - 0.3 time/30 s. Narrow passages are 2.0-2.8 m wide.

10 random and 10 CMA-ES seeds; each seed is evaluated on the same 270 test scenes (9 kinds x 30). Cells: mean over seeds ± standard error; *Change* is the per-seed difference CMA-ES minus its own random start, averaged over seeds. Metrics from Table I of Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation Algorithms* (2023); distances are measured edge to edge; T averages successful episodes only; CD averages scenes with walls only. Aggregated time (AT) is omitted: our pedestrians have no cooperative goals.

| Metric | Unit | Better | Random weights | CMA-ES | Change (paired) |
| --- | --- | --- | --- | --- | --- |
| Success rate (`S`) |  | high | 0.584 ± 0.089 | 0.977 ± 0.003 | 0.393 ± 0.090 |
| Collision rate (`C`) |  | low | 0.090 ± 0.014 | 0.012 ± 0.002 | -0.077 ± 0.013 |
| Human collisions (`HC`) |  | low | 0.090 ± 0.014 | 0.012 ± 0.002 | -0.077 ± 0.013 |
| Wall collisions (`WC`) |  | low | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Timeouts (`TO`) |  | low | 0.327 ± 0.096 | 0.011 ± 0.003 | -0.316 ± 0.097 |
| Failures to progress (5 s, 0.1 m) (`FP`) | per episode | low | 1.635 ± 0.597 | 0.074 ± 0.011 | -1.561 ± 0.596 |
| Stalled time (< 0.05 m/s) (`ST`) | s | low | 8.463 ± 2.837 | 0.335 ± 0.044 | -8.128 ± 2.849 |
| Time to reach goal (successes) (`T`) | s | low | 15.799 ± 1.443 | 11.232 ± 0.130 | -4.568 ± 1.473 |
| Path length (`PL`) | m |  | 7.001 ± 0.891 | 9.909 ± 0.060 | 2.909 ± 0.917 |
| Success weighted by path length (`SPL`) |  | high | 0.549 ± 0.086 | 0.887 ± 0.002 | 0.338 ± 0.086 |
| Average speed (`Vavg`) | m/s |  | 0.500 ± 0.084 | 0.880 ± 0.011 | 0.380 ± 0.085 |
| Maximum speed (`Vmax`) | m/s |  | 0.763 ± 0.113 | 0.997 ± 0.001 | 0.234 ± 0.112 |
| Average acceleration magnitude (`Aavg`) | m/s^2 | low | 0.147 ± 0.025 | 0.167 ± 0.008 | 0.020 ± 0.027 |
| Minimum acceleration (`Amin`) | m/s^2 |  | -0.794 ± 0.107 | -0.928 ± 0.058 | -0.134 ± 0.133 |
| Maximum acceleration (`Amax`) | m/s^2 |  | 0.846 ± 0.103 | 0.998 ± 0.001 | 0.152 ± 0.102 |
| Average jerk magnitude (`Javg`) | m/s^3 | low | 0.791 ± 0.142 | 0.784 ± 0.088 | -0.007 ± 0.185 |
| Minimum jerk (`Jmin`) | m/s^3 |  | -11.307 ± 1.537 | -11.604 ± 0.253 | -0.296 ± 1.543 |
| Maximum jerk (`Jmax`) | m/s^3 |  | 10.693 ± 1.541 | 11.507 ± 0.939 | 0.815 ± 2.032 |
| Minimum clearance to walls (scenes with walls) (`CDmin`) | m | high | 1.508 ± 0.116 | 1.110 ± 0.017 | -0.398 ± 0.120 |
| Average clearance to walls (scenes with walls) (`CDavg`) | m |  | 2.403 ± 0.050 | 2.261 ± 0.010 | -0.142 ± 0.050 |
| Minimum distance to a human (`DHmin`) | m | high | 0.927 ± 0.292 | 0.584 ± 0.010 | -0.344 ± 0.291 |
| Personal space compliance (>= 0.5 m) (`SC`) |  | high | 0.875 ± 0.026 | 0.965 ± 0.004 | 0.090 ± 0.025 |
| Minimum time to collision (constant velocity, capped at 10 s) (`TTC`) | s | high | 4.525 ± 0.490 | 4.314 ± 0.029 | -0.212 ± 0.486 |
| Training objective J (not Table I) (`objective`) |  | high | 1.035 ± 0.076 | 1.711 ± 0.008 | 0.676 ± 0.076 |

## Success / collision rate by scene kind (mean over seeds)

| Kind | Random weights | CMA-ES |
| --- | --- | --- |
| Crossing | 0.75 / 0.05 | 1.00 / 0.00 |
| Head-on | 0.76 / 0.05 | 0.96 / 0.04 |
| Overtaking a slow walker | 0.44 / 0.03 | 0.98 / 0.01 |
| Following | 0.80 / 0.00 | 1.00 / 0.00 |
| Oncoming group | 0.61 / 0.17 | 1.00 / 0.00 |
| Narrow passage | 0.67 / 0.13 | 0.98 / 0.02 |
| Standing group | 0.25 / 0.00 | 1.00 / 0.00 |
| Two-way corridor | 0.60 / 0.31 | 0.97 / 0.03 |
| Recorded STEPS crowd | 0.38 / 0.06 | 0.90 / 0.02 |
