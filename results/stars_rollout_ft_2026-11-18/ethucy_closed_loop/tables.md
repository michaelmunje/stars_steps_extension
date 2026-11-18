**All windows (549)**

| method | ADE 4.8 s | ADE 8 s | ADE 12 s | colliding pairs % (8 s) | HACS (8 s) |
|---|---|---|---|---|---|
| cv | 0.497 | 0.907 | 1.428 | 3.54 | 0.1298 |
| orca | 0.559 | 0.896 | 1.235 | 0.13 | 0.0018 |
| sfm | 0.574 | 0.968 | 1.354 | 1.13 | 0.0137 |
| stars_improved | 0.521 | 0.989 | 1.567 | 2.78 | 0.1163 |
| stars_improved_social | 0.576 | 1.043 | 1.621 | 0.44 | 0.0054 |
| stars_rft | 0.443 | 0.809 | 1.263 | 1.99 | 0.1006 |
| stars_rft_os4 | 0.445 | 0.812 | 1.273 | 1.73 | 0.0904 |
| stars_rft_social | 0.517 | 0.928 | 1.422 | 0.96 | 0.0259 |

**Pooled split `train` recordings**

| method | ADE 4.8 s | ADE 8 s | ADE 12 s | colliding pairs % (8 s) | HACS (8 s) |
|---|---|---|---|---|---|
| cv | 0.456 | 0.888 | 1.471 | 3.39 | 0.1796 |
| orca | 0.532 | 0.877 | 1.23 | 0.16 | 0.0031 |
| sfm | 0.566 | 0.953 | 1.346 | 0.88 | 0.0152 |
| stars_improved | 0.484 | 0.951 | 1.571 | 2.69 | 0.1612 |
| stars_improved_social | 0.546 | 1.012 | 1.641 | 0.46 | 0.0073 |
| stars_rft | 0.396 | 0.755 | 1.23 | 2.08 | 0.1615 |
| stars_rft_os4 | 0.399 | 0.758 | 1.242 | 1.86 | 0.1458 |
| stars_rft_social | 0.495 | 0.908 | 1.413 | 0.94 | 0.039 |

**Pooled split `val` recordings**

| method | ADE 4.8 s | ADE 8 s | ADE 12 s | colliding pairs % (8 s) | HACS (8 s) |
|---|---|---|---|---|---|
| cv | 0.616 | 1.159 | 1.869 | 3.93 | 0.1052 |
| orca | 0.719 | 1.145 | 1.657 | 0.08 | 0.0006 |
| sfm | 0.691 | 1.187 | 1.748 | 1.51 | 0.0146 |
| stars_improved | 0.666 | 1.298 | 2.182 | 3.22 | 0.0947 |
| stars_improved_social | 0.718 | 1.342 | 2.109 | 0.39 | 0.0049 |
| stars_rft | 0.587 | 1.093 | 1.778 | 1.92 | 0.0467 |
| stars_rft_os4 | 0.588 | 1.094 | 1.787 | 1.47 | 0.0401 |
| stars_rft_social | 0.65 | 1.202 | 1.933 | 1.03 | 0.0164 |

**Pooled split `test` recordings**

| method | ADE 4.8 s | ADE 8 s | ADE 12 s | colliding pairs % (8 s) | HACS (8 s) |
|---|---|---|---|---|---|
| cv | 0.444 | 0.612 | 0.779 | 3.86 | 0.0322 |
| orca | 0.431 | 0.606 | 0.737 | 0.0 | 0.0 |
| sfm | 0.455 | 0.705 | 0.897 | 3.11 | 0.0084 |
| stars_improved | 0.43 | 0.661 | 0.813 | 1.98 | 0.0273 |
| stars_improved_social | 0.47 | 0.712 | 0.976 | 0.38 | 0.0012 |
| stars_rft | 0.375 | 0.56 | 0.725 | 1.13 | 0.0141 |
| stars_rft_os4 | 0.378 | 0.562 | 0.729 | 0.94 | 0.0133 |
| stars_rft_social | 0.408 | 0.601 | 0.826 | 0.94 | 0.0041 |

**Kinematics at 8 s (W1 to ground truth)**

| method | speed_mean | speed_w1_vs_gt | acceleration_mean | acceleration_w1_vs_gt | turning_mean | turning_w1_vs_gt |
|---|---|---|---|---|---|---|
| ground_truth | 0.466 | 0.0 | 0.159 | 0.0 | 12.767 | 0.0 |
| cv | 0.481 | 0.017 | 0.0 | 0.171 | 0.0 | 13.167 |
| orca | 0.437 | 0.033 | 0.048 | 0.118 | 3.407 | 9.285 |
| sfm | 0.448 | 0.051 | 0.121 | 0.08 | 11.554 | 2.552 |
| stars_improved | 0.456 | 0.052 | 0.076 | 0.131 | 5.72 | 7.907 |
| stars_improved_social | 0.46 | 0.016 | 0.076 | 0.125 | 6.032 | 7.968 |
| stars_rft | 0.438 | 0.04 | 0.061 | 0.144 | 3.666 | 10.101 |
| stars_rft_os4 | 0.439 | 0.04 | 0.061 | 0.144 | 3.664 | 10.082 |
| stars_rft_social | 0.402 | 0.067 | 0.071 | 0.147 | 4.889 | 9.329 |

**Paired micro ADE differences (a - b, 95% block-bootstrap CI)**

| horizon_s | a | b | a_minus_b | ci_low | ci_high |
|---|---|---|---|---|---|
| 2.0 | stars_rft | cv | -0.007 | -0.013 | -0.002 |
| 2.0 | stars_rft | orca | -0.08 | -0.089 | -0.071 |
| 2.0 | stars_rft_os4 | cv | -0.007 | -0.012 | -0.002 |
| 2.0 | stars_rft_os4 | orca | -0.08 | -0.089 | -0.07 |
| 2.0 | stars_rft_social | cv | 0.026 | 0.02 | 0.031 |
| 2.0 | stars_rft_social | orca | -0.047 | -0.054 | -0.039 |
| 4.0 | stars_rft | cv | -0.04 | -0.05 | -0.029 |
| 4.0 | stars_rft | orca | -0.108 | -0.127 | -0.088 |
| 4.0 | stars_rft_os4 | cv | -0.038 | -0.049 | -0.028 |
| 4.0 | stars_rft_os4 | orca | -0.106 | -0.126 | -0.086 |
| 4.0 | stars_rft_social | cv | 0.021 | 0.01 | 0.031 |
| 4.0 | stars_rft_social | orca | -0.047 | -0.061 | -0.032 |
| 4.8 | stars_rft | cv | -0.054 | -0.066 | -0.041 |
| 4.8 | stars_rft | orca | -0.116 | -0.142 | -0.093 |
| 4.8 | stars_rft_os4 | cv | -0.051 | -0.064 | -0.039 |
| 4.8 | stars_rft_os4 | orca | -0.114 | -0.14 | -0.09 |
| 4.8 | stars_rft_social | cv | 0.021 | 0.007 | 0.033 |
| 4.8 | stars_rft_social | orca | -0.042 | -0.06 | -0.026 |
| 8.0 | stars_rft | cv | -0.098 | -0.123 | -0.07 |
| 8.0 | stars_rft | orca | -0.087 | -0.128 | -0.044 |
| 8.0 | stars_rft_os4 | cv | -0.096 | -0.12 | -0.068 |
| 8.0 | stars_rft_os4 | orca | -0.085 | -0.127 | -0.04 |
| 8.0 | stars_rft_social | cv | 0.02 | -0.002 | 0.043 |
| 8.0 | stars_rft_social | orca | 0.031 | 0.003 | 0.061 |
| 12.0 | stars_rft | cv | -0.164 | -0.219 | -0.1 |
| 12.0 | stars_rft | orca | 0.028 | -0.044 | 0.102 |
| 12.0 | stars_rft_os4 | cv | -0.155 | -0.209 | -0.092 |
| 12.0 | stars_rft_os4 | orca | 0.037 | -0.031 | 0.114 |
| 12.0 | stars_rft_social | cv | -0.005 | -0.059 | 0.057 |
| 12.0 | stars_rft_social | orca | 0.187 | 0.127 | 0.25 |
