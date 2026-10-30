# Optuna study `steps_cvae_scratch`

475 trials: 23 complete, 452 pruned. Objective: validation macro agent ADE (best-of-K, K from the study command), lower is better.

## Top 8 trials

| trial | value | cvae_latent_dim | decoder_dropout | decoder_hidden_dim | dropout | hidden_dim | kl_weight | knn | learning_rate | max_nodes | representation | warmup_epochs | weight_decay |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 53 | 0.0798 | 64 | 0.0633 | 512 | 0.0905 | 128 | 0.0799 | 24 | 0.000963 | 4096 | hidden | 4.31 | 0.00217 |
| 83 | 0.0800 | 8 | 0.0679 | 512 | 0.154 | 128 | 0.0631 | 24 | 0.000761 | 4096 | hidden | 0.542 | 9.49e-05 |
| 153 | 0.0801 | 8 | 0.0791 | 512 | 0.0334 | 128 | 0.0671 | 24 | 0.00101 | 4096 | hidden | 0.529 | 0.000289 |
| 470 | 0.0802 | 64 | 0.0329 | 512 | 0.244 | 128 | 0.077 | 16 | 0.000835 | 4096 | hidden | 1.78 | 0.00143 |
| 443 | 0.0802 | 64 | 0.00761 | 512 | 0.159 | 128 | 0.0601 | 16 | 0.0015 | 4096 | hidden | 2.78 | 0.0068 |
| 38 | 0.0803 | 64 | 0.0119 | 512 | 0.152 | 128 | 0.0741 | 16 | 0.00103 | 4096 | hidden | 4.23 | 0.00181 |
| 106 | 0.0803 | 8 | 0.0177 | 512 | 0.0595 | 128 | 0.0854 | 16 | 0.00107 | 4096 | hidden | 3.45 | 6.35e-06 |
| 33 | 0.0803 | 8 | 0.236 | 512 | 0.17 | 128 | 0.0687 | 16 | 0.000668 | 4096 | hidden | 3.06 | 0.000162 |

## Parameter importance (fANOVA)

| parameter | importance |
| --- | ---: |
| kl_weight | 0.521 |
| dropout | 0.215 |
| max_nodes | 0.078 |
| learning_rate | 0.061 |
| cvae_latent_dim | 0.044 |
| decoder_hidden_dim | 0.037 |
| weight_decay | 0.033 |
| representation | 0.004 |
| decoder_dropout | 0.004 |
| knn | 0.002 |
| hidden_dim | 0.001 |
| warmup_epochs | 0.000 |
