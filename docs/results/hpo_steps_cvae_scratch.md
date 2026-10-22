# Optuna study `steps_cvae_scratch`

73 trials: 16 complete, 53 pruned. Objective: validation macro agent ADE (best-of-K, K from the study command), lower is better.

## Top 8 trials

| trial | value | cvae_latent_dim | decoder_dropout | decoder_hidden_dim | dropout | hidden_dim | kl_weight | knn | learning_rate | max_nodes | representation | warmup_epochs | weight_decay |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 53 | 0.0798 | 64 | 0.0633 | 512 | 0.0905 | 128 | 0.0799 | 24 | 0.000963 | 4096 | hidden | 4.31 | 0.00217 |
| 38 | 0.0803 | 64 | 0.0119 | 512 | 0.152 | 128 | 0.0741 | 16 | 0.00103 | 4096 | hidden | 4.23 | 0.00181 |
| 33 | 0.0803 | 8 | 0.236 | 512 | 0.17 | 128 | 0.0687 | 16 | 0.000668 | 4096 | hidden | 3.06 | 0.000162 |
| 25 | 0.0804 | 8 | 0.121 | 512 | 0.16 | 128 | 0.0796 | 16 | 0.000633 | 4096 | hidden | 1.99 | 5.85e-05 |
| 59 | 0.0806 | 16 | 0.299 | 512 | 0.0605 | 128 | 0.0781 | 16 | 0.000983 | 4096 | mu | 3.99 | 0.000607 |
| 43 | 0.0807 | 16 | 0.244 | 512 | 0.183 | 128 | 0.0862 | 16 | 0.00116 | 4096 | mu | 2.36 | 0.000256 |
| 30 | 0.0811 | 64 | 0.0333 | 512 | 0.0944 | 128 | 0.139 | 16 | 0.000706 | 4096 | hidden | 3.97 | 3.76e-05 |
| 0 | 0.0812 | 8 | 0.119 | 512 | 0.0594 | 128 | 0.115 | 16 | 0.000413 | 4096 | mu | 3.58 | 3.86e-06 |

## Parameter importance (fANOVA)

| parameter | importance |
| --- | ---: |
| kl_weight | 0.577 |
| dropout | 0.190 |
| warmup_epochs | 0.047 |
| max_nodes | 0.038 |
| decoder_hidden_dim | 0.032 |
| decoder_dropout | 0.028 |
| weight_decay | 0.026 |
| knn | 0.022 |
| learning_rate | 0.021 |
| cvae_latent_dim | 0.016 |
| representation | 0.004 |
| hidden_dim | 0.000 |
