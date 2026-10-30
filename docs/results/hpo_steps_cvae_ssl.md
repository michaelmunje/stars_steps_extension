# Optuna study `steps_cvae_ssl`

213 trials: 82 complete, 131 pruned. Objective: validation macro agent ADE (best-of-K, K from the study command), lower is better.

## Top 8 trials

| trial | value | cvae_latent_dim | decoder_dropout | decoder_hidden_dim | encoder_lr_ratio | encoder_mode | kl_weight | learning_rate | max_nodes | representation | warmup_epochs | weight_decay |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 165 | 0.0792 | 8 | 0.0183 | 128 | 0.189 | finetune | 0.0607 | 0.00236 | 4096 | hidden | 1.8 | 0.000112 |
| 210 | 0.0793 | 8 | 0.00814 | 128 | 0.134 | finetune | 0.0605 | 0.00212 | 4096 | hidden | 1.68 | 0.000188 |
| 138 | 0.0793 | 64 | 0.0495 | 256 | 0.198 | finetune | 0.0622 | 0.00152 | 4096 | hidden | 4.9 | 0.00506 |
| 116 | 0.0793 | 32 | 0.164 | 256 | 0.0823 | finetune | 0.0667 | 0.00255 | 4096 | hidden | 4.77 | 0.00298 |
| 55 | 0.0793 | 64 | 0.044 | 256 | 0.149 | finetune | 0.0533 | 0.00234 | 4096 | hidden | 4.57 | 0.00639 |
| 172 | 0.0796 | 64 | 0.0615 | 256 | 0.24 | finetune | 0.063 | 0.0015 | 4096 | hidden | 4.9 | 0.00971 |
| 85 | 0.0796 | 64 | 0.172 | 256 | 0.127 | finetune | 0.0647 | 0.00142 | 4096 | hidden | 4.96 | 0.00759 |
| 212 | 0.0796 | 8 | 0.121 | 256 | 0.143 | finetune | 0.0659 | 0.00155 | 4096 | hidden | 1.75 | 0.00114 |

## Parameter importance (fANOVA)

| parameter | importance |
| --- | ---: |
| kl_weight | 0.558 |
| warmup_epochs | 0.183 |
| cvae_latent_dim | 0.056 |
| decoder_dropout | 0.055 |
| decoder_hidden_dim | 0.035 |
| encoder_mode | 0.035 |
| weight_decay | 0.027 |
| learning_rate | 0.024 |
| representation | 0.020 |
| max_nodes | 0.008 |
