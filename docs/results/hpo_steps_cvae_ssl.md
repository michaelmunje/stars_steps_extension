# Optuna study `steps_cvae_ssl`

50 trials: 31 complete, 15 pruned. Objective: validation macro agent ADE (best-of-K, K from the study command), lower is better.

## Top 8 trials

| trial | value | cvae_latent_dim | decoder_dropout | decoder_hidden_dim | encoder_lr_ratio | encoder_mode | kl_weight | learning_rate | max_nodes | representation | warmup_epochs | weight_decay |
| ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 0.0798 | 16 | 0.0921 | 256 | 0.389 | finetune | 0.1 | 0.00108 | 4096 | hidden | 2.79 | 0.000927 |
| 30 | 0.0803 | 32 | 0.186 | 256 | 0.414 | finetune | 0.117 | 0.00156 | 4096 | hidden | 1.55 | 0.00139 |
| 44 | 0.0804 | 16 | 0.0249 | 256 | 0.305 | finetune | 0.0682 | 0.0028 | 4096 | hidden | 4.54 | 0.000353 |
| 23 | 0.0804 | 16 | 0.273 | 128 | 0.293 | finetune | 0.11 | 0.00116 | 4096 | hidden | 1.85 | 0.00131 |
| 45 | 0.0808 | 16 | 0.0613 | 512 | 0.341 | finetune | 0.14 | 0.00104 | 4096 | hidden | 2.67 | 0.0013 |
| 36 | 0.0812 | 16 | 0.129 | 256 | 0.456 | finetune | 0.152 | 0.00134 | 4096 | hidden | 4.49 | 0.000144 |
| 32 | 0.0815 | 16 | 0.284 | 128 | 0.311 | finetune | 0.175 | 0.00152 | 4096 | hidden | 1.83 | 0.00243 |
| 38 | 0.0817 | 32 | 0.102 | 256 | 0.85 | finetune | 0.131 | 0.00129 | 4096 | hidden | 2.02 | 0.00126 |

## Parameter importance (fANOVA)

| parameter | importance |
| --- | ---: |
| kl_weight | 0.512 |
| warmup_epochs | 0.128 |
| cvae_latent_dim | 0.103 |
| decoder_dropout | 0.076 |
| decoder_hidden_dim | 0.061 |
| weight_decay | 0.043 |
| learning_rate | 0.031 |
| max_nodes | 0.022 |
| encoder_mode | 0.012 |
| representation | 0.012 |
