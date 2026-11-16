### ETH/UCY leave-one-out: minADE20/minFDE20 (m)

| Method | eth | hotel | univ | zara1 | zara2 | avg | leakage |
|---|---:|---:|---:|---:|---:|---:|---|
| Constant velocity | 1.08/2.28 | 0.32/0.61 | 0.52/1.17 | 0.43/0.95 | 0.32/0.72 | 0.53/1.15 | none |
| SocialVAE (official ckpt) | 0.50/0.86 | 0.15/0.22 | 0.25/0.48 | 0.20/0.36 | 0.15/0.29 | 0.25/0.44 | none (official LOO) |
| &nbsp;&nbsp;published | 0.47/0.76 | 0.14/0.22 | 0.25/0.47 | 0.20/0.37 | 0.14/0.28 | 0.24/0.42 | paper |
| MoFlow (official ckpt) | 0.42/0.61 | 0.11/0.17 | 0.25/0.41 | 0.17/0.29 | 0.12/0.22 | 0.22/0.34 | none (official LOO) |
| &nbsp;&nbsp;published | 0.40/0.57 | 0.11/0.17 | 0.23/0.39 | 0.15/0.26 | 0.12/0.22 | 0.20/0.32 | paper |
| Trajectron++ (official ckpt) | 0.77/1.34 | 0.24/0.36 | 0.30/0.58 | 0.23/0.43 | 0.17/0.33 | 0.34/0.61 | none (official LOO) |
| &nbsp;&nbsp;published | 0.39/0.83 | 0.12/0.21 | 0.20/0.44 | 0.15/0.33 | 0.11/0.25 | 0.19/0.41 | paper (future-leaking features) |
| AgentFormer (official ckpt) | 0.45/0.75 | 0.14/0.22 | 0.25/0.45 | 0.18/0.30 | 0.14/0.24 | 0.23/0.39 | none (official LOO) |
| &nbsp;&nbsp;published | 0.45/0.75 | 0.14/0.22 | 0.25/0.45 | 0.18/0.30 | 0.14/0.24 | 0.23/0.39 | paper |
| Social-GAN (official ckpt) | 0.61/1.01 | 0.33/0.65 | 0.40/0.72 | 0.23/0.43 | 0.22/0.39 | 0.36/0.64 | none (official LOO) |
| &nbsp;&nbsp;published | 0.77/1.39 | 0.43/0.88 | 0.75/1.50 | 0.34/0.68 | 0.36/0.73 | 0.53/1.04 | MODEL_ZOO (scene-joint min: compare minJoint) |
| SocialVAE (ours, STEPS-trained) | 0.50/0.82 | 0.12/0.20 | 0.29/0.53 | 0.24/0.43 | 0.18/0.33 | 0.27/0.46 | none (zero-shot) |
| MoFlow (ours, STEPS-trained) | 0.62/0.97 | 0.13/0.22 | 0.30/0.53 | 0.24/0.42 | 0.16/0.28 | 0.29/0.49 | none (zero-shot) |
| STARS LOO (no-ETH/UCY pretrain + LOO fine-tune) | 0.53/1.00 | 0.12/0.19 | 0.24/0.47 | 0.19/0.37 | 0.15/0.29 | 0.25/0.46 | none (LOO) |
| STARS STEPS-only | 0.47/0.81 | 0.12/0.21 | 0.29/0.57 | 0.25/0.48 | 0.19/0.36 | 0.27/0.49 | none (zero-shot) |
| STARS pooled w/o ETH/UCY | 0.45/0.80 | 0.10/0.18 | 0.27/0.53 | 0.24/0.49 | 0.18/0.36 | 0.25/0.47 | none (zero-shot) |
| STARS pooled w/o ETH/UCY + oracle goal | 0.37/0.50 | 0.09/0.13 | 0.24/0.41 | 0.21/0.37 | 0.17/0.32 | 0.22/0.34 | none (zero-shot; privileged goal input) |
| STARS pooled (6 datasets) | 0.46/0.83 | 0.10/0.16* | 0.22/0.42* | 0.22/0.44* | 0.16/0.31 | 0.23/0.43 | LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean |
| STARS STEPS-only, per-agent diffusion | 0.55/0.83 | 0.15/0.22 | 0.35/0.64 | 0.29/0.51 | 0.21/0.38 | 0.31/0.52 | none (zero-shot) |
| STARS STEPS-only, joint diffusion | 0.50/0.80 | 0.14/0.21 | 0.34/0.60 | 0.30/0.49 | 0.21/0.35 | 0.30/0.49 | none (zero-shot) |

### ETH/UCY: mean-over-20-samples ADE/FDE (m)

| Method | eth | hotel | univ | zara1 | zara2 | avg | leakage |
|---|---:|---:|---:|---:|---:|---:|---|
| Constant velocity | 1.08/2.28 | 0.32/0.61 | 0.52/1.17 | 0.43/0.95 | 0.32/0.72 | 0.53/1.15 | none |
| SocialVAE (official ckpt) | 1.09/2.24 | 0.45/0.95 | 0.70/1.55 | 0.60/1.34 | 0.44/1.01 | 0.66/1.42 | none (official LOO) |
| MoFlow (official ckpt) | 1.40/2.89 | 0.85/1.90 | 1.07/2.08 | 0.82/1.80 | 0.79/1.81 | 0.99/2.09 | none (official LOO) |
| Trajectron++ (official ckpt) | 1.33/2.72 | 0.58/1.17 | 0.69/1.56 | 0.60/1.38 | 0.46/1.07 | 0.73/1.58 | none (official LOO) |
| AgentFormer (official ckpt) | 1.88/4.09 | 0.88/2.01 | 1.01/2.26 | 0.73/1.64 | 0.75/1.71 | 1.05/2.34 | none (official LOO) |
| Social-GAN (official ckpt) | 1.24/2.49 | 0.76/1.60 | 1.08/2.19 | 0.68/1.48 | 0.57/1.21 | 0.86/1.79 | none (official LOO) |
| SocialVAE (ours, STEPS-trained) | 1.24/2.53 | 0.43/0.91 | 0.83/1.73 | 0.77/1.68 | 0.53/1.16 | 0.76/1.60 | none (zero-shot) |
| MoFlow (ours, STEPS-trained) | 1.78/3.44 | 1.24/2.53 | 1.33/2.69 | 1.05/2.19 | 1.16/2.42 | 1.31/2.66 | none (zero-shot) |
| STARS LOO (no-ETH/UCY pretrain + LOO fine-tune) | 1.03/2.15 | 0.29/0.56 | 0.61/1.34 | 0.47/1.04 | 0.34/0.75 | 0.55/1.17 | none (LOO) |
| STARS STEPS-only | 1.09/2.17 | 0.29/0.58 | 0.68/1.43 | 0.59/1.28 | 0.41/0.89 | 0.61/1.27 | none (zero-shot) |
| STARS pooled w/o ETH/UCY | 1.12/2.31 | 0.24/0.47 | 0.62/1.31 | 0.49/1.04 | 0.36/0.77 | 0.56/1.18 | none (zero-shot) |
| STARS pooled w/o ETH/UCY + oracle goal | 0.78/1.32 | 0.15/0.25 | 0.51/1.00 | 0.37/0.70 | 0.30/0.60 | 0.42/0.77 | none (zero-shot; privileged goal input) |
| STARS pooled (6 datasets) | 1.17/2.39 | 0.24/0.47* | 0.56/1.20* | 0.48/1.03* | 0.36/0.78 | 0.56/1.17 | LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean |
| STARS STEPS-only, per-agent diffusion | 1.47/2.90 | 0.51/1.05 | 0.89/1.86 | 0.89/1.94 | 0.65/1.41 | 0.88/1.83 | none (zero-shot) |
| STARS STEPS-only, joint diffusion | 1.30/2.58 | 0.49/0.98 | 0.84/1.71 | 0.86/1.83 | 0.60/1.27 | 0.82/1.67 | none (zero-shot) |

### ETH/UCY: first-sample ADE/FDE (m)

| Method | eth | hotel | univ | zara1 | zara2 | avg | leakage |
|---|---:|---:|---:|---:|---:|---:|---|
| Constant velocity | 1.08/2.28 | 0.32/0.61 | 0.52/1.17 | 0.43/0.95 | 0.32/0.72 | 0.53/1.15 | none |
| SocialVAE (official ckpt) | 1.11/2.26 | 0.45/0.96 | 0.71/1.55 | 0.60/1.36 | 0.44/1.01 | 0.66/1.43 | none (official LOO) |
| MoFlow (official ckpt) | 1.30/2.75 | 0.48/1.04 | 0.95/1.87 | 0.80/1.69 | 1.37/2.98 | 0.98/2.07 | none (official LOO) |
| Trajectron++ (official ckpt) | 1.35/2.76 | 0.57/1.14 | 0.69/1.55 | 0.60/1.39 | 0.46/1.07 | 0.73/1.58 | none (official LOO) |
| AgentFormer (official ckpt) | 1.03/2.03 | 1.41/3.29 | 0.67/1.42 | 0.82/1.88 | 0.70/1.61 | 0.93/2.05 | none (official LOO) |
| Social-GAN (official ckpt) | 1.23/2.50 | 0.75/1.58 | 1.07/2.19 | 0.66/1.45 | 0.57/1.20 | 0.86/1.78 | none (official LOO) |
| SocialVAE (ours, STEPS-trained) | 1.25/2.55 | 0.45/0.95 | 0.83/1.73 | 0.76/1.66 | 0.53/1.15 | 0.76/1.61 | none (zero-shot) |
| MoFlow (ours, STEPS-trained) | 1.42/2.66 | 0.76/1.61 | 0.76/1.52 | 0.74/1.65 | 0.77/1.69 | 0.89/1.82 | none (zero-shot) |
| STARS LOO (no-ETH/UCY pretrain + LOO fine-tune) | 1.02/2.12 | 0.30/0.58 | 0.61/1.34 | 0.47/1.02 | 0.34/0.75 | 0.55/1.16 | none (LOO) |
| STARS STEPS-only | 1.09/2.15 | 0.29/0.59 | 0.68/1.42 | 0.58/1.25 | 0.41/0.89 | 0.61/1.26 | none (zero-shot) |
| STARS pooled w/o ETH/UCY | 1.12/2.32 | 0.23/0.45 | 0.62/1.31 | 0.50/1.05 | 0.36/0.78 | 0.57/1.18 | none (zero-shot) |
| STARS pooled w/o ETH/UCY + oracle goal | 0.74/1.24 | 0.15/0.25 | 0.51/1.00 | 0.37/0.70 | 0.30/0.59 | 0.42/0.76 | none (zero-shot; privileged goal input) |
| STARS pooled (6 datasets) | 1.20/2.46 | 0.25/0.49* | 0.56/1.21* | 0.47/1.01* | 0.35/0.77 | 0.57/1.19 | LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean |
| STARS STEPS-only, per-agent diffusion | 1.44/2.87 | 0.53/1.08 | 0.88/1.84 | 0.89/1.96 | 0.65/1.41 | 0.88/1.83 | none (zero-shot) |
| STARS STEPS-only, joint diffusion | 1.27/2.52 | 0.50/1.00 | 0.84/1.70 | 0.86/1.82 | 0.60/1.27 | 0.81/1.66 | none (zero-shot) |

### ETH/UCY: minJointADE20/minJointFDE20 (m)

| Method | eth | hotel | univ | zara1 | zara2 | avg | leakage |
|---|---:|---:|---:|---:|---:|---:|---|
| Constant velocity | 1.12/2.30 | 0.31/0.61 | 0.54/1.21 | 0.41/0.92 | 0.32/0.72 | 0.54/1.15 | none |
| SocialVAE (official ckpt) | 0.57/1.01 | 0.20/0.34 | 0.58/1.27 | 0.31/0.65 | 0.25/0.55 | 0.38/0.76 | none (official LOO) |
| MoFlow (official ckpt) | 0.46/0.71 | 0.20/0.36 | 0.69/1.34 | 0.30/0.61 | 0.28/0.60 | 0.39/0.72 | none (official LOO) |
| Trajectron++ (official ckpt) | 0.85/1.46 | 0.29/0.50 | 0.58/1.30 | 0.32/0.68 | 0.27/0.59 | 0.46/0.90 | none (official LOO) |
| AgentFormer (official ckpt) | 0.48/0.79 | 0.24/0.46 | 0.62/1.31 | 0.28/0.56 | 0.30/0.62 | 0.38/0.75 | none (official LOO) |
| Social-GAN (official ckpt) | 0.61/1.00 | 0.41/0.82 | 0.73/1.47 | 0.31/0.62 | 0.34/0.69 | 0.48/0.92 | none (official LOO) |
| SocialVAE (ours, STEPS-trained) | 0.55/0.92 | 0.16/0.30 | 0.68/1.42 | 0.38/0.77 | 0.29/0.60 | 0.41/0.80 | none (zero-shot) |
| MoFlow (ours, STEPS-trained) | 0.62/1.03 | 0.23/0.42 | 0.62/1.27 | 0.39/0.76 | 0.35/0.71 | 0.44/0.84 | none (zero-shot) |
| STARS LOO (no-ETH/UCY pretrain + LOO fine-tune) | 0.59/1.10 | 0.14/0.23 | 0.51/1.10 | 0.26/0.54 | 0.20/0.43 | 0.34/0.68 | none (LOO) |
| STARS STEPS-only | 0.53/0.91 | 0.14/0.24 | 0.57/1.19 | 0.33/0.67 | 0.25/0.50 | 0.36/0.70 | none (zero-shot) |
| STARS pooled w/o ETH/UCY | 0.51/0.91 | 0.12/0.22 | 0.53/1.12 | 0.30/0.63 | 0.23/0.49 | 0.34/0.67 | none (zero-shot) |
| STARS pooled w/o ETH/UCY + oracle goal | 0.42/0.60 | 0.10/0.14 | 0.43/0.82 | 0.25/0.44 | 0.22/0.42 | 0.28/0.48 | none (zero-shot; privileged goal input) |
| STARS pooled (6 datasets) | 0.52/0.93 | 0.12/0.19* | 0.48/1.02* | 0.29/0.59* | 0.22/0.46 | 0.32/0.64 | LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean |
| STARS STEPS-only, per-agent diffusion | 0.64/0.99 | 0.19/0.32 | 0.72/1.48 | 0.44/0.87 | 0.34/0.69 | 0.47/0.87 | none (zero-shot) |
| STARS STEPS-only, joint diffusion | 0.56/0.91 | 0.17/0.28 | 0.68/1.36 | 0.41/0.75 | 0.30/0.58 | 0.43/0.78 | none (zero-shot) |

### ETH/UCY social plausibility (mean over 5 folds)

| Method | col20_agent | col40_agent | col20_scene | col40_scene | speed_mean | speed_gt | speed_w1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ground truth (reference) | 0.005 | 0.081 | 0.045 | 0.226 | 0.675 | 0.675 | 0.000 |
| Constant velocity | 0.069 | 0.180 | 0.227 | 0.353 | 0.708 | 0.675 | 0.037 |
| SocialVAE (official ckpt) | 0.097 | 0.231 | 0.274 | 0.425 | 0.669 | 0.675 | 0.044 |
| MoFlow (official ckpt) | 0.076 | 0.199 | 0.231 | 0.383 | 0.759 | 0.675 | 0.149 |
| Trajectron++ (official ckpt) | 0.103 | 0.238 | 0.283 | 0.431 | 0.731 | 0.675 | 0.078 |
| AgentFormer (official ckpt) | 0.074 | 0.204 | 0.226 | 0.371 | 0.729 | 0.675 | 0.153 |
| Social-GAN (official ckpt) | 0.102 | 0.240 | 0.289 | 0.429 | 0.683 | 0.675 | 0.141 |
| SocialVAE (ours, STEPS-trained) | 0.102 | 0.233 | 0.279 | 0.425 | 0.656 | 0.675 | 0.058 |
| MoFlow (ours, STEPS-trained) | 0.105 | 0.251 | 0.286 | 0.448 | 0.903 | 0.675 | 0.288 |
| STARS LOO (no-ETH/UCY pretrain + LOO fine-tune) | 0.050 | 0.140 | 0.187 | 0.294 | 0.685 | 0.675 | 0.045 |
| STARS STEPS-only | 0.073 | 0.189 | 0.226 | 0.360 | 0.626 | 0.675 | 0.058 |
| STARS pooled w/o ETH/UCY | 0.051 | 0.140 | 0.188 | 0.296 | 0.682 | 0.675 | 0.041 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.051 | 0.154 | 0.190 | 0.315 | 0.643 | 0.675 | 0.045 |
| STARS pooled (6 datasets) | 0.050 | 0.142 | 0.190 | 0.299 | 0.689 | 0.675 | 0.037 |
| STARS STEPS-only, per-agent diffusion | 0.123 | 0.295 | 0.318 | 0.503 | 0.696 | 0.675 | 0.091 |
| STARS STEPS-only, joint diffusion | 0.114 | 0.283 | 0.300 | 0.485 | 0.700 | 0.675 | 0.085 |

### Our datasets: minADE20/minFDE20 (m)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant velocity | 0.26/0.49 | 0.85/1.59 | 0.62/1.17 | 0.67/1.36 | 0.96/1.79 | 0.69/1.37 | 0.36/0.81 |
| SocialVAE (ours, STEPS-trained) | 0.07/0.10 | 0.35/0.54 | 0.29/0.43 | 0.34/0.55 | 0.37/0.53 | 0.30/0.50 | 0.22/0.41 |
| MoFlow (ours, STEPS-trained) | 0.08/0.11 | 0.34/0.50 | 0.33/0.52 | 0.28/0.45 | 0.37/0.55 | 0.31/0.53 | 0.20/0.37 |
| STARS STEPS-only | 0.08/0.12 | 0.35/0.55 | 0.26/0.41 | 0.29/0.46 | 0.35/0.50 | 0.28/0.48 | 0.24/0.46 |
| STARS pooled w/o ETH/UCY | 0.07/0.12 | 0.26/0.39 | 0.21/0.33 | 0.23/0.38 | 0.29/0.40 | 0.23/0.39 | 0.22/0.44 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.07/0.10 | 0.25/0.34 | 0.17/0.21 | 0.20/0.29 | 0.27/0.35 | 0.20/0.27 | 0.20/0.39 |
| STARS pooled (6 datasets) | 0.07/0.11 | 0.27/0.39 | 0.20/0.31 | 0.23/0.38 | 0.30/0.41 | 0.24/0.41 | 0.19/0.39 |
| STARS STEPS-only, per-agent diffusion | 0.07/0.09 | 0.40/0.60 | 0.31/0.52 | 0.36/0.55 | 0.40/0.55 | 0.33/0.51 | 0.27/0.50 |
| STARS STEPS-only, joint diffusion | 0.07/0.10 | 0.39/0.59 | 0.32/0.48 | 0.37/0.54 | 0.40/0.55 | 0.32/0.49 | 0.25/0.43 |
| *scored agents / windows* | 22989 / 1500 | 13866 / 500 | 447 / 4 | 21195 / 300 | 18284 / 300 | 733 / 188 | 1856 / 268 |

### Our datasets: mean-over-samples ADE/FDE (m)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant velocity | 0.26/0.49 | 0.85/1.59 | 0.62/1.17 | 0.67/1.36 | 0.96/1.79 | 0.69/1.37 | 0.36/0.81 |
| SocialVAE (ours, STEPS-trained) | 0.22/0.45 | 0.92/1.77 | 0.84/1.64 | 1.12/2.37 | 1.08/2.10 | 1.01/2.20 | 0.61/1.36 |
| MoFlow (ours, STEPS-trained) | 1.30/2.66 | 1.51/2.98 | 1.47/2.85 | 1.07/2.21 | 1.48/2.88 | 1.27/2.59 | 1.25/2.61 |
| STARS STEPS-only | 0.13/0.23 | 0.71/1.32 | 0.68/1.31 | 0.82/1.70 | 0.81/1.52 | 0.68/1.42 | 0.48/1.04 |
| STARS pooled w/o ETH/UCY | 0.13/0.23 | 0.62/1.15 | 0.59/1.13 | 0.58/1.16 | 0.64/1.14 | 0.55/1.11 | 0.41/0.88 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.11/0.18 | 0.58/1.02 | 0.45/0.80 | 0.45/0.80 | 0.58/0.97 | 0.40/0.65 | 0.35/0.70 |
| STARS pooled (6 datasets) | 0.13/0.24 | 0.64/1.17 | 0.57/1.10 | 0.58/1.17 | 0.82/1.50 | 0.59/1.19 | 0.41/0.88 |
| STARS STEPS-only, per-agent diffusion | 0.21/0.39 | 0.93/1.77 | 0.85/1.66 | 1.17/2.45 | 1.11/2.11 | 1.14/2.41 | 0.79/1.72 |
| STARS STEPS-only, joint diffusion | 0.20/0.36 | 0.91/1.70 | 0.89/1.69 | 1.10/2.22 | 1.07/2.03 | 0.95/1.94 | 0.68/1.44 |
| *scored agents / windows* | 22989 / 1500 | 13866 / 500 | 447 / 4 | 21195 / 300 | 18284 / 300 | 733 / 188 | 1856 / 268 |

### Our datasets: minJointADE20/FDE20 (m)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant velocity | 0.27/0.50 | 0.81/1.52 | 0.62/1.16 | 0.68/1.36 | 0.95/1.77 | 0.65/1.27 | 0.36/0.81 |
| SocialVAE (ours, STEPS-trained) | 0.12/0.22 | 0.65/1.23 | 0.76/1.48 | 0.96/2.00 | 0.92/1.77 | 0.46/0.92 | 0.36/0.77 |
| MoFlow (ours, STEPS-trained) | 0.14/0.25 | 0.64/1.18 | 0.58/1.03 | 0.62/1.22 | 0.84/1.56 | 0.56/1.07 | 0.47/0.95 |
| STARS STEPS-only | 0.09/0.15 | 0.52/0.95 | 0.63/1.20 | 0.71/1.44 | 0.69/1.27 | 0.36/0.70 | 0.31/0.64 |
| STARS pooled w/o ETH/UCY | 0.09/0.15 | 0.44/0.79 | 0.53/1.03 | 0.51/1.00 | 0.55/0.95 | 0.31/0.56 | 0.28/0.59 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.08/0.12 | 0.40/0.68 | 0.41/0.72 | 0.40/0.69 | 0.50/0.81 | 0.25/0.37 | 0.26/0.51 |
| STARS pooled (6 datasets) | 0.09/0.15 | 0.44/0.80 | 0.51/0.96 | 0.51/1.02 | 0.69/1.23 | 0.32/0.60 | 0.27/0.57 |
| STARS STEPS-only, per-agent diffusion | 0.12/0.20 | 0.66/1.22 | 0.75/1.46 | 0.98/2.01 | 0.93/1.73 | 0.52/1.00 | 0.45/0.94 |
| STARS STEPS-only, joint diffusion | 0.11/0.18 | 0.64/1.17 | 0.78/1.48 | 0.92/1.83 | 0.89/1.64 | 0.45/0.82 | 0.38/0.74 |
| *scored agents / windows* | 22989 / 1500 | 13866 / 500 | 447 / 4 | 21195 / 300 | 18284 / 300 | 733 / 188 | 1856 / 268 |

### Our datasets: P(agent within 0.4 m of another prediction, same sample)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ground truth (reference) | 0.058 | 0.054 | 0.000 | 0.128 | 0.002 | 0.009 | 0.104 |
| Constant velocity | 0.138 | 0.120 | 0.065 | 0.294 | 0.233 | 0.146 | 0.181 |
| SocialVAE (ours, STEPS-trained) | 0.114 | 0.153 | 0.074 | 0.321 | 0.263 | 0.221 | 0.239 |
| MoFlow (ours, STEPS-trained) | 0.287 | 0.187 | 0.125 | 0.296 | 0.263 | 0.173 | 0.313 |
| STARS STEPS-only | 0.056 | 0.100 | 0.057 | 0.310 | 0.218 | 0.206 | 0.220 |
| STARS pooled w/o ETH/UCY | 0.054 | 0.074 | 0.046 | 0.254 | 0.175 | 0.087 | 0.140 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.055 | 0.077 | 0.025 | 0.251 | 0.177 | 0.065 | 0.133 |
| STARS pooled (6 datasets) | 0.056 | 0.081 | 0.039 | 0.264 | 0.197 | 0.105 | 0.152 |
| STARS STEPS-only, per-agent diffusion | 0.125 | 0.171 | 0.071 | 0.347 | 0.314 | 0.292 | 0.320 |
| STARS STEPS-only, joint diffusion | 0.106 | 0.164 | 0.086 | 0.344 | 0.310 | 0.279 | 0.304 |

### Our datasets: P(agent within 0.2 m ...)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ground truth (reference) | 0.010 | 0.007 | 0.000 | 0.062 | 0.000 | 0.000 | 0.011 |
| Constant velocity | 0.048 | 0.038 | 0.036 | 0.129 | 0.090 | 0.067 | 0.062 |
| SocialVAE (ours, STEPS-trained) | 0.037 | 0.057 | 0.029 | 0.142 | 0.111 | 0.099 | 0.096 |
| MoFlow (ours, STEPS-trained) | 0.121 | 0.071 | 0.047 | 0.127 | 0.105 | 0.076 | 0.116 |
| STARS STEPS-only | 0.012 | 0.033 | 0.021 | 0.134 | 0.085 | 0.092 | 0.087 |
| STARS pooled w/o ETH/UCY | 0.011 | 0.021 | 0.017 | 0.110 | 0.064 | 0.028 | 0.040 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.011 | 0.021 | 0.009 | 0.109 | 0.066 | 0.018 | 0.035 |
| STARS pooled (6 datasets) | 0.011 | 0.023 | 0.016 | 0.115 | 0.073 | 0.035 | 0.048 |
| STARS STEPS-only, per-agent diffusion | 0.039 | 0.062 | 0.025 | 0.147 | 0.131 | 0.121 | 0.129 |
| STARS STEPS-only, joint diffusion | 0.031 | 0.057 | 0.032 | 0.147 | 0.127 | 0.120 | 0.121 |

### Our datasets: P(scene has a <0.4 m contact)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ground truth (reference) | 0.284 | 0.399 | 0.000 | 0.846 | 0.057 | 0.020 | 0.321 |
| Constant velocity | 0.583 | 0.563 | 0.750 | 0.936 | 0.983 | 0.255 | 0.431 |
| SocialVAE (ours, STEPS-trained) | 0.506 | 0.608 | 0.975 | 0.948 | 0.981 | 0.348 | 0.537 |
| MoFlow (ours, STEPS-trained) | 0.758 | 0.638 | 0.963 | 0.948 | 0.988 | 0.283 | 0.643 |
| STARS STEPS-only | 0.287 | 0.505 | 0.950 | 0.943 | 0.959 | 0.326 | 0.481 |
| STARS pooled w/o ETH/UCY | 0.277 | 0.438 | 0.975 | 0.929 | 0.953 | 0.160 | 0.339 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.278 | 0.443 | 0.775 | 0.930 | 0.950 | 0.127 | 0.331 |
| STARS pooled (6 datasets) | 0.285 | 0.455 | 0.900 | 0.931 | 0.960 | 0.190 | 0.368 |
| STARS STEPS-only, per-agent diffusion | 0.535 | 0.652 | 0.988 | 0.958 | 0.993 | 0.433 | 0.652 |
| STARS STEPS-only, joint diffusion | 0.475 | 0.634 | 1.000 | 0.953 | 0.992 | 0.426 | 0.631 |

### Our datasets: speed W1 to GT (m/s)

| Method | steps | ptdpes | madras | grand_central | atc | eth | ucy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ground truth (reference) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Constant velocity | 0.003 | 0.008 | 0.013 | 0.008 | 0.003 | 0.038 | 0.006 |
| SocialVAE (ours, STEPS-trained) | 0.037 | 0.083 | 0.078 | 0.128 | 0.106 | 0.079 | 0.047 |
| MoFlow (ours, STEPS-trained) | 0.469 | 0.365 | 0.368 | 0.053 | 0.210 | 0.163 | 0.301 |
| STARS STEPS-only | 0.061 | 0.137 | 0.072 | 0.052 | 0.125 | 0.044 | 0.061 |
| STARS pooled w/o ETH/UCY | 0.061 | 0.099 | 0.065 | 0.057 | 0.122 | 0.051 | 0.025 |
| STARS pooled w/o ETH/UCY + oracle goal | 0.060 | 0.097 | 0.085 | 0.059 | 0.122 | 0.067 | 0.027 |
| STARS pooled (6 datasets) | 0.059 | 0.098 | 0.073 | 0.060 | 0.074 | 0.053 | 0.022 |
| STARS STEPS-only, per-agent diffusion | 0.036 | 0.033 | 0.114 | 0.073 | 0.068 | 0.054 | 0.067 |
| STARS STEPS-only, joint diffusion | 0.026 | 0.027 | 0.097 | 0.064 | 0.058 | 0.036 | 0.062 |

### Closed loop (ETH/UCY, 5 scenes pooled, micro-averaged)

| Method | 4.8 s | 8 s | 12 s | 16 s | 20 s |
|---|---:|---:|---:|---:|---:|
| **ADE/FDE (m)** | | | | | |
| agentformer_official@4 | 0.76/1.67 | 1.37/3.08 | 2.20/4.87 | 3.12/6.78 | 3.89/8.08 |
| cv | 0.50/1.10 | 0.94/2.10 | 1.47/3.32 | 1.94/4.32 | 2.54/5.48 |
| ground_truth | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| moflow_official@4 | 0.96/2.03 | 1.85/4.01 | 3.04/6.35 | 4.16/8.09 | 5.14/9.60 |
| sgan_official@4 | 1.11/2.28 | 1.96/4.07 | 2.86/5.74 | 3.56/6.67 | 4.16/7.15 |
| socialvae_official@4 | 0.65/1.39 | 1.17/2.56 | 1.74/3.76 | 2.22/4.75 | 2.73/5.66 |
| stars_loo@4 | 0.55/1.19 | 1.03/2.29 | 1.63/3.55 | 2.17/4.67 | 2.81/5.99 |
| stars_loo_native | 0.75/1.77 | 1.63/3.91 | 2.86/6.68 | 4.16/9.50 | 5.45/12.19 |
| stars_noethucy@4 | 0.55/1.14 | 0.99/2.15 | 1.55/3.38 | 2.04/4.40 | 2.63/5.49 |
| stars_noethucy_goal_native | 0.49/0.99 | 0.88/1.81 | 1.29/2.55 | 1.52/2.84 | 1.77/3.10 |
| stars_noethucy_native | 0.56/1.21 | 1.07/2.38 | 1.66/3.72 | 2.19/4.86 | 2.83/6.09 |
| stars_pooled_native | 0.54/1.20 | 1.04/2.37 | 1.59/3.48 | 1.92/4.06 | 2.39/4.80 |
| stars_steps_native | 0.67/1.45 | 1.20/2.67 | 1.75/3.74 | 2.08/4.25 | 2.55/4.98 |
| trajectron_official@4 | 0.68/1.51 | 1.29/2.98 | 2.11/4.90 | 2.93/6.73 | 3.95/8.97 |
| **agents in collision (<0.4 m)** | | | | | |
| agentformer_official@4 | 0.364 | 0.489 | 0.560 | 0.575 | 0.660 |
| cv | 0.310 | 0.374 | 0.391 | 0.333 | 0.308 |
| ground_truth | 0.198 | 0.267 | 0.305 | 0.325 | 0.331 |
| moflow_official@4 | 0.397 | 0.476 | 0.506 | 0.491 | 0.500 |
| sgan_official@4 | 0.422 | 0.590 | 0.650 | 0.676 | 0.717 |
| socialvae_official@4 | 0.372 | 0.455 | 0.464 | 0.460 | 0.468 |
| stars_loo@4 | 0.263 | 0.343 | 0.358 | 0.335 | 0.321 |
| stars_loo_native | 0.330 | 0.440 | 0.451 | 0.423 | 0.425 |
| stars_noethucy@4 | 0.260 | 0.321 | 0.309 | 0.284 | 0.268 |
| stars_noethucy_goal_native | 0.228 | 0.269 | 0.250 | 0.202 | 0.185 |
| stars_noethucy_native | 0.241 | 0.294 | 0.283 | 0.266 | 0.249 |
| stars_pooled_native | 0.265 | 0.332 | 0.316 | 0.297 | 0.283 |
| stars_steps_native | 0.296 | 0.335 | 0.307 | 0.258 | 0.240 |
| trajectron_official@4 | 0.402 | 0.465 | 0.473 | 0.474 | 0.462 |
| **mean speed (m/s)** | | | | | |
| agentformer_official@4 | 0.579 | 0.510 | 0.485 | 0.477 | 0.473 |
| cv | 0.618 | 0.477 | 0.378 | 0.319 | 0.316 |
| ground_truth | 0.606 | 0.456 | 0.340 | 0.260 | 0.235 |
| moflow_official@4 | 0.817 | 0.779 | 0.778 | 0.774 | 0.791 |
| sgan_official@4 | 0.679 | 0.646 | 0.621 | 0.594 | 0.586 |
| socialvae_official@4 | 0.582 | 0.437 | 0.347 | 0.300 | 0.295 |
| stars_loo@4 | 0.627 | 0.490 | 0.405 | 0.367 | 0.370 |
| stars_loo_native | 0.758 | 0.714 | 0.700 | 0.716 | 0.735 |
| stars_noethucy@4 | 0.597 | 0.456 | 0.360 | 0.310 | 0.308 |
| stars_noethucy_goal_native | 0.538 | 0.359 | 0.225 | 0.151 | 0.127 |
| stars_noethucy_native | 0.626 | 0.493 | 0.393 | 0.338 | 0.336 |
| stars_pooled_native | 0.538 | 0.371 | 0.259 | 0.210 | 0.200 |
| stars_steps_native | 0.572 | 0.403 | 0.279 | 0.215 | 0.208 |
| trajectron_official@4 | 0.660 | 0.554 | 0.502 | 0.493 | 0.523 |
