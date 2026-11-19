# Two-stage (self-supervised encoder, then forecasting) vs end-to-end STARS on the ETH/UCY protocol

Branch `ethucy-protocol`, 2026-11-19. Question: does initialising the STARS encoder from its self-supervised
(SSL) objective help, compared with training the forecaster end to end from scratch? All else is matched.

## Setup

| | End-to-end (A, B) | Two-stage (A', B') |
|---|---|---|
| Stage 1 | none | STARS SSL encoder `E3_s{1,2,3}_long` (branch `stars-ablation`): per-window multi-agent graph, native 8 steps, 24-NN / 10 m, same heading rule, same 4 datasets (STEPS, PTDPES, MADRAS, Grand Central train splits), 15 + 25 = 40 epochs, best epoch 23-24 of the continuation |
| Protocol A / A' (zero-shot) | `ep_pre4_s{42,1,2,3}`: CVAE (latent 64, decoder 512), KL 0.0799, LR 3e-4 cosine, 60 epochs, macro val selection, `configs/stars_ma/four_no_ethucy.yaml` | `ep_ts_pre4_s{1,2,3}`: identical, plus `--encoder-checkpoint E3_s<s>_long --encoder-mode finetune` (encoder LR = 0.1 x the decoder LR, i.e. 3e-5); seed s pairs SSL seed s with forecaster seed s. Frozen-encoder variant `ep_tsf_pre4_s1` (seed 1 only) |
| Protocol B / B' (LOSO) | the same 12-config sweep (LR 3e-5 / 1e-4 / 3e-4 x encoder-LR multiplier 1 / 0.1 x 10 / 30 epochs, cosine, patience 8) per fold, from each A checkpoint | the same sweep from each A' checkpoint |
| LOSO from scratch | seeds 0, 1, 2: same architecture re-initialised, LR 3e-4 / 1e-3 x 100 / 200 epochs, per-fold selection | – |

- **Encoder dropout:** the SSL checkpoint's recipe has encoder dropout 0.2. A' uses the forecasting recipe's
  0.0905, the same as A (`train_stars_ma.py` now lets `--dropout` override it, since dropout is not a weight).
- **Selection:** the per-fold B and B' configurations were selected on the fold's val split by **minADE20**
  (best of 20), not by single-sample ADE. The held-out scene is never used.
- **Pretraining validation** (macro minADE20 over the 4 datasets' val splits, selection score):
  - A: 0.2013 / 0.2028 / 0.2022 (seeds 1 / 2 / 3).
  - A': 0.2037 / 0.2041 / 0.2041.
  - A' frozen: 0.2205 (seed 1).
- **Extra compute of two-stage:** 40 SSL epochs at 122-128 s per epoch, i.e. about 1.4 GPU-hours per seed (4.2
  for 3 seeds). The A forecaster itself took 60 epochs at 136-232 s (2.3-3.9 GPU-hours) per seed, and A' took
  145-185 s per epoch, about the same. The epoch times vary with how many runs share a GPU. Two-stage
  therefore costs about 40-60% more GPU time per seed.

**Metrics** (agreed protocol; the 5 standard Social-GAN / SocialVAE test scenes; scored agents 364 / 1197 /
24334 / 2356 / 5910; 5-fold mean):
- single-sample ADE/FDE: one random sample, averaged over the K = 20 draws;
- mode ADE/FDE: z = prior mean;
- KDE NLL: Gaussian KDE with Scott bandwidth of the 20 samples per agent and step, NLL of the truth, log
  density clipped at -20 as in Trajectron++, mean over the steps up to the horizon;
- predicted collisions: share of (agent, sample) pairs with another scored agent's same-sample prediction
  within 0.2 m.

Rows give mean ± std over seeds. Paired differences are 95% block-bootstrap CIs: blocks of 50 consecutive
window starts of one recording, resampled within each fold, 2000 reps, with the same blocks for both models.
Each side is the per-agent mean over its listed seeds. **These CIs cover test-set (window) variability, not
seed variability**, so the per-seed differences are listed as well. "ns" means the CI covers 0.

## Findings (only what the CIs and seeds support)

1. **A' vs A (zero-shot, seeds 1-3 paired):**
   - **Short horizons:** two-stage is slightly more accurate at short horizons. Single-sample ADE@1.2 s is
     -0.004 m [-0.005, -0.003], about 3% of 0.132, and lower in all 3 seed pairs (-0.005, -0.005, -0.002).
     Mode ADE@1.2 s is -0.004 [-0.005, -0.003], also 3/3 seeds.
   - **4.8 s:** single-sample ADE is -0.009 [-0.014, -0.004], but the per-seed differences are mixed
     (-0.011, -0.021, +0.004), and the seed std of either model (~0.009) is the same size. **Mode ADE/FDE at
     4.8 s: no significant difference.**
   - **KDE NLL:** lower for A' at every horizon (4.8 s: -0.67 nats [-0.88, -0.46]; 3/3 seeds lower), but both
     stay far above the baselines (5.0 vs about 1-1.5).
   - **Collisions@0.2 m:** no significant difference at 4.8 s. Slightly fewer at 2.4-3.6 s (about -0.1
     percentage points).
2. **B' vs B (LOSO, seeds 1-3 paired):**
   - **Single-sample ADE/FDE:** no significant difference at any horizon after 1.2 s. At 1.2 s, B' is +0.001 m
     worse.
   - **Mode ADE:** slightly worse for B' (4.8 s: +0.007 [+0.000, +0.014]; 3/3 seeds worse, +0.003 to +0.011).
   - **KDE NLL:** lower for B' at 1.2, 3.6 and 4.8 s (4.8 s: -0.18 [-0.27, -0.08]). Per seed, however, it is
     +0.26, +0.39 and -1.18, so this is not consistent across seeds.
   - **Collisions:** no significant difference.
   - **Overall:** after LOSO fine-tuning, the SSL initialisation gives no consistent advantage, and a small
     mode-ADE disadvantage.
3. **Frozen SSL encoder (A' frozen vs A, seed 1):**
   - **Accuracy:** clearly worse. Single-sample ADE@4.8 is +0.050 [+0.043, +0.056], mode ADE +0.022, and
     collisions +1.6 points.
   - **KDE NLL:** much lower (-2.4 nats), because its samples are more spread out.
   - **Caveat:** one seed only.
4. **Pretraining vs LOSO from scratch (protocol context, both B and B' vs 3 scratch seeds):**
   - **Accuracy:** the pretrained models are clearly better at every horizon (B: single-sample ADE@4.8 -0.072
     [-0.094, -0.043], mode ADE -0.060, collisions -2.8 points).
   - **KDE NLL:** clearly worse than scratch (+1.3 nats). Scratch samples are wider.
5. **B vs A:**
   - LOSO fine-tuning lowers mode ADE (4.8 s: -0.011 [-0.020, -0.004]; 3/3 seeds) and KDE NLL (-1.8 nats;
     3/3 seeds).
   - Single-sample ADE@4.8: no significant difference.
   - Single-sample FDE@4.8 rises (+0.025 [+0.011, +0.039]), and collisions rise by +0.5 points.

**Related evidence** (separate experiment, not re-tested here): the ablation ladder on branch `stars-ablation`
found no accuracy gain from SSL pretraining when the decoder sees only ETH/UCY. The two results agree in
direction, but the setups differ (there the decoder was trained on ETH/UCY only; here the forecaster is
pretrained on the 4 datasets).

**Bottom line:**
- On this protocol, two-stage training buys a small, seed-consistent gain only at short horizons and in KDE NLL
  zero-shot (A').
- It gives no consistent gain after LOSO fine-tuning (B'), where its mode ADE is slightly worse.
- It costs about 40-60% more GPU time.
- A frozen SSL encoder is clearly worse for accuracy.

## Tables

Full tables (all horizons, every row, per fold, paired CIs): [`two_stage_tables.md`](two_stage_tables.md).
Data:
- `primary_per_fold.csv`: per row, fold and seed.
- `primary.csv`: 5-fold mean ± std over seeds.
- `two_stage_paired.csv`: every paired difference and its CI at every horizon.

Summary at 4.8 s (mean ± std over seeds):

| Row | seeds | single-sample ADE / FDE | mode ADE / FDE | KDE NLL | collisions < 0.2 m |
|---|---:|---|---|---:|---:|
| A, end-to-end | 1-3 | 0.581 ± 0.010 / 1.200 ± 0.015 | 0.467 ± 0.002 / 0.976 ± 0.004 | 5.72 ± 0.35 | 5.0% |
| A', two-stage | 1-3 | 0.572 ± 0.009 / 1.189 ± 0.015 | 0.464 ± 0.001 / 0.980 ± 0.003 | 5.05 ± 0.29 | 4.9% |
| A', frozen encoder | 1 | 0.625 / 1.288 | 0.489 / 1.017 | 2.99 | 6.3% |
| B, end-to-end + LOSO | 1-3 | 0.577 ± 0.003 / 1.225 ± 0.006 | 0.455 ± 0.001 / 0.959 ± 0.006 | 3.92 ± 0.62 | 5.5% |
| B', two-stage + LOSO | 1-3 | 0.580 ± 0.007 / 1.234 ± 0.016 | 0.462 ± 0.005 / 0.974 ± 0.009 | 3.74 ± 0.31 | 5.5% |
| LOSO from scratch | 0-2 | 0.649 ± 0.009 / 1.374 ± 0.020 | 0.515 ± 0.004 / 1.071 ± 0.015 | 2.62 ± 0.05 | 8.3% |
| SocialVAE (official) | – | 0.656 / 1.419 | – | 1.46 | 9.7% |
| MoFlow (official) | – | 0.985 / 2.094 | – | 1.05 | 7.6% |
| Constant velocity | – | 0.534 / 1.148 | 0.534 / 1.148 | – | 6.9% |
| Ground truth | – | | | | 0.5% |

The A and B rows over all 4 seeds (42, 1, 2, 3) are in `two_stage_tables.md`.

## Commands

```bash
bash runs/ethucy_protocol/pretrain_ts.sh <seed> finetune   # A' (frozen: ... <seed> frozen)
bash runs/ethucy_protocol/queue_sweep.sh <seed> 0 loso_ts  # B' sweep; then: python -m forecast_bench.protocol select
bash runs/ethucy_protocol/predict.sh ep_zs_ts_s<seed> ep_zs_ts_s<seed>_mode ep_loso_ts_s<seed> ep_loso_ts_s<seed>_mode
python -m forecast_bench.protocol primary --out results/ethucy_protocol_2026-11-18
python -m forecast_bench.protocol paired --out results/ethucy_protocol_2026-11-18
python results/ethucy_protocol_2026-11-18/scripts/two_stage_tables.py > results/ethucy_protocol_2026-11-18/two_stage_tables.md
```
(scripts copied under `results/ethucy_protocol_2026-11-18/scripts/`; the `runs/` copies are what the queue ran.)
