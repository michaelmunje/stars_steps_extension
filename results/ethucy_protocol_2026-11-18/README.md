# ETH/UCY evaluation protocol for STARS: zero-shot (A) and leave-one-scene-out (B)

Branch `ethucy-protocol`. Dates: 2026-11-18/19. Code: `src/forecast_bench/{finetune_stars_loo,predict,protocol,report}.py`,
`simplenavsim/scripts/ethucy_benchmark.py`, `src/simplenavsim/realism/sim.py`. Scripts as run: `scripts/`.

## Protocols

- **A, zero-shot.** STARS is pretrained on STEPS, PTDPES, MADRAS and Grand Central only
  (`configs/stars_ma/four_no_ethucy.yaml`; 175,538 training windows). There is no ATC and no ETH/UCY
  recording at all (eth, hotel, students001/003, uni_examples, zara01-03, arxiepiskopi). It is evaluated on the
  5 standard ETH/UCY test scenes (Social-GAN / SocialVAE `test` files; 364 / 1197 / 24334 / 2356 / 5910 scored
  agents).
  - Recipe (the stars_improved recipe): CVAE latent 64, decoder 512, KL 0.0799, LR 3e-4 cosine, 60 epochs,
    kNN 24 / 10 m, macro val minADE20 selection.
  - Seeds: 42, 1, 2, 3 (`experiments/stars_ma/runs/ep_pre4_s*`).
- **B, leave-one-scene-out (the protocol of the official baselines).** For each held-out scene, the A checkpoint
  is fine-tuned on the standard `train` files of the other 4 scenes and evaluated on the held-out scene.
  - Sweep per fold and seed: 12 configs (LR 3e-5 / 1e-4 / 3e-4 x encoder-LR multiplier 1 / 0.1 x 10 / 30
    epochs, warmup + cosine, early-stopping patience 8).
  - Selection: the config with the best **minADE20 on that fold's standard `val` files** (other 4 scenes only).
  - Total: 240 runs, `experiments/forecast_bench/loso/`, selection in `loso/selection.csv`.
- **LOSO from scratch (control).** Same architecture, re-initialised, trained on the same fold files.
  - Configs: LR 3e-4 / 1e-3 x 100 / 200 epochs, KL warmup, same val selection.
  - Seeds 0, 1, 2.
- **+ RFT (closed-loop rollout fine-tuning, branch `stars-rollout-ft`).** Applied to seed 1 only, the best
  pretraining validation score.
  - A + RFT: `rft.sh` on the 4 pretraining datasets.
  - B + RFT: each fold's B checkpoint, on that fold's 4 training scenes only, selected on the fold's val split
    by 12-step closed-loop ADE.
- **Two-stage (A', B').** SSL-pretrained encoder, otherwise identical. See [`two_stage.md`](two_stage.md).

The ETH/UCY test scene is never used for training or selection in any of these rows.

**Metrics** (agreed). Horizons are 1.2 / 2.4 / 3.6 / 4.8 s, and every value is a 5-fold mean.
- **Single-sample ADE/FDE:** one random sample, averaged over K = 20 draws.
- **Mode ADE/FDE:** z = prior mean.
- **KDE NLL:** Gaussian KDE (Scott bandwidth) of the 20 samples per agent and step; log density clipped at -20;
  mean over the steps up to the horizon.
- **Collisions:** share of (agent, sample) pairs within 0.2 m of another scored agent's same-sample prediction.
- **Uncertainty:** mean ± std over seeds. Paired comparisons use 95% block-bootstrap CIs (blocks of 50
  consecutive windows of a recording, 2000 reps; `protocol.py paired` / `ci`).
- **Best-of-20 minADE/FDE** is reported only in the "comparison with published methods" table.

**Fairness of the comparison with the official baselines.**
- Same test windows, scored agents and 20 samples.
- B uses the same LOSO data as the baselines.
- A and B additionally use 4 non-ETH/UCY datasets for pretraining, which the baselines do not.

## Open loop: primary metrics

Single-sample ADE / FDE (m), mean ± std over seeds:

| Row | seeds | 1.2 s | 2.4 s | 3.6 s | 4.8 s |
|---|---:|---:|---:|---:|---:|
| **A, zero-shot** | 4 | 0.131±0.002 / 0.206±0.005 | 0.262±0.005 / 0.495±0.009 | 0.415±0.007 / 0.837±0.011 | 0.581±0.008 / 1.201±0.012 |
| **B, LOSO** | 4 | 0.116±0.001 / 0.185±0.002 | 0.245±0.003 / 0.477±0.006 | 0.401±0.004 / 0.838±0.009 | 0.574±0.006 / 1.221±0.010 |
| A + RFT | 1 | 0.131 / 0.204 | 0.258 / 0.483 | 0.406 / 0.815 | 0.568 / 1.174 |
| B + RFT | 1 | 0.111 / 0.177 | 0.234 / 0.458 | 0.385 / 0.806 | 0.552 / 1.175 |
| LOSO from scratch | 3 | 0.134±0.003 / 0.213±0.004 | 0.281±0.004 / 0.545±0.007 | 0.456±0.006 / 0.942±0.012 | 0.649±0.009 / 1.374±0.020 |
| SocialVAE (official) | – | 0.131 / 0.212 | 0.278 / 0.540 | 0.455 / 0.951 | 0.656 / 1.419 |
| Trajectron++ (official) | – | 0.148 / 0.240 | 0.313 / 0.606 | 0.510 / 1.062 | 0.733 / 1.580 |
| Social-GAN (official) | – | 0.201 / 0.312 | 0.394 / 0.734 | 0.616 / 1.234 | 0.863 / 1.794 |
| AgentFormer (official) | – | 0.185 / 0.299 | 0.408 / 0.827 | 0.707 / 1.558 | 1.051 / 2.343 |
| MoFlow (official) | – | 0.198 / 0.311 | 0.417 / 0.818 | 0.689 / 1.444 | 0.985 / 2.094 |
| Constant velocity (deterministic) | – | 0.111 / 0.178 | 0.230 / 0.442 | 0.372 / 0.770 | 0.534 / 1.148 |

Mode ADE / FDE at 4.8 s, KDE NLL at 4.8 s, and collisions < 0.2 m at 4.8 s. "4x" = 4x joint-sample
oversampling, keeping the 20 least-colliding samples. Its accuracy is within 0.006 m of the plain row.

| Row | mode ADE@1.2 | mode ADE / FDE @4.8 | KDE NLL@4.8 | collisions | collisions, 4x |
|---|---:|---:|---:|---:|---:|
| A | 0.113±0.001 | 0.467±0.002 / 0.979±0.006 | 5.76±0.30 | 5.0±0.2% | 3.4±0.2% |
| B | 0.105±0.001 | 0.455±0.001 / 0.960±0.005 | 4.16±0.70 | 5.5±0.1% | 3.7±0.1% |
| A + RFT (seed 1) | 0.118 | 0.477 / 0.997 | 7.15 | 4.8% | 3.2% |
| B + RFT (seed 1) | 0.104 | 0.456 / 0.964 | 5.57 | 5.5% | 3.7% |
| LOSO from scratch | 0.121±0.002 | 0.515±0.004 / 1.071±0.015 | 2.62±0.05 | 8.3±0.4% | 6.1±0.1% |
| SocialVAE | – | – | 1.46 | 9.7% | |
| MoFlow | – | – | 1.05 | 7.6% | |
| AgentFormer | – | – | 1.23 | 7.4% | |
| Trajectron++ | – | – | 3.22 | 10.3% | |
| Social-GAN | – | – | 3.32 | 10.2% | |
| Constant velocity | 0.111 | 0.534 / 1.148 | – | 6.9% | |
| Ground truth | | | | 0.5% | |

Per fold, B at 4.8 s (single-sample ADE/FDE; mode ADE; KDE NLL; collisions):

| eth | hotel | univ | zara1 | zara2 |
|---|---|---|---|---|
| 1.08/2.25; 0.91; 6.81; 3.3% | 0.32/0.62; 0.23; 6.95; 1.2% | 0.64/1.40; 0.49; 1.90; 16.9% | 0.48/1.05; 0.37; 2.44; 2.0% | 0.35/0.79; 0.28; 2.69; 4.1% |

The baselines per fold, and A per fold, are in `two_stage_tables.md`. The data files are `primary_per_fold.csv`
and `primary.csv`.

### What the open-loop data supports

The paired CIs are in `two_stage_paired.csv` (seeds 1-3) and the 4.8 s numbers below come from there.
- **B vs the learned baselines.** B has lower single-sample ADE/FDE than every official baseline at every
  horizon.
  - vs SocialVAE, the best of them: -0.079 m [-0.107, -0.048] ADE at 4.8 s, and -0.014 [-0.018, -0.011] at 1.2 s.
  - B also has fewer collisions (5.5% vs 7.4-10.3%).
- **B vs constant velocity.** CV beats B on single-sample ADE: +0.043 [+0.026, +0.064] at 4.8 s, and already
  +0.006 at 1.2 s.
  - B's *mode* prediction beats CV at every horizon on the 5-fold mean (0.455 vs 0.534 at 4.8 s).
  - Per fold, B's mode is better on eth (0.91 vs 1.08), hotel (0.23 vs 0.32), univ (0.49 vs 0.52), zara1
    (0.37 vs 0.43) and zara2 (0.28 vs 0.32).
- **KDE NLL is STARS's weak point.**
  - A, B and RFT have a 4.8 s KDE NLL of 4-7, against 1.0-1.5 for MoFlow, AgentFormer and SocialVAE.
  - For A (seed 42), 16% of agent-steps on eth and 20% on zara1 hit the -20 log-density clip at 4.8 s.
  - The 20 samples are narrower than the error: median sample spread at 4.8 s on zara1 is 0.33 m vs 1.47 m for
    MoFlow.
  - LOSO fine-tuning lowers NLL (B vs A: -1.8 nats [-2.2, -1.4], 3/3 seeds), but it stays above the baselines.
- **B vs A.**
  - Lower: mode ADE at every horizon (4.8 s: -0.011 [-0.020, -0.004]) and single-sample ADE up to 3.6 s.
  - Single-sample ADE at 4.8 s: no significant difference.
  - Higher: single-sample FDE at 4.8 s (+0.025 [+0.011, +0.039]) and collisions (+0.5 points).
  - Per fold, LOSO mainly helps univ, zara1 and zara2 in NLL, and makes hotel worse in ADE (0.24 to 0.32,
    mean over seeds).
- **Pretraining (B) vs LOSO from scratch.** B is better on all accuracy metrics (single-sample ADE@4.8 -0.072
  [-0.094, -0.043]; mode ADE -0.060) and on collisions (-2.8 points). It is worse on KDE NLL (+1.3 nats).
- **RFT, seed 1 only, no CIs.** B + RFT has the lowest single-sample ADE of our rows (0.552 at 4.8 s) but a
  higher NLL than B. A + RFT has the highest NLL (7.15).
- **Two-stage training (A', B').** See `two_stage.md`. In short: a small gain zero-shot at short horizons and in
  NLL, no consistent gain after LOSO.

### Comparison with published methods (per-agent best-of-20, the papers' metric; 95% block-bootstrap CI)

| Row | minADE / minFDE (5-fold mean) | 95% CI (minADE) | scene-level best-of-20 (minJointADE) |
|---|---|---|---|
| A, zero-shot (4 seeds) | 0.255 / 0.477 | [0.228, 0.295] | 0.344 |
| **B, LOSO (4 seeds)** | 0.241 / 0.451 | [0.213, 0.284] | 0.340 |
| B' two-stage + LOSO (3 seeds) | 0.239 / 0.439 | [0.215, 0.274] | 0.339 |
| LOSO from scratch (3 seeds) | 0.275 / 0.492 | [0.250, 0.310] | 0.395 |
| MoFlow (official) | 0.217 / 0.341 | [0.200, 0.233] | 0.385 |
| AgentFormer (official) | 0.233 / 0.393 | [0.216, 0.249] | 0.384 |
| SocialVAE (official) | 0.251 / 0.442 | [0.230, 0.278] | 0.382 |
| Trajectron++ (official, observed-only inputs) | 0.343 / 0.606 | [0.311, 0.386] | 0.463 |
| Social-GAN (official) | 0.360 / 0.641 | [0.327, 0.388] | 0.481 |
| [LEAKED] STARS improved (trained on hotel/univ/zara1) | 0.232 / 0.425 | – | 0.321 |

Paired differences, B minus baseline, 95% block-bootstrap CIs (`ci_paired.csv`):
- **Per-agent minADE: no significant difference** from SocialVAE (-0.009 [-0.021, +0.009]), AgentFormer (+0.008
  [-0.015, +0.041]) or MoFlow (+0.024 [-0.002, +0.059]).
- Per-agent minFDE: B is behind MoFlow (unpaired CIs do not overlap).
- **Scene-level best-of-20 (minJointADE): B is better than every baseline**, e.g. SocialVAE -0.042 [-0.055, -0.030]
  and MoFlow -0.045 [-0.075, -0.016].
- Per-fold best-of-20 values are in `ci.csv`.

## Closed loop (ETH/UCY realism simulator, `results/ethucy_2026-11-16_improved`)

Setup:
- 549 windows over the 5 scenes (stride 10). STARS re-forecasts every 0.4 s and executes the first step.
- Each window uses the checkpoint that never saw its scene (B: per-scene LOSO checkpoint).
- Pretraining seed 1, 5 rollout seeds, 4x rollout oversampling (least-colliding of 4) for the sampled rows.
- Micro ADE over agents, with 95% block-bootstrap CIs (20 s blocks).
- † = reads the oracle goal (privileged).

| Method | ADE 4.8 s | ADE 8 s | ADE 12 s | FDE 12 s | contact < 0.2 m (8 s) | agents in collision (8 s) |
|---|---|---|---|---|---|---|
| A | 0.559 [0.526, 0.593] | 1.005 [0.931, 1.083] | 1.501 [1.382, 1.629] | 3.26 | 4.5% | 22.4% |
| A, mode | 0.496 [0.464, 0.532] | 0.881 [0.809, 0.959] | 1.319 [1.204, 1.443] | 2.88 | 5.4% | 23.6% |
| B | 0.697 [0.669, 0.724] | 1.491 [1.403, 1.569] | 2.654 [2.444, 2.840] | 6.41 | 11.8% | 40.3% |
| B, mode | 0.621 [0.595, 0.647] | 1.315 [1.237, 1.388] | 2.361 [2.176, 2.533] | 5.84 | 13.3% | 42.0% |
| **A + RFT** | 0.492 [0.461, 0.527] | **0.876 [0.807, 0.954]** | 1.336 [1.223, 1.459] | 2.92 | 6.1% | 21.8% |
| **B + RFT** | **0.489 [0.464, 0.515]** | 0.953 [0.889, 1.016] | 1.587 [1.467, 1.700] | 3.60 | 8.3% | 26.1% |
| Constant velocity | 0.497 [0.464, 0.533] | 0.907 [0.832, 0.987] | 1.428 [1.304, 1.558] | 3.27 | 14.6% | 38.4% |
| ORCA (calibrated) † | 0.559 [0.519, 0.603] | 0.896 [0.817, 0.980] | **1.235 [1.113, 1.365]** | 2.03 | 0.0% | 2.1% |
| Social force (calibrated) † | 0.574 [0.538, 0.612] | 0.968 [0.892, 1.048] | 1.354 [1.232, 1.479] | 2.26 | 1.1% | 15.8% |

Realism at 8 s: Wasserstein-1 distance to the recorded distributions (lower is better) and mean speed.

| Method | mean speed (GT 0.466 m/s) | W1 speed | W1 accel | W1 turning (deg/s) | W1 NN distance |
|---|---:|---:|---:|---:|---:|
| A | 0.398 | 0.074 | 0.068 | 5.06 | 0.062 |
| B | **0.706** | 0.239 | 0.121 | 8.89 | 0.125 |
| A + RFT | 0.421 | 0.047 | 0.148 | 9.95 | 0.036 |
| B + RFT | 0.530 | 0.073 | 0.143 | 10.17 | 0.034 |
| Constant velocity | 0.481 | 0.017 | 0.171 | 13.17 | 0.146 |
| ORCA † | 0.437 | 0.033 | 0.118 | 9.28 | 0.062 |
| Social force † | 0.448 | 0.051 | 0.080 | 2.55 | 0.238 |

### B's closed-loop degradation is real, and what the data shows about it

- **The gap is large.** B's closed-loop ADE is 0.697 [0.669, 0.724] at 4.8 s vs A's 0.559 [0.526, 0.593], with
  non-overlapping CIs. The gap grows with the horizon: 2.65 vs 1.50 m at 12 s. It holds for every rollout seed
  (0.690-0.709 vs 0.553-0.563).
- **It is a closed-loop effect.** In open loop, B's predicted speed matches ground truth on every fold (seed 1,
  4.8 s mean speed):

  | | eth | hotel | univ | zara1 | zara2 |
  |---|---|---|---|---|---|
  | B | 0.97 | 0.46 | 0.55 | 0.95 | 0.53 |
  | GT | 0.88 | 0.46 | 0.54 | 0.97 | 0.54 |

- **In closed loop, B's agents speed up.** Mean speed over 6.4-8 s of the rollout (rollout seed 0):

  | | eth | hotel | univ | zara1 | zara2 |
  |---|---|---|---|---|---|
  | B | 0.80 | 0.23 | **0.95** | 1.02 | **0.54** |
  | GT | 0.60 | 0.23 | 0.46 | 0.94 | 0.35 |
  | A | 0.63 | 0.19 | 0.33 | 0.83 | 0.28 |

  Pooled over all scenes at 8 s, B's mean speed is 0.706 m/s against 0.466 for GT.
- **Where it shows.** Closed-loop ADE at 4.8 s:

  | | eth | hotel | univ | zara1 | zara2 |
  |---|---|---|---|---|---|
  | B | 0.64 | 0.33 | 0.80 | 0.55 | 0.43 |
  | A | 0.60 | 0.25 | 0.60 | 0.59 | 0.42 |

  B is worse on univ and hotel, and level on the other three scenes. Contacts and agents in collision roughly
  double (40% vs 22% in collision at 8 s), consistent with the faster agents in the densest scene (univ).
- **What these runs do not establish.** They do not show why LOSO fine-tuning causes the speed-up under its own
  rollouts. Only the seed-1 B checkpoints were run in closed loop.
- **RFT on B removes it.** B + RFT has mean speed 0.530, W1 speed 0.073, and ADE 0.489 [0.464, 0.515] at 4.8 s.
  That is level with CV and A + RFT at 4.8 s. At 8 and 12 s it is worse than A + RFT (0.953 vs 0.876; 1.587 vs
  1.336).
- **Closed-loop ranking.**
  - A + RFT and A mode are the best STARS rows at 8 and 12 s, with overlapping CIs. Both are within CI of CV.
  - At 12 s, ORCA † has the lowest ADE.
  - All STARS rows have far more contacts than ORCA and social force (both use the oracle goal), but fewer than
    CV.
  - RFT raises turning and acceleration W1 (straighter, smoother rollouts), as seen on branch `stars-rollout-ft`.

## Files

- `primary.csv`, `primary_per_fold.csv`: agreed metrics, every row / fold / seed.
- `ci.csv`, `ci_paired.csv`: best-of-20 and older metrics with CIs.
- `two_stage.md`, `two_stage_tables.md`, `two_stage_paired.csv`: two-stage study, all horizons, paired CIs.
- Closed loop: `../ethucy_2026-11-16_improved/{report.md, metrics/aggregate.csv, metrics/per_scene.csv, metrics/kinematics.csv}`.
- Benchmark report with the new rows: `../forecasting_2026-11-19_ethucy_protocol/` (report.md, tables.md, figures).
- Checkpoints, all gitignored:
  - `experiments/stars_ma/runs/{ep_pre4_s*, ep_ts_pre4_s*, ep_tsf_pre4_s1, ep_rft_zs_s1}`;
  - `experiments/forecast_bench/loso/{sweep, ep_<kind>_s<seed>/<fold>}`;
  - predictions in `experiments/forecast_bench/preds/ethucy_<fold>/ep_*.npz`.

## Commands

```bash
# A: pretraining (resumable; 30-min Slurm jobs via scripts/node1h.slurm + keeper.sh + worker.sh)
bash scripts/pretrain.sh <seed> 60
# B: LOSO sweep, then per-fold selection on val minADE20
bash scripts/queue_sweep.sh <seed>        # -> loso.sh loso <seed> <fold> lr<LR>_enc<M>_e<E>
python -m forecast_bench.protocol select
# scratch control / B + RFT
bash scripts/loso.sh scratch <seed> <fold> lr1e-3_e100
bash scripts/loso.sh loso_rft 1 <fold> lr5e-5_e12
bash scripts/rft.sh experiments/stars_ma/runs/ep_pre4_s1/checkpoints/best.pt experiments/stars_ma/runs/ep_rft_zs_s1 --config configs/stars_ma/four_no_ethucy.yaml
# predictions (ep_<zs|loso|scratch|zs_rft|loso_rft|zs_ts|loso_ts>_s<seed>[_mode|_os4])
python -m forecast_bench.predict --method ep_loso_s1 --sets ethucy_eth,ethucy_hotel,ethucy_univ,ethucy_zara1,ethucy_zara2
# metrics
python -m forecast_bench.protocol primary --out results/ethucy_protocol_2026-11-18
python -m forecast_bench.protocol paired --out results/ethucy_protocol_2026-11-18
python -m forecast_bench.protocol ci --out results/ethucy_protocol_2026-11-18 --methods ...
python -m forecast_bench.report --out results/forecasting_2026-11-19_ethucy_protocol
# closed loop (per-scene LOSO checkpoints; EP_SEED = pretraining seed)
EP_SEED=1 python simplenavsim/scripts/ethucy_benchmark.py run --run results/ethucy_2026-11-16_improved --append --device cuda \
  --seeds 5 --methods stars_ep_zs stars_ep_zs_mode stars_ep_loso stars_ep_loso_mode stars_ep_zs_rft stars_ep_loso_rft
EP_SEED=1 python simplenavsim/scripts/ethucy_benchmark.py aggregate --run results/ethucy_2026-11-16_improved
EP_SEED=1 python simplenavsim/scripts/ethucy_benchmark.py report --run results/ethucy_2026-11-16_improved
```

Compute: about 9 node-hours (~36 GPU-hours). That covers pretraining (4 seeds plus 3 two-stage seeds and 1
frozen), 420 LOSO runs, the scratch control, RFT, predictions and the closed loop. It excludes the SSL encoders
(about 4.2 GPU-hours, branch `stars-ablation`).
