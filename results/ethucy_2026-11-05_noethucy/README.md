# Goal-conditioned STARS trained without ETH/UCY (2026-11-06)

Two CVAE forecasters with the pooled recipe of `cvae_all_lr3e-4_seed42` (scratch encoder, 60 epochs,
LR 3e-4 cosine, KL 0.0799, latent 64, decoder 512, kNN 24 / 10 m, seed 42, macro-ADE selection):

| run | data | goal input |
|---|---|---|
| `cvae_noethucy_nogoal` (`stars_noethucy`) | `configs/stars_ma/all_no_ethucy.yaml` | none (control) |
| `cvae_noethucy_goal` (`stars_noethucy_goal`) | same | yes |

**Data:** STEPS, PTDPES, MADRAS, Grand Central (temporal split) and the ATC subset, with ATC on a
held-out recording (2013-09-29 = test) and capped at 80k/10k/50k windows. There are 225,072 training
windows and 25,417 validation windows. No ETH or UCY data was used for training or for checkpoint
selection, so all five benchmark scenes are zero-shot.

**Goal:** for each agent, the last observed position of its track in the recording (an oracle goal,
the same rule as the benchmark). It is given relative to the agent's current position, in the agent's
local frame, as `[dx, dy, distance] / 10 m` plus a `valid` flag. It is concatenated to the decoder
context, so the CVAE prior, the posterior and the decoder all see it.
- Every agent in every window has a valid goal.
- Some tracks end inside the 12-step future, so the goal equals the last supervised position:
  1.6% of training targets and 11.5% of all training agents. These windows are kept.
- The Grand Central temporal split: a train-window goal can lie in time after the split boundary. It is
  the same track's own end, never another split's agents.

**Training:** both runs on one GB200 node (job 83626): about 3.2 h each (about 175 s per epoch).
- Best epoch: 48 with goals, 45 without.
- After training, `evaluate_stars_ma` took about 4.3 h.

## Forecasting test metrics (5 training datasets, best-of-20, agent ADE / FDE in m)

| | all | STEPS | PTDPES | MADRAS | Grand Central | ATC |
|---|---|---|---|---|---|---|
| control | 0.249 / 0.387 | 0.079 / 0.125 | 0.283 / 0.411 | 0.213 / 0.323 | 0.241 / 0.399 | 0.297 / 0.416 |
| goal-conditioned | **0.222 / 0.306** | **0.072 / 0.105** | **0.261 / 0.347** | **0.180 / 0.225** | **0.208 / 0.300** | **0.278 / 0.357** |

- Joint ADE / FDE: 0.445 / 0.834 for the control and 0.376 / 0.646 with goals.
- Closed-loop agent ADE, executing 4 steps per round: 0.253 for the control and 0.229 with goals.
- Regression: in the 20.8 s free simulation the goal model's agents are slower (mean speed 0.63 m/s,
  against 0.81 for the control and 0.90 in the real data). Agents near their track end stop.
- Files: `forecasting/*.json`.

## ETH/UCY benchmark rows ([`plots/goal_conditioning.png`](plots/goal_conditioning.png), `report.md`)

Same windows, horizons, five seeds and blocked-bootstrap CIs as `ethucy_2026-11-05_4981b49`. The
earlier rows were copied over and checked by window hash; the new rows were added with `--append`.
The adapter weight for `stars_noethucy_goal_adapter` (0.2) was calibrated on the benchmark's
calibration recordings, as for the other adapters. Micro ADE in m:

| | 4.8 s | 8 s | 12 s |
|---|---|---|---|
| `stars_noethucy` | 0.556 [0.522, 0.591] | 1.033 [0.953, 1.113] | 1.634 [1.497, 1.769] |
| `stars_noethucy_goal` | 0.486 [0.456, 0.517] | 0.870 [0.801, 0.942] | 1.282 [1.163, 1.400] |
| `stars_noethucy_goal_adapter` | 0.481 [0.450, 0.511] | 0.837 [0.772, 0.904] | 1.220 [1.111, 1.328] |
| `stars_pooled_goal` (in-domain) | 0.476 [0.447, 0.506] | 0.837 [0.774, 0.906] | 1.185 [1.070, 1.303] |
| `orca` | 0.559 [0.519, 0.603] | 0.896 [0.817, 0.980] | 1.235 [1.113, 1.365] |

- **Goal input against the control:** ADE is lower by 0.07, 0.16 and 0.35 m at 4.8, 8 and 12 s,
  and every CI of the difference is below 0.
- **Against the STEPS checkpoint with the goal adapter:** lower at 4.8 s (by 0.045 m). At 8 and 12 s
  the CIs include 0.
- **Against `cv`:** lower at 8 and 12 s.
- **Against `orca`:** ADE is lower at 4.8 s (by 0.07 m) and level at 8 and 12 s. FDE is still higher
  at long horizons (2.53 against 2.03 m at 12 s).
- **The adapter on top of the goal model** helps a little more at 8 and 12 s.
- **Colliding pair-time** rises with goals (0.71% against 0.63% at 12 s, overlapping CIs) and
  stays far above SFM and ORCA.

Reproduce: `scripts/train_eval.sh {goal,nogoal}`, `scripts/benchmark.sh`, `scripts/plot.py` (from the
repository root). No preference clips were exported for the new rows.
