# Multi-agent STARS on STEPS

STARS extended from a single-target representation probe into a multi-agent pedestrian dynamics
model: pedestrian history -> STARS graph encoder -> temporal dynamics decoder -> every
pedestrian's future, rolled out closed-loop for simulation. The original single-target STARS
probe (`baselines.train_stars`, `baselines.stars`) is unchanged and remains the baseline.

| Module | Contents |
|---|---|
| `src/baselines/stars_ma_graph.py` | Torch graph builder: node selection, local-frame node motion features, STARS SE(2) edge features, kNN/radius topology; differentiable in positions |
| `src/baselines/stars_ma_model.py` | `StarsMotionEncoder`, phase-1 `StarsMotionVAE`, `TrajectoryDecoder` (deterministic / CVAE), `StarsMultiAgentForecaster`, closed-loop `rollout` |
| `src/baselines/stars_ma_data.py` | `StarsMultiAgentDataset`: one graph per scene window over the shared `Scene` samples |
| `src/baselines/stars_ma_batch.py` | Training data path: cached `WindowStore` per split (GPU-resident), batched on-device graph construction (`build_batch`, `GraphBatch.rebuild` for rollouts), node-budget batching |
| `src/baselines/stars_ma_metrics.py` | Best-of-K ADE/FDE (target / agent / joint), interacting vs isolated, density bins, collisions, closest approach, diversity, motion distributions |
| `src/baselines/train_stars_ma.py` | `--phase ssl` and `--phase forecast` (frozen / finetune / scratch, deterministic / CVAE, multi-step rollout loss); dataset configs, DDP, bf16, resume, `--benchmark` |
| `configs/stars_ma/*.yaml` | Dataset lists: `steps`, `ptdpes`, `all` (every converted dataset), `steps_plus_ptdpes` (weighted mixture) |
| `src/baselines/evaluate_stars_ma.py` | Teacher-forced, closed-loop and long-horizon simulation report (JSON) |
| `tests/test_stars_ma.py` | Contracts below |

## Design

**Graph (one per scene window).** Nodes are pedestrians visible at the last observed step and
seen at least twice in the 8 observed steps (2.5 Hz, 3.2 s); all of them are forecast. No
resampling to STARS's 40-step grid: sequences are 8 steps long.

- *Node features* `[N, 8, 7]`, in each agent's own frame (origin at its last observed position, x
  along its last heading): position, velocity (m/s), speed, cos/sin of heading relative to the
  current heading. Invariant to the scene frame.
- *Edge features* `[E, 8, 8]`: STARS's SE(2) relative pose of destination in source frame plus its
  per-step change (relative motion), unchanged except for the heading fix below.
- *Topology*: each destination keeps its `--knn 16` nearest sources within `--radius 10` m at the
  last observed step; `--fully-connected` restores STARS's topology. `steps` windows have a median
  ~18 / p90 ~28 / max ~37 nodes (validation split), so full graphs carry a few hundred edges, most
  of them between distant pedestrians.
- *Headings*: direction of travel as in STARS, but steps before an agent's first movement take that
  movement's direction instead of 0 (the scene x axis). STARS's rule made the first steps'
  features depend on the scene frame; the tests check the new features are SE(2)-invariant.
- *Robot slot*: node type 0 is reserved for a robot; pedestrians are type 1.

**Encoder.** The pinned upstream `CombinedHGNNEncoder` modules, unmodified: type embedding,
`TemporalEdgeTransformer` over edges, two `TransformerConv` rounds, `to_mu`/`to_logvar`. The new
node branch is a second `TemporalEdgeTransformer` over node features added through a bias-free
projection -- the same mechanism STARS uses for SEAN robot features. `--edge-only` removes it; the
tests check that the edge-only encoder equals upstream STARS exactly on 40-step graphs. Both
temporal transformers have their positional embedding resized to 8 steps.

**Phase 1 (`--phase ssl`).** STARS's VAE objective at the native length: valid-masked edge
reconstruction + `beta` KL (STARS warm-up), plus valid-masked node-motion reconstruction
(`--node-loss-weight`). STARS's optimiser recipe.

**Decoder.** Per agent: `[representation ‖ current motion state ‖ z]` -> one social
`TransformerConv` round over the same graph (reusing encoder edge embeddings) -> GRU emitting
per-step displacements in the agent's local frame; positions are their cumulative sum, mapped back
to the scene frame. Permutation-equivariant and size-agnostic (tested).

- *Deterministic*: no `z`.
- *CVAE* (`--decoder cvae`): learnt prior `p(z | context)` and posterior `q(z | context, future)`
  per agent, trained with reconstruction + KL (warm-up). This is a new latent for future
  diversity, not the representation VAE's latent. `z` joins the context before the social round,
  so a joint draw of all agents' `z` decodes into mutually consistent futures.

**Closed loop (`rollout`).** Encode scene -> predict 12 steps -> execute the first `exec_steps` for
every agent -> shift the 8-step window -> rebuild graph -> repeat. A CVAE agent's `z` is its intent:
drawn once and held across rounds. The agent set is closed (no entries or exits). With
`--rollout-exec-steps s` training adds a multi-step loss: the model is rolled out over the 12-step
horizon on its own predictions (posterior `z` held fixed) and the rollout is supervised against
ground truth; gradients flow through the rebuilt graphs.

**Data.** The benchmark population has one sample per (window, target); training uses unique
windows (`steps` train split: 101,032 samples -> 76,675 windows). Evaluation defaults to `--windows all`,
so `target_*` is the same population and best-of-K definition as the single-target baselines.

## Commands

All on `steps` (198 indoor atrium recordings) with the `steps_default` preset; run from the repository root with the project `.venv`.

```bash
# Phase 1: self-supervised encoder
python -m baselines.train_stars_ma --phase ssl --dataset steps --run-dir experiments/stars_ma/ssl

# Phase 2: deterministic multi-agent forecaster, frozen vs fine-tuned vs scratch encoder
python -m baselines.train_stars_ma --phase forecast --dataset steps \
  --encoder-checkpoint experiments/stars_ma/ssl/checkpoints/best.pt --encoder-mode frozen \
  --run-dir experiments/stars_ma/det_frozen
#   --encoder-mode finetune ... --run-dir experiments/stars_ma/det_finetune
#   --encoder-mode scratch (no --encoder-checkpoint) ... --run-dir experiments/stars_ma/det_scratch

# Phase 3: CVAE
python -m baselines.train_stars_ma --phase forecast --dataset steps --decoder cvae \
  --encoder-checkpoint experiments/stars_ma/ssl/checkpoints/best.pt --encoder-mode finetune \
  --run-dir experiments/stars_ma/cvae_finetune

# Phase 4: multi-step closed-loop training (4 steps executed per round)
python -m baselines.train_stars_ma --phase forecast --dataset steps --decoder cvae \
  --encoder-checkpoint experiments/stars_ma/ssl/checkpoints/best.pt --encoder-mode finetune \
  --rollout-exec-steps 4 --run-dir experiments/stars_ma/cvae_finetune_rollout4

# Evaluation: teacher-forced, closed-loop at 1/4/12 executed steps, 20 s simulation
python -m baselines.evaluate_stars_ma --checkpoint experiments/stars_ma/cvae_finetune/checkpoints/best.pt \
  --datasets steps --split test --output experiments/stars_ma/cvae_finetune/eval_test.json
```

| Ablation | Flags |
|---|---|
| STARS representation vs scratch | `--encoder-mode frozen` / `finetune` vs `scratch` |
| Edge-only vs node + edge | `--edge-only` on both phases vs default |
| Frozen vs fine-tuned | `--encoder-mode frozen` vs `finetune` |
| Deterministic vs multimodal | `--decoder deterministic` vs `cvae` |
| One-step vs multi-step training | `--rollout-exec-steps 0` vs `4` (or `1`) |
| STARS topology | `--fully-connected` on both phases |

`--max-scenes`, `--max-batches`, `--epochs`, `--hidden-dim`, `--latent-dim` and `--device cpu`
give quick integration runs.

## Status (2026-10-21)

Validated on CPU only: unit tests (`tests/test_stars_ma.py`) and tiny end-to-end runs of every
phase and the evaluator on `steps`. **No GPU training has been run, so there are no results
yet.**

Tests: torch graph equals the numpy STARS adapter; SE(2) invariance of features; kNN / radius
topology; edge-only encoder equals upstream STARS at 40 steps; no future leakage into predictions
or rollouts; permutation equivariance; CVAE shapes and positive KL; differentiable rollouts;
frame round trip; metrics on known predictions; window de-duplication.

## Open items

1. **GPU runs and budgets.** Time an epoch at the default sizes; tune `--batch-size`, `--knn`,
   `--num-workers`. Rollout training rebuilds graphs per graph in Python each round, so it will be
   the bottleneck; batching `build_graph` across graphs is the first optimisation.
2. **Stationary pedestrians.** On `steps` the forecast agents' mean future speed is 0.12 m/s and
   90% of them move slower than 0.2 m/s (validation split, 300 windows), so most agents are
   standing. Headings of standing agents follow tracker jitter (STARS's `|dx|+|dy| > 1e-8` rule),
   and near-zero predictions score well on ADE. Report metrics split by moving / stationary
   agents, and consider a speed threshold for heading updates.
3. **Collision threshold.** `--collision-distance 0.2` m is a placeholder; calibrate against the
   ground-truth closest-approach distribution so `collision_rate_gt` is near zero.
4. **Social realism metrics not yet implemented**: relative trajectory change during interactions
   (e.g. deviation from constant velocity when a neighbour is within the interaction radius).
5. **Agents entering / leaving** in long rollouts: the simulated set is fixed at the first round.
6. **Baseline tables.** Not wired into `scripts/pretrain_transfer_model.sh`, `run_transfer_experiment.sh`
   or the aggregate tables; no `--resume`; no transfer (`--target-dataset`) stage yet.
7. **Diffusion decoder** (after the CVAE baseline).
8. **Robot conditioning** (not implemented, by design): add a type-0 node whose features include
   its planned trajectory over the prediction horizon (an extra robot-only temporal branch,
   injected like the pedestrian node branch), and condition the rollout on a supplied robot plan
   each round instead of predicting the robot. STEPS has no robot, so this needs interaction data.
