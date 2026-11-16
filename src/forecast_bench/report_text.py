"""Static parts of ``report.md`` (protocol, methods, reproduction, caveats, commands); tables come from report.py."""

INTRO = """# Open-loop trajectory forecasting benchmark: STARS vs. official baselines

Branch `forecast-benchmark`, code in `src/forecast_bench/`. Main task: **open-loop forecasting**, 8 observed /
12 predicted steps at 0.4 s (3.2 s -> 4.8 s), K = 20 samples. A secondary closed-loop rollout to 20 s is at the
end. Every number: `metrics.csv` / `metrics.json` / `closed_loop_metrics.csv`; all tables: `tables.md`.
"""

PROTOCOL = """## Protocol

**ETH/UCY leave-one-out (comparable to papers).** Test windows come from the standard Social-GAN / SocialVAE
test files (`references/social-vae/data/<scene>/test`): every 20-frame window (stride 1); scored agents = present
in all 20 frames, i.e. eth 364, hotel 1197, univ 24334, zara1 2356, zara2 5910 -- identical to the standard
counts. Windows with one scored pedestrian are kept (SocialVAE's rule; Social-GAN / MoFlow's official loaders drop
them, which only matters on eth: 181 vs 364 agents). Context agents = everyone visible at the last observed step
with >= 2 observed points (observed steps only).

**Our datasets.** Each dataset's test split as defined by our STARS configs (`windows=unique`), seeded subsets of
~20k scored agents (madras has only 4 test windows); `eth` test = seq_eth, `ucy` test = crowds_zara02.

**Metrics (scored agents).** minADE20/minFDE20 (standard best-of-20 per agent); mean-over-samples ADE/FDE
(expected single-sample error) and first-sample ADE/FDE; minJointADE/FDE (per window, min over k of the mean
agent error: sample k as one joint scene); collisions = P(an agent's prediction comes within 0.2 / 0.4 m of another
scored agent's prediction of the *same sample* at the same step), per (agent, sample) and per (scene, sample), over
windows with >= 2 scored agents; the "Ground truth" row is the same rate on the real futures. Speed: mean predicted
speed vs GT and the 1-D Wasserstein distance of per-step speeds.

**Sample structure.** Joint scene samples: STARS CVAE (one latent per agent but decoded jointly), STARS joint
diffusion, AgentFormer, Social-GAN. Independent per agent: SocialVAE, Trajectron++, STARS per-agent diffusion
(decoded per agent). MoFlow (ETH setting) is single-agent. For independent samplers the joint / collision metrics
pair agents' k-th samples arbitrarily.
"""

METHODS = """## Methods and leakage

| Row | What | Held-out ETH/UCY scene in training? |
|---|---|---|
| Constant velocity | last observed displacement | - |
| SocialVAE / MoFlow / Trajectron++ / AgentFormer / Social-GAN (official ckpt) | official code + official per-fold pretrained checkpoints, adapted to the common windows | no (official LOO) |
| SocialVAE / MoFlow (STEPS-trained) | our earlier STEPS-trained checkpoints | no (zero-shot) |
| STARS LOO | `cvae_noethucy_nogoal`, fine-tuned per fold on the standard `train` files, selected on `val` (`finetune_stars_loo.py`, 30 epochs) | no |
| STARS STEPS-only (CVAE / per-agent diffusion / joint diffusion) | `cvae_kl01_finetune`, `diffusion_finetune`, `joint_diffusion_finetune` | no (zero-shot) |
| STARS pooled w/o ETH/UCY (+ oracle goal) | `cvae_noethucy_{nogoal,goal}` (5 datasets) | no (zero-shot); the goal model receives each agent's track end (privileged) |
| STARS pooled (6 datasets) | `cvae_all_lr3e-4_seed42` | **yes for hotel, zara1, univ** (students001 train, students003 val) -> cells marked `*`; eth and zara2 clean |
"""

REPRO = """## Reproduction of the official baselines (ETH/UCY minADE20/minFDE20, m)

| Model | source | eth | hotel | univ | zara1 | zara2 |
|---|---|---|---|---|---|---|
| SocialVAE (no FPC) | paper | .47/.76 | .14/.22 | .25/.47 | .20/.37 | .14/.28 |
| | official `main.py --test`, GPU | .478/.770 | .145/.223 | .253/.476 | .201/.371 | .149/.282 |
| | our adapter, common windows | .504/.861 | .146/.223 | .254/.477 | .197/.364 | .152/.286 |
| MoFlow (teacher) | paper | .40/.57 | .11/.17 | .23/.39 | .15/.26 | .12/.22 |
| | official `eval_eth.py`, HF ckpts | .403/.535 | .115/.173 | .255/.406 | .170/.296 | .125/.220 |
| | our adapter, same agents (SGAN subset) | .403/.535 | .115/.173 | .255/.406 | .170/.296 | .125/.220 |
| Trajectron++ (`*_vel`) | paper | .39/.83 | .12/.21 | .20/.44 | .15/.33 | .11/.25 |
| | official eval, **leaky** preprocessing | .402/.841 | .117/.200 | .199/.444 | .152/.331 | .114/.250 |
| | our adapter, observed steps only | .773/1.338 | .244/.357 | .295/.578 | .230/.426 | .174/.329 |
| AgentFormer | paper | .45/.75 | .14/.22 | .25/.45 | .18/.30 | .14/.24 |
| | official = our adapter | .451/.748 | .142/.225 | .254/.454 | .177/.304 | .140/.236 |
| Social-GAN (SGAN-20VP-20) | MODEL_ZOO (scene-joint min) | .77/1.39 | .43/.88 | .75/1.50 | .34/.68 | .36/.73 |
| | our adapter, scene-joint, >= 2 agents | .754/1.375 | .438/.885 | .746/1.500 | .346/.693 | .359/.725 |

* **Trajectron++'s published ETH/UCY numbers rely on future leakage.** The released weights predate upstream
  03712b7 ("Fixed the gradient calculations", issues #40/#26): features were `np.gradient` over whole tracks, so the
  current velocity uses x[t+1] and the acceleration x[t+2]. With that preprocessing the official numbers reproduce;
  with features from observed steps only (our adapter, the fair setting) the same weights score far worse. Our
  tables use the leak-free adapter.
* Social-GAN's official metric is a scene-joint min (sum of agent errors); our per-agent minADE table is lower for
  it, the minJoint table is the closer analogue.
* SocialVAE eth differences are sampling noise (364 agents; seeds .49-.51; the official script gives .501 on CPU).
* MoFlow's released checkpoints reproduce slightly worse than the paper on univ / zara1 (checkpoint, not adapter).
"""

CLOSED = """## Closed loop (secondary): ETH/UCY up to 20 s

`closed_loop.py` reuses the `simplenavsim.realism` closed-loop windows and metrics (cohort = everyone present at the
last observed step, simulated jointly; a window starts every 10 s; an agent is scored at a horizon only if its
recorded track covers it; 2 seeds x 1 sample, micro-averaged over the 5 scenes). Schemes: `<method>@4` = receding
horizon (predict 12, execute the first 4 steps = 1.6 s for everyone, append to history, repeat); `*_native` = STARS'
own loop (re-forecast every 0.4 s, execute 1 step, temporally correlated CVAE intent). Collision = agent within 0.4 m
of another scored agent (simulated) at the same step. The scored population changes with the horizon (only agents
whose recorded track lasts that long), so ground-truth speed falls with horizon (long tracks are slow / lingering
pedestrians) and numbers are comparable across methods within a column, not along a row. MoFlow's receding-horizon
rollout (100 flow steps per call) ran in a separate per-fold job; if its row is missing it was still running.
"""

CAVEATS = """## Caveats

* Leakage: `STARS pooled` saw hotel / zara1 / univ during training or selection (marked `*`); STARS LOO, no-ETH/UCY,
  STEPS-only and all official baselines are clean.
* The goal-conditioned STARS row uses an oracle goal (track end): not comparable to unconditioned methods.
* Our pooled-dataset windows are STARS' target-centred scene frames; MoFlow official reads absolute positions and is
  therefore only run on the world-frame ETH/UCY sets. Official ETH/UCY checkpoints are not run on our datasets.
* minADE20 rewards diversity; mean-over-samples ADE and the collision rates are the single-sample view.
* Collision rates count contacts between *predicted* agents only (no ground-truth or context agents); ground-truth
  rates show how many close contacts real data contains (several datasets have 5-13 % of agents within 0.4 m).
* The closed-loop study is small (stride 10 s, 2 seeds); receding-horizon baselines use a fresh sample every 1.6 s.
"""

COMMANDS = """## Commands

From the worktree root, `.venv` active, `PYTHONPATH=$PWD/src`:

```
python -m forecast_bench.windows                                   # common windows -> experiments/forecast_bench/windows
sbatch --time=04:00:00 --job-name=fc_bench scripts/horizon/gpu_jobs.slurm \\
  "bash runs/forecast_bench/stars_gpu.sh gpu0" "bash runs/forecast_bench/stars_gpu.sh gpu1" \\
  "bash runs/forecast_bench/stars_gpu.sh gpu2" "bash runs/forecast_bench/stars_gpu.sh gpu3"
    # STARS LOO fine-tunes, CV + STARS predictions, STARS LOO predictions, closed-loop rollouts
python -m forecast_bench.predict_socialvae --method socialvae_official|socialvae_steps --sets <sets>
python -m forecast_bench.predict_moflow --method moflow_official|moflow_steps --sets <sets>   # HF fyxfelixfu/moflow
python -m forecast_bench.predict_trajectron --sets <ethucy sets>    # weights from git history (module docstring)
python -m forecast_bench.predict_agentformer --sets <ethucy sets>   # official Google-Drive models (module docstring)
python -m forecast_bench.predict_sgan --sets <ethucy sets>          # MODEL_ZOO models
python -m forecast_bench.official_eval socialvae|moflow --fold <fold>   # official-script reproductions
python -m forecast_bench.report --rescore --out results/forecasting_<date>_<sha>
```
"""
