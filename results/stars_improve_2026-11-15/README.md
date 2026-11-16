# STARS improvement: collisions and speed (2026-11-15/16, IN PROGRESS)

Branch `stars-improve` (worktree /scratch/09865/mmunje/stars-improve). Open-loop evaluation, obs 8 / pred 12 at 0.4 s.
Tool: `python -m baselines.stars_ma_openloop` (single-sample, mean-of-K, mode, minADE, joint ADE; pair collision
rates at 0.2/0.3/0.4 m per sample, on the mean, and with agents' samples shuffled; path and net speed; GT and
window statistics). JSONs in `diagnosis/` and `val/`.

## Diagnosis (validation split, unique windows)
- **Data:** GT close-pair rates (pairs ever < d over 4.8 s) are low: 0.03% at 0.2 m, 0.15% at 0.4 m pooled
  (MADRAS 1.9%, STEPS 0.45%, UCY 0.6%, ATC 0.01%). 50-88% of GT close pairs are groups (within 1 m and co-moving
  at the last observed step). No duplicate tracks (< 0.05 m) of note. So the data does not explain the excess.
- **Pooled STARS (`cvae_all_lr3e-4_seed42`), per sample:** 0.14% at 0.2 m and 0.42% at 0.4 m: 2-5x GT (UCY 0.021
  vs 0.006, ATC zero-shot 0.010 vs 0.0001), about 70% of constant velocity's rate. The sample *mean* and the
  prior-mean (mode) collide less. Agents' samples are not coordinated: shuffling sample indices across agents
  gives the same rate (0.46% vs 0.42%). Cause: per-agent latents mixed by one social layer over *observed*
  history; the decoder never sees neighbours' predicted futures, and no loss penalises overlap.
- **Diffusion decoders (STEPS):** more collisions than the CVAE (0.86%/0.75% joint vs 0.39% at 0.4 m; GT 0.45%),
  joint diffusion only slightly better than per-agent.
- **Speed:** open-loop it is mostly fine. Net speed (|x12-x0|/4.8 s) 0.42 vs GT 0.43; ratio for moving agents
  0.93-0.96 per sample, 0.93 for the mode. Lower sampling temperature/mode is slower (regression to the mean).
  Path speed looks slower on ATC (0.29 vs 0.49) and STEPS because GT is jittery (ATC GT net speed is only 0.29);
  STEPS is 74% stationary agents, which pulls pooled speeds down. The closed-loop slowness is compounding,
  not an open-loop bias.

## Fixes (15-epoch screening, all 7 datasets, val; table in `val/table.txt`)
| run | single ADE (macro) | minADE | joint | net speed | pair 0.2 | pair 0.4 | shuffled 0.4 |
|---|---|---|---|---|---|---|---|
| base | 0.545 | 0.232 | 0.437 | 0.415 | 0.19% | 0.55% | 0.58% |
| refine 1 round (future-aware social layer) | 0.549 | 0.231 | 0.439 | 0.414 | 0.20% | 0.55% | 0.59% |
| collision hinge 10 on prior sample (prior trainable) | 1.366 | 0.411 | 0.767 | 0.461 | 0.08% | 0.28% | 0.28% |
| refine + collision 10 (prior trainable) | 0.947 | 0.435 | 0.631 | 0.308 | 0.02% | 0.10% | 0.15% |
GT: 0.03% / 0.15%. Refinement alone changes nothing; with the collision loss it makes samples coordinated
(joint below shuffled) and reaches GT rates, but the hinge through a trainable prior widened the prior / slowed
agents and hurt ADE. Now the default detaches the prior latent (decoder-only signal). Pending runs:
`r2_col10fix`, `r2_refine_col10fix`, `r2_refine_col3fix`, `r2_refine_col10_bom` (+best-of-many prior loss),
`r2_refine_col10_kl003`.
