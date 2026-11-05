# Agent-judged CMA-ES (2026-11-04)

CMA-ES whose fitness is an agent judge's labels of rendered episodes, instead of a hand-crafted
reward (AutoEval's loop, Zhou et al. 2025). The judge is a headless Claude Code agent (`claude -p`,
Sonnet) that labels top-down episode pictures with the rubric in `src/simplenavsim/optimization/judge.py`:
excellent +1, acceptable +0.5, stalled 0, unsafe -1. A candidate's fitness is its mean label.

- **Setting:** CPU-sized (`judged_cmaes_poc.yaml`): reactive-replay pedestrians, constant-velocity
  predictor (the STARS-crowd version needs a GPU node; none was free). 6 controller weights tuned
  from the default weights.
- **Run:** 6 candidates x 2 scenes per generation, every episode judged; 60 generations in 41 minutes
  (720 judged episodes).
- **Hand-crafted reward** = SPL - 2 collision + personal-space compliance - 0.3 time / 30 s (the
  Principles objective of `cmaes_crowd_principles.yaml`). Logged for every episode, never optimised.
- **Usage:** judging cost an estimated $6.9 at API prices (61 calls, including one aborted first try); checked after every generation with
  `claude -p /usage` (`scripts/usage_guard.py`, stop at 60% of the 5-hour session or 20% of the week);
  the session stayed at or below 26%, the week at 9%.

## Results on 210 held-out scenes ([`comparison.png`](comparison.png))

Random weights and CMA-ES are the earlier seed study's 10 seeds (CMA-ES seed s started from random
seed s and was trained on the hand-crafted reward in the STARS crowd: 11 weights, ~460k episodes per
seed), scored here as they are. Mean ± standard deviation over seeds (`cost_comparison.json`):

| | Success | Time within 0.45 m of a person | Hand-crafted reward |
| --- | --- | --- | --- |
| Random weights (10 seeds) | 77.8 ± 32.7% | 6.1 ± 3.4% | 1.45 ± 0.36 |
| CMA-ES on the hand-crafted reward (10 seeds) | 98.7 ± 1.0% | 0.4 ± 0.2% | **1.79 ± 0.01** |
| Agent-judged CMA-ES | 91.0% | **0.4%** | 1.68 |

- The agent-judged policy is as comfortable as reward-tuned CMA-ES (0.4% of the time within 0.45 m,
  0.69 m minimum gap, no collisions) but stalls more often (9% timeouts), with 640x fewer episodes
  and 6 tuned weights instead of 11.
- The judge agrees with the hand-crafted reward: over the 720 judged episodes the Spearman
  correlation of the label with the reward is +0.77 (with jerk +0.02: the judge ignores smoothness).

## Failure cases ([`failure_cases/`](failure_cases/))

The held-out scenes where the agent-judged policy fails (19 of 210, all timeouts), worst reward first,
at most two per kind: two-way corridors #20 and #5, narrow passage #7, standing groups #2 and #4.
`failure_cases.gif` shows the start weights, random weights (seed 0), CMA-ES (seed 0) and the
agent-judged policy on them; there is also one GIF per scene. Everyone else gets through, and the
agent-judged policy loops or freezes in front of passable people. The corridor and narrow-passage
cases are hard in general: only 6-7 of the 10 CMA-ES seeds and 7-8 of the 10 random weight sets reach
the goal there (`scenes.json`). Likely causes: two scenes per generation rarely show the cost of
stalling, and the rubric rates a stall (0) above a close pass that reaches the goal (+0.5).

## Files

`comparison.png`, `cost_comparison.json` (every metric per policy), `failure_cases/`,
`convergence.png` (judge fitness and hand-crafted reward per generation), `results.json` (start vs
agent-judged, all metrics), `judged_cmaes_policy.yaml` (final weights), `judged_episode_labels.jsonl`
(every judged episode: parameters, label, reason, all metrics, hand-crafted reward), `scripts/`.
