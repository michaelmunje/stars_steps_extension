# Agent judge vs cost function: two 1-hour CMA-ES runs (2026-11-04)

Both runs tune the same 6 controller weights (goal weight, pedestrian weight, personal-space radius,
preferred speed, prediction horizon, smoothness) from the same default weights, in the same
CPU-sized setting (`judged_cmaes_poc.yaml`: reactive-replay pedestrians, constant-velocity predictor;
the STARS-crowd version needs a GPU node, none was free). The judge is a headless Claude Code agent
(`claude -p`, Sonnet) that labels top-down episode pictures with the rubric in
`src/simplenavsim/optimization/judge.py`: excellent +1, acceptable +0.5, stalled 0, unsafe -1.

| | 1. Agent-judged CMA-ES | 2. Cost-function CMA-ES + agent checks |
| --- | --- | --- |
| Fitness | the judge's labels | the cost in `objective.json`, revised by the agent |
| Generation | 6 candidates x 2 scenes, all judged | 12 candidates x 4 scenes |
| Run | 60 generations, 41 min (generation cap) | 161 generations, 60 min |
| Judge use | every generation (720 episodes) | 4 checks of the 3 best candidates (48 episodes) |

**Experiment 2's cost function** (`objective_history.jsonl`, reasons in `objective.json`):
start J = SPL - 2 C + SC - 0.3 time/30 s;
gen 40: judge flagged brief close passes and sharp turns, so + (-3 x intimate-zone rate) + (-0.05 x jerk);
gen 78: close passes gone, judge flagged slow detours around groups, so time weight -0.3 -> -0.6;
gens 112 and 140: the remaining 'acceptable' episodes were already the lowest-cost ones, so no change.

## Held-out test (7 kinds x 30 scenes, never seen in training)

| Policy | Success | Collisions | Timeouts | SPL | Time to goal | Personal space (>= 0.5 m) | Time < 0.45 m | Min gap | Jerk | Principles J | Judge (7 test pictures) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Start (defaults) | 99.5% | 0.5% | 0% | 0.97 | 10.5 s | 0.89 | 8.9% | 0.34 m | 0.93 | 1.745 | +0.57 |
| 1. Agent-judged (final mean) | 91.0% | **0%** | 9.0% | 0.83 | 12.0 s | **0.99** | **0.4%** | **0.69 m** | 1.08 | 1.684 | **+0.93** |
| 1. Agent-judged (best candidate) | 90.5% | 0% | 9.5% | 0.82 | 12.1 s | 0.99 | 0.5% | 0.70 m | 1.06 | 1.674 | +0.93 |
| 2. Cost function + agent (final mean) | **99.0%** | **0%** | 1.0% | **0.93** | 10.8 s | 0.95 | 2.9% | 0.51 m | **0.57** | **1.770** | +0.79 |

- **Agent-judged CMA-ES learns the most comfortable robot** (99% of the time >= 0.5 m from everyone,
  0.69 m minimum gap, judge +0.93) but becomes over-cautious: 9% timeouts, mostly backing off and
  detouring around oncoming groups (22 s in `policies.gif`). With 2 scenes per generation it rarely
  saw the cost of stalling, and the rubric rates a stall (0) above a close pass that reaches the goal
  only by 0.5.
- **Cost function + agent checks gives the best all-round policy:** 99% success, no collisions,
  comfortable clearance (min gap 0.51 m), the smoothest motion, and the best Principles J. The
  agent's revisions removed close passes (time < 0.45 m: 8.9% at the start -> 2.9%) without losing
  task performance, at about 1/15 of the judge calls.
- **The judge agrees with the cost function:** over experiment 1's 720 episodes, Spearman correlation
  of the label with Principles J +0.77 and with the original J +0.71; with jerk +0.02 (the judge
  ignores smoothness that the metrics penalise).
- Judge usage for both runs and the test labels: 69 `claude -p` calls, estimated $7.91 at API
  prices; the subscription went from 17% to 26% of the 5-hour session and stayed at 9% of the week.

## Files

`policies.gif` / `.png` (held-out scenes, rows = policies), `sheet.png` (the judge's view of the
test episodes) and `test_judge_labels.json`, `convergence.png`, `results.json` (every metric),
`judged_cmaes_policy.yaml` and `costfn_agent_policy.yaml` (final weights),
`judged_episode_labels.jsonl` (experiment 1: every episode's parameters, label, reason, metrics and
both cost functions), `evals.jsonl` (experiment 2's checks), `scripts/` (drivers, usage guard,
configs). Usage was checked after every generation with `claude -p /usage` (`usage_guard.py`).
