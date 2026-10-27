"""Train the PPO arc-choice policy (``simplenavsim.policies.ppo``) in a SimpleNavSim crowd.

1. Behaviour cloning: imitate the tuned controller's own arc choice (``--policy``, e.g. v4) on its
   rollouts, so PPO starts at the controller's level; the critic is fitted to those returns too.
2. PPO on fresh training scenes every iteration (clip 0.2, GAE), the best greedy policy on the fixed
   validation scenes (the config's objective) is kept.

Per-step reward, scaled so that an episode sums to roughly the config's objective:
    progress toward the goal (potential shaping, 1 over the whole route), +success, -collision,
    -time, -comfort (time within 0.45 m), -disruption (robot_influence share), -circling (per turn).

    python simplenavsim/scripts/train_ppo.py --config simplenavsim/configs/cmaes_crowd_v4.yaml \\
        --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --device cuda --out-dir checkpoints/simplenavsim/ppo
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from simplenavsim.cli import build, setup_logging
from simplenavsim.core.geometry import wrap_angle
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.optimization.evaluator import Evaluator, Objective
from simplenavsim.policies.ppo import ActorCritic, PPOPolicy

INTIMATE = 0.45


def collect(env, ppo, batch, objective, *, expert: bool):
    """One synchronous episode per environment; returns per-step tensors ``[T, B, ...]``."""
    dt, duration = batch.dt, batch.duration
    state = env.reset(batch)
    ppo.reset(batch.size)
    goal = batch.robot_goal
    route = (torch.linalg.norm(goal - batch.robot_start[:, :2], dim=-1) - env.sim.goal_tolerance).clamp(min=1.0)
    replan = getattr(env.pedestrians, "replan_every", 1)
    out = {k: [] for k in ("x", "valid", "action", "logp", "value", "reward", "alive", "done")}
    for _ in range(batch.steps - 1):
        alive = ~state.done
        if not bool(alive.any()):
            break
        obs = env.observe(state, batch)
        x, valid, choice = ppo.observe(obs)
        with torch.no_grad():
            logits, value = ppo.net(x, valid)
        dist = torch.distributions.Categorical(logits=logits)
        action = choice if expert else dist.sample()
        command = torch.where(alive[:, None], ppo.command(action, valid), torch.zeros(batch.size, 2, device=x.device))
        before = state
        state = env.step(state, batch, command)
        ended = state.done & ~before.done
        progress = (torch.linalg.norm(goal - before.robot.pos, dim=-1) - torch.linalg.norm(goal - state.robot.pos, dim=-1)) / route
        gap = torch.linalg.norm(state.peds.pos - state.robot.pos[:, None], dim=-1) - batch.ped_radius - env.robot.radius
        near = (torch.where(state.peds.active, gap, torch.full_like(gap, 1e3)).min(-1).values < INTIMATE).float()
        turn = wrap_angle(state.robot.pose[:, 2] - before.robot.pose[:, 2]).abs() / (2 * math.pi)
        influence = getattr(env.pedestrians, "last_influence", None)
        if influence is None:
            influence = torch.zeros_like(near)
        else:
            influence = (influence * before.peds.active).sum(-1) / before.peds.active.sum(-1).clamp(min=1) * replan
        reward = (progress + objective.success * (state.success & ended) - objective.collision * (
                  (state.ped_collision | state.obstacle_collision) & ended)
                  - (objective.time + objective.comfort * near + objective.disruption * influence) * dt / duration
                  - objective.circling * turn) * alive
        for key, value_ in (("x", x), ("valid", valid), ("action", action), ("logp", dist.log_prob(action)),
                            ("value", value), ("reward", reward), ("alive", alive), ("done", state.done)):
            out[key].append(value_)
    return {k: torch.stack(v) for k, v in out.items()}


def advantages(data, gamma, lam):
    T = data["reward"].shape[0]
    adv, last = torch.zeros_like(data["reward"]), torch.zeros_like(data["reward"][0])
    for t in reversed(range(T)):
        nonterminal = (~data["done"][t]).float()
        next_value = data["value"][t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = data["reward"][t] + gamma * next_value * nonterminal - data["value"][t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + data["value"]


def update(net, optim, data, adv, returns, args, *, imitate: bool):
    keep = data["alive"]
    x, valid, action, logp = data["x"][keep], data["valid"][keep], data["action"][keep], data["logp"][keep]
    adv, returns = adv[keep], returns[keep]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    for _ in range(args.epochs):
        for idx in torch.randperm(x.shape[0], device=x.device).split(args.minibatch):
            logits, value = net(x[idx], valid[idx])
            dist = torch.distributions.Categorical(logits=logits)
            if imitate:
                policy_loss, entropy = -dist.log_prob(action[idx]).mean(), dist.entropy().mean()
            else:
                ratio = torch.exp(dist.log_prob(action[idx]) - logp[idx])
                policy_loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx]).mean()
                entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * (value - returns[idx]).pow(2).mean() - args.entropy * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()
            stats.append((policy_loss.item(), entropy.item()))
    return [sum(s) / len(stats) for s in zip(*stats)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", required=True, help="Controller policy YAML: arcs, masks, forecasts and the BC expert")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--bc-iterations", type=int, default=15)
    parser.add_argument("--per-kind", type=int, default=24, help="Training scenes per kind per iteration")
    parser.add_argument("--val-per-kind", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/simplenavsim/ppo"))
    args = parser.parse_args()
    setup_logging()
    torch.manual_seed(args.seed)
    config, env, controller, _, pool, device = build(args.config, policy=args.policy, device=args.device)
    cm, objective = config.get("cmaes") or {}, Objective(**(config.get("objective") or {}))
    kinds, dt, duration = cm.get("kinds"), cm.get("dt", 0.1), cm.get("duration", 30.0)
    expert_values = config["policy_parameters"]
    net = ActorCritic(controller.candidates(device).shape[0]).to(device)
    ppo = PPOPolicy(controller, expert_values, net)
    evaluator = Evaluator(env, ppo, objective, device=device, max_envs=config.get("max_envs", 4096))
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    val = scenario_set("val", kinds, args.val_per_kind, dt=dt, duration=duration, pool=pool)
    expert_val = Evaluator(env, controller, objective, device=device).evaluate([expert_values], val)[0].objective
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out_dir / "progress.jsonl").open("a")
    best = -math.inf
    for it in range(args.bc_iterations + args.iterations):
        imitate = it < args.bc_iterations
        scenes = scenario_set("train", kinds, args.per_kind, start=(args.seed * 1000 + it) * args.per_kind,
                              dt=dt, duration=duration, pool=pool)
        batch = collate(scenes, device)
        net.eval()
        data = collect(env, ppo, batch, objective, expert=imitate)
        adv, returns = advantages(data, args.gamma, args.lam)
        net.train()
        policy_loss, entropy = update(net, optim, data, adv, returns, args, imitate=imitate)
        net.eval()
        episode_return = float(data["reward"].sum(0).mean())
        row = {"iteration": it, "phase": "bc" if imitate else "ppo", "return": episode_return,
               "policy_loss": policy_loss, "entropy": entropy, "expert_val_objective": expert_val}
        if it % args.eval_every == args.eval_every - 1 or it == args.bc_iterations - 1:
            result = evaluator.evaluate([{}], val)[0]
            row.update(val_objective=result.objective, val_success=result.metrics["success"],
                       val_collision=result.metrics["collision"])
            if result.objective > best:
                best = result.objective
                ppo.save(args.out_dir)
            ppo.save(args.out_dir, "latest")
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"best validation objective {best:.3f} (controller {expert_val:.3f}); saved {args.out_dir / 'best_policy.yaml'}")


if __name__ == "__main__":
    main()
