"""Train learned subgoals for the tuned controller with PPO (``simplenavsim.policies.ppo``).

The reward is the config's objective, the same number CMA-ES maximises, given at the end of each
episode, plus progress shaping toward the goal with zero net effect: per-decision potential
differences of ``-distance / route``, closed so that the shaping sums to a constant per scene. The
untrained policy equals the controller (``--policy``), and validation every ``--eval-every``
iterations keeps the best greedy policy (iteration 0 is the controller itself).

    python simplenavsim/scripts/train_ppo.py --config simplenavsim/configs/cmaes_crowd_v4.yaml \\
        --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml --device cuda --out-dir checkpoints/simplenavsim/subgoal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simplenavsim.cli import build, setup_logging
from simplenavsim.core.metrics import all_metrics
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import collate, scenario_set
from simplenavsim.optimization.evaluator import Evaluator
from simplenavsim.policies.ppo import SubgoalNet, SubgoalPolicy


def episode(env, policy, batch, objective):
    """Sampled rollout -> per-decision tensors ``[D, B]`` (features ``[D, B, F]``, raw actions ``[D, B, 2]``)."""
    policy.sample = True
    run = rollout(env, batch, policy, policy.parameter_tensor({}, batch.size, batch.robot_start.device))
    policy.sample = False
    score = objective(all_metrics(run, None, batch, env.robot.radius, env.robot.max_speed, env.sim.goal_tolerance))
    d, B = policy.decisions, batch.size
    k = torch.tensor([x["k"] for x in d], device=score.device)
    alive = k[:, None] < run.end_step[None]                                       # [D, B]
    goal, rows = batch.robot_goal, torch.arange(B, device=score.device)
    route = (torch.linalg.norm(goal - batch.robot_start[:, :2], dim=-1) - env.sim.goal_tolerance).clamp(min=1.0)
    phi = lambda step: -torch.linalg.norm(run.robot_pose[rows, step, :2] - goal, dim=-1) / route
    last = (alive.long() * torch.arange(1, len(d) + 1, device=score.device)[:, None]).argmax(0)  # last live decision
    rewards = torch.zeros(alive.shape, device=score.device)
    for i in range(len(d)):
        here = phi(k[i].expand(B))
        nxt = phi(torch.minimum(k[i + 1].expand(B), run.end_step)) if i + 1 < len(d) else torch.zeros_like(here)
        final = last == i
        rewards[i] = (torch.where(final, -here, nxt - here) + final * score) * alive[i]
    done = torch.zeros_like(alive)
    done[last, rows] = True
    stack = lambda key: torch.stack([x[key] for x in d])
    return {"x": stack("x"), "raw": stack("raw"), "logp": stack("logp"), "value": stack("value"),
            "reward": rewards, "alive": alive, "done": done}, float(score.mean())


def advantages(data, gamma, lam):
    adv, last = torch.zeros_like(data["reward"]), torch.zeros_like(data["reward"][0])
    T = adv.shape[0]
    for t in reversed(range(T)):
        nonterminal = (~data["done"][t]).float()
        next_value = data["value"][t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = data["reward"][t] + gamma * next_value * nonterminal - data["value"][t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + data["value"]


def update(net, optim, data, adv, returns, args):
    m = data["alive"]
    x, raw, old, adv, returns = data["x"][m], data["raw"][m], data["logp"][m], adv[m], returns[m]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(args.epochs):
        for idx in torch.randperm(x.shape[0], device=x.device).split(args.minibatch):
            dist, value = net(x[idx])
            ratio = torch.exp(dist.log_prob(raw[idx]).sum(-1) - old[idx])
            surrogate = torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx])
            loss = -surrogate.mean() + 0.5 * (value - returns[idx]).pow(2).mean() - args.entropy * dist.entropy().sum(-1).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", required=True, help="Tuned controller policy YAML (executes the subgoals)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--per-kind", type=int, default=24, help="Fresh training scenes per kind per iteration")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hold", type=int, default=10, help="Control steps per subgoal (10 = 1 s)")
    parser.add_argument("--max-offset", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/simplenavsim/subgoal"))
    args = parser.parse_args()
    setup_logging()
    torch.manual_seed(args.seed)
    config, env, controller, evaluator, pool, device = build(args.config, policy=args.policy, device=args.device)
    cm = config.get("cmaes") or {}
    kinds, dt, duration = cm.get("kinds"), cm.get("dt", 0.1), cm.get("duration", 30.0)
    net = SubgoalNet().to(device)
    policy = SubgoalPolicy(controller, config["policy_parameters"], net, hold=args.hold, max_offset=args.max_offset)
    evaluator = Evaluator(env, policy, evaluator.objective, device=device, max_envs=config.get("max_envs", 4096))
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    val = scenario_set("val", kinds, cm.get("val_per_kind", 12), dt=dt, duration=duration, pool=pool)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = (args.out_dir / "progress.jsonl").open("a")
    best = -float("inf")
    for it in range(args.iterations + 1):
        row = {"iteration": it, "std": net.log_std.exp().tolist()}
        if it % args.eval_every == 0 or it == args.iterations:
            result = evaluator.evaluate([{}], val)[0]
            row.update(val_objective=result.objective, val_success=result.metrics["success"],
                       val_collision=result.metrics["collision"])
            if result.objective > best:
                best = result.objective
                policy.save(args.out_dir)
            policy.save(args.out_dir, "latest")
        if it < args.iterations:
            scenes = scenario_set("train", kinds, args.per_kind, start=(args.seed * 1000 + it) * args.per_kind,
                                  dt=dt, duration=duration, pool=pool)
            data, train_objective = episode(env, policy, collate(scenes, device), evaluator.objective)
            adv, returns = advantages(data, args.gamma, args.lam)
            update(net, optim, data, adv, returns, args)
            row["train_objective"] = train_objective
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    print(f"best validation objective {best:.3f}; saved {args.out_dir / 'best_policy.yaml'}")


if __name__ == "__main__":
    main()
