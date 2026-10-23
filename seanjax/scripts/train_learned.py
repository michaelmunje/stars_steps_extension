#!/usr/bin/env python3
"""Train the learned navigation policy (``seanjax.policies.learned``): behaviour cloning from a
teacher policy, then evolution strategies on the navigation objective.

    python seanjax/scripts/train_learned.py --config seanjax/configs/cmaes_crowd_v4.yaml \\
        --teacher checkpoints/seanjax/cmaes_crowd_v4/best_policy.yaml \\
        --out-dir checkpoints/seanjax/learned_v1 --device cuda

1. **Behaviour cloning.** The teacher drives ``--bc-per-kind`` training scenarios of each kind (the
   config's ``cmaes.kinds``; seeds disjoint from the ES draws). Every control step's observation
   features and teacher command become a regression pair; the MLP is fitted by MSE.
2. **Evolution strategies** (OpenAI-ES: antithetic Gaussian perturbations, centred ranks, Adam).
   Every generation evaluates ``--population`` perturbed networks on one fresh draw of training
   scenarios (shared by all of them), as one simulation batch. The mean network is validated on the
   config's fixed validation set every ``--validate-every`` generations, and the best is kept.

Writes ``bc.npy``, ``best.npy`` and ``best_policy.yaml`` (loadable by evaluate.py, compare_policies.py
and visualize_rollout.py), ``history.jsonl`` and ``state.pt`` (``--resume``). Decision 91.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from seanjax.cli import build, load_learned_policy, setup_logging
from seanjax.core.config import make_predictor
from seanjax.core.metrics import all_metrics, summarize
from seanjax.core.rollout import rollout
from seanjax.core.scenarios import collate, scenario_set
from seanjax.optimization.evaluator import _repeat_rollout
from seanjax.policies.learned import LearnedConfig, LearnedPolicy


def _write_policy(out: Path, name: str, weights: np.ndarray, policy: LearnedPolicy, **extra) -> Path:
    np.save(out / f"{name}.npy", weights)
    config = {k: v for k, v in vars(policy.config).items()}
    path = out / f"{name}_policy.yaml"
    path.write_text(yaml.safe_dump({"policy": "learned", "weights": f"{name}.npy", "learned": config, **extra},
                                   sort_keys=False))
    return path


def collect(env, teacher, teacher_params, student, scenarios, device, chunk: int):
    """(features, teacher raw targets) from the teacher driving ``scenarios``."""
    xs, ys = [], []
    act = teacher.act

    def recording_act(obs, params):
        command = act(obs, params)
        features, _ = student.features(obs)
        xs.append(features.float().cpu())
        ys.append(student.from_command(command).float().cpu())
        return command

    teacher.act = recording_act
    try:
        for start in range(0, len(scenarios), chunk):
            batch = collate(scenarios[start:start + chunk], device)
            rollout(env, batch, teacher, teacher.parameter_tensor(teacher_params, batch.size, device))
    finally:
        teacher.act = act
    return torch.cat(xs), torch.cat(ys)


def behaviour_clone(student: LearnedPolicy, x, y, device, *, epochs: int, seed: int, log) -> np.ndarray:
    theta = torch.tensor(student.init_weights(seed), device=device, requires_grad=True)
    optimiser = torch.optim.Adam([theta], lr=3e-3)
    x, y = x.to(device), y.to(device)
    n, batch = x.shape[0], 4096
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            pred = student.forward(x[idx], theta[None].expand(len(idx), -1))
            loss = ((pred - y[idx]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 5 == 0 or epoch == epochs - 1:
            log(f"bc epoch {epoch}: mse {total / n:.4f}")
    return theta.detach().cpu().numpy()


def evaluate(env, policy, objective, thetas: np.ndarray, scenarios, cf_cache, device, max_envs: int):
    """Mean objective per weight vector ``[N]`` (and the mean metrics of the first one)."""
    base = collate(scenarios, device)
    key = tuple(s.name for s in scenarios)
    if key not in cf_cache:
        cf_cache.clear()
        cf_cache[key] = rollout(env, base, robot_present=False)
    cf = cf_cache[key]
    S = len(scenarios)
    per_chunk = max(1, max_envs // S)
    scores, first = [], None
    for start in range(0, len(thetas), per_chunk):
        chunk = thetas[start:start + per_chunk]
        n = len(chunk)
        batch = base.repeat(n)
        params = torch.as_tensor(np.repeat(chunk, S, axis=0), device=device)
        run = rollout(env, batch, policy, params)
        metrics = all_metrics(run, _repeat_rollout(cf, n), batch, env.robot.radius)
        score = objective(metrics).view(n, S).mean(1)
        scores.append(score.cpu().numpy())
        if first is None:
            first = summarize({k: v[:S] for k, v in metrics.items()})
    return np.concatenate(scores), first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, help="Policy YAML to clone (skipped with --resume or --init)")
    parser.add_argument("--init", type=Path, help="Start ES from these weights (.npy) instead of cloning")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bc-per-kind", type=int, default=40)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=48, help="Even: antithetic pairs")
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-per-kind", type=int, default=6)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_logging()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    history = open(out / "history.jsonl", "a")
    log = lambda message: print(message, flush=True)

    config, env, teacher, evaluator, pool, device = build(args.config, policy=str(args.teacher) if args.teacher else None,
                                                           device=args.device)
    cmaes = config.get("cmaes") or {}
    kinds, duration, dt = cmaes["kinds"], cmaes.get("duration", 30.0), cmaes.get("dt", 0.1)
    robot = env.robot
    student = LearnedPolicy(LearnedConfig(hidden=args.hidden, max_speed=robot.max_speed, min_speed=robot.min_speed,
                                          max_angular_speed=robot.max_angular_speed,
                                          max_acceleration=robot.max_acceleration,
                                          max_angular_acceleration=robot.max_angular_acceleration,
                                          robot_radius=robot.radius),
                            make_predictor((config.get("policy") or {}).get("predictor"), device))
    objective, max_envs = evaluator.objective, int(config.get("max_envs", 4096))
    scen = lambda split, per_kind, start: scenario_set(split, kinds, per_kind, start=start, dt=dt, duration=duration,
                                                       pool=pool)
    val = scen("val", int(cmaes.get("val_per_kind", 12)), 0)
    cf_cache: dict = {}

    state_path = out / "state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, weights_only=False)
        theta, m, v, start_gen, best = state["theta"], state["m"], state["v"], state["generation"] + 1, state["best"]
        log(f"resumed at generation {start_gen}")
    else:
        if args.init:
            theta = np.load(args.init)
        else:
            started = time.time()
            x, y = collect(env, teacher, config["policy_parameters"], student,
                           scen("train", args.bc_per_kind, 500_000), device, chunk=max(1, max_envs // 4))
            log(f"bc data: {x.shape[0]} samples in {time.time() - started:.0f} s")
            theta = behaviour_clone(student, x, y, device, epochs=args.bc_epochs, seed=args.seed, log=log)
            np.save(out / "bc.npy", theta)
        m, v, start_gen = np.zeros_like(theta), np.zeros_like(theta), 1
        score, metrics = evaluate(env, student, objective, theta[None], val, cf_cache, device, max_envs)
        best = {"objective": float(score[0]), "generation": 0, "metrics": metrics}
        np.save(out / "best.npy", theta)
        _write_policy(out, "best", theta, student, validation_objective=best["objective"], generation=0)
        log(f"start: validation objective {best['objective']:.3f}")
        history.write(json.dumps({"generation": 0, "kind": "val", "objective": best["objective"], **metrics}) + "\n")

    rng = np.random.default_rng(args.seed)
    half = args.population // 2
    beta1, beta2 = 0.9, 0.999
    for g in range(start_gen, args.generations + 1):
        started = time.time()
        eps = rng.standard_normal((half, theta.size)).astype(np.float32)
        thetas = np.concatenate([theta + args.sigma * eps, theta - args.sigma * eps])
        train = scen("train", args.train_per_kind, g * args.train_per_kind)
        scores, _ = evaluate(env, student, objective, thetas, train, cf_cache, device, max_envs)
        ranks = np.empty_like(scores)
        ranks[np.argsort(scores)] = np.arange(len(scores))
        ranks = ranks / (len(scores) - 1) - 0.5
        grad = ((ranks[:half] - ranks[half:])[:, None] * eps).sum(0) / (args.population * args.sigma)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        step = args.lr * (m / (1 - beta1 ** g)) / (np.sqrt(v / (1 - beta2 ** g)) + 1e-8)
        theta = (theta + step).astype(np.float32)                               # ascent: higher objective
        record = {"generation": g, "kind": "train", "mean": float(scores.mean()), "best": float(scores.max()),
                  "seconds": time.time() - started}
        if g % args.validate_every == 0 or g == args.generations:
            score, metrics = evaluate(env, student, objective, theta[None], val, cf_cache, device, max_envs)
            record.update({"kind": "val", "objective": float(score[0]), **metrics})
            if score[0] > best["objective"]:
                best = {"objective": float(score[0]), "generation": g, "metrics": metrics}
                np.save(out / "best.npy", theta)
                _write_policy(out, "best", theta, student, validation_objective=best["objective"], generation=g)
            log(f"gen {g}: train mean {scores.mean():.3f}, val {score[0]:.3f} (best {best['objective']:.3f} @ {best['generation']})")
            torch.save({"theta": theta, "m": m, "v": v, "generation": g, "best": best}, state_path)
        history.write(json.dumps(record) + "\n")
        history.flush()
    (out / "summary.json").write_text(json.dumps({"best": best, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
    log(f"best validation objective {best['objective']:.3f} at generation {best['generation']}")


if __name__ == "__main__":
    main()
