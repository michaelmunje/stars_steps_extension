#!/usr/bin/env python3
"""Why does the robot circle instead of stopping? Per scenario, the fraction of control steps in which
standing still was ruled out by the hard safety constraint, and how the robot turned meanwhile.

    python simplenavsim/scripts/diagnose_waiting.py --config simplenavsim/configs/cmaes_crowd_v3.yaml \
        --policy checkpoints/simplenavsim/cmaes_crowd_v2/best_policy.yaml --kinds dense_flow narrow_passage --seeds 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simplenavsim.cli import DEFAULT_CONFIG, build, setup_logging
from simplenavsim.core.rollout import rollout
from simplenavsim.core.scenarios import SPLIT_OFFSETS, collate, make_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--kinds", nargs="+", default=["dense_flow", "narrow_passage"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    setup_logging()
    config, env, policy, evaluator, pool, device = build(args.config, policy=args.policy, device=args.device)
    scenarios = [make_scenario(k, SPLIT_OFFSETS[args.split] + s, pool=pool, split=args.split)
                 for k in args.kinds for s in range(args.seeds)]
    batch = collate(scenarios, device)
    log = []
    act = policy.act

    def recording_act(obs, params):
        command = act(obs, params)
        log.append((policy.last_info["stop_valid"].clone(), command.clone()))
        return command

    policy.act = recording_act
    run = rollout(env, batch, policy, policy.parameter_tensor(config["policy_parameters"], batch.size, device))
    stop_valid = torch.stack([s for s, _ in log], 1).float()                    # [B, T]
    commands = torch.stack([c for _, c in log], 1)                             # [B, T, 2]
    live = torch.arange(stop_valid.shape[1], device=device)[None] < run.end_step[:, None]
    report = {}
    for i, kind in enumerate(args.kinds):
        rows = slice(i * args.seeds, (i + 1) * args.seeds)
        m = live[rows].float()
        v, w = commands[rows, :, 0], commands[rows, :, 1]
        blocked = (1 - stop_valid[rows]) * m
        turning = ((w.abs() > 0.3) & (v.abs() > 0.05)).float() * m
        report[kind] = {
            "stop_ruled_out": float(blocked.sum() / m.sum()),
            "turning_while_stop_ruled_out": float((turning * blocked).sum() / blocked.sum().clamp(min=1)),
            "turning_while_stop_allowed": float((turning * (1 - stop_valid[rows]).logical_not() * m).sum()
                                                / (m.sum() - blocked.sum()).clamp(min=1)),
            "net_turn_rad": float((w * m).sum(1).mul(run.dt).abs().mean()),
            "success": float(run.success[rows].float().mean()),
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
