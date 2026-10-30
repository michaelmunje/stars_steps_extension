"""Collect multi-agent STARS test evaluations into one markdown table and figure.

    python -m baselines.report_stars_ma experiments/stars_ma/runs --output docs/results/stars_ma_steps.md

Reads every ``<run>/eval_test.json`` (written by ``evaluate_stars_ma``) and reports, per run and for
the constant-velocity reference: best-of-K target / all-agent / joint ADE and FDE, moving vs
stationary agents, closed-loop ADE at each executed-steps setting, and the simulation realism
numbers. A bar chart of the headline metrics is written next to the markdown file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = [("agent_ade", "agent ADE"), ("agent_fde", "agent FDE"), ("moving_ade", "moving ADE"),
           ("moving_fde", "moving FDE"), ("stationary_ade", "stationary ADE"), ("target_ade", "target ADE"),
           ("target_fde", "target FDE"), ("joint_ade", "joint ADE"), ("collision_rate_pred", "pred. collisions")]


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="Directory containing run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="eval_test.json")
    args = parser.parse_args()
    reports = {path.parent.name: json.loads(path.read_text()) for path in sorted(args.runs.glob(f"*/{args.name}"))}
    if not reports:
        raise SystemExit(f"No {args.name} under {args.runs}")
    first = next(iter(reports.values()))
    lines = [f"# Multi-agent STARS: {', '.join(first['datasets'])} {first['split']} split",
             "", f"{first['n_graphs']} windows ({first['windows']}); CVAE rows are best-of-K over K = {first['k']} "
             "samples, deterministic rows K = 1. Distances in metres. Moving agents: ground-truth future mean "
             "speed > 0.2 m/s.", "",
             "| run | " + " | ".join(label for _, label in COLUMNS) + " |",
             "| --- |" + " ---: |" * len(COLUMNS)]
    rows = {"constant velocity": first["constant_velocity"], **{name: r["teacher_forced"] for name, r in reports.items()}}
    for name, metrics in rows.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(metrics.get(key)) for key, _ in COLUMNS) + " |")
    lines += ["", "## Closed loop (re-encoding the scene from the model's own predictions)", "",
              "| run | " + " | ".join(f"agent ADE, exec {k.split('_')[1]}" for k in first["closed_loop"]) + " |",
              "| --- |" + " ---: |" * len(first["closed_loop"])]
    for name, r in reports.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(v.get("agent_ade")) for v in r["closed_loop"].values()) + " |")
    if any("simulation" in r for r in reports.values()):
        keys = ["sim_implausible_agent_rate", "sim_collision_rate", "sim_speed_wasserstein", "sim_acceleration_wasserstein",
                "sim_speed_mean", "real_speed_mean"]
        lines += ["", f"## {first.get('simulation', {}).get('sim_seconds', 20):.0f} s simulation realism", "",
                  "| run | " + " | ".join(keys) + " |", "| --- |" + " ---: |" * len(keys)]
        for name, r in reports.items():
            sim = r.get("simulation", {})
            lines.append(f"| {name} | " + " | ".join(_fmt(sim.get(k)) for k in keys) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ["agent_ade", "agent_fde", "moving_ade", "moving_fde"]
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(rows), 4))
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax.bar(np.arange(len(rows)) + i * width, [m.get(key, np.nan) for m in rows.values()], width, label=key)
    ax.set_xticks(np.arange(len(rows)) + 0.4 - width / 2, list(rows), rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("metres"), ax.grid(alpha=0.3, axis="y"), ax.legend(fontsize=8)
    ax.set_title(f"{', '.join(first['datasets'])} {first['split']}: best-of-K errors")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=130)
    print(args.output)


if __name__ == "__main__":
    main()
