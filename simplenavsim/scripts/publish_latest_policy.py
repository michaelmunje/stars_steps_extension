#!/usr/bin/env python3
"""Rebuild ``docs/latest_policy/``: the current reference navigation policy, its test results and its
visuals, as a folder GitHub renders in the browser (README.md with inline GIFs and figures).

    python simplenavsim/scripts/publish_latest_policy.py --name "Crowd v4" \\
        --policy checkpoints/simplenavsim/cmaes_crowd_v4/best_policy.yaml \\
        --config simplenavsim/configs/cmaes_crowd_v4.yaml \\
        --evaluation runs/simplenavsim/crowd_v6/evaluate/evaluation_test.json \\
        --compare docs/results/navigation_compare_crowd_v6.png \\
        --visuals runs/simplenavsim/crowd_v6/visualize --decisions 75

The output folder is deleted and recreated every time, so it only ever holds one version.
Visuals come from ``--visuals`` (a ``visualize_rollout.py`` output directory: ``*.gif``, ``*.png``,
``convergence.png``). With ``--render`` they are rendered first into that directory instead (one
GPU, the scenarios in ``--scenarios``, seed 0, plus recorded seeds 0-3), using ``--run-dir`` for the
convergence plot.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = (("objective", "Objective", "{:.3f}"), ("success", "Success", "{:.0%}"), ("collision", "Collision", "{:.1%}"),
           ("intimate_violation_rate", "< 0.45 m", "{:.1%}"), ("path_efficiency", "Path eff.", "{:.2f}"),
           ("loops", "Loops", "{:.2f}"), ("time_to_goal", "Time (s)", "{:.1f}"))
CAPTIONS = {"recorded": "Recorded STEPS traffic", "dense_flow": "Dense cross flow in a corridor",
            "narrow_passage": "Narrow passage with an oncoming pedestrian", "group_toward": "Group walking toward the robot",
            "crossing": "Single pedestrian crossing", "head_on": "Head-on pedestrian"}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row(label: str, metrics: dict) -> str:
    cells = [fmt.format(metrics[key]) if isinstance(metrics.get(key), (int, float)) else "—" for key, _, fmt in COLUMNS]
    return f"| {label} | " + " | ".join(cells) + " |"


def _render(args) -> None:
    script = ROOT / "simplenavsim/scripts/visualize_rollout.py"
    jobs = [(kind, 0) for kind in args.scenarios] + [("recorded", seed) for seed in range(args.recorded)]
    for i, (kind, seed) in enumerate(jobs):
        command = [sys.executable, str(script), "--config", str(args.config), "--checkpoint", str(args.policy),
                   "--scenario", kind, "--seed", str(seed), "--out-dir", str(args.visuals)]
        if args.device:
            command += ["--device", args.device]
        if i == 0 and args.run_dir:
            command += ["--run-dir", str(args.run_dir)]
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Human-readable version name, e.g. 'Crowd v4'")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="The CMA-ES config the policy was tuned and evaluated with")
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluate.py evaluation_test.json")
    parser.add_argument("--compare", type=Path, help="compare_policies.py figure")
    parser.add_argument("--visuals", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="Render the visuals into --visuals first")
    parser.add_argument("--scenarios", nargs="+", default=["dense_flow", "narrow_passage", "group_toward"])
    parser.add_argument("--recorded", type=int, default=4, help="Recorded STEPS scenes to render (seeds 0..n-1)")
    parser.add_argument("--run-dir", type=Path, help="CMA-ES run directory (convergence plot when rendering)")
    parser.add_argument("--device")
    parser.add_argument("--decisions", default="", help="docs/decisions.md entries behind this version, e.g. '75'")
    parser.add_argument("--notes", default="", help="Extra Markdown placed under the headline")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/latest_policy")
    args = parser.parse_args()
    if args.render:
        _render(args)

    evaluation = json.loads(args.evaluation.read_text())
    key = next((k for k in evaluation if Path(k).parent.name == args.policy.parent.name), None)
    if key is None:
        raise SystemExit(f"{args.policy} is not in {args.evaluation} (policies: {list(evaluation)})")

    shutil.rmtree(args.out, ignore_errors=True)
    (args.out / "visuals").mkdir(parents=True)
    shutil.copy(args.policy, args.out / "policy.yaml")
    shutil.copy(args.config, args.out / "config.yaml")
    shutil.copy(args.evaluation, args.out / "evaluation_test.json")
    if args.compare:
        shutil.copy(args.compare, args.out / "visuals" / "compare.png")
    gifs = sorted(args.visuals.glob("*.gif"), key=lambda p: (not p.name.startswith("recorded"), p.name))
    for path in [*gifs, *args.visuals.glob("convergence.png")]:          # rollout stills duplicate the GIFs
        shutil.copy(path, args.out / "visuals" / path.name)

    ours = evaluation[key]
    kinds = sorted(ours["per_kind"])
    lines = [f"# Latest navigation policy: {args.name}", "",
             f"Published {date.today().isoformat()} from commit `{_git_commit()}`. This folder is regenerated by "
             "`simplenavsim/scripts/publish_latest_policy.py` for every new reference policy; older versions live in git "
             "history.", ""]
    if args.decisions:
        lines += [f"Reasoning: [`docs/decisions.md`](../decisions.md), entries {args.decisions}.", ""]
    if args.notes:
        lines += [args.notes, ""]
    lines += ["## Test results", "",
              f"{ours['n_scenarios']} test scenarios in the STARS-driven crowd (`{args.config.name}`); the objective is the "
              "one this policy was tuned on. `< 0.45 m` is the fraction of time a pedestrian is in the robot's intimate "
              "zone; loops are full turns beyond half a turn per episode.", "",
              "| Policy | " + " | ".join(c[1] for c in COLUMNS) + " |", "| --- |" + " ---: |" * len(COLUMNS)]
    lines.append(_row(f"**{args.name}**", ours["overall"]))
    lines += [_row(name, result["overall"]) for name, result in evaluation.items() if name != key]
    lines += ["", "### By scenario", "", "| Scenario | " + " | ".join(c[1] for c in COLUMNS) + " |",
              "| --- |" + " ---: |" * len(COLUMNS)]
    lines += [_row(kind, ours["per_kind"][kind]) for kind in kinds]
    lines += ["", "## Rollouts", "",
              "The view follows the robot (blue disc, white tick = heading; square: start, star: goal). "
              "**Blue line**: where it has been. **Blue dashed**: the plan it just chose. **Thin coloured lines**: "
              "what it predicts each nearby pedestrian (within 4 m) will do; that forecast is what the plan is "
              "scored against. Pedestrians have fading 2 s trails and turn red-outlined inside the 0.45 m intimate "
              "zone (the dotted ring). The status line gives time, speed, the nearest gap and whether the robot is "
              "driving, waiting or stopping; the final frame states the outcome. Playback is about 1.8x real time. "
              "Single episodes show behaviour, not rates: the outcome of a close encounter can change between runs "
              "(decision 83).", ""]
    for gif in gifs:
        kind = gif.stem.split("-")[0]
        lines += [f"### {CAPTIONS.get(kind, kind)} (`{gif.stem.rsplit('-', 1)[0]}`)", "",
                  f"![{gif.stem}](visuals/{gif.name})", ""]
    if args.compare:
        lines += ["## Side by side with earlier policies", "", "![comparison](visuals/compare.png)", ""]
    if (args.out / "visuals" / "convergence.png").exists():
        lines += ["## Tuning convergence", "", "![convergence](visuals/convergence.png)", ""]
    config = args.config.resolve().relative_to(ROOT) if args.config.resolve().is_relative_to(ROOT) else args.config
    lines += ["## Use it", "", "From the repository root:", "", "```bash",
              f"python simplenavsim/scripts/evaluate.py --config {config} \\",
              "  --policy docs/latest_policy/policy.yaml --split test --device cuda",
              f"python simplenavsim/scripts/visualize_rollout.py --config {config} \\",
              "  --checkpoint docs/latest_policy/policy.yaml --scenario dense_flow --device cuda", "```", "",
              "`config.yaml` here is a copy for reference; it names its simulator, pedestrian and policy configs relative "
              "to `simplenavsim/configs/`, so run with the original path above.", "",
              "Parameters:", "", "```yaml", args.policy.read_text().strip(), "```", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    print(args.out / "README.md")


if __name__ == "__main__":
    main()
