"""Per-dataset test ADE: one model per dataset against one model trained on all of them.

    python -m baselines.plot_stars_ma_single_vs_all --output docs/results/stars_ma_single_vs_all.png

Single-dataset reports are ``<runs>/<prefix><dataset>/eval_test.json``; datasets without one are
skipped. STEPS's own model is ``cvae_hpo53_scratch_seed42`` (its report: ``eval_test.json``).
Bars: constant velocity (grey reference), the single-dataset model and the all-dataset model, with
the training-window count of each dataset under its name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.plot_stars_ma_summary import GRID, INK, LABELS, MUTED, REFERENCE, SERIES

SINGLE_RUN = {"steps": "cvae_hpo53_scratch_seed42"}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("experiments/stars_ma/runs"))
    parser.add_argument("--all-report", type=Path,
                        default=Path("experiments/stars_ma/runs/cvae_hpo53_scratch_all/eval_test_all.json"))
    parser.add_argument("--prefix", default="cvae_hpo53_scratch_")
    parser.add_argument("--metric", default="agent_ade")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = json.loads(args.all_report.read_text())
    rows = []
    for name in combined["datasets"]:
        run = args.runs / SINGLE_RUN.get(name, args.prefix + name)
        report = run / "eval_test.json"
        if not report.exists():
            continue
        single = json.loads(report.read_text())
        windows = json.loads((run / "status.json").read_text()).get("n_train_windows")
        key = f"{name}/{args.metric}"
        rows.append((name, windows, combined["constant_velocity"][key], single["teacher_forced"][key],
                     combined["teacher_forced"][key]))

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.6))
    series = [("Constant velocity", 2, REFERENCE), ("Trained on this dataset only", 3, SERIES[1]),
              ("Trained on all six datasets", 4, SERIES[0])]
    height = 0.8 / len(series)
    for r, (label, column, color) in enumerate(series):
        ys = [i + (r - 1) * height for i in range(len(rows))]
        values = [row[column] for row in rows]
        ax.barh(ys, values, height=height - 0.04, color=color, label=label, edgecolor="white", linewidth=1)
        for y, v in zip(ys, values):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [f"{LABELS.get(n, n)}\n{w:,} windows" if w else LABELS.get(n, n)
                                     for n, w, *_ in rows], color=INK, fontsize=8)
    ax.invert_yaxis()
    unit = args.metric.split("_")[1].upper()
    ax.set_xlabel(f"Agent {unit} on the test split (m), best of {combined.get('k', 20)}", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8), ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("One model per dataset vs one model for all", fontsize=10, color=INK, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(args.output, [r[0] for r in rows])


if __name__ == "__main__":
    main()
