"""Per-dataset test error of a multi-agent STARS model against constant velocity, as one figure.

    python -m baselines.plot_stars_ma_summary \\
        --report experiments/stars_ma/runs/cvae_hpo53_scratch_all/eval_test_all.json \\
        --output docs/results/stars_ma_all_test.png

Bars: constant velocity (grey reference), STARS best-of-K, and STARS re-forecasting from its own
predictions every model step (``closed_loop/exec_1``, the setting the SEAN-JAX crowd uses).
Left panel agent ADE, right panel agent FDE (closed loop: ADE only, so the right panel has two).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"steps": "STEPS", "ptdpes": "PTDPES", "madras": "Madras", "grand_central": "Grand Central",
          "ucy": "UCY", "eth": "ETH"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
REFERENCE, SERIES = "#a9a89f", ("#2a78d6", "#eb6834")          # neutral, categorical slots 1-2


def plot(report: dict, output: Path, title: str | None = None) -> Path:
    """Draw the per-dataset summary for an ``evaluate_stars_ma`` report to ``output``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = report["datasets"]
    tf, cv = report["teacher_forced"], report["constant_velocity"]
    closed = report.get("closed_loop", {}).get("exec_1")
    k = report.get("k", 20)

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6), sharey=True)
    for ax, metric in zip(axes, ("agent_ade", "agent_fde")):
        rows = [("Constant velocity", cv, REFERENCE), (f"STARS, best of {k}", tf, SERIES[0])]
        if closed is not None and metric == "agent_ade":
            rows.append(("STARS, re-forecast every step", closed, SERIES[1]))
        height = 0.8 / len(rows)
        for r, (label, source, color) in enumerate(rows):
            values = [source[f"{n}/{metric}"] for n in names]
            ys = [i + (r - (len(rows) - 1) / 2) * height for i in range(len(names))]
            ax.barh(ys, values, height=height - 0.04, color=color, label=label, edgecolor="white", linewidth=1)
            for y, v in zip(ys, values):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names], color=INK)
        ax.set_xlabel(f"{metric.split('_')[1].upper()} (m)", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8), ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"Agent {metric.split('_')[1].upper()}, test split", fontsize=10, color=INK, loc="left")
    axes[0].invert_yaxis()                      # shared y: invert once, first dataset on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5)
    if title:
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95 if title else 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    print(plot(json.loads(args.report.read_text()), args.output, args.title))


if __name__ == "__main__":
    main()
