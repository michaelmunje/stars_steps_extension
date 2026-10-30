"""Matplotlib figures: CMA-ES convergence and GIF grids of rollouts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROBOT, GOAL = "#1f4e9c", "#2e8b57"


def plot_convergence(generations_file, path) -> None:
    """Training objective (best / mean of each generation) and validation objective over generations."""
    records = [json.loads(line) for line in Path(generations_file).read_text().splitlines() if line.strip()]
    train = [r for r in records if r["kind"] == "train"]
    val = [r for r in records if r["kind"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if train:
        g = [r["generation"] for r in train]
        axes[0].plot(g, [r["objective_best"] for r in train], label="generation best")
        axes[0].plot(g, [r["objective_mean"] for r in train], label="generation mean")
        axes[0].set_title("Training objective J (fresh scenarios each generation)")
        twin = axes[0].twinx()
        twin.plot(g, [r["sigma"] for r in train], color="gray", ls=":", label="sigma")
        twin.set_ylabel("CMA-ES step size", color="gray")
    for candidate, style in (("default", "o"), ("mean", "-"), ("generation_best", "--")):
        rows = [r for r in val if r.get("candidate") == candidate]
        if rows:
            axes[1].plot([r["generation"] for r in rows], [r["val_objective"] for r in rows], style, label=candidate)
    axes[1].set_title("Validation objective J (fixed scenarios)")
    for ax in axes:
        ax.set_xlabel("generation"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_grid(snapshots: list[dict], row_labels: list[str], path, *, title: str = "", every: int = 3,
                 fps: int = 8, trail_s: float = 2.0, hold_s: float = 1.5) -> None:
    """GIF grid: one row per snapshot (e.g. a pedestrian condition), one column per scene.

    Each snapshot is ``evaluate.snapshot`` output for the same scenes. Columns share a view,
    so rows are directly comparable; each cell shows walls, pedestrians with ``trail_s`` trails
    (red outline within 0.45 m of the robot), the robot, its path and current plan, and the outcome.
    With ``ped_ghost`` (the robot-absent rollout), a dashed circle shows where a pedestrian would be
    without the robot whenever that is over 0.1 m away.
    """
    R, C = len(snapshots), len(snapshots[0]["names"])
    views = []
    for c in range(C):
        pts = np.concatenate([np.concatenate([s["robot_pose"][c, : int(s["end_step"][c]) + 1, :2], s["goal"][c][None]])
                              for s in snapshots])
        lo, hi = pts.min(0) - 3.0, pts.max(0) + 3.0
        span, centre = np.maximum(hi - lo, 8.0), (lo + hi) / 2
        views.append((centre - span / 2, centre + span / 2))
    aspect = max((hi - lo)[1] / (hi - lo)[0] for lo, hi in views)
    height = 3.4 * aspect * R * 1.12 + 1.3                                   # rows + title + legend (inches)
    fig, axes = plt.subplots(R, C, figsize=(3.4 * C, height), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.75 / height, top=1 - 0.55 / height, wspace=0.05, hspace=0.12)
    fig.suptitle(title, fontsize=11)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    legend = [Line2D([], [], marker="o", color=ROBOT, ls="", ms=8, label="robot"),
              Line2D([], [], color=ROBOT, alpha=0.6, label="robot path"),
              Line2D([], [], color=ROBOT, ls="--", label="current plan"),
              Line2D([], [], marker="*", color=GOAL, ls="", ms=11, label="goal"),
              Line2D([], [], marker="o", color=colors[0], ls="", ms=8, label="pedestrian (2 s trail)"),
              Line2D([], [], marker="o", color=colors[0], mec="red", mew=1.5, ls="", ms=8, label="within 0.45 m of robot")]
    if any("ped_ghost" in s for s in snapshots):
        legend.append(Line2D([], [], marker="o", mfc="none", mec=colors[0], ls=":", color="gray", ms=8,
                             label="same pedestrian without the robot"))
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    trail = max(1, int(round(trail_s / snapshots[0]["dt"])))
    last = int(max(s["end_step"].max() for s in snapshots))
    frames = list(range(0, last + 1, every)) + [last] * int(hold_s * fps)

    def update(k):
        for r, s in enumerate(snapshots):
            for c in range(C):
                ax, (lo, hi) = axes[r, c], views[c]
                ax.clear()
                ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
                ax.set_xticks([]), ax.set_yticks([])
                end = int(s["end_step"][c])
                kk = min(k, end)
                for seg in s["segments"][c][s["segment_mask"][c]]:
                    ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color="black", lw=2)
                ax.plot(*s["goal"][c], marker="*", color=GOAL, ms=11)
                x, y, th = s["robot_pose"][c, kk]
                for p in np.flatnonzero(s["ped_mask"][c]):
                    if not s["ped_active"][c, p, kk]:
                        continue
                    k0 = max(0, kk - trail)
                    seg = s["ped_pos"][c, p, k0:kk + 1][s["ped_active"][c, p, k0:kk + 1]]
                    ax.plot(*seg.T, color=colors[p % 20], lw=1, alpha=0.5)
                    near = np.hypot(*(s["ped_pos"][c, p, kk] - (x, y))) - s["ped_radius"][c, p] - 0.3 < 0.45
                    ax.add_patch(Circle(s["ped_pos"][c, p, kk], s["ped_radius"][c, p], facecolor=colors[p % 20],
                                        alpha=0.85, edgecolor="red" if near else "none", lw=1.5))
                    ghost = s.get("ped_ghost")
                    if ghost is not None and s["ghost_active"][c, p, kk] \
                            and np.hypot(*(ghost[c, p, kk] - s["ped_pos"][c, p, kk])) > 0.1:
                        ax.plot(*np.stack([ghost[c, p, kk], s["ped_pos"][c, p, kk]]).T, color="gray", lw=0.8, ls=":")
                        ax.add_patch(Circle(ghost[c, p, kk], s["ped_radius"][c, p], fill=False,
                                            edgecolor=colors[p % 20], lw=1.0, ls="--"))
                ax.plot(*s["robot_pose"][c, : kk + 1, :2].T, color=ROBOT, lw=1.5, alpha=0.6)
                if s["plans"] and kk < len(s["plans"]):
                    plan, steps = s["plans"][kk]
                    ax.plot(*np.concatenate([[[x, y]], plan[c, : int(steps[c])]]).T, color=ROBOT, lw=1.2, ls="--")
                ax.add_patch(Circle((x, y), 0.3, color=ROBOT))
                ax.plot([x, x + 0.4 * np.cos(th)], [y, y + 0.4 * np.sin(th)], color="white", lw=1.5)
                status = f"{kk * s['dt']:.1f} s"
                if k >= end:
                    status += "  " + ("reached" if s["success"][c] else "COLLISION" if s["collision"][c] else "timeout")
                ax.text(0.02, 0.97, status, transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
                if r == 0:
                    ax.set_title(s["names"][c].rsplit("-", 1)[0], fontsize=9)
                if c == 0:
                    ax.set_ylabel(row_labels[r], fontsize=8)

    FuncAnimation(fig, update, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps))
    update(last)
    fig.savefig(Path(path).with_suffix(".png"), dpi=90)
    plt.close(fig)
