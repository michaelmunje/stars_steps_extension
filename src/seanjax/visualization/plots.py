"""Matplotlib figures: rollouts (static and animated), CMA-ES convergence, policy comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

ROBOT, GOAL, REFERENCE, COUNTERFACTUAL = "#1f4e9c", "#2e8b57", "#9a9a9a", "#555555"


def _np(t):
    return t.detach().cpu().numpy()


def _extent(batch, run, b, margin=1.0):
    pts = [_np(run.robot_pose[b, :, :2]), _np(batch.robot_goal[b])[None]]
    active = _np(run.ped_active[b])
    if active.any():
        pts.append(_np(run.ped_pos[b])[active])
    segs = _np(batch.segments[b])[_np(batch.segment_mask[b])]
    if len(segs):
        pts += [segs[:, :2], segs[:, 2:]]
    allp = np.concatenate([p.reshape(-1, 2) for p in pts])
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    return lo, hi


def _static(ax, batch, b):
    for seg in _np(batch.segments[b])[_np(batch.segment_mask[b])]:
        ax.plot([seg[0], seg[2]], [seg[1], seg[3]], color="black", lw=2.5, solid_capstyle="round")
    for x, y, r in _np(batch.circles[b])[_np(batch.circle_mask[b])]:
        ax.add_patch(Circle((x, y), r, color="black", alpha=0.6))
    start, goal = _np(batch.robot_start[b]), _np(batch.robot_goal[b])
    ax.plot(*start[:2], marker="s", color=ROBOT, ms=8, label="robot start")
    ax.plot(*goal, marker="*", color=GOAL, ms=16, label="goal")


def plot_rollout(batch, run, b: int = 0, *, counterfactual=None, path=None, title=None, show_reference=True):
    """Robot path, pedestrian paths (solid), their robot-absent counterfactuals (dashed) and references."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _static(ax, batch, b)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    pos, active = _np(run.ped_pos[b]), _np(run.ped_active[b])
    ref, valid = _np(batch.ped_ref[b]), _np(batch.ped_valid[b])
    for p in np.flatnonzero(_np(batch.ped_mask[b])):
        color = colors[p % 20]
        if show_reference and valid[p].any():
            ax.plot(*ref[p, valid[p]].T, ls=":", lw=1, color=REFERENCE)
        if counterfactual is not None:
            cf_active = _np(counterfactual.ped_active[b, p])
            if cf_active.any():
                ax.plot(*_np(counterfactual.ped_pos[b, p])[cf_active].T, ls="--", lw=1, color=color, alpha=0.6)
        if active[p].any():
            track = pos[p, active[p]]
            ax.plot(*track.T, lw=1.6, color=color)
            ax.add_patch(Circle(track[-1], float(batch.ped_radius[b, p]), color=color, alpha=0.5))
    end = int(run.end_step[b])
    robot = _np(run.robot_pose[b, : end + 1, :2])
    ax.plot(*robot.T, lw=3, color=ROBOT, label="robot")
    ax.add_patch(Circle(robot[-1], 0.3, color=ROBOT, alpha=0.4))
    outcome = "success" if bool(run.success[b]) else ("collision" if bool(run.ped_collision[b] | run.obstacle_collision[b]) else "timeout")
    ax.set_title(title or f"{batch.names[b]}: {outcome} at {end * run.dt:.1f} s")
    lo, hi = _extent(batch, run, b)
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal"), ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=ROBOT, lw=3, label="robot"),
               plt.Line2D([], [], color="0.35", lw=1.6, label="pedestrian (robot present)")]
    if counterfactual is not None:
        handles.append(plt.Line2D([], [], color="0.35", ls="--", label="pedestrian (robot absent)"))
    if show_reference:
        handles.append(plt.Line2D([], [], color=REFERENCE, ls=":", label="reference"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def animate_rollout(batch, run, b: int = 0, *, path, counterfactual=None, every: int = 2, fps: int = 15):
    """GIF of one environment: robot and pedestrian discs with short trails."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _static(ax, batch, b)
    lo, hi = _extent(batch, run, b)
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal"), ax.grid(alpha=0.3)
    pos, active = _np(run.ped_pos[b]), _np(run.ped_active[b])
    peds = np.flatnonzero(_np(batch.ped_mask[b]))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    discs = [ax.add_patch(Circle((0, 0), float(batch.ped_radius[b, p]), color=colors[p % 20], alpha=0.7)) for p in peds]
    ghosts = []
    if counterfactual is not None:
        ghosts = [ax.add_patch(Circle((0, 0), float(batch.ped_radius[b, p]), fill=False, ls="--", color=colors[p % 20]))
                  for p in peds]
    robot_pose = _np(run.robot_pose[b])
    robot = ax.add_patch(Circle(robot_pose[0, :2], 0.3, color=ROBOT, alpha=0.8))
    heading, = ax.plot([], [], color="white", lw=2)
    trail, = ax.plot([], [], color=ROBOT, lw=2)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    frames = range(0, run.steps, every)

    def update(k):
        for disc, p in zip(discs, peds):
            disc.set_visible(bool(active[p, k]))
            disc.center = pos[p, k]
        for ghost, p in zip(ghosts, peds):
            ghost.set_visible(bool(counterfactual.ped_active[b, p, k]))
            ghost.center = _np(counterfactual.ped_pos[b, p, k])
        x, y, th = robot_pose[k]
        robot.center = (x, y)
        heading.set_data([x, x + 0.3 * np.cos(th)], [y, y + 0.3 * np.sin(th)])
        trail.set_data(robot_pose[: k + 1, 0], robot_pose[: k + 1, 1])
        label.set_text(f"t = {k * run.dt:4.1f} s   v = {float(run.robot_v[b, k]):.2f} m/s")
        return [*discs, *ghosts, robot, heading, trail, label]

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


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


def plot_comparison(summaries: dict[str, dict[str, float]], path, metrics=None) -> None:
    """Bar chart of selected metrics for several policies (e.g. default vs optimised)."""
    metrics = metrics or ["success", "collision", "time_to_goal", "min_ped_clearance", "personal_violation_rate",
                          "disruption_displacement", "peds_affected", "command_smoothness"]
    names = list(summaries)
    fig, axes = plt.subplots(2, (len(metrics) + 1) // 2, figsize=(3.2 * ((len(metrics) + 1) // 2), 6))
    for ax, metric in zip(axes.flat, metrics):
        values = [summaries[name].get(metric, np.nan) for name in names]
        ax.bar(range(len(names)), values, color=[f"C{i}" for i in range(len(names))])
        ax.set_xticks(range(len(names)), names, rotation=20, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
