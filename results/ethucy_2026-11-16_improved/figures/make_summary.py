"""Summary figures of the closed-loop ETH/UCY benchmark (run results/ethucy_2026-11-16_improved).

    python results/ethucy_2026-11-16_improved/figures/make_summary.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
agg = pd.read_csv(D.parent / "metrics/aggregate.csv").query("level == 'micro'")
kin = pd.read_csv(D.parent / "metrics/kinematics.csv").set_index("method")
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a95"
# method -> (label, colour, line style, marker filled)
M = {"cv": ("Constant velocity", "#3d3c39", ":", True),
     "sfm": ("Social force (calibrated) †", GRAY, "-", True),
     "orca": ("ORCA (calibrated) †", GRAY, "--", True),
     "stars_steps_joint_diffusion": ("STARS joint diffusion (STEPS)", "#b9b8b3", "-.", True),
     "stars_pooled": ("STARS (previous)", "#2a78d6", "-", True),
     "stars_pooled_mode": ("STARS (previous), mode", "#2a78d6", "--", False),
     "stars_improved": ("STARS improved", "#eb6834", "-", True),
     "stars_improved_mode": ("STARS improved, mode", "#eb6834", "--", False),
     "stars_improved_social": ("STARS collision-aware", "#1baf7a", "-", True),
     "stars_noethucy_goal": ("STARS goal-conditioned †", "#8f4bd1", "-", True)}
M = {m: v for m, v in M.items() if m in set(agg.method)}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def value(m, metric, h):
    r = agg[(agg.method == m) & (agg.metric == metric) & (agg.horizon_s == h)]
    return float(r.value.iloc[0]) if len(r) else np.nan


# ADE vs horizon, and at 8 s split into near-straight / non-linear agents.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
hs = sorted(agg.horizon_s.unique())
for m, (name, c, ls, filled) in M.items():
    if "diffusion" in m:   # far off the scale (1.3 m above CV at 4.8 s); shown in the bars
        continue
    axes[0].plot(hs, [value(m, "ade", h) - value("cv", "ade", h) for h in hs], color=c, ls=ls, lw=2, marker="o", ms=5,
                 mfc=c if filled else "white", label=name)
axes[0].set(xlabel="simulated horizon (s)", ylabel="ADE minus constant-velocity ADE (m)")
axes[0].set_title("Position error relative to constant velocity (one rollout)", loc="left", color=INK, fontsize=11)
axes[0].set_ylim(-0.22, 0.44)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
names = list(M)
y = np.arange(len(names))[::-1]
for off, metric, alpha, lab in ((0.2, "lade", 1.0, "near-straight agents"), (-0.2, "nade", 0.45, "non-linear agents")):
    v = [value(m, metric, 8.0) for m in names]
    axes[1].barh(y + off, v, height=0.38, color=[M[m][1] for m in names], alpha=alpha, edgecolor="white", lw=1, label=lab)
    for yy, vv in zip(y + off, v):
        axes[1].text(vv, yy, f" {vv:.2f}", va="center", fontsize=8, color=MUTED)
axes[1].set_yticks(y, [M[m][0] for m in names])
axes[1].set_xlabel("ADE at 8 s (m)")
axes[1].set_title("At 8 s: near-straight (solid) vs non-linear (pale) agents", loc="left", color=INK, fontsize=10.5)
for ax in axes:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
fig.suptitle("ETH/UCY closed loop: accuracy", x=0.01, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.01, "Micro average over all ETH/UCY windows; below zero = more accurate than constant velocity. Dashed STARS "
         "lines / hollow markers: mode (latent = prior mean, deterministic).\n† uses the oracle goal (track end), which constant "
         "velocity and goal-free STARS do not have. STARS joint diffusion is off the left panel's scale.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig(D / "closed_loop_accuracy.png", dpi=160)
plt.close(fig)

# Realism at 8 s: contacts and distribution distances to the recording.
PANELS = [("contact40", "Agents within 0.4 m (%)"), ("speed", "W1 speed (m/s)"), ("acceleration", "W1 acceleration (m/s²)"),
          ("turning", "W1 turning rate (deg/s)"), ("nn_distance", "W1 nearest-neighbour dist. (m)")]
fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.8), sharey=True)
for ax, (key, title) in zip(axes, PANELS):
    if key == "contact40":
        vals = [100 * value(m, key, 8.0) for m in names]
        gt = 100 * value("ground_truth", key, 8.0)
        ax.axvline(gt, color=MUTED, ls="--", lw=1.2)
        ax.text(gt, len(names) - 0.1, " real pedestrians", color=MUTED, fontsize=9, ha="left", va="center")
        ax.set_ylim(-0.6, len(names) + 0.2)
    else:
        vals = [kin.loc[m, f"{key}_w1_vs_gt"] for m in names]
    ax.barh(y, vals, height=0.62, color=[M[m][1] for m in names], edgecolor="white", lw=1,
            hatch=None)
    for bar, m in zip(ax.patches, names):
        if not M[m][3]:
            bar.set_facecolor("white"), bar.set_edgecolor(M[m][1]), bar.set_linewidth(1.6)
    for yy, vv in zip(y, vals):
        ax.text(vv, yy, f" {vv:.1f}" if key in ("contact40", "turning") else f" {vv:.2f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlim(0, 1.25 * np.nanmax(vals))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(y, [M[m][0] for m in names])
fig.suptitle("ETH/UCY closed loop at 8 s: realism vs real pedestrians (lower W1 is better)", x=0.01, ha="left", fontsize=14,
             color=INK)
fig.text(0.01, 0.01, "Contacts: scored agents within 0.4 m of another at some step, observed walking groups ignored. W1 = "
         "Wasserstein-1 distance between simulated and real per-step distributions. Hollow bars: STARS mode. † oracle goal.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(D / "closed_loop_realism.png", dpi=160)
print("ok")
