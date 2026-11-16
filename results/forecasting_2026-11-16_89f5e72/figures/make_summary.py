"""Polished summary figures of the forecasting benchmark (ETH/UCY, mean over the 5 leave-one-out folds).

    python results/forecasting_2026-11-16_89f5e72/figures/make_summary.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(D.parent / "metrics.csv")))
cl = list(csv.DictReader(open(D.parent / "closed_loop_metrics.csv")))
FOLDS = ["ethucy_eth", "ethucy_hotel", "ethucy_univ", "ethucy_zara1", "ethucy_zara2"]
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a95"
STARS = {"stars_pooled": ("STARS (previous)", "#2a78d6"), "stars_improved": ("STARS improved", "#eb6834"),
         "stars_improved_social": ("STARS collision-aware", "#1baf7a")}
BASE = {"cv": "Constant velocity", "sgan_official": "Social-GAN", "trajectron_official": "Trajectron++†",
        "socialvae_official": "SocialVAE", "agentformer_official": "AgentFormer", "moflow_official": "MoFlow"}


def mean(method, key):
    v = [float(r[key]) for r in rows if r["method"] == method and r["set"] in FOLDS and r[key] not in ("", "nan")]
    return np.mean(v) if len(v) == 5 else np.nan


plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
gt = mean("ground_truth", "col40_agent")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
OFF = {(0, "moflow_official"): (4, -15, "left"), (0, "agentformer_official"): (8, 4, "left"),
       (0, "trajectron_official"): (0, -17, "center"), (0, "sgan_official"): (8, 2, "left"),
       (1, "trajectron_official"): (0, -17, "center"), (1, "socialvae_official"): (-8, -12, "right"),
       (1, "agentformer_official"): (-4, -17, "center"), (1, "moflow_official"): (-8, 5, "right")}
for i, (ax, key, title) in enumerate([(axes[0], "min_ade", "Best of 20 samples"), (axes[1], "mean_ade", "Typical single sample (mean over 20)")]):
    ax.axhline(100 * gt, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(ax.get_xlim()[0], 100 * gt, "", color=MUTED)
    for m, name in BASE.items():
        x, y = mean(m, key), 100 * mean(m, "col40_agent")
        ax.scatter(x, y, s=70, color=GRAY, edgecolor="white", lw=2, zorder=3)
        dx, dy, ha = OFF.get((i, m), (7, 4, "left"))
        ax.annotate(name.replace("\n", " "), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, color=MUTED, ha=ha)
    for m, (name, c) in STARS.items():
        x, y = mean(m, key), 100 * mean(m, "col40_agent")
        ax.scatter(x, y, s=110, color=c, edgecolor="white", lw=2, zorder=4)
        ax.annotate(name, (x, y), xytext=(8, -4), textcoords="offset points", fontsize=10.5, color=INK, weight="bold")
    ax.set_title(title, color=INK, fontsize=12.5, loc="left")
    ax.set_xlabel("ADE (m), lower is better")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0.99, 100 * gt + 0.6, "real pedestrians", transform=ax.get_yaxis_transform(), ha="right", color=MUTED, fontsize=9.5)
axes[0].set_ylabel("Predicted agents within 0.4 m of another (%)")
axes[0].set_ylim(3, 26)
axes[1].set_xlim(right=1.12)
fig.suptitle("ETH/UCY open-loop forecasting (4.8 s): accuracy vs collisions", x=0.01, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.005, "Mean over the 5 leave-one-out scenes. Baselines are official pretrained checkpoints. "
         "STARS (previous) and improved saw hotel, univ and zara1 in training.\n†Trajectron++ inputs from observed steps only (the released model relies on future-leaking features).", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(D / "accuracy_vs_collisions.png", dpi=160)

# Closed loop: ADE and contacts vs horizon, selected methods.
CL = {"ground_truth": ("Real pedestrians", INK, "--"), "cv": ("Constant velocity", GRAY, ":"),
      "socialvae_official@4": ("SocialVAE", "#9b9a95", "-"), "moflow_official@4": ("MoFlow", "#9b9a95", "-."),
      "agentformer_official@4": ("AgentFormer", "#9b9a95", "--"),
      "stars_pooled_native": ("STARS (previous)", "#2a78d6", "-")}
MK = {"socialvae_official@4": "s", "moflow_official@4": "^", "agentformer_official@4": "D", "cv": "o"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for m, (name, c, ls) in CL.items():
    r = sorted([x for x in cl if x["method"] == m], key=lambda x: float(x["horizon_s"]))
    if not r:
        continue
    h = [float(x["horizon_s"]) for x in r]
    if m != "ground_truth":
        axes[0].plot(h, [float(x["ade"]) for x in r], color=c, ls=ls, lw=2, marker=MK.get(m, "o"), ms=6, label=name)
    axes[1].plot(h, [100 * float(x["collision_agent"]) for x in r], color=c, ls=ls, lw=2, marker=MK.get(m, "o"), ms=6, label=name)
for ax, t, y in [(axes[0], "Position error, one rollout", "ADE (m)"), (axes[1], "Agents in contact (< 0.4 m)", "% of agents")]:
    ax.set_title(t, loc="left", color=INK)
    ax.set_xlabel("simulated horizon (s)")
    ax.set_ylabel(y)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[1].legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("ETH/UCY closed loop (predictions fed back in), up to 20 s; STARS improved not yet run in this loop", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(D / "closed_loop.png", dpi=160)
print("ok")
