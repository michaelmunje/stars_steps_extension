"""Polished summary figures of the forecasting benchmark (ETH/UCY, mean over the 5 leave-one-out folds).

    python results/forecasting_2026-11-19_ethucy_protocol/figures/make_summary.py
"""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(D.parent / "metrics.csv")))
cl = list(csv.DictReader(open(D.parent / "closed_loop_metrics.csv")))
FOLDS = ["ethucy_eth", "ethucy_hotel", "ethucy_univ", "ethucy_zara1", "ethucy_zara2"]
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a95"
# ETH/UCY protocol rows (seed-averaged): A = zero-shot (no ETH/UCY in training), B = A + leave-one-scene-out
# fine-tune (the baselines' protocol); both with 4x sample selection. The leaked row trained on hotel/univ/zara1.
STARS = {"ep_loso_os4": ("STARS B (LOSO)", "#eb6834"), "ep_zs_os4": ("STARS A (zero-shot)", "#2a78d6"),
         "stars_improved": ("STARS improved [leaked]", "#c9c8c3")}
MODE = {"ep_loso_os4": "ep_loso_mode", "ep_zs_os4": "ep_zs_mode", "stars_improved": "stars_improved_mode"}
BASE = {"cv": "Constant velocity", "sgan_official": "Social-GAN", "trajectron_official": "Trajectron++†",
        "socialvae_official": "SocialVAE", "agentformer_official": "AgentFormer", "moflow_official": "MoFlow"}


def mean(method, key):
    """5-fold mean; ``ep_<kind>[_os4|_mode]`` = mean over the seed rows ``ep_<kind>_s<seed>[...]``."""
    m = re.fullmatch(r"(ep_[a-z_]+?)(_os4|_mode)?", method)
    pat = re.compile(rf"{m[1]}_s\d+{m[2] or ''}") if m else None
    match = (lambda x: bool(pat.fullmatch(x))) if pat else (lambda x: x == method)
    v = [float(r[key]) for r in rows if match(r["method"]) and r["set"] in FOLDS and r[key] not in ("", "nan")]
    return np.mean(v) if len(v) % 5 == 0 and v else np.nan


plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
COL = "contact40_agent"   # within 0.4 m of another predicted agent, observed walking groups ignored
gt = mean("ground_truth", COL)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
OFF = {(0, "moflow_official"): (4, -15, "left"), (0, "agentformer_official"): (8, -13, "left"),
       (0, "socialvae_official"): (6, 5, "left"),
       (0, "trajectron_official"): (0, -17, "center"), (0, "sgan_official"): (8, 2, "left"),
       (1, "trajectron_official"): (0, -17, "center"), (1, "socialvae_official"): (-8, -12, "right"),
       (1, "agentformer_official"): (0, 8, "center"), (1, "moflow_official"): (0, -17, "center")}
for i, (ax, key, title) in enumerate([(axes[0], "min_ade", "Best of 20 samples"), (axes[1], "mean_ade", "Typical single sample (mean over 20)")]):
    ax.axhline(100 * gt, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(ax.get_xlim()[0], 100 * gt, "", color=MUTED)
    for m, name in BASE.items():
        x, y = mean(m, key), 100 * mean(m, COL)
        ax.scatter(x, y, s=70, color=GRAY, edgecolor="white", lw=2, zorder=3)
        dx, dy, ha = OFF.get((i, m), (7, 4, "left"))
        ax.annotate(name.replace("\n", " "), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, color=MUTED, ha=ha)
    for m, (name, c) in STARS.items():
        x, y = mean(m, key), 100 * mean(m, COL)
        ax.scatter(x, y, s=110, color=c, edgecolor="white", lw=2, zorder=4)
        dx, dy = {"ep_loso_os4": (8, 9), "ep_zs_os4": (8, -7), "stars_improved": (8, -19)}[m]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10.5, va="center",
                    color=MUTED if "leaked" in name else INK, weight="normal" if "leaked" in name else "bold")
        if i == 1 and not np.isnan(mean(MODE[m], key)):   # deterministic mode: hollow marker, joined to the sampled one
            xm, ym = mean(MODE[m], key), 100 * mean(MODE[m], COL)
            ax.plot([x, xm], [y, ym], color=c, lw=1, alpha=0.6, zorder=2)
            ax.scatter(xm, ym, s=90, facecolor="white", edgecolor=c, lw=2, zorder=4)
    ax.set_title(title, color=INK, fontsize=12.5, loc="left")
    ax.set_xlabel("ADE (m), lower is better")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0.99, 100 * gt + 0.6, "real pedestrians", transform=ax.get_yaxis_transform(), ha="right", color=MUTED, fontsize=9.5)
axes[0].set_ylabel("Predicted agents within 0.4 m of another (%)\nwalking groups excluded")
axes[1].set_xlim(right=1.12)
fig.suptitle("ETH/UCY open-loop forecasting (4.8 s): accuracy vs collisions", x=0.01, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.005, "Mean over the 5 leave-one-out scenes. Baselines are official pretrained checkpoints. "
         "STARS A/B never saw the test scene; STARS improved [leaked] saw hotel, univ and zara1 in training.\nHollow: STARS mode (latent = prior mean, one deterministic prediction). "
         "†Trajectron++ inputs from observed steps only (the released model relies on future-leaking features).", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(D / "accuracy_vs_collisions.png", dpi=160)

# Realism: contacts and distribution distances to the real data, one panel per metric.
REAL = [("cv", "Constant velocity", GRAY), ("sgan_official", "Social-GAN", GRAY), ("trajectron_official", "Trajectron++†", GRAY),
        ("socialvae_official", "SocialVAE", GRAY), ("agentformer_official", "AgentFormer", GRAY), ("moflow_official", "MoFlow", GRAY),
        ("stars_steps_joint_diffusion", "STARS joint diffusion (STEPS)", "#b9b8b3"),
        ("stars_improved", "STARS improved [leaked]", "#c9c8c3"), ("ep_zs_os4", "STARS A (zero-shot)", "#2a78d6"),
        ("ep_loso_os4", "STARS B (LOSO)", "#eb6834"), ("ep_loso_mode", "STARS B, mode", "#f4a582")]
PANELS = [(COL, "Agents within 0.4 m (%)", 100), ("speed_w1", "W1 speed (m/s)", 1), ("accel_w1", "W1 acceleration (m/s²)", 1),
          ("turn_w1", "W1 turning rate (deg/s)", 1), ("nn_w1", "W1 nearest-neighbour dist. (m)", 1)]
fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.6), sharey=True)
ys = np.arange(len(REAL))[::-1]
for ax, (key, title, scale) in zip(axes, PANELS):
    vals = [scale * mean(m, key) for m, _, _ in REAL]
    ax.barh(ys, vals, height=0.62, color=[c for _, _, c in REAL], edgecolor="white", lw=1)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:.1f}" if scale == 100 or key == "turn_w1" else f" {v:.2f}", va="center", fontsize=8.5, color=MUTED)
    if key == COL:
        ax.axvline(100 * gt, color=MUTED, ls="--", lw=1.2)
        ax.text(100 * gt, len(REAL) - 0.1, " real pedestrians", color=MUTED, fontsize=9, ha="left", va="center")
    ax.set_ylim(-0.6, len(REAL) + 0.2)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlim(0, 1.25 * np.nanmax(vals))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(ys, [n for _, n, _ in REAL])
fig.suptitle("ETH/UCY open-loop forecasting: realism vs real pedestrians (lower is better for W1)", x=0.01, ha="left",
             fontsize=14, color=INK)
fig.text(0.01, 0.01, "Mean over the 5 leave-one-out scenes, all 20 samples pooled (STARS mode: one deterministic prediction). "
         "W1 = Wasserstein-1 distance between the predicted and real per-step distributions; contacts ignore observed walking groups.",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(D / "realism.png", dpi=160)

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
