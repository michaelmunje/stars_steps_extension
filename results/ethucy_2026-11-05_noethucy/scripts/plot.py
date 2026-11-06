"""Goal-conditioned vs control (both trained without ETH/UCY) on the ETH/UCY benchmark.

    python results/ethucy_2026-11-05_noethucy/scripts/plot.py   # -> results/ethucy_2026-11-05_noethucy/plots/goal_conditioning.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "results/ethucy_2026-11-05_noethucy"
agg = pd.read_csv(f"{R}/metrics/aggregate.csv")
SERIES = [("stars_noethucy", "STARS, no ETH/UCY (control)", "#2a78d6"),
          ("stars_noethucy_goal", "STARS, no ETH/UCY, goal-conditioned", "#eb6834"),
          ("stars_noethucy_goal_adapter", "goal-conditioned + goal adapter", "#1baf7a"),
          ("stars_pooled_goal", "STARS pooled (in-domain) + goal adapter", "#eda100"),
          ("orca", "ORCA (calibrated)", "#e87ba4")]
PANELS = [("ade", "ADE (m)", 1), ("fde", "FDE (m)", 1), ("collision_pair_time", "Colliding pair-time (%)", 100)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (metric, title, scale) in zip(axes, PANELS):
    for method, label, color in SERIES:
        f = agg[(agg.method == method) & (agg.metric == metric) & (agg.level == "micro")].sort_values("horizon_s")
        ax.fill_between(f.horizon_s, f.ci_low * scale, f.ci_high * scale, color=color, alpha=0.12, lw=0)
        ax.plot(f.horizon_s, f.value * scale, color=color, lw=2, marker="o", ms=4, label=label)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("horizon (s)")
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("ETH/UCY closed-loop benchmark, all five scenes zero-shot for the no-ETH/UCY models "
             "(micro average, 95% block-bootstrap CI, mean of 5 seeds)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{R}/plots/goal_conditioning.png", dpi=160)
