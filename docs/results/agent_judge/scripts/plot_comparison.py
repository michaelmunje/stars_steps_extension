"""Bar plot: random weights and the 10 CMA-ES seeds of the seed study (mean ± std, dots = seeds) against
the agent-judged run, all on the same 210 held-out scenes of the CPU setting (cost_comparison.json)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "docs/results/agent_judge/"
r = json.load(open(D + "cost_comparison.json"))
rows = [("Random weights\n(10 seeds)", [r[f"random s{i}"] for i in range(10)]),
        ("CMA-ES\n(10 seeds)", [r[f"seed study s{i}"] for i in range(10)]),
        ("Agent-judged\nCMA-ES", [r["A: agent-judged"]])]
metrics = [("S", "Success rate (%) - higher is better", 100, (0, 108)),
           ("intimate_violation_rate", "Time within 0.45 m of a person (%) - lower is better", 100, (0, None)),
           ("pj", "Hand-crafted reward (higher is better)", 1, (0.8, 2.0))]
colours = ["#9aa7b8", "#1f4e9c", "#d9822b"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, (key, title, scale, ylim) in zip(axes, metrics):
    for i, ((name, group), c) in enumerate(zip(rows, colours)):
        v = np.array([g[key] for g in group]) * scale
        lo_cap, hi_cap = (0, 100) if scale == 100 else (-np.inf, np.inf)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        err = [[v.mean() - max(lo_cap, v.mean() - sd)], [min(hi_cap, v.mean() + sd) - v.mean()]]
        ax.bar(i, v.mean(), color=c, alpha=0.85, width=0.65, yerr=err if len(v) > 1 else None, capsize=6)
        if len(v) > 1:
            ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, color="black", s=10, zorder=3)
        digits = 2 if scale == 1 else 1
        label = f"{v.mean():.{digits}f}" + (f" ± {sd:.{digits}f}" if len(v) > 1 else "")
        top = min(hi_cap, v.mean() + sd)
        ax.annotate(label, (i, max(top, v.max())), ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_xticks(range(len(rows)), [n for n, _ in rows], fontsize=9)
    ax.set_title(title, fontsize=10.5), ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(ylim[0], ylim[1] if ylim[1] else ax.get_ylim()[1] * 1.15)
fig.suptitle("Held-out test, 210 scenes (CPU setting)\nRandom weights and CMA-ES: mean ± std over 10 seeds (dots: seeds; "
             "CMA-ES seed s started from random seed s; trained on the hand-crafted reward). Agent-judged: one run", fontsize=11)
fig.tight_layout()
fig.savefig(D + "comparison.png", dpi=130)
print("saved", D + "comparison.png")
