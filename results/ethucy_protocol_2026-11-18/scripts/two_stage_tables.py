"""Tables for two_stage.md from primary_per_fold.csv (mean ± std over seeds) and two_stage_paired.csv (paired CIs).

    python results/ethucy_protocol_2026-11-18/scripts/two_stage_tables.py > results/ethucy_protocol_2026-11-18/two_stage_tables.md
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1]
H = ["1.2", "2.4", "3.6", "4.8"]
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]
ROWS = [("ep_zs", None, "A: end-to-end, zero-shot (all 4 seeds)"), ("ep_zs", {1, 2, 3}, "A, seeds 1-3"),
        ("ep_zs_ts", None, "A': two-stage (SSL encoder, fine-tuned), zero-shot"),
        ("ep_zs_tsf", None, "A' frozen SSL encoder (seed 1)"), ("ep_zs", {1}, "A, seed 1"),
        ("ep_loso", None, "B: A + LOSO (all 4 seeds)"), ("ep_loso", {1, 2, 3}, "B, seeds 1-3"),
        ("ep_loso_ts", None, "B': A' + LOSO"), ("ep_scratch", None, "LOSO from scratch (seeds 0-2)"),
        ("socialvae_official", None, "SocialVAE (official)"), ("moflow_official", None, "MoFlow (official)"),
        ("agentformer_official", None, "AgentFormer (official)"), ("cv", None, "Constant velocity"),
        ("ground_truth", None, "Ground truth")]

df = pd.read_csv(D / "primary_per_fold.csv")
avg = df.groupby(["row", "seed"]).mean(numeric_only=True)


def cell(r, k, pct=False, d=3):
    v = r[k].dropna() * (100 if pct else 1)
    if v.empty:
        return "–"
    s = f"{v.mean():.{d}f}"
    return s + (f" ± {v.std():.{d}f}" if len(v) > 1 else "")


def table(title, keys):
    print(f"### {title}\n")
    print("| Row | seeds | " + " | ".join(f"{h} s" for h in H) + " |")
    print("|---|---:|" + "---:|" * len(H))
    for row, seeds, label in ROWS:
        if row not in avg.index.get_level_values(0):
            continue
        r = avg.loc[row]
        if seeds:
            r = r[r.index.isin(seeds)]
        cells = []
        for h in H:
            parts = [cell(r, k.format(h=h), pct=k.startswith("col"), d=1 if k.startswith("col") else 3) for k in keys]
            cells.append(" / ".join(parts))
        print(f"| {label} | {len(r)} | " + " | ".join(cells) + " |")
    print()


table("Single-sample ADE / FDE (m; one random sample, averaged over the K = 20 draws)", ["ade@{h}", "fde@{h}"])
table("Most-likely (mode, z = prior mean) ADE / FDE (m)", ["mode_ade@{h}", "mode_fde@{h}"])
table("KDE NLL (nats; Gaussian KDE of the 20 samples per agent and step, mean over steps up to the horizon)", ["nll@{h}"])
table("Predicted collisions < 0.2 m (% of agent-samples; ground truth in the last row)", ["col20@{h}"])

print("### Per fold at 4.8 s (single-sample ADE / mode ADE / KDE NLL; mean over seeds)\n")
print("| Row | " + " | ".join(FOLDS) + " |")
print("|---|" + "---:|" * len(FOLDS))
g = df.groupby(["row", "fold"]).mean(numeric_only=True)
for row, seeds, label in ROWS:
    if seeds or row == "ground_truth" or row not in g.index.get_level_values(0):
        continue
    print(f"| {label} | " + " | ".join(
        f"{g.loc[(row, f), 'ade@4.8']:.2f} / " + (f"{g.loc[(row, f), 'mode_ade@4.8']:.2f}" if pd.notna(g.loc[(row, f)].get('mode_ade@4.8')) else "–")
        + f" / {g.loc[(row, f), 'nll@4.8']:.2f}" for f in FOLDS) + " |")
print()

p = pd.read_csv(D / "two_stage_paired.csv")
METRICS = [("ade", "single-sample ADE"), ("fde", "single-sample FDE"), ("mode_ade", "mode ADE"),
           ("mode_fde", "mode FDE"), ("nll", "KDE NLL"), ("col20", "collisions < 0.2 m")]
print("### Paired differences a - b (5-fold mean; 95% block-bootstrap CI; ns = CI covers 0)\n")
for h in H:
    print(f"**{h} s**\n")
    print("| Comparison (a vs b) | " + " | ".join(m[1] for m in METRICS) + " |")
    print("|---|" + "---:|" * len(METRICS))
    for _, r in p.iterrows():
        cells = []
        for m, _ in METRICS:
            k = f"{m}@{h}"
            if f"{k}_diff" not in r or pd.isna(r[f"{k}_diff"]):
                cells.append("–")
                continue
            f = 100 if m == "col20" else 1
            d, lo, hi = r[f"{k}_diff"] * f, r[f"{k}_lo"] * f, r[f"{k}_hi"] * f
            sig = "" if lo > 0 or hi < 0 else " ns"
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")
        print(f"| {r['comparison']} | " + " | ".join(cells) + " |")
    print()
