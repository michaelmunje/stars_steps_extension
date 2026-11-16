"""Score every prediction file, then write metrics (CSV/JSON), markdown tables and plots.

    python -m forecast_bench.report --root experiments/forecast_bench --out results/forecasting_<date>_<sha>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bench.metrics import ground_truth, score
from forecast_bench.windows import FOLDS, ROOT, load

# Method -> (label, ETH/UCY leakage note). Leakage: was the held-out ETH/UCY scene in training/selection?
METHODS = {
    "ground_truth": ("Ground truth (reference)", ""),
    "cv": ("Constant velocity", "none"),
    "socialvae_official": ("SocialVAE (official ckpt)", "none (official LOO)"),
    "moflow_official": ("MoFlow (official ckpt)", "none (official LOO)"),
    "trajectron_official": ("Trajectron++ (official ckpt)", "none (official LOO)"),
    "agentformer_official": ("AgentFormer (official ckpt)", "none (official LOO)"),
    "sgan_official": ("Social-GAN (official ckpt)", "none (official LOO)"),
    "socialvae_steps": ("SocialVAE (ours, STEPS-trained)", "none (zero-shot)"),
    "moflow_steps": ("MoFlow (ours, STEPS-trained)", "none (zero-shot)"),
    "stars_loo": ("STARS LOO (no-ETH/UCY pretrain + LOO fine-tune)", "none (LOO)"),
    "stars_steps": ("STARS STEPS-only", "none (zero-shot)"),
    "stars_noethucy": ("STARS pooled w/o ETH/UCY", "none (zero-shot)"),
    "stars_noethucy_goal": ("STARS pooled w/o ETH/UCY + oracle goal", "none (zero-shot; privileged goal input)"),
    "stars_pooled": ("STARS pooled (6 datasets)", "LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean"),
    "stars_steps_diffusion": ("STARS STEPS-only, per-agent diffusion", "none (zero-shot)"),
    "stars_steps_joint_diffusion": ("STARS STEPS-only, joint diffusion", "none (zero-shot)"),
}
# Published minADE20 / minFDE20 (m) on the standard splits, from the papers (no FPC/test-time tricks unless noted).
PUBLISHED = {
    "socialvae_official": {"eth": (0.47, 0.76), "hotel": (0.14, 0.22), "univ": (0.25, 0.47), "zara1": (0.20, 0.37),
                           "zara2": (0.14, 0.28)},
}
LEAKED = {"stars_pooled": {"hotel", "univ", "zara1"}}
POOLED = ("steps", "ptdpes", "madras", "grand_central", "atc", "eth", "ucy")


def score_all(root: Path) -> pd.DataFrame:
    rows = []
    for wpath in sorted((root / "windows").glob("*.npz")):
        name = wpath.stem
        windows = load(wpath)
        rows.append({"set": name, "method": "ground_truth", **ground_truth(windows)})
        for ppath in sorted((root / "preds" / name).glob("*.npz")):
            with np.load(ppath) as f:
                pred = f["pred"]
            rows.append({"set": name, "method": ppath.stem, **score(windows, pred)})
            print(name, ppath.stem, f"{rows[-1]['min_ade']:.3f}/{rows[-1]['min_fde']:.3f}", flush=True)
    return pd.DataFrame(rows)


def _order(methods) -> list[str]:
    known = [m for m in METHODS if m in set(methods)]
    return known + sorted(set(methods) - set(known))


def fmt(a, b) -> str:
    return "–" if pd.isna(a) else f"{a:.2f}/{b:.2f}"


def ethucy_table(df: pd.DataFrame, a: str, b: str, *, published: bool = False) -> str:
    sets = [f"ethucy_{f}" for f in FOLDS]
    sub = df[df.set.isin(sets) & (df.method != "ground_truth")]
    lines = ["| Method | " + " | ".join(FOLDS) + " | avg | leakage |", "|---|" + "---:|" * 6 + "---|"]
    for method in _order(sub.method.unique()):
        cells, vals = [], []
        for fold in FOLDS:
            r = sub[(sub.method == method) & (sub.set == f"ethucy_{fold}")]
            if r.empty:
                cells.append("–")
                continue
            x, y = float(r[a].iloc[0]), float(r[b].iloc[0])
            vals.append((x, y))
            cells.append(fmt(x, y) + ("*" if fold in LEAKED.get(method, ()) else ""))
        avg = fmt(*np.mean(vals, 0)) if len(vals) == len(FOLDS) else "–"
        lines.append(f"| {METHODS.get(method, (method,))[0]} | " + " | ".join(cells) + f" | {avg} | "
                     f"{METHODS.get(method, ('', ''))[1]} |")
        if published and method in PUBLISHED:
            p = PUBLISHED[method]
            avg = fmt(*np.mean([p[f] for f in FOLDS], 0))
            lines.append(f"| &nbsp;&nbsp;published | " + " | ".join(fmt(*p[f]) for f in FOLDS) + f" | {avg} | paper |")
    return "\n".join(lines)


def social_table(df: pd.DataFrame, sets: list[str], cols: list[str]) -> str:
    sub = df[df.set.isin(sets)]
    lines = ["| Method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for method in _order(sub.method.unique()):
        r = sub[sub.method == method]
        if r.set.nunique() < len(sets):
            continue
        # macro average over sets
        lines.append(f"| {METHODS.get(method, (method,))[0]} | " +
                     " | ".join(f"{r[c].mean():.3f}" for c in cols) + " |")
    return "\n".join(lines)


def pooled_table(df: pd.DataFrame, a: str, b: str) -> str:
    sets = [s for s in (f"pooled_{d}" for d in POOLED) if s in set(df.set)]
    sub = df[df.set.isin(sets) & (df.method != "ground_truth")]
    head = [s.removeprefix("pooled_") for s in sets]
    lines = ["| Method | " + " | ".join(head) + " |", "|---|" + "---:|" * len(head)]
    for method in _order(sub.method.unique()):
        cells = []
        for s in sets:
            r = sub[(sub.method == method) & (sub.set == s)]
            cells.append("–" if r.empty else fmt(float(r[a].iloc[0]), float(r[b].iloc[0])))
        lines.append(f"| {METHODS.get(method, (method,))[0]} | " + " | ".join(cells) + " |")
    counts = df[df.set.isin(sets) & (df.method == "ground_truth")].set_index("set")
    lines.append("| *scored agents / windows* | " + " | ".join(
        f"{int(counts.loc[s, 'agents'])} / {int(counts.loc[s, 'windows'])}" for s in sets) + " |")
    return "\n".join(lines)


def pooled_social(df: pd.DataFrame, col: str) -> str:
    sets = [s for s in (f"pooled_{d}" for d in POOLED) if s in set(df.set)]
    sub = df[df.set.isin(sets)]
    head = [s.removeprefix("pooled_") for s in sets]
    lines = ["| Method | " + " | ".join(head) + " |", "|---|" + "---:|" * len(head)]
    for method in _order(sub.method.unique()):
        cells = []
        for s in sets:
            r = sub[(sub.method == method) & (sub.set == s)]
            cells.append("–" if r.empty or pd.isna(r[col].iloc[0]) else f"{float(r[col].iloc[0]):.3f}")
        lines.append(f"| {METHODS.get(method, (method,))[0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def closed_loop_tables(path: Path) -> tuple[str, pd.DataFrame]:
    files = sorted(path.glob("*.csv"))
    if not files:
        return "", pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files])
    g = df.groupby(["method", "horizon_s"])
    agg = pd.DataFrame({
        "ade": g.ade_sum.sum() / g.agents.sum(), "fde": g.fde_sum.sum() / g.agents.sum(),
        "collision_agent": g.agents_in_collision.sum() / g.agents.sum(),
        "speed": g.speed_sum.sum() / g.speed_n.sum(), "speed_gt": g.speed_gt_sum.sum() / g.speed_n.sum(),
        "speed_w1": g.speed_w1_sum.sum() / g.agents.sum(), "agents": g.agents.sum() / g.seed.nunique(),
    }).reset_index()
    hs = sorted(agg.horizon_s.unique())
    lines = ["| Method | " + " | ".join(f"{h:g} s" for h in hs) + " |", "|---|" + "---:|" * len(hs)]
    for metric, title in (("ade", "ADE/FDE (m)"), ("collision_agent", "agents in collision (<0.4 m)"),
                          ("speed", "mean speed (m/s)")):
        lines.append(f"| **{title}** |" + " |" * len(hs))
        for method in agg.method.unique():
            r = agg[agg.method == method].set_index("horizon_s")
            if metric == "ade":
                cells = [fmt(r.loc[h, "ade"], r.loc[h, "fde"]) if h in r.index else "–" for h in hs]
            else:
                cells = [f"{r.loc[h, metric]:.3f}" if h in r.index else "–" for h in hs]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines), agg


def plots(df: pd.DataFrame, closed: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [f"ethucy_{f}" for f in FOLDS]
    sub = df[df.set.isin(sets)].groupby("method")[["min_ade", "mean_ade", "col40_agent", "col20_agent"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, r in sub.iterrows():
        if method == "ground_truth":
            for a in ax:
                a.axhline(r.col40_agent, color="k", ls="--", lw=1)
            ax[0].text(0.02, r.col40_agent, " ground truth", va="bottom", transform=ax[0].get_yaxis_transform())
            continue
        ax[0].scatter(r.min_ade, r.col40_agent)
        ax[0].annotate(method, (r.min_ade, r.col40_agent), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].scatter(r.mean_ade, r.col40_agent)
        ax[1].annotate(method, (r.mean_ade, r.col40_agent), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set(xlabel="minADE20 (m), mean over 5 folds", ylabel="P(agent within 0.4 m of another prediction)",
              title="ETH/UCY: accuracy vs predicted close contacts")
    ax[1].set(xlabel="mean-over-samples ADE (m)", title="single-sample accuracy vs close contacts")
    fig.tight_layout()
    fig.savefig(out / "ethucy_accuracy_vs_collisions.png", dpi=150)
    plt.close(fig)
    if not closed.empty:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for method in closed.method.unique():
            r = closed[closed.method == method]
            style = dict(ls="--", color="k") if method == "ground_truth" else {}
            if method != "ground_truth":
                ax[0].plot(r.horizon_s, r.ade, marker="o", label=method)
            ax[1].plot(r.horizon_s, r.collision_agent, marker="o", label=method, **style)
            ax[2].plot(r.horizon_s, r.speed, marker="o", label=method, **style)
        ax[0].set(xlabel="horizon (s)", ylabel="ADE (m)", title="Closed-loop ADE (1 sample)")
        ax[1].set(xlabel="horizon (s)", ylabel="fraction of agents", title="Agents in collision (<0.4 m)")
        ax[2].set(xlabel="horizon (s)", ylabel="m/s", title="Mean speed")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "closed_loop_horizons.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/forecast_bench")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "metrics.csv"
    df = pd.read_csv(cache) if cache.exists() and not args.rescore else score_all(args.root)
    df.to_csv(cache, index=False)
    (args.out / "metrics.json").write_text(json.dumps(df.to_dict(orient="records"), indent=1))
    closed_md, closed = closed_loop_tables(args.root / "closed_loop")
    if not closed.empty:
        closed.to_csv(args.out / "closed_loop_metrics.csv", index=False)
    eth = [f"ethucy_{f}" for f in FOLDS]
    pooled = [s for s in (f"pooled_{d}" for d in POOLED) if s in set(df.set)]
    sections = [
        ("ETH/UCY leave-one-out: minADE20/minFDE20 (m)", ethucy_table(df, "min_ade", "min_fde", published=True)),
        ("ETH/UCY: mean-over-20-samples ADE/FDE (m)", ethucy_table(df, "mean_ade", "mean_fde")),
        ("ETH/UCY: first-sample ADE/FDE (m)", ethucy_table(df, "s0_ade", "s0_fde")),
        ("ETH/UCY: minJointADE20/minJointFDE20 (m)", ethucy_table(df, "min_joint_ade", "min_joint_fde")),
        ("ETH/UCY social plausibility (mean over 5 folds)",
         social_table(df, eth, ["col20_agent", "col40_agent", "col20_scene", "col40_scene", "speed_mean", "speed_gt", "speed_w1"])),
        ("Our datasets: minADE20/minFDE20 (m)", pooled_table(df, "min_ade", "min_fde")),
        ("Our datasets: mean-over-samples ADE/FDE (m)", pooled_table(df, "mean_ade", "mean_fde")),
        ("Our datasets: minJointADE20/FDE20 (m)", pooled_table(df, "min_joint_ade", "min_joint_fde")),
        ("Our datasets: P(agent within 0.4 m of another prediction, same sample)", pooled_social(df, "col40_agent")),
        ("Our datasets: P(agent within 0.2 m ...)", pooled_social(df, "col20_agent")),
        ("Our datasets: P(scene has a <0.4 m contact)", pooled_social(df, "col40_scene")),
        ("Our datasets: speed W1 to GT (m/s)", pooled_social(df, "speed_w1")),
        ("Closed loop (ETH/UCY, 5 scenes pooled, micro-averaged)", closed_md),
    ]
    (args.out / "tables.md").write_text("\n\n".join(f"### {t}\n\n{body}" for t, body in sections if body) + "\n")
    plots(df, closed, args.out)
    print((args.out / "tables.md").read_text())


if __name__ == "__main__":
    main()
