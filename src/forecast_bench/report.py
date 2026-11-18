"""Score every prediction file, then write metrics (CSV/JSON), markdown tables and plots.

    python -m forecast_bench.report --root experiments/forecast_bench --out results/forecasting_<date>_<sha>
"""
from __future__ import annotations

import argparse
import json
import re
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
    # ETH/UCY protocol (results/ethucy_protocol_*): seed-averaged rows (mean over the pretraining seeds' metrics)
    "ep_loso": ("**STARS B: 4-dataset pretrain + LOSO fine-tune**", "none (standard LOSO)"),
    "ep_loso_os4": ("**STARS B + 4x sample selection**", "none (standard LOSO)"),
    "ep_loso_mode": ("&nbsp;&nbsp;STARS B, mode (z = prior mean)", "none (standard LOSO)"),
    "ep_zs": ("**STARS A: 4-dataset pretrain, zero-shot**", "none (no ETH/UCY at all)"),
    "ep_zs_os4": ("**STARS A + 4x sample selection**", "none (no ETH/UCY at all)"),
    "ep_zs_mode": ("&nbsp;&nbsp;STARS A, mode (z = prior mean)", "none (no ETH/UCY at all)"),
    "ep_scratch": ("STARS LOSO from scratch (no pretraining)", "none (standard LOSO)"),
    "ep_scratch_os4": ("STARS LOSO from scratch + 4x sample selection", "none (standard LOSO)"),
    "ep_scratch_mode": ("&nbsp;&nbsp;STARS LOSO from scratch, mode", "none (standard LOSO)"),
    "socialvae_steps": ("SocialVAE (ours, STEPS-trained)", "none (zero-shot)"),
    "moflow_steps": ("MoFlow (ours, STEPS-trained)", "none (zero-shot)"),
    "stars_loo": ("STARS LOO, first attempt (5-dataset pretrain incl. ATC + 15-epoch LOO fine-tune)", "none (LOO)"),
    "stars_steps": ("STARS STEPS-only", "none (zero-shot)"),
    "stars_noethucy": ("STARS pooled w/o ETH/UCY", "none (zero-shot)"),
    "stars_noethucy_mode": ("&nbsp;&nbsp;mode (z = prior mean)", "none (zero-shot)"),
    "stars_noethucy_goal": ("STARS pooled w/o ETH/UCY + oracle goal †", "none (zero-shot; privileged goal input)"),
    "stars_noethucy_goal_mode": ("&nbsp;&nbsp;mode (z = prior mean) †", "none (zero-shot; privileged goal input)"),
    "stars_pooled": ("[LEAKED] STARS pooled (6 datasets)", "LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean"),
    "stars_pooled_mode": ("&nbsp;&nbsp;mode (z = prior mean)", "LEAK: as STARS pooled"),
    "stars_improved": ("[LEAKED] STARS improved (7 datasets, 4x sample selection)",
                       "LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean"),
    "stars_improved_mode": ("&nbsp;&nbsp;mode (z = prior mean)", "LEAK: as STARS pooled"),
    "stars_improved_social": ("[LEAKED] STARS improved, collision-aware (refine + collision loss)",
                              "LEAK: hotel, zara1, univ(001 train, 003 val); eth, zara2 clean"),
    "stars_improved_social_mode": ("&nbsp;&nbsp;mode (z = prior mean)", "LEAK: as STARS pooled"),
    "stars_steps_diffusion": ("STARS STEPS-only, per-agent diffusion", "none (zero-shot)"),
    "stars_steps_joint_diffusion": ("STARS STEPS-only, joint diffusion", "none (zero-shot)"),
}
# Published minADE20 / minFDE20 (m) on the standard splits, from the papers (no FPC/test-time tricks unless noted).
PUBLISHED = {
    "socialvae_official": {"eth": (0.47, 0.76), "hotel": (0.14, 0.22), "univ": (0.25, 0.47), "zara1": (0.20, 0.37),
                           "zara2": (0.14, 0.28)},
    "moflow_official": {"eth": (0.40, 0.57), "hotel": (0.11, 0.17), "univ": (0.23, 0.39), "zara1": (0.15, 0.26),
                        "zara2": (0.12, 0.22)},
    # T++ published numbers rely on future-leaking features (np.gradient over whole tracks, fixed upstream in 03712b7)
    "trajectron_official": {"eth": (0.39, 0.83), "hotel": (0.12, 0.21), "univ": (0.20, 0.44), "zara1": (0.15, 0.33),
                            "zara2": (0.11, 0.25)},
    "agentformer_official": {"eth": (0.45, 0.75), "hotel": (0.14, 0.22), "univ": (0.25, 0.45), "zara1": (0.18, 0.30),
                             "zara2": (0.14, 0.24)},
    # SGAN MODEL_ZOO (SGAN-20VP-20) uses a scene-joint min: compare with the minJoint table, not per-agent minADE
    "sgan_official": {"eth": (0.77, 1.39), "hotel": (0.43, 0.88), "univ": (0.75, 1.50), "zara1": (0.34, 0.68),
                      "zara2": (0.36, 0.73)},
}
PUB_NOTE = {"trajectron_official": "paper (future-leaking features)",
            "sgan_official": "MODEL_ZOO (scene-joint min: compare minJoint)"}
LEAKED = {m: {"hotel", "univ", "zara1"} for b in ("stars_pooled", "stars_improved", "stars_improved_social")
          for m in (b, f"{b}_mode")}
POOLED = ("steps", "ptdpes", "madras", "grand_central", "atc", "eth", "ucy")


def score_all(root: Path, done: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score every prediction file; with ``done`` (a cached metrics table) only the (set, method) pairs it lacks."""
    have = set() if done is None else set(zip(done.set, done.method))
    rows = [] if done is None else done.to_dict("records")
    for wpath in sorted((root / "windows").glob("*.npz")):
        name = wpath.stem
        todo = [p for p in sorted((root / "preds" / name).glob("*.npz")) if (name, p.stem) not in have]
        if (name, "ground_truth") in have and not todo:
            continue
        windows = load(wpath)
        if (name, "ground_truth") not in have:
            rows.append({"set": name, "method": "ground_truth", **ground_truth(windows)})
        for ppath in todo:
            with np.load(ppath) as f:
                pred = f["pred"]
            rows.append({"set": name, "method": ppath.stem, **score(windows, pred)})
            print(name, ppath.stem, f"{rows[-1]['min_ade']:.3f}/{rows[-1]['min_fde']:.3f}", flush=True)
    return pd.DataFrame(rows)


SEEDED = re.compile(r"(ep_[a-z]+)_s\d+(_os4|_mode)?")


def seed_average(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the per-seed rows ``ep_<kind>_s<seed>[_os4|_mode]`` by their mean over seeds (``ep_<kind>[...]``)."""
    match = df.method.str.fullmatch(SEEDED.pattern)
    if not match.any():
        return df
    seeded = df[match].copy()
    seeded["seeds"] = 1
    seeded["method"] = seeded.method.map(lambda m: "".join(g or "" for g in SEEDED.fullmatch(m).groups()))
    num = seeded.select_dtypes("number").columns.drop("seeds")
    avg = seeded.groupby(["set", "method"])[list(num)].mean().join(seeded.groupby(["set", "method"]).seeds.sum())
    return pd.concat([df[~match], avg.reset_index()], ignore_index=True)


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
            lines.append(f"| &nbsp;&nbsp;published | " + " | ".join(fmt(*p[f]) for f in FOLDS) + f" | {avg} | {PUB_NOTE.get(method, 'paper')} |")
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


# Headline columns: (header, metric a, metric b or None, kind) -- "m": metres a/b, "%": percent, "x": plain
HEADLINE_ACC = [("minADE/FDE (per agent)", "min_ade", "min_fde", "m"),
                ("minJointADE/FDE (scene)", "min_joint_ade", "min_joint_fde", "m"),
                ("single-sample ADE/FDE", "mean_ade", "mean_fde", "m"),
                ("ADE straight / non-linear", "mean_ade_straight", "mean_ade_nonlinear", "m")]
HEADLINE_REAL = [("contact < 0.2 m", "contact20_agent", None, "%"), ("contact < 0.4 m", "contact40_agent", None, "%"),
                 ("W1 speed (m/s)", "speed_w1", None, "x"), ("W1 accel (m/s²)", "accel_w1", None, "x"),
                 ("W1 turn (deg/s)", "turn_w1", None, "x1"), ("W1 NN dist (m)", "nn_w1", None, "x")]


def headline_table(df: pd.DataFrame, sets: list[str], cols: list) -> str:
    """Macro average over ``sets`` per method; methods missing a set are skipped."""
    sub = df[df.set.isin(sets)]
    lines = ["| Method | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for method in _order(sub.method.unique()):
        r = sub[sub.method == method]
        if r.set.nunique() < len(sets):
            continue
        cells = []
        for _, a, b, kind in cols:
            if method == "ground_truth" and kind == "m" or a not in r or r[a].isna().all():
                cells.append("")
            elif kind == "m":
                cells.append(f"{r[a].mean():.2f}/{r[b].mean():.2f}")
            elif kind == "%":
                cells.append(f"{100 * r[a].mean():.1f}%")
            else:
                cells.append(f"{r[a].mean():.{1 if kind == 'x1' else 3}f}")
        lines.append(f"| {METHODS.get(method, (method,))[0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


HEADLINE_NOTE = ("minADE/FDE: per-agent best of 20 (each agent picks its own best sample); minJointADE/FDE: "
                 "scene-level best of 20 (one sample index for all agents of a window). Single-sample = "
                 "mean over the 20 samples of the per-sample error (deterministic methods: their one prediction). "
                 "*mode* rows decode the STARS CVAE prior mean (z = μ, no sampling): one deterministic prediction, the "
                 "fair comparison to constant velocity. Straight / non-linear: agents whose real future stays within / "
                 "departs from its start-end chord by more than 0.2 m on average. Contact: share of agents within "
                 "0.2 / 0.4 m of another scored agent's prediction (same sample, same step), ignoring pairs that are a "
                 "walking group in the observation (within 1 m, velocities within 0.3 m/s at the last observed step). "
                 "W1: Wasserstein-1 distance to the real per-step speed, acceleration, turning-rate and "
                 "nearest-neighbour-distance (clipped at 5 m) distributions, pooled over samples. † oracle goal input "
                 "(privileged). STARS pooled / improved rows (and their modes) trained on hotel, univ and zara1.")


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
    parser.add_argument("--rescore", action="store_true", help="score every prediction again (default: only new ones)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "metrics.csv"
    df = score_all(args.root, pd.read_csv(cache) if cache.exists() and not args.rescore else None)
    df.to_csv(cache, index=False)
    (args.out / "metrics.json").write_text(json.dumps(df.to_dict(orient="records"), indent=1))
    df = seed_average(df)
    closed_md, closed = closed_loop_tables(args.root / "closed_loop")
    if not closed.empty:
        closed.to_csv(args.out / "closed_loop_metrics.csv", index=False)
    eth = [f"ethucy_{f}" for f in FOLDS]
    pooled = [s for s in (f"pooled_{d}" for d in POOLED) if s in set(df.set)]
    share = df[df.set.isin(eth) & (df.method == "ground_truth")].get("nonlinear_share", pd.Series(dtype=float)).mean()
    sections = [
        ("ETH/UCY headline: accuracy (mean over 5 folds, m)", headline_table(df, eth, HEADLINE_ACC) + "\n\n" + HEADLINE_NOTE
         + f" Non-linear agents: {100 * share:.0f}% of ETH/UCY scored agents."),
        ("ETH/UCY headline: realism (mean over 5 folds)", headline_table(df, eth, HEADLINE_REAL)),
        ("Our datasets headline: accuracy (mean over datasets, m)", headline_table(df, pooled, HEADLINE_ACC)),
        ("Our datasets headline: realism (mean over datasets)", headline_table(df, pooled, HEADLINE_REAL)),
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
    body = "\n\n".join(f"### {t}\n\n{b}" for t, b in sections if b)
    from forecast_bench import report_text as T
    headline = (args.out / "headline.md").read_text() if (args.out / "headline.md").exists() else ""
    (args.out / "report.md").write_text("\n".join([T.INTRO, headline, T.PROTOCOL, T.METHODS, "## Results\n", body,
                                                   "", T.REPRO, T.CLOSED, T.CAVEATS, T.COMMANDS]))
    (args.out / "tables.md").write_text("\n\n".join(f"### {t}\n\n{body}" for t, body in sections if body) + "\n")
    plots(df, closed, args.out)
    print((args.out / "tables.md").read_text())


if __name__ == "__main__":
    main()
