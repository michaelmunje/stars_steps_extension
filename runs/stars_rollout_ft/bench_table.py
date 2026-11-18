"""Markdown tables of the ETH/UCY closed-loop benchmark (results/ethucy_rft_eval): ADE / collisions per pooled split,
kinematics W1 at 8 s, paired ADE differences. Usage: python bench_table.py RUN_DIR"""
import sys
from pathlib import Path

import pandas as pd

M = ['cv', 'orca', 'sfm', 'stars_improved', 'stars_improved_social', 'stars_rft', 'stars_rft_os4', 'stars_rft_social']
run = Path(sys.argv[1]) / "metrics"


def md(frame, index=True):
    frame = frame.reset_index() if index else frame
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


d = pd.read_csv(run / "per_rollout.csv")
d = d[d.method.isin(M)]


def tab(f, title):
    g = f.groupby(['method', 'horizon_s'])[['ade_sum', 'agents', 'colliding_pairs', 'pairs', 'hacs_sum', 'agent_steps']].sum()
    a = (g.ade_sum / g.agents).unstack()[[4.8, 8.0, 12.0]].round(3)
    a.columns = ['ADE 4.8 s', 'ADE 8 s', 'ADE 12 s']
    c = (100 * g.colliding_pairs / g.pairs).unstack()[[8.0]].round(2)
    c.columns = ['colliding pairs % (8 s)']
    h = (g.hacs_sum / g.agent_steps).unstack()[[8.0]].round(4)
    h.columns = ['HACS (8 s)']
    out = pd.concat([a, c, h], axis=1).reindex([m for m in M if m in a.index])
    out.index.name = "method"
    print(f"**{title}**\n\n{md(out)}\n")


tab(d, "All windows (549)")
for s in ['train', 'val', 'test']:
    tab(d[d.pooled_split == s], f"Pooled split `{s}` recordings")
k = pd.read_csv(run / "kinematics.csv").set_index('method')
k = k.reindex([m for m in ['ground_truth'] + M if m in k.index])[
    ['speed_mean', 'speed_w1_vs_gt', 'acceleration_mean', 'acceleration_w1_vs_gt', 'turning_mean', 'turning_w1_vs_gt']].round(3)
print(f"**Kinematics at 8 s (W1 to ground truth)**\n\n{md(k)}\n")
p = pd.read_csv(run / "paired.csv")
q = p[(p.metric == 'ade') & (p.level == 'micro') & p.a.isin(['stars_rft', 'stars_rft_os4', 'stars_rft_social'])
      & p.b.isin(['cv', 'orca', 'stars_improved', 'stars_improved_social'])]
print(f"**Paired micro ADE differences (a - b, 95% block-bootstrap CI)**\n\n"
      f"{md(q[['horizon_s', 'a', 'b', 'a_minus_b', 'ci_low', 'ci_high']].round(3), index=False)}")
