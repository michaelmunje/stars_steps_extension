"""Test table: macro (mean over datasets) accuracy, pooled collisions/speed; seeds as mean +- sd."""
import json, sys
import numpy as np

def rows(path, suffix=""):
    d = json.load(open(path)); m = d["metrics"]; ds = [k for k in m if k != "all"]
    out = {}
    for name in d["checkpoints"]:
        g = lambda c: float(np.mean([m[k][f"{name}_{c}"] for k in ds]))
        a = m["all"]
        out[name + suffix] = {
            "ADE": g("ade"), "FDE": g("fde"), "minADE": g("minade"), "minFDE": g("minfde"),
            "jADE": g("joint_ade"), "jFDE": g("joint_fde"),
            "pair0.2": 100 * a[f"{name}_allk_pair0.2"], "pair0.4": 100 * a[f"{name}_allk_pair0.4"],
            "ng0.2": 100 * a[f"{name}_allk_nogroup_pair0.2"], "ng0.4": 100 * a[f"{name}_allk_nogroup_pair0.4"],
            "netspeed": a[f"{name}_netspeed"],
            "speedW1": float(np.mean([m[k].get(f"speed_{name}_w1", np.nan) for k in ds])),
        }
    gt = m["all"]
    return out, {"pair0.2": 100 * gt["gt_pair0.2"], "pair0.4": 100 * gt["gt_pair0.4"],
                 "ng0.2": 100 * gt["gt_nogroup_pair0.2"], "ng0.4": 100 * gt["gt_nogroup_pair0.4"],
                 "netspeed": gt["netspeed_gt"]}, ds

allrows, gt = {}, None
for arg in sys.argv[1:]:
    path, _, suffix = arg.partition(":")
    r, gt, ds = rows(path, suffix); allrows.update(r)
cols = ["ADE", "FDE", "minADE", "minFDE", "jADE", "jFDE", "pair0.2", "pair0.4", "ng0.2", "ng0.4", "netspeed", "speedW1"]
print("Datasets (macro mean):", ", ".join(ds))
print("| model | " + " | ".join(cols) + " |"); print("|---" * (len(cols) + 1) + "|")
def fmt(vals): return f"{np.mean(vals):.3f}" + (f" ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "")
groups = {}
for name, r in allrows.items():
    key = (
        "new, 3 seeds" + (name[name.index("_os"):] if "_os" in name else "") if name.startswith("base_s") else name)
    groups.setdefault(key, []).append(r)
for key, rs in groups.items():
    print(f"| {key} | " + " | ".join(fmt([r[c] for r in rs]) for c in cols) + " |")
print("| ground truth | " + " | ".join(f"{gt[c]:.3f}" if c in gt else "" for c in cols) + " |")
