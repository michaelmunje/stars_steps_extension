"""Speed summary: per-dataset mean/median/p999 in-track speed, in meters/second.

For each recording, samples are stride-aligned to ``canonical_stride`` and consecutive
in-track pairs are formed; each pair's speed is its displacement divided by the elapsed
time (``delta_step / storage_hz``). Pairs from every recording in a dataset are pooled
before computing mean/median/p999. If the pooled pair count exceeds ``max_pairs``, the pool
is subsampled (fixed seed) before the quantile computation only -- this bounds the cost of
``np.median``/``np.quantile``, not the peak memory used to build the pool. Writes
``paper/stats/{dataset_dir.name.lower()}_speed_summary.json`` per input directory by
default (lower-cased to match the existing per-dataset artifacts, e.g. ``ptdpes_*``).

Example:
    python -m data_quality datasets/eth datasets/ucy datasets/atc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def speed_summary(dataset_dir:Path,storage_hz:float,canonical_stride:int,max_pairs:int=1_000_000)->dict:
    speed_chunks=[]
    for path in sorted((dataset_dir/"converted").glob("*.traj.parquet")):
        table=pq.read_table(path,columns=["track_id","step","x","y"]);ids=table["track_id"].to_numpy();steps=table["step"].to_numpy();xy=np.c_[table["x"].to_numpy(),table["y"].to_numpy()]
        order=np.lexsort((steps,ids));ids,steps,xy=ids[order],steps[order],xy[order]
        task=np.flatnonzero(steps%canonical_stride==0);ids,steps,xy=ids[task],steps[task],xy[task]
        delta_step=np.diff(steps);valid=(ids[1:]==ids[:-1])&(delta_step==canonical_stride);index=np.flatnonzero(valid)
        if len(index):speed_chunks.append(np.linalg.norm(xy[index+1]-xy[index],axis=1)/(delta_step[index]/storage_hz))
    values=np.concatenate(speed_chunks) if speed_chunks else np.array([])
    pairs=len(values)
    # max_pairs bounds only the cost of the median/quantile computation below,
    # not the peak memory used to pool `values` above (that scales with `pairs`,
    # i.e. with the full in-track pair count across every recording).
    subsampled=pairs>max_pairs
    if subsampled:values=np.random.default_rng(0).choice(values,max_pairs,replace=False)
    sampled_pairs=len(values)
    return {"pairs":pairs,"sampled_pairs":sampled_pairs,"subsampled":subsampled,"mean_mps":float(values.mean()) if len(values) else float("nan"),"median_mps":float(np.median(values)) if len(values) else float("nan"),"p999_mps":float(np.quantile(values,.999)) if len(values) else float("nan")}


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset_dirs",nargs="+",type=Path)
    parser.add_argument("--output-dir",type=Path,default=None,help="Directory for per-dataset JSON output; defaults to paper/stats/, one {dataset}_speed_summary.json per dataset_dir")
    args=parser.parse_args()
    from canonical import CANONICAL_HZ
    from dataset_paths import PAPER_STATS_ROOT, TASK_CONFIG_PATH
    storage_hz=CANONICAL_HZ
    canonical_stride=int(json.loads(TASK_CONFIG_PATH.read_text())["canonical_stride"])
    output_dir=args.output_dir or PAPER_STATS_ROOT
    for directory in args.dataset_dirs:
        result=speed_summary(directory,storage_hz,canonical_stride)
        if result["pairs"]==0:
            raise ValueError(f"{directory.name}: no in-track pairs found under {directory / 'converted'}; refusing to write a NaN-valued (invalid JSON) speed summary")
        mean=result["mean_mps"]
        level="OK" if 1.0<=mean<=1.8 else "WARN"
        note="; retained source jumps allowed" if directory.name.lower()=="ptdpes" else ""
        print(f"[{level}] {directory.name:8s} mean={mean:.3f} median={result['median_mps']:.3f} p999={result['p999_mps']:.3f} m/s{note}")
        output_path=output_dir/f"{directory.name.lower()}_speed_summary.json"
        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(json.dumps(result,indent=2)+"\n")
        print(output_path)


if __name__=="__main__":main()
