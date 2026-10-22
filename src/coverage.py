"""Codebook-coverage curves: how much of a shared feature codebook a recording's data
covers as a function of how much of that recording is retained, under two retention
policies ("random": scattered time bins; "contiguous": one unbroken block).

Per recording, the timeline is discretized into ``n_bins = clamp(ceil(hours * 10), 1, 100)``
bins of roughly 6-minute width (the lower clamp keeps a zero-hour recording from yielding
``n_bins = 0``); a requested ``fraction`` is converted to a bin count via
``bins_to_keep = max(1, round_half_up(n_bins * fraction))`` and that many bins are drawn
without replacement for the "random" curve. Because ``bins_to_keep`` is an integer, short
recordings (few bins) round several requested fractions onto the same retained bin
count -- the curve's x-axis reports the hours actually retained
(``hours * bins_to_keep / n_bins``), not the requested fraction times hours, so the
random and contiguous series stay honestly comparable even where the discretization is
coarse. The lower clamp on ``bins_to_keep`` also means the smallest requested fractions of
short recordings are plotted at ``hours / n_bins`` rather than at zero coverage: e.g. a
3-bin recording at ``fraction = 0.1`` would round to 0 bins unclamped, but keeps 1 and
plots at ``hours / 3``. The "contiguous" curve instead selects one exact block of
``hours * fraction`` duration and is unaffected by the bin discretization.

Example:
    CUDA_VISIBLE_DEVICES="" python -m coverage datasets/eth datasets/ucy \\
        --output paper/figures/coverage/coverage.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]


def recording_features(path:Path,storage_hz:float)->tuple[np.ndarray,np.ndarray,float]:
    table=pq.read_table(path,columns=["track_id","step","x","y"]);ids=table["track_id"].to_numpy();steps=table["step"].to_numpy();xy=np.c_[table["x"].to_numpy(),table["y"].to_numpy()]
    neighbors=np.zeros(len(ids),float)
    step_order=np.argsort(steps,kind="stable")
    step_boundaries=np.r_[0,np.flatnonzero(np.diff(steps[step_order]))+1,len(steps)]
    for left,right in zip(step_boundaries[:-1],step_boundaries[1:]):
        index=step_order[left:right];tree=cKDTree(xy[index]);neighbors[index]=tree.query_ball_point(xy[index],5.0,return_length=True)-1
    features=[];times=[]
    track_order=np.lexsort((steps,ids))
    track_boundaries=np.r_[0,np.flatnonzero(np.diff(ids[track_order]))+1,len(ids)]
    for left,right in zip(track_boundaries[:-1],track_boundaries[1:]):
        index=track_order[left:right]
        if len(index)<4:continue
        delta=np.diff(xy[index],axis=0);dt=np.diff(steps[index])/storage_hz;valid=dt>0
        if not valid.any():continue
        speed=np.linalg.norm(delta[valid],axis=1)/dt[valid];length=np.linalg.norm(delta,axis=1).sum();angles=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
        features.append([speed.mean(),np.quantile(speed,.9),neighbors[index].mean(),np.linalg.norm(xy[index[-1]]-xy[index[0]])/max(length,1e-9),np.abs(np.diff(angles)).sum(),np.mean(speed<.2)])
        times.append(float((np.median(steps[index])-steps.min())/storage_hz/3600))
    span=(float(steps.max()-steps.min())/storage_hz/3600) if len(steps) else 0.0
    return np.asarray(features),np.asarray(times),span


def coverage_curves(dataset_dirs:list[Path],storage_hz:float,seeds=range(5),fractions=np.linspace(.1,1,10),k:int=256,codebook_sample_limit:int=100_000)->dict:
    records=[];union=[]
    for directory in dataset_dirs:
        recording_paths=sorted((directory/"converted").glob("*.traj.parquet"))
        for path in tqdm(recording_paths,total=len(recording_paths),desc=f"coverage features: {directory.name}",unit="recording"):
            features,times,hours=recording_features(path,storage_hz)
            if len(features):records.append((directory.name,path.stem,features,times,hours));union.append(features)
    all_features=np.concatenate(union);mean=all_features.mean(0);scale=np.where(all_features.std(0)>1e-9,all_features.std(0),1)
    normalized=(all_features-mean)/scale;clusters=min(k,len(normalized))
    codebook_training=normalized
    if len(codebook_training)>codebook_sample_limit:
        codebook_training=codebook_training[np.random.default_rng(0).choice(len(codebook_training),codebook_sample_limit,replace=False)]
    centroids,_=kmeans2(codebook_training,clusters,minit="++",seed=0,iter=50)
    output={"codebook_size":clusters,"union_feature_count":len(normalized),"codebook_training_count":len(codebook_training),"codebook_sampling":"all union features" if len(codebook_training)==len(normalized) else "deterministic uniform sample of union features","recordings":{}}
    centroid_tree=cKDTree(centroids)
    for dataset,recording,features,times,hours in tqdm(records,total=len(records),desc="coverage curves: fraction x seed sweep",unit="recording"):
        labels=centroid_tree.query((features-mean)/scale)[1];key=f"{dataset}/{recording}";curves={"random":[],"contiguous":[]}
        # n_bins/track_bins depend only on hours/times, which are fixed per recording.
        n_bins=max(1,min(100,int(np.ceil(hours*10))))
        track_bins=np.minimum((times/max(hours,1e-12)*n_bins).astype(int),n_bins-1)
        for fraction in fractions:
            random_values=[];contiguous_values=[]
            # Round-half-up (not Python/NumPy's round-half-to-even) so a tie such as
            # n_bins=5, fraction=0.5 keeps 3 bins, matching the plain-English "round" a
            # reader expects; this only changes the exact bin count at half-integer ties.
            bins_to_keep=max(1,int(np.floor(n_bins*fraction+0.5)));block_hours=hours*fraction
            retained_hours=hours*bins_to_keep/n_bins
            for seed in seeds:
                rng=np.random.default_rng(seed)
                chosen_bins=rng.choice(n_bins,bins_to_keep,replace=False);random_selected=np.flatnonzero(np.isin(track_bins,chosen_bins))
                random_values.append(int(np.bincount(labels[random_selected],minlength=clusters).astype(bool).sum())/clusters if len(random_selected) else 0.0)
                start=float(rng.uniform(0,max(0.0,hours-block_hours)))
                selected=np.flatnonzero((times>=start)&(times<=start+block_hours))
                contiguous_values.append(int(np.bincount(labels[selected],minlength=clusters).astype(bool).sum())/clusters if len(selected) else 0.0)
            # "random" reports the hours actually retained (bins_to_keep/n_bins of the
            # recording), not the requested fraction times hours, so it is plotted at the
            # x-value it truly represents rather than a rounding-distorted one; "contiguous"
            # retains exactly block_hours=hours*fraction by construction, so it is unchanged.
            curves["random"].append({"hours":retained_hours,"mean":float(np.mean(random_values)),"std":float(np.std(random_values))})
            curves["contiguous"].append({"hours":hours*fraction,"mean":float(np.mean(contiguous_values)),"std":float(np.std(contiguous_values))})
        output["recordings"][key]=curves
    return output


def plot(result:dict,output:Path)->Path:
    """Render the coverage-curve figure. Writes `output.with_suffix(".pdf")` as the primary
    vector artifact and `output.with_suffix(".png")` as a raster preview; returns the PDF path."""
    figure,axis=plt.subplots(figsize=(7,4.5))
    for key,curves in result["recordings"].items():
        for mode,style in (("random","-"),("contiguous","--")):
            values=curves[mode];axis.plot([v["hours"] for v in values],[v["mean"] for v in values],style,label=f"{key} ({mode})")
    axis.set(xlabel="Hours within recording",ylabel="Codebook coverage",ylim=(0,1));axis.legend(fontsize=6,ncol=2);figure.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    pdf_path=output.with_suffix(".pdf");figure.savefig(pdf_path)
    figure.savefig(output.with_suffix(".png"),dpi=180)
    plt.close(figure);return pdf_path


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset_dirs",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,default=Path("paper/figures/coverage/coverage.pdf"),help="Base path shared by the outputs: .pdf (primary vector figure), .png (raster preview), and .json (underlying curve data)")
    args=parser.parse_args()
    from canonical import CANONICAL_HZ
    storage_hz=CANONICAL_HZ
    result=coverage_curves(args.dataset_dirs,storage_hz)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    path=args.output.with_suffix(".json");path.write_text(json.dumps(result,indent=2)+"\n")
    pdf_path=plot(result,args.output);print(pdf_path);print(pdf_path.with_suffix(".png"));print(path)


if __name__=="__main__":main()
