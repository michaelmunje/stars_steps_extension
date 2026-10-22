from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from tqdm.auto import tqdm

from artifact_provenance import implementation_hash, sha256_file
from canonical import CANONICAL_DT
from data import _analysis_index_path
from duplicate_annotations import (
    DUPLICATE_MIN_FRAMES,
    DUPLICATE_SEPARATION_M,
    duplicate_pairs,
    duplicate_summary,
    duplicate_tracks,
)
from opentraj_indicators import (
    aligned_angular_sample,
    collision_energy,
    frame_interactions,
    frame_local_density,
    global_density,
    motion_properties,
    path_efficiency_index,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOG_DIR = REPO_ROOT / "logs"
logger = logging.getLogger(__name__)
OPENTRAJ_REFERENCE = "crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a"
INDICATOR_CONTRACT = "opentraj-retained-trajlet-indicators-v1"
INDICATOR_IMPLEMENTATION_HASH = implementation_hash(
    (ROOT / "opentraj_indicators.py", ROOT / "opentraj_report.py", ROOT / "duplicate_annotations.py"),
    base=ROOT,
)
METRIC_COLUMNS = (
    "speed_mean", "speed_range", "acceleration_mean_abs", "acceleration_max_abs",
    "path_efficiency", "local_density_max", "local_density_max_deduplicated",
    "ttc_min", "dca_min", "collision_energy",
)
# Descriptive, unit-bearing axis labels for the raw METRIC_COLUMNS names (see
# opentraj_dataset_stats.py's LaTeX header / paper_dataset_metrics.py's
# metric_units payload for the same unit conventions).
METRIC_LABELS = {
    "speed_mean": "Mean speed (m/s)",
    "speed_range": "Speed range (m/s)",
    "acceleration_mean_abs": "Mean |acceleration| (m/s²)",
    "acceleration_max_abs": "Max |acceleration| (m/s²)",
    "path_efficiency": "Path efficiency",
    "local_density_max": "Max local density (ped/m²)",
    "local_density_max_deduplicated": "Max local density, duplicates removed (ped/m²)",
    "ttc_min": "Min TTC (s)",
    "dca_min": "Min DCA (m)",
    "collision_energy": "Collision energy",
}
# Shared house style for every figure this module renders (F-005): applied as a local context
# manager rather than the global, order-dependent `sns.set()` so the result never depends on
# which plotting function happens to run first.
FIGURE_STYLE = sns.axes_style("whitegrid")


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _scene_bounds(views: list[Path]) -> dict[str, tuple[float, float, float, float]]:
    extrema: dict[str, list[float]] = {}
    for path in tqdm(views, desc="Scene bounds", unit="recording"):
        parquet = pq.ParquetFile(path);metadata={k.decode():v.decode() for k,v in (parquet.schema_arrow.metadata or {}).items()}
        table=parquet.read(columns=["smoothed_x","smoothed_y"])
        x=table["smoothed_x"].to_numpy();y=table["smoothed_y"].to_numpy();scene=metadata["scene_id"]
        values=[float(x.min()),float(x.max()),float(y.min()),float(y.max())]
        if scene not in extrema:extrema[scene]=values
        else:
            old=extrema[scene];extrema[scene]=[min(old[0],values[0]),max(old[1],values[1]),min(old[2],values[2]),max(old[3],values[3])]
    return {key:tuple(value) for key,value in extrema.items()}


def _filtered_density(state: np.ndarray, retained: np.ndarray) -> np.ndarray:
    """Local density recomputed with duplicate-annotation tracks out of the frame.

    Excluded pedestrians get NaN rather than a value: they have no
    de-duplicated density of their own, while their retained twin is measured
    against the rest of the crowd as if it had only ever been annotated once.
    """
    values = np.full(len(state), np.nan)
    values[retained] = frame_local_density(state[retained, :2])
    return values


def compute_opentraj_report(
    dataset_dir: Path,
    *,
    index_path: Path | None = None,
    sequences: set[str] | None = None,
    output_dir: Path | None = None,
    report_name: str | None = None,
) -> tuple[Path, Path]:
    index_path=index_path or dataset_dir/"indices/opentraj-analysis-trajlets.parquet"
    if not index_path.exists():raise FileNotFoundError(f"Missing {index_path}; run python -m analysis_pipeline first")
    index_revision = sha256_file(index_path)
    index=pq.read_table(index_path).to_pandas();index=index[index.retained_after_static_filter]
    if sequences is not None:index=index[index.scene_sequence.isin(sequences)]
    views=sorted((dataset_dir/"analysis/opentraj_kalman_smooth").glob("*.analysis.parquet"))
    if sequences is not None:
        views=[
            path for path in views
            if (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"scene_sequence",b"").decode() in sequences
        ]
    if not views:raise ValueError(f"No OpenTraj analysis views selected for {report_name or dataset_dir.name}")
    bounds=_scene_bounds(views);metrics=[];frame_density=[];duplicates_by_recording=[]
    aligned_sum=np.zeros((12,2));angle_sum=np.zeros(12);angle_square_sum=np.zeros(12);angular_count=0
    by_recording={key:value for key,value in index.groupby(["scene_sequence", "recording"],sort=False)}
    for path in tqdm(views,desc=f"OpenTraj indicators: {dataset_dir.name}",unit="recording"):
        parquet=pq.ParquetFile(path);metadata={k.decode():v.decode() for k,v in (parquet.schema_arrow.metadata or {}).items()}
        sequence=metadata["scene_sequence"];scene_id=metadata["scene_id"];recording=metadata["recording"]
        frame=parquet.read().to_pandas().sort_values(["step","track_id"],kind="stable")
        # One pedestrian annotated under two track ids inflates the 1/b² density
        # kernel quadratically (C-002). Detect those identities here and carry a
        # filtered density alongside the upstream-parity value, leaving both the
        # canonical data and the pinned kernel untouched.
        pairs=duplicate_pairs(frame.track_id.to_numpy(),frame.step.to_numpy(),frame.smoothed_x.to_numpy(),frame.smoothed_y.to_numpy())
        excluded=duplicate_tracks(pairs)
        duplicates_by_recording.append({"scene_sequence":sequence,"recording":recording,**duplicate_summary(pairs,excluded),"pairs":[asdict(pair) for pair in pairs]})
        interaction={};density={};filtered_density={}
        frame_groups = frame.groupby("step",sort=True)
        for frame_number, (step,current) in enumerate(frame_groups, start=1):
            state=current[["smoothed_x","smoothed_y","smoothed_vx","smoothed_vy"]].to_numpy()
            ttc,dca=frame_interactions(state);local=frame_local_density(state[:,:2])
            filtered=local if not excluded else _filtered_density(state,~current.track_id.isin(excluded).to_numpy())
            frame_density.append({"dataset":report_name or metadata["dataset"],"scene_id":scene_id,"scene_sequence":sequence,"step":int(step),"global_density":global_density(len(current),bounds[scene_id])})
            for track_id,t,d,l,f in zip(current.track_id,ttc,dca,local,filtered):interaction[(int(track_id),int(step))]=(t,d);density[(int(track_id),int(step))]=l;filtered_density[(int(track_id),int(step))]=f
            if frame_number % 10_000 == 0:
                logger.info(
                    "%s: computed %s/%s analysis frames",
                    recording, f"{frame_number:,}", f"{frame_groups.ngroups:,}",
                )
        lookup={(int(agent),int(step)):row for row,(agent,step) in enumerate(zip(frame.track_id,frame.step))}
        recording_trajlets = by_recording.get((sequence, recording),pd.DataFrame())
        for trajlet_number, item in enumerate(
            recording_trajlets.itertuples(index=False), start=1
        ):
            steps=np.arange(int(item.source_start_step),int(item.source_returned_end_step)+1)
            indices=[lookup[(int(item.agent_id),int(step))] for step in steps]
            values=frame.iloc[indices]
            xy=values[["smoothed_x","smoothed_y"]].to_numpy();velocity=values[["smoothed_vx","smoothed_vy"]].to_numpy()
            aligned,angles=aligned_angular_sample(np.c_[xy,velocity])
            if np.isfinite(aligned).all() and np.isfinite(angles).all():
                aligned_sum+=aligned;angle_sum+=angles;angle_square_sum+=angles**2;angular_count+=1
            # dt is derived from the trajlet's own (canonical-rate) timestamps rather than
            # hardcoded, extending motion_properties/path_efficiency_index's pinned OpenTraj
            # parity (tests/test_opentraj_compat.py) to this production report path (BC-004).
            timestamps=steps.astype(float)*CANONICAL_DT
            trajlet_batch=np.concatenate([xy,velocity,timestamps[:,None]],axis=1)[None,:,:]
            motion=motion_properties(trajlet_batch);efficiency=float(path_efficiency_index(trajlet_batch)[0])
            ttc=np.asarray([interaction[(int(item.agent_id),int(step))][0] for step in steps]);dca=np.asarray([interaction[(int(item.agent_id),int(step))][1] for step in steps]);local=np.asarray([density[(int(item.agent_id),int(step))] for step in steps]);local_filtered=np.asarray([filtered_density[(int(item.agent_id),int(step))] for step in steps])
            finite_ttc=ttc[np.isfinite(ttc)];finite_dca=dca[np.isfinite(dca)];energy=collision_energy(finite_ttc)
            metrics.append({"dataset":report_name or metadata["dataset"],"scene_id":scene_id,"scene_sequence":sequence,"recording":recording,"split":item.split,"agent_id":int(item.agent_id),"source_start_step":int(item.source_start_step),
                "speed_mean":float(motion["speed_mean"][0]),"speed_range":float(motion["speed_range"][0]),"acceleration_mean_abs":float(motion["acceleration_mean_abs"][0]),"acceleration_max_abs":float(motion["acceleration_max_abs"][0]),"path_efficiency":efficiency,"local_density_max":float(np.nanmax(local)) if np.isfinite(local).any() else np.nan,"local_density_max_deduplicated":float(np.nanmax(local_filtered)) if np.isfinite(local_filtered).any() else np.nan,
                "ttc_min":float(finite_ttc.min()) if len(finite_ttc) else np.nan,"dca_min":float(finite_dca.min()) if len(finite_dca) else np.nan,"collision_energy":float(energy.max()) if len(energy) else np.nan})
            if trajlet_number % 10_000 == 0:
                logger.info(
                    "%s: aggregated %s/%s retained trajlets",
                    recording, f"{trajlet_number:,}", f"{len(recording_trajlets):,}",
                )
    duplicate_totals={
        "duplicate_separation_m":DUPLICATE_SEPARATION_M,"duplicate_min_frames":DUPLICATE_MIN_FRAMES,
        "duplicate_pairs":sum(record["duplicate_pairs"] for record in duplicates_by_recording),
        "duplicate_tracks_excluded":sum(record["duplicate_tracks_excluded"] for record in duplicates_by_recording),
        "recordings_with_duplicates":sum(1 for record in duplicates_by_recording if record["duplicate_pairs"]),
    }
    output_dir=output_dir or dataset_dir/"analysis/opentraj_reports";output_dir.mkdir(parents=True,exist_ok=True)
    metric_path=output_dir/"trajlet_indicators.parquet";density_path=output_dir/"frame_global_density.parquet"
    provenance = {
        b"indicator_contract": INDICATOR_CONTRACT.encode(),
        b"analysis_preprocessing": b"opentraj_kalman_smooth",
        b"opentraj_reference": OPENTRAJ_REFERENCE.encode(),
        b"indicator_implementation_hash": INDICATOR_IMPLEMENTATION_HASH.encode(),
        b"source_analysis_index_sha256": index_revision.encode(),
        b"duplicate_separation_m": str(DUPLICATE_SEPARATION_M).encode(),
        b"duplicate_min_frames": str(DUPLICATE_MIN_FRAMES).encode(),
        b"duplicate_pairs": str(duplicate_totals["duplicate_pairs"]).encode(),
        b"duplicate_tracks_excluded": str(duplicate_totals["duplicate_tracks_excluded"]).encode(),
    }
    metric_table = pa.Table.from_pylist(metrics).replace_schema_metadata(provenance)
    density_table = pa.Table.from_pylist(frame_density).replace_schema_metadata(provenance)
    _atomic_write_table(metric_table,metric_path);_atomic_write_table(density_table,density_path)
    angular={}
    if angular_count:
        keypoints=aligned_sum/angular_count;mean_degrees=np.rad2deg(np.arctan2(keypoints[:,0],keypoints[:,1]))
        variance=np.maximum(angle_square_sum/angular_count-(angle_sum/angular_count)**2,0)
        std_degrees=np.sqrt(variance)
        # These indices intentionally match upstream, including nominal 4.8 s
        # mapping to returned sample 11 (timestamp 4.4 s).
        for nominal_s in (1.6,2.4,4.8):
            index=int(round(nominal_s/.4))-1
            angular[str(nominal_s)]={"returned_sample_index":index,"returned_timestamp_s":index*.4,"mean_degrees":float(mean_degrees[index]),"std_degrees":float(std_degrees[index])}
    def summarize(rows:list[dict],key:str)->dict:
        values=np.asarray([row[key] for row in rows],dtype=float);finite=values[np.isfinite(values)]
        return {"mean":float(finite.mean()),"median":float(np.median(finite)),"n":int(len(finite))} if len(finite) else {"mean":None,"median":None,"n":0}
    summary={"dataset":report_name or dataset_dir.name,"recordings":sorted(sequences) if sequences is not None else "all canonical recordings","analysis_preprocessing":"opentraj_kalman_smooth","retained_trajlets":len(metrics),"frames":len(frame_density),"indicators":{key:summarize(metrics,key) for key in METRIC_COLUMNS},"global_density":summarize(frame_density,"global_density"),"angular_deviation":angular,"duplicate_annotations":{**duplicate_totals,"by_recording":[record for record in duplicates_by_recording if record["duplicate_pairs"]]},"definitions":"Faithful OpenTraj-compatible analysis values; separate from forecasting statistics. local_density_max reproduces OpenTraj exactly; local_density_max_deduplicated recomputes it with duplicate-annotation track identities removed from each frame."}
    summary_path = output_dir / "summary.json"
    temporary_summary = summary_path.with_suffix(".tmp.json")
    temporary_summary.write_text(json.dumps(summary,indent=2)+"\n")
    temporary_summary.replace(summary_path)
    return metric_path,density_path


def _histogram_samples(groups:list[np.ndarray],maximum:int, bins:int,quantiles:tuple[float,float])->list[np.ndarray]:
    finite=[np.asarray(group)[np.isfinite(group)] for group in groups]
    union=np.concatenate([group for group in finite if len(group)])
    edges=np.linspace(*np.quantile(union,quantiles),bins+1)
    output=[]
    for group in finite:
        if not len(group):output.append(np.array([]));continue
        counts=np.round(np.histogram(group,bins=edges)[0]/(len(group)/maximum)).astype(int)
        output.append(np.repeat(edges[:-1],counts))
    return output


def plot_opentraj_report(metric_paths: list[Path], output: Path) -> Path:
    frames=[]
    for path in metric_paths:
        frame=pq.read_table(path).to_pandas()
        if "dataset" not in frame or frame["dataset"].nunique()!=1:frame["dataset"]=path.parent.name
        frames.append(frame)
    data=pd.concat(frames,ignore_index=True);metrics=["speed_mean","speed_range","acceleration_mean_abs","acceleration_max_abs","path_efficiency","local_density_max","ttc_min","dca_min","collision_energy"]
    with FIGURE_STYLE:
        figure,axes=plt.subplots(3,3,figsize=(13,10))
        for axis,key in zip(axes.flat,metrics):
            finite=data[np.isfinite(data[key])]
            sns.boxplot(data=finite,x="dataset",y=key,ax=axis,showfliers=False);axis.tick_params(axis="x",rotation=25);axis.set_xlabel("");axis.set_ylabel(METRIC_LABELS[key])
        figure.tight_layout();output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(output,bbox_inches="tight");plt.close(figure)
    return output


def plot_opentraj_compatible_figures(metric_paths:list[Path],output_dir:Path)->list[Path]:
    """Render the five principal upstream OpenTraj indicator figure layouts."""
    frames=[];names=[]
    for path in metric_paths:
        frame=pq.read_table(path).to_pandas()
        name=str(frame["dataset"].iloc[0]) if len(frame) and "dataset" in frame else path.parent.name
        frame["dataset"]=name;frames.append(frame);names.append(name)
    output_dir.mkdir(parents=True,exist_ok=True);outputs=[]
    def sampled_frame(key:str,maximum:int,quantiles:tuple[float,float],bins:int=50)->pd.DataFrame:
        samples=_histogram_samples([frame[key].to_numpy() for frame in frames],maximum,bins,quantiles)
        return pd.concat([pd.DataFrame({"dataset":name,key:value}) for name,value in zip(names,samples)],ignore_index=True)
    # Long labels are wrapped onto two lines: each stacked subplot's rotated
    # ylabel is otherwise taller than its own row and bleeds into the
    # neighbouring subplot regardless of tight_layout/constrained_layout.
    motion_properties_labels={
        "speed_mean":"Mean speed (m/s)","speed_range":"Speed range (m/s)",
        "acceleration_mean_abs":"Mean |acceleration|\n(m/s²)","acceleration_max_abs":"Max |acceleration|\n(m/s²)",
    }
    with FIGURE_STYLE:
        figure,axes=plt.subplots(4,1,figsize=(12,8),sharex=True,constrained_layout=True)
        for axis,key,limit,q in zip(axes,("speed_mean","speed_range","acceleration_mean_abs","acceleration_max_abs"),((0,2.4),(0,2),(0,.5),(0,1.2)),((0,.92),(0,.92),(.02,.92),(.02,.92))):
            sns.swarmplot(data=sampled_frame(key,300,q),x="dataset",y=key,size=1,ax=axis);axis.set_ylabel(motion_properties_labels[key]);axis.set_ylim(*limit);axis.set_xlabel("")
        path=output_dir/"motion_properties.pdf";figure.savefig(path,dpi=500,bbox_inches="tight");plt.close(figure);outputs.append(path)
        figure,axis=plt.subplots(figsize=(12,1.8));data=sampled_frame("path_efficiency",800,(.1,1),100);data.path_efficiency*=100
        sns.swarmplot(data=data,x="dataset",y="path_efficiency",size=1,ax=axis);axis.set_ylabel("Path efficiency (%)");axis.set_ylim(90,100);axis.set_xlabel("");figure.tight_layout();path=output_dir/"path_eff.pdf";figure.savefig(path,dpi=500,bbox_inches="tight");plt.close(figure);outputs.append(path)
        figure,axes=plt.subplots(3,1,figsize=(12,5),sharex=True,constrained_layout=True)
        for axis,key in zip(axes,("ttc_min","dca_min","collision_energy")):
            sns.swarmplot(data=sampled_frame(key,500,(.05,.95)),x="dataset",y=key,size=1,ax=axis);axis.set_ylabel(METRIC_LABELS[key]);axis.set_xlabel("")
        path=output_dir/"collision.pdf";figure.savefig(path,dpi=500,bbox_inches="tight");plt.close(figure);outputs.append(path)
        local=sampled_frame("local_density_max",800,(.05,.95));global_groups=[]
        for metric_path in metric_paths:
            density_path=metric_path.parent/"frame_global_density.parquet";global_groups.append(pq.read_table(density_path)["global_density"].to_numpy())
        global_sampled=_histogram_samples(global_groups,800,50,(.05,.98));global_frame=pd.concat([pd.DataFrame({"dataset":name,"global_density":values}) for name,values in zip(names,global_sampled)],ignore_index=True)
        figure,axes=plt.subplots(2,1,figsize=(12,3.6),sharex=True,constrained_layout=True);sns.swarmplot(data=global_frame,x="dataset",y="global_density",size=1,ax=axes[0]);sns.swarmplot(data=local,x="dataset",y="local_density_max",size=1,ax=axes[1]);axes[0].set_ylabel("Global density, duplicates retained\n(ped/m²)");axes[1].set_ylabel("Local density, duplicates retained\n(ped/m²)");axes[0].set_xlabel("");axes[1].set_xlabel("");path=output_dir/"density.pdf";figure.savefig(path,dpi=500,bbox_inches="tight");plt.close(figure);outputs.append(path)
        summaries=[json.loads((path.parent/"summary.json").read_text()) for path in metric_paths]
        figure,axes=plt.subplots(2,1,figsize=(max(6,len(names)+2),5),sharex=True,constrained_layout=True)
        for axis,time in zip(axes,("4.8","2.4")):
            mean=[summary["angular_deviation"].get(time,{}).get("mean_degrees",np.nan) for summary in summaries];std=[summary["angular_deviation"].get(time,{}).get("std_degrees",np.nan) for summary in summaries]
            # Both panels plot the signed mean angular deviation. Upstream
            # (references/opentraj/opentraj/toolkit/indicators/traj_deviation.py:84)
            # plots the 4.8 s panel signed but takes abs() of the 2.4 s panel
            # (traj_deviation.py:94); that asymmetry made the two panels'
            # identical "Mean angular deviation (deg)" axis label false for
            # the 2.4 s one. STEPS deliberately diverges from upstream here
            # (F-010) and keeps both panels signed so the shared label is
            # accurate for both.
            axis.bar(np.arange(len(names)),mean,yerr=std,alpha=.7,color="red",error_kw={"ecolor":"blue","lw":2,"capsize":5});axis.set_ylabel("Mean angular\ndeviation (deg)");axis.set_title(f"t = {time} s",fontsize=10,loc="right");axis.grid(axis="y",linestyle="--")
        axes[1].set_xticks(np.arange(len(names)),names,rotation=-20);path=output_dir/"traj_deviation.pdf";figure.savefig(path,dpi=400,bbox_inches="tight");plt.close(figure);outputs.append(path)
    return outputs


def _configure_logging(log_name: str) -> None:
    """Log to console and to logs/<log_name> so unattended runs keep a record."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / log_name)],
    )


def main() -> None:
    _configure_logging("opentraj_report.log")
    parser=argparse.ArgumentParser(description="Compute OpenTraj-compatible indicators from the analysis view")
    parser.add_argument("dataset_dirs",type=Path,nargs="+");parser.add_argument("--plot",type=Path);parser.add_argument("--plot-dir",type=Path)
    parser.add_argument("--split-strategy",choices=("grouped_scene_sequence","leave_one_scene_out"),default="grouped_scene_sequence")
    parser.add_argument("--held-out-scene")
    parser.add_argument("--artifact-variant", help="Read/write a named publication artifact variant")
    parser.add_argument("--scene-sequences", nargs="+")
    args=parser.parse_args();paths=[]
    if args.split_strategy=="leave_one_scene_out" and not args.held_out_scene:
        parser.error("--held-out-scene is required for leave_one_scene_out")
    for dataset_dir in args.dataset_dirs:
        if args.artifact_variant:
            index_path=dataset_dir/"indices"/f"opentraj-analysis-trajlets-{args.artifact_variant}.parquet"
            output_dir=dataset_dir/"analysis"/f"opentraj_reports_{args.artifact_variant}"
        else:
            index_path=_analysis_index_path(dataset_dir,args.split_strategy,args.held_out_scene)
            output_dir=None
        metric,density=compute_opentraj_report(
            dataset_dir,index_path=index_path,
            sequences=set(args.scene_sequences) if args.scene_sequences else None,
            output_dir=output_dir,
        );paths.append(metric);logger.info("Wrote %s",metric);logger.info("Wrote %s",density)
    if args.plot:logger.info("Wrote %s",plot_opentraj_report(paths,args.plot))
    if args.plot_dir:
        for path in plot_opentraj_compatible_figures(paths,args.plot_dir):logger.info("Wrote %s",path)


if __name__=="__main__":main()
