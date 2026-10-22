from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from baselines.eth_ucy import fold_datasets, protocol
from config import Config, SeedConfig
from data import _scene_arrays
from metrics import collision_metrics
from models import ConstantVelocityModel
from opentraj_report import (
    METRIC_COLUMNS,
    compute_opentraj_report,
    plot_opentraj_compatible_figures,
    plot_opentraj_report,
)


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
logger = logging.getLogger(__name__)
FOLD_ORDER = ("eth", "hotel", "univ", "zara1", "zara2")
DISPLAY = {"eth": "ETH", "hotel": "HOTEL", "univ": "UNIV", "zara1": "ZARA1", "zara2": "ZARA2"}
OFFICIAL_RECORDING = {"seq_eth": "biwi_eth", "seq_hotel": "biwi_hotel"}


def _dataset_name(recording: str) -> str:
    return "eth" if recording.startswith("seq_") else "ucy"


def _neighbor_cv(scene, obs_len: int, pred_len: int) -> np.ndarray:
    predictions=[]
    for agent in range(1,scene.xy.shape[1]):
        if scene.valid[obs_len-2:obs_len,agent].all():
            velocity=scene.xy[obs_len-1,agent]-scene.xy[obs_len-2,agent]
            predictions.append(scene.xy[obs_len-1,agent]+np.arange(1,pred_len+1)[:,None]*velocity)
        else:
            predictions.append(np.full((pred_len,2),np.nan))
    return np.asarray(predictions)


def evaluate_fold_cv(fold: str, cfg: Config) -> dict:
    train,val,test=fold_datasets(fold,cfg,"none")
    ades=[];fdes=[];pred_cols=[];gt_cols=[]
    for dataset in test.datasets:
        rows_by_recording=defaultdict(list)
        for row in dataset.rows:rows_by_recording[row["recording"]].append(row)
        for recording,rows in rows_by_recording.items():
            path=dataset.recordings[recording]
            table=pq.read_table(path,columns=["track_id","step","x","y"])
            ids,steps,x,y=(table[column].combine_chunks().to_numpy() for column in ("track_id","step","x","y"))
            model=ConstantVelocityModel(dataset.history_steps,dataset.future_steps)
            for row in rows:
                final=int(row["start_step"])+(dataset.total_steps-1)*dataset.canonical_stride
                lo=np.searchsorted(steps,int(row["start_step"]),side="left")
                hi=np.searchsorted(steps,final,side="right")
                arrays=_scene_arrays(
                    ids[lo:hi],steps[lo:hi],x[lo:hi],y[lo:hi],int(row["start_step"]),
                    dataset.history_steps,dataset.future_steps,dataset.canonical_stride,
                    int(row["primary_track_id"]),
                )
                scene=SimpleNamespace(xy=arrays["xy"],valid=arrays["valid"])
                prediction=model.predict(scene)[0]
                truth=scene.xy[dataset.history_steps:,0]
                distance=np.linalg.norm(prediction-truth,axis=-1)
                ades.append(float(distance.mean()));fdes.append(float(distance[-1]))
                predicted_neighbors=_neighbor_cv(scene,dataset.history_steps,dataset.future_steps)
                truth_neighbors=np.moveaxis(scene.xy[dataset.history_steps:,1:],1,0)
                pred_col,gt_col=collision_metrics(prediction,predicted_neighbors,truth_neighbors)
                pred_cols.append(pred_col);gt_cols.append(gt_col)
    return {
        "n_train_windows":len(train),"n_val_windows":len(val) if val is not None else 0,"n_eval_windows":len(test),
        "ade":float(np.mean(ades)),"fde":float(np.mean(fdes)),
        "pred_col_pct":100*float(np.mean(pred_cols)),"gt_col_pct":100*float(np.mean(gt_cols)),
        "cv_definition":"last observed displacement extrapolated one canonical timestep at a time",
    }


def _canonical_summary(recordings: set[str], data_seed: int) -> dict:
    hours=0.0;track_durations=[];raw_tracks=0;occupancy=[];forecast_rows=[];analysis_rows=[]
    for dataset_name in ("eth","ucy"):
        selected={recording for recording in recordings if _dataset_name(recording)==dataset_name}
        if not selected:continue
        dataset_dir=ROOT/"datasets"/dataset_name
        index=pq.read_table(dataset_dir/"indices"/f"scenes-seed{data_seed}.parquet").to_pylist()
        forecast_rows.extend(row for row in index if row["recording"] in selected)
        analysis=pq.read_table(dataset_dir/"indices/opentraj-analysis-trajlets.parquet").to_pylist()
        analysis_rows.extend(row for row in analysis if row["scene_sequence"] in selected)
        for recording in sorted(selected):
            table=pq.read_table(dataset_dir/"converted"/f"{recording}.traj.parquet",columns=["track_id","step"])
            ids=table["track_id"].to_numpy();steps=table["step"].to_numpy()
            if len(steps):
                hours+=(float(steps.max()-steps.min())/2.5)/3600.0
                unique,counts=np.unique(steps,return_counts=True)
                count_by_step=dict(zip(unique.tolist(),counts.tolist()))
                occupancy.extend(count_by_step.get(step,0) for step in range(int(unique.min()),int(unique.max())+1))
            for track_id in np.unique(ids):
                values=steps[ids==track_id]
                raw_tracks+=1
                track_durations.append((float(values.max()-values.min())+1)/2.5)
    type_counts=Counter(int(row["scene_tag"]) for row in forecast_rows)
    return {
        "hours":hours,"raw_tracks":raw_tracks,
        "median_track_span_s":float(np.median(track_durations)),
        "occupancy_mean":float(np.mean(occupancy)),
        "forecast_windows":len(forecast_rows),
        "type_counts":{str(tag):type_counts[tag] for tag in range(1,5)},
        "type3_fraction":type_counts[3]/len(forecast_rows) if forecast_rows else float("nan"),
        "valid_opentraj_trajlets":len(analysis_rows),
        "retained_opentraj_trajlets":sum(bool(row["retained_after_static_filter"]) for row in analysis_rows),
    }


def _opentraj_fold_report(fold: str, recordings: set[str]) -> tuple[Path,dict]:
    dataset_names={_dataset_name(recording) for recording in recordings}
    if len(dataset_names)!=1:raise ValueError(f"Fold {fold} spans multiple canonical datasets")
    dataset_name=next(iter(dataset_names))
    output_dir=ROOT/"paper/stats/eth_ucy_folds"/fold/"opentraj"
    metric_path,_=compute_opentraj_report(
        ROOT/"datasets"/dataset_name,
        sequences=recordings,
        output_dir=output_dir,
        report_name=DISPLAY[fold],
    )
    return metric_path,json.loads((output_dir/"summary.json").read_text())


def _write_tables(rows: list[dict]) -> None:
    paper=ROOT/"paper"
    cv_rows=[];summary_rows=[];indicator_rows=[]
    for row in rows:
        heldout=" + ".join(row["held_out_test_data"])
        cv=row["constant_velocity"];canonical=row["canonical"];ind=row["opentraj"]["indicators"]
        cv_rows.append(f"{row['fold']} & {heldout} & {cv['n_eval_windows']} & {cv['ade']:.3f} & {cv['fde']:.3f} & {cv['pred_col_pct']:.2f} & {cv['gt_col_pct']:.2f} \\\\")
        summary_rows.append(f"{row['fold']} & {heldout} & {canonical['hours']:.3f} & {canonical['raw_tracks']} & {canonical['forecast_windows']} & {canonical['retained_opentraj_trajlets']} & {100*canonical['type3_fraction']:.1f} \\\\")
        indicator_rows.append(f"{row['fold']} & {ind['speed_mean']['mean']:.3f} & {ind['acceleration_mean_abs']['mean']:.3f} & {ind['path_efficiency']['mean']:.3f} & {ind['local_density_max']['mean']:.3f} & {ind['ttc_min']['mean']:.3f} & {ind['dca_min']['mean']:.3f} \\\\")
    (paper/"constant_velocity_eth_ucy_folds.tex").write_text(
        "\\begin{tabular}{llrrrrr}\nFold & Held-out data & N & ADE & FDE & Pred. col. (\\%) & GT col. (\\%) \\\\\n\\hline\n"+"\n".join(cv_rows)+"\n\\end{tabular}\n"
    )
    (paper/"table_eth_ucy_fold_stats.tex").write_text(
        "\\begin{tabular}{llrrrrr}\nFold & Held-out data & Hours & Tracks & Forecast windows & Analysis trajlets & Type-3 (\\%) \\\\\n\\hline\n"+"\n".join(summary_rows)+"\n\\end{tabular}\n"
    )
    (paper/"table_eth_ucy_fold_indicators.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\nFold & Speed & Acceleration & Path efficiency & Local density & TTC & DCA \\\\\n\\hline\n"+"\n".join(indicator_rows)+"\n\\end{tabular}\n"
    )


def generate(data_seed: int = 42) -> list[dict]:
    definition=protocol();cfg=Config(seeds=SeedConfig(data_seed,42),train_rotation=False)
    outputs=[];metric_paths=[]
    for fold in FOLD_ORDER:
        recordings=set(definition["folds"][fold])
        metric_path,opentraj=_opentraj_fold_report(fold,recordings);metric_paths.append(metric_path)
        result={
            "fold":DISPLAY[fold],
            "held_out_recordings":sorted(recordings),
            "held_out_test_data":[OFFICIAL_RECORDING.get(recording,recording) for recording in sorted(recordings)],
            "canonical":_canonical_summary(recordings,data_seed),
            "opentraj":opentraj,
            "constant_velocity":evaluate_fold_cv(fold,cfg),
        }
        output=ROOT/"paper/stats/eth_ucy_folds"/f"{fold}.json"
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        outputs.append(result);logger.info("Wrote %s",output)
    figure_dir=ROOT/"paper/figures/eth_ucy_folds"
    plot_opentraj_report(metric_paths,figure_dir/"all_indicators.pdf")
    plot_opentraj_compatible_figures(metric_paths,figure_dir)
    _write_tables(outputs)
    manifest=ROOT/"paper/eth_ucy_fold_metrics.json"
    manifest.write_text(json.dumps(outputs,indent=2,allow_nan=False)+"\n")
    logger.info("Wrote %s",manifest)
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
    _configure_logging("generate_eth_ucy_folds.log")
    parser=argparse.ArgumentParser(description="Generate per-fold ETH/UCY statistics, CV metrics, and paper figures")
    parser.add_argument("--data-seed",type=int,default=42)
    args=parser.parse_args();generate(args.data_seed)


if __name__=="__main__":main()
