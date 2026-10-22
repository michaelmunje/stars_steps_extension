from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from artifact_provenance import git_provenance
from config import Config, DatasetSpec, SeedConfig, scoped_task_config, stable_config_hash
from data import SceneDataset, _recordings, _scene_arrays
from experiments import ExperimentSpec
from metrics import ade, collision_metrics, fde
from models import ConstantVelocityModel
from opentraj_dataset_stats import generate as generate_opentraj_dataset_stats
from paper_dataset_metrics import generate_publication_tables
from results import RUN_COLUMNS, append_run
from sizing import write_sizing_table
from stats import STATS_SCHEMA_VERSION, analyze as analyze_stats


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
logger = logging.getLogger(__name__)
PAPER_STATS_SCENE_LIMIT=50_000
PAPER_STATS_REQUIRED_FIELDS={
    "hours", "n_trajectories_raw", "n_trajectories_usable",
    "all_segment_length_median_s", "occupancy_mean", "area_m2", "area_method",
    "density_mean_ped_m2", "density_p95_ped_m2", "density_max_ped_m2",
    "scenes_analyzed", "scene_population", "cv_fde_mean", "path_efficiency_mean",
    "heading_entropy_mean", "min_tca_p05_steps_r033_s", "rms_jerk_mean",
    "mean_neighbors_within_2m", "mean_neighbors_within_5m", "type3_fraction",
}


def _stats_cache_is_current(value:dict|None,spec:DatasetSpec,scene_limit:int|None)->bool:
    if (
        value is None
        or value.get("scene_limit") != scene_limit
        or not PAPER_STATS_REQUIRED_FIELDS.issubset(value)
    ):
        return False
    return int(value.get("stats_schema_version",0))>=STATS_SCHEMA_VERSION


def _run_identity(spec:DatasetSpec,cfg:Config)->tuple[str,str]:
    task=json.loads(cfg.task_path.read_text())
    index_path=spec.path/"indices"/f"scenes-seed{cfg.seeds.data_seed}.parquet"
    dataset_revision={
        "index":(index_path.stat().st_size,index_path.stat().st_mtime_ns),
        "recordings":[(path.name,path.stat().st_size,path.stat().st_mtime_ns) for path in _recordings(spec.path)],
    }
    config_hash=stable_config_hash({"config":cfg,"task":scoped_task_config(task,spec.name),"dataset_revision":dataset_revision})
    experiment=ExperimentSpec("constant_velocity",spec.name,(spec.name,),None,0.0,1.0,cfg.seeds.data_seed,cfg.seeds.model_seed)
    return stable_config_hash({"experiment":experiment,"config_hash":config_hash}),config_hash


def _neighbor_cv(scene, obs_len: int, pred_len: int) -> np.ndarray:
    predictions=[]
    for agent in range(1,scene.xy.shape[1]):
        if scene.valid[obs_len-2:obs_len,agent].all():
            velocity=scene.xy[obs_len-1,agent]-scene.xy[obs_len-2,agent]
            predictions.append(scene.xy[obs_len-1,agent]+np.arange(1,pred_len+1)[:,None]*velocity)
        else: predictions.append(np.full((pred_len,2),np.nan))
    return np.asarray(predictions)


def evaluate_cv(spec: DatasetSpec, cfg: Config, run_id: str, run_config_hash: str) -> dict:
    train=SceneDataset(spec,"train",cfg); test=SceneDataset(spec,"test",cfg)
    model=ConstantVelocityModel(test.history_steps,test.future_steps); all_ade=[]; all_fde=[]; pred_cols=[]; gt_cols=[]
    by_type=defaultdict(lambda:{"ade":[],"fde":[]})
    rows_by_recording=defaultdict(list)
    for row in test.rows: rows_by_recording[row["recording"]].append(row)
    paths={path.name.removesuffix(".traj.parquet"):path for path in _recordings(spec.path)}
    for recording,rows in rows_by_recording.items():
        table=pq.read_table(paths[recording],columns=["track_id","step","x","y"])
        ids,steps,x,y=(table[column].combine_chunks().to_numpy() for column in ("track_id","step","x","y"))
        for row in rows:
            final=int(row["start_step"])+(test.total_steps-1)*test.canonical_stride
            lo=np.searchsorted(steps,int(row["start_step"]),side="left");hi=np.searchsorted(steps,final,side="right")
            arrays=_scene_arrays(ids[lo:hi],steps[lo:hi],x[lo:hi],y[lo:hi],int(row["start_step"]),test.history_steps,test.future_steps,test.canonical_stride,int(row["primary_track_id"]))
            scene=SimpleNamespace(xy=arrays["xy"],valid=arrays["valid"])
            prediction=model.predict(scene)[0];ground_truth=scene.xy[test.history_steps:,0]
            a,f=ade(prediction,ground_truth),fde(prediction,ground_truth); all_ade.append(a); all_fde.append(f)
            by_type[int(row["scene_tag"])]["ade"].append(a); by_type[int(row["scene_tag"])]["fde"].append(f)
            predicted_neighbors=_neighbor_cv(scene,test.history_steps,test.future_steps)
            gt_neighbors=np.moveaxis(scene.xy[test.history_steps:,1:],1,0)
            pc,gc=collision_metrics(prediction,predicted_neighbors,gt_neighbors); pred_cols.append(pc); gt_cols.append(gc)
    tracks={row["primary_track_id"] for row in train.rows}
    task=cfg.task_path.read_text(); task=json.loads(task); dataset_task=task.get("datasets",{}).get(spec.name,{})
    split_path=spec.path/"split_manifests"/f"grouped-seed{cfg.seeds.data_seed}.json"
    split_hash="sha256:"+hashlib.sha256(split_path.read_bytes()).hexdigest()
    provenance=git_provenance(ROOT)
    record={
        "run_id":run_id, "model":"constant_velocity", "target_dataset":spec.name,
        "steps_fraction":0.0, "target_fraction":1.0, "data_seed":cfg.seeds.data_seed,
        "model_seed":cfg.seeds.model_seed, "pretrain_source":"none", "n_train_scenes":len(train),
        "n_train_tracks":len(tracks), "iterations":0, "ade":float(np.mean(all_ade)),
        "fde":float(np.mean(all_fde)), "top3_ade":float(np.mean(all_ade)),
        "top3_fde":float(np.mean(all_fde)), "k":1, "pred_col":100*float(np.mean(pred_cols)),
        "gt_col":100*float(np.mean(gt_cols)), "config_hash":run_config_hash,
        "canonical_release":cfg.canonical_release,"canonical_schema_version":3,
        "analysis_preprocessing":cfg.analysis_preprocessing,
        "forecast_preset":cfg.forecasting.requested_preset,
        "forecast_preset_modified":cfg.forecasting.preset_modified,
        "forecast_literature_exact":cfg.forecasting.literature_exact,
        "forecast_input_preprocessing":cfg.forecast_input_preprocessing,
        "forecast_target_preprocessing":cfg.forecast_target_preprocessing,
        "model_normalization":cfg.model_normalization,"split_manifest_hash":split_hash,
        "coordinate_frame":cfg.forecasting.coordinates.frame,
        "feature_representation":cfg.forecasting.features.representation,
        "derivative_mode":cfg.forecasting.features.derivative_mode,
        "standardization_mode":cfg.forecasting.standardization.mode,
        "augmentation_rotation":cfg.forecasting.augmentation.random_rotation.enabled,
        "augmentation_flip":cfg.forecasting.augmentation.random_flip.enabled,
        "augmentation_scale":cfg.forecasting.augmentation.random_scale.enabled,
        "scene_membership":cfg.forecasting.scene_membership,
        "min_complete_agents":cfg.forecasting.min_complete_agents,
        "forecast_preprocessing_hash":cfg.forecasting.forecast_preprocessing_hash,
        "upstream_repository":cfg.forecasting.upstream_repo,
        "upstream_pinned_commit":cfg.forecasting.upstream_commit,
        "normalization_stats_path":"",
        "history_steps":task["history_steps"], "future_steps":task["future_steps"],
        "sample_hz":dataset_task.get("sample_hz",task["sample_hz"]), "canonical_stride":dataset_task.get("canonical_stride",task["canonical_stride"]),
        "chunk_stride":str(dataset_task.get("chunk_stride","unspecified")),
        "min_length":task.get("min_length",0.0), "static_threshold":cfg.category.static_threshold,
        "linear_threshold":cfg.category.linear_threshold, "inter_dist_thresh":cfg.category.inter_dist_thresh,
        "inter_pos_range":cfg.category.inter_pos_range, "grp_dist_thresh":cfg.category.grp_dist_thresh,
        "grp_std_thresh":cfg.category.grp_std_thresh,
        "w_target":dataset_task.get("w_target",cfg.sizing.w_target),
        "n_target":cfg.sizing.n_target if cfg.sizing.n_target is not None else -1,
        "batch_size":cfg.batch_size, "num_workers":cfg.num_workers,
        "train_rotation":cfg.train_rotation, "eval_stride":cfg.eval_stride if cfg.eval_stride is not None else -1,
        "git_commit":provenance["git_commit"], "git_dirty":provenance["git_dirty"],
    }
    for tag,value in enumerate(cfg.category.acceptance,1): record[f"acceptance_type{tag}"]=value
    for tag in range(1,5):
        record[f"ade_type{tag}"]=float(np.mean(by_type[tag]["ade"])) if by_type[tag]["ade"] else float("nan")
        record[f"fde_type{tag}"]=float(np.mean(by_type[tag]["fde"])) if by_type[tag]["fde"] else float("nan")
    assert set(record)==set(RUN_COLUMNS); return record


def _write_cv_table(records:list[dict],path:Path)->None:
    rows=[f"{r['target_dataset']} & {r['ade']:.3f} & {r['fde']:.3f} & {r['pred_col']:.2f} & {r['gt_col']:.2f} \\\\" for r in records]
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\\begin{tabular}{lrrrr}\nDataset & ADE & FDE & Pred. col. (\\%) & GT col. (\\%) \\\\\n\\hline\n"+"\n".join(rows)+"\n\\end{tabular}\n")


def _configure_logging(log_name: str) -> None:
    """Log to console and to logs/<log_name> so unattended runs keep a record."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / log_name)],
    )


def main()->None:
    _configure_logging("generate_tables.log")
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=("constant_velocity",),required=True); parser.add_argument("--all-datasets",action="store_true"); parser.add_argument("--datasets",nargs="+",help="Generate only these dataset names (for example: eth ucy madras)"); parser.add_argument("--allow-partial",action="store_true",help="Skip converted datasets whose Phase-1 scene index is missing"); parser.add_argument("--data-seed",type=int,default=42); parser.add_argument("--scene-limit",type=int,default=PAPER_STATS_SCENE_LIMIT,help=f"Maximum scenes for expensive statistics (default: {PAPER_STATS_SCENE_LIMIT})"); parser.add_argument("--no-scene-cap",action="store_true",help="Analyze every indexed scene")
    args=parser.parse_args(); cfg=Config(seeds=SeedConfig(args.data_seed,42),train_rotation=False); scene_limit=None if args.no_scene_cap else args.scene_limit
    if not args.all_datasets and not args.datasets: parser.error("one of --all-datasets or --datasets is required")
    requested={name.lower() for name in args.datasets} if args.datasets else None
    dataset_root=ROOT/"datasets"; specs=[];task_config=json.loads((ROOT/"configs"/"task.json").read_text()); allowed=set(task_config.get("datasets",{}))
    missing=[]
    for path in sorted(dataset_root.iterdir()):
        dataset_name=path.name.lower()
        if dataset_name not in allowed or (requested is not None and dataset_name not in requested) or not path.is_dir() or not any((path/"converted").glob("*.traj.parquet")): continue
        if (path/"indices"/f"scenes-seed{args.data_seed}.parquet").exists(): specs.append(DatasetSpec(dataset_name,path))
        else: missing.append(dataset_name)
    if args.all_datasets and missing and not args.allow_partial:
        raise FileNotFoundError(f"Missing preprocessed scene indices for: {', '.join(missing)}. Run preprocess.py for each, or pass --allow-partial for a smoke test.")
    if not specs: raise FileNotFoundError("No preprocessed dataset indices found")
    found={spec.name for spec in specs}
    if requested is not None and found != requested:
        unavailable=sorted(requested-found)
        raise FileNotFoundError(f"Requested datasets are not ready: {', '.join(unavailable)}")
    runs_path=ROOT/"experiments"/"results"/"runs.parquet"; existing={}
    if runs_path.exists():
        existing_table=pq.read_table(runs_path)
        if existing_table.schema.names==list(RUN_COLUMNS): existing={row["run_id"]:row for row in existing_table.to_pylist()}
        else:
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive=runs_path.with_name(f"runs.schema-archive-{stamp}.parquet")
            runs_path.replace(archive)
            logger.warning("Archived incompatible run schema to %s", archive)
    records=[]
    for spec in specs:
        expected_run_id,expected_config_hash=_run_identity(spec,cfg)
        record=existing.get(expected_run_id)
        if record is None:
            record=evaluate_cv(spec,cfg,expected_run_id,expected_config_hash);append_run(runs_path,record)
        records.append(record)
    _write_cv_table(records,ROOT/"paper"/"constant_velocity.tex")
    write_sizing_table(ROOT/"paper"/"stats",ROOT/"paper"/"sizing.tex",{spec.name for spec in specs})
    statistics=[]
    for spec in specs:
        path=ROOT/"paper"/"stats"/f"{spec.name}_paper_stats.json"
        value=json.loads(path.read_text()) if path.exists() else None
        if not _stats_cache_is_current(value,spec,scene_limit):
            value=analyze_stats(spec,cfg,scene_limit);path.write_text(json.dumps(value,indent=2)+"\n")
        statistics.append(value)
    generate_opentraj_dataset_stats(ROOT)
    generate_publication_tables(allow_partial=args.allow_partial)
    logger.info("Wrote %s", ROOT/"paper"/"constant_velocity.tex")
    logger.info("Wrote %s", runs_path)


if __name__=="__main__": main()
