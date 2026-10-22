#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
SOURCE_ROOT=ROOT/"src"
if str(SOURCE_ROOT) not in sys.path:sys.path.insert(0,str(SOURCE_ROOT))

from analysis_trajlets import extract_opentraj_trajlets
from preprocessing_views import OpenTrajKalmanModel
from opentraj_indicators import collision_energy, dca_ttc_matrices, frame_local_density, global_density, motion_properties, path_efficiency_index


PINNED_COMMIT="97ec1d5e1b579f26febeaa9083705952deac338a"


def load_module(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(module);return module


def validate_view(path:Path,opentraj_root:Path)->dict:
    sys.path.insert(0,str(opentraj_root/"opentraj"))
    from toolkit.core.trajlet import split_trajectories
    table=pq.read_table(path);metadata={k.decode():v.decode() for k,v in (table.schema.metadata or {}).items()}
    frame=table.to_pandas().rename(columns={"track_id":"agent_id","step":"frame_id","smoothed_x":"pos_x","smoothed_y":"pos_y","smoothed_vx":"vel_x","smoothed_vy":"vel_y"})
    frame["scene_id"]=metadata["scene_id"];frame["timestamp"]=frame.frame_id/2.5;frame["label"]="pedestrian"
    grouped=frame.groupby(["scene_id","agent_id"])
    upstream_all=split_trajectories(grouped,4.8,2.,0.,False);upstream_retained=split_trajectories(grouped,4.8,2.,1.,False)
    ours=[]
    for agent_id,track in frame.groupby("agent_id"):
        ours.extend(extract_opentraj_trajlets(dataset=metadata["dataset"],scene_id=metadata["scene_id"],scene_sequence=metadata["scene_sequence"],agent_id=int(agent_id),steps=track.frame_id.to_numpy(),xy=track[["pos_x","pos_y"]].to_numpy()))
    upstream_keys=sorted((int(row.agent_id.iloc[0]),int(row.frame_id.iloc[0]),tuple(row.frame_id)) for row in upstream_all)
    our_keys=sorted((row.agent_id,row.source_start_step,tuple(range(row.source_start_step,row.source_returned_end_step+1))) for row in ours)
    result={"recording":metadata["recording"],"upstream_candidates":len(upstream_all),"ours_candidates":len(ours),"upstream_retained":len(upstream_retained),"ours_retained":sum(row.retained_after_static_filter for row in ours),"returned_indices_equal":upstream_keys==our_keys}
    result["passed"]=result["upstream_candidates"]==result["ours_candidates"] and result["upstream_retained"]==result["ours_retained"] and result["returned_indices_equal"]
    return result


def validate_kalman(opentraj_root:Path)->dict:
    upstream=load_module(opentraj_root/"opentraj/toolkit/utils/kalman_smoother.py","pinned_opentraj_kalman")
    xy=np.c_[np.linspace(0,4,20),np.sin(np.linspace(0,2,20))]
    expected_pos,expected_vel=upstream.KalmanModel(.4,n_dim=2,n_iter=7).smooth(xy)
    actual_pos,actual_vel=OpenTrajKalmanModel(.4,n_dim=2,n_iter=7).smooth(xy)
    return {"position_max_abs_error":float(np.max(np.abs(expected_pos-actual_pos))),"velocity_max_abs_error":float(np.max(np.abs(expected_vel-actual_vel)))}


def validate_indicators(opentraj_root:Path)->dict:
    sys.path.insert(0,str(opentraj_root))
    sys.path.insert(0,str(opentraj_root/"opentraj"))
    motion=load_module(opentraj_root/"opentraj/toolkit/indicators/motion_properties.py","pinned_opentraj_motion")
    efficiency=load_module(opentraj_root/"opentraj/toolkit/indicators/path_efficiency.py","pinned_opentraj_efficiency")
    collision=load_module(opentraj_root/"opentraj/toolkit/indicators/collision_energy.py","pinned_opentraj_collision")
    # crowd_density imports every dataset loader at module import time, including
    # an unused OpenCV dependency. A minimal stub is sufficient for exercising
    # the upstream density function itself when OpenCV is not installed.
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules["cv2"] = types.ModuleType("cv2")
    density_module=load_module(opentraj_root/"opentraj/toolkit/indicators/crowd_density.py","pinned_opentraj_density")
    rng=np.random.default_rng(17);xy=np.cumsum(rng.normal(size=(5,12,2)),axis=1);velocity=rng.normal(size=(5,12,2));time=np.broadcast_to(np.arange(12)*.4,(5,12))
    values=np.concatenate((xy,velocity,time[...,None]),axis=2);ours=motion_properties(values)
    expected={"speed_mean":motion.speed_avg_of_trajs(values),"speed_range":motion.speed_var_of_trajs(values)}
    expected["acceleration_mean_abs"],expected["acceleration_max_abs"]=motion.acceleration_of_tarjs(values)
    state=rng.normal(size=(9,4));expected_dca,expected_ttc=collision.DCA_MTX(state);actual_dca,actual_ttc=dca_ttc_matrices(state)
    errors={key:float(np.max(np.abs(ours[key]-expected[key]))) for key in expected}
    errors.update(path_efficiency_max_abs_error=float(np.max(np.abs(path_efficiency_index(values)-efficiency.path_efficiency_index(values)))),dca_max_abs_error=float(np.max(np.abs(actual_dca-expected_dca))),ttc_max_abs_error=float(np.max(np.abs(actual_ttc-expected_ttc))))
    frames=[];ours_by_agent={agent:[] for agent in range(3)}
    for step in range(20):
        rows=[];positions=[]
        for agent,offset in enumerate((0.0,0.7,2.0)):
            rows.append({"scene_id":"synthetic","agent_id":agent,"frame_id":step,"timestamp":step*.4,"label":"pedestrian","pos_x":step*.2+offset,"pos_y":agent*.1,"vel_x":.5,"vel_y":0.})
            positions.append([step*.2+offset,agent*.1])
        for agent,value in enumerate(frame_local_density(np.asarray(positions))):
            ours_by_agent[agent].append(value)
        frames.append(pd.DataFrame(rows))
    upstream_density=np.asarray(density_module.local_density(frames,{},"synthetic"))
    our_density=np.asarray([
        max(ours_by_agent[agent][start:start+12])
        for agent in range(3) for start in (0,7)
    ])
    errors["local_density_max_abs_error"]=float(np.max(np.abs(our_density-upstream_density)))
    ttc_sample=rng.uniform(0.,4.,size=25);expected_energy=collision.energy(ttc_sample,3.0,0.2);actual_energy=collision_energy(ttc_sample,upper=3.0,lower=0.2)
    errors["collision_energy_max_abs_error"]=float(np.max(np.abs(actual_energy-expected_energy)))
    density_count=7;density_area=23.5;density_frame=pd.DataFrame({"scene_id":["synthetic"]*density_count});density_area_table=pd.DataFrame({"area":[density_area]},index=["synthetic"])
    expected_global_density=density_module.global_density([density_frame],density_area_table)[0];actual_global_density=global_density(density_count,(0.,density_area,0.,1.))
    errors["global_density_max_abs_error"]=float(np.abs(actual_global_density-expected_global_density))
    errors["passed"]=all(value<1e-12 for key,value in errors.items() if key.endswith("error") or key in expected)
    return errors


def main()->None:
    parser=argparse.ArgumentParser(description="Numerically compare analysis views with pinned upstream OpenTraj")
    parser.add_argument("opentraj_root",type=Path);parser.add_argument("analysis_views",type=Path,nargs="+");parser.add_argument("--output",type=Path)
    args=parser.parse_args();results={"pinned_commit":PINNED_COMMIT,"kalman":validate_kalman(args.opentraj_root),"indicators":validate_indicators(args.opentraj_root),"trajlets":[validate_view(path,args.opentraj_root) for path in args.analysis_views]}
    kalman_failures={key:value for key,value in results["kalman"].items() if not value<1e-12}
    if kalman_failures:raise AssertionError(f"kalman parity exceeded threshold 1e-12: {', '.join(f'{key}={value!r}' for key,value in kalman_failures.items())}\n"+json.dumps(results,indent=2))
    if not results["indicators"]["passed"]:raise AssertionError(json.dumps(results,indent=2))
    if not all(row["passed"] for row in results["trajlets"]):raise AssertionError(json.dumps(results,indent=2))
    payload=json.dumps(results,indent=2)+"\n"
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(payload)
    print(payload,end="")


if __name__=="__main__":main()
