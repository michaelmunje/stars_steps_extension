from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.device import training_device
from baselines.status import RunStatus, atomic_json
from baselines.provenance import snapshot_environment,snapshot_split_inputs
from baselines.registry import resolve
from baselines.qualitative import contact_sheet,render_scene
from baselines.trajectron_data import fold_environment, reference_hyperparams, validate_trajectron_preprocessing
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from baselines.preprocessing_options import add_preprocessing_arguments, handle_preprocessing_introspection, resolved_forecasting_from_args
from forecasting_config import config_as_dict


def best_of_k(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    distances=np.linalg.norm(pred-gt[None],axis=-1)
    return float(distances.mean(axis=1).min()),float(distances[:,-1].min())


def evaluate(model: Trajectron, env, *, k: int=20, stride: int=20, max_scenes: int|None=None,visual_count:int=0,visual_dir:Path|None=None):
    ades=[];fdes=[];n_scenes=0;visuals=[];visual_metrics=[]
    node_type=env.NodeType.PEDESTRIAN
    for scene in env.scenes[:max_scenes]:
        timesteps=np.arange(7,scene.timesteps-12,stride,dtype=np.int64)
        if not len(timesteps):continue
        outputs=model.predict(scene,timesteps,12,num_samples=k,min_future_timesteps=12,
                              min_history_timesteps=7,full_dist=False)
        for timestep,nodes in outputs.items():
            for node,prediction in nodes.items():
                # Official output is (1, K, T, 2), in scene coordinates.
                pred=np.asarray(prediction)[0]
                gt=node.get(np.array([int(timestep)+1,int(timestep)+12]),{"position":["x","y"]})
                if pred.shape==(k,12,2) and np.isfinite(gt).all():
                    ade,fde=best_of_k(pred,gt);ades.append(ade);fdes.append(fde)
                    if visual_dir is not None and len(visuals)<visual_count:
                        present=scene.present_nodes(np.array([int(timestep)]),type=node_type,min_history_timesteps=7,min_future_timesteps=12).get(int(timestep),[])
                        ordered=[node]+[other for other in present if other!=node]
                        xy=np.stack([other.get(np.array([int(timestep)-7,int(timestep)+12]),{"position":["x","y"]}) for other in ordered],axis=1)
                        number=len(visuals)+1;stem=visual_dir/f"scene_{number:02d}"
                        visual_metrics.append(render_scene(xy,np.isfinite(xy).all(-1),pred,stem,f"Held-out scene {number}"));visuals.append(stem.with_suffix(".png"))
        n_scenes+=1
    if visual_dir is not None:
        contact_sheet(visuals,visual_dir/"contact_sheet.png");atomic_json(visual_dir/"metrics.json",{"scenes":visual_metrics})
    if not ades:raise RuntimeError("Trajectron++ evaluation population is empty")
    return {"k":k,"n_predictions":len(ades),"n_recordings":n_scenes,"visual_scenes":len(visuals),
            "topk_ade":float(np.mean(ades)),
            "topk_fde":float(np.mean(fdes))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",required=True,choices=["eth","hotel","univ","zara1","zara2"])
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--k",type=int,default=20)
    parser.add_argument("--stride",type=int,default=20)
    parser.add_argument("--max-recordings",type=int)
    parser.add_argument("--data-seed",type=int,default=42)
    parser.add_argument("--visual-scenes",type=int,default=10)
    parser.add_argument("--visual-seed",type=int,default=42)
    add_preprocessing_arguments(parser)
    handle_preprocessing_introspection(default_preset="trajectron_pp")
    args=parser.parse_args();forecasting=resolved_forecasting_from_args(args,default_preset="trajectron_pp");validate_trajectron_preprocessing(forecasting);args.visual_scenes=max(0,min(30,args.visual_scenes));torch.manual_seed(args.visual_seed);np.random.seed(args.visual_seed);device=training_device(args.device);hp=reference_hyperparams();checkpoint=resolve(args.checkpoint)
    run=RunStatus(args.run_dir,{**vars(args),"checkpoint":str(checkpoint),"checkpoint_reference":args.checkpoint,"run_dir":str(args.run_dir),
                               "device_realized":str(device),"model":"trajectron++",**forecasting.flat_provenance(),"forecasting_config":config_as_dict(forecasting)})
    run.update(status="running",stage="preparing_test_environment")
    env=fold_environment(args.fold,"test",max_recordings=args.max_recordings,data_seed=args.data_seed)
    snapshot_split_inputs(args.run_dir);snapshot_environment(args.run_dir,"test",env)
    for scene in env.scenes:
        scene.calculate_scene_graph(env.attention_radius,hp["edge_addition_filter"],hp["edge_removal_filter"])
    registrar=ModelRegistrar(str(args.run_dir/"checkpoints"),device)
    model=Trajectron(registrar,hp,None,device);model.set_environment(env);model.set_annealing_params()
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    registrar.model_dict.load_state_dict(payload["models"]);registrar.to(device)
    run.update(status="running",stage="evaluation")
    with torch.no_grad():result=evaluate(model,env,k=args.k,stride=args.stride,visual_count=args.visual_scenes,visual_dir=args.run_dir/"visuals")
    atomic_json(args.run_dir/"evaluation.json",result)
    run.metric({"stage":"test",**result});run.update(status="complete",stage="evaluation",**result)


if __name__=="__main__":main()
