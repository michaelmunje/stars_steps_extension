from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.agentformer import build,build_dlow,identity_collate,topk,topk_dlow,sample_dlow
from baselines.device import training_device
from baselines.status import RunStatus,atomic_json
from baselines.provenance import snapshot_dataset,snapshot_split_inputs
from baselines.registry import resolve
from baselines.qualitative import contact_sheet,render_scene
from baselines.train_agentformer import sources
from baselines.preprocessing_options import add_preprocessing_arguments,config_preprocessing,handle_preprocessing_introspection
from forecasting_config import config_as_dict


def main():
    p=argparse.ArgumentParser();p.add_argument("--fold",required=True,choices=["eth","hotel","univ","zara1","zara2"])
    p.add_argument("--checkpoint",required=True);p.add_argument("--predictor-checkpoint")
    p.add_argument("--model",choices=["vae","dlow"],default="dlow");p.add_argument("--run-dir",required=True,type=Path)
    p.add_argument("--device",default="cuda");p.add_argument("--max-scenes",type=int);p.add_argument("--data-seed",type=int,default=42);p.add_argument("--visual-scenes",type=int,default=10);p.add_argument("--visual-seed",type=int,default=42);add_preprocessing_arguments(p);handle_preprocessing_introspection();a=p.parse_args();a.visual_scenes=max(0,min(30,a.visual_scenes));torch.manual_seed(a.visual_seed);np.random.seed(a.visual_seed);device=training_device(a.device)
    if a.model=="dlow" and not a.predictor_checkpoint:p.error("--predictor-checkpoint required for DLow")
    checkpoint=resolve(a.checkpoint);predictor_checkpoint=resolve(a.predictor_checkpoint) if a.predictor_checkpoint else None
    preprocessing=config_preprocessing(a);forecasting=preprocessing["forecasting"];*_,test=sources("scratch",a.fold,a.max_scenes,a.data_seed,1,preprocessing=preprocessing);loader=DataLoader(test,batch_size=1,collate_fn=identity_collate)
    if a.model=="vae":
        _,model=build(device,a.fold);metric=topk
    else:
        _,model=build_dlow(device,predictor_checkpoint,a.fold);metric=topk_dlow
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"]);model.eval()
    run=RunStatus(a.run_dir,{**vars(a),"checkpoint":str(checkpoint),"checkpoint_reference":a.checkpoint,"predictor_checkpoint":str(predictor_checkpoint) if predictor_checkpoint else None,"predictor_checkpoint_reference":a.predictor_checkpoint,
        **forecasting.flat_provenance(),"forecasting_config":config_as_dict(forecasting),
                               "run_dir":str(a.run_dir),"k":20});run.update(status="running",stage="evaluation")
    snapshot_split_inputs(a.run_dir);snapshot_dataset(a.run_dir,"test",test)
    ades=[];fdes=[];visuals=[];visual_metrics=[]
    for data in loader:
        ade,fde=metric(model,data);ades.append(ade);fdes.append(fde)
        if len(visuals)<a.visual_scenes:
            if a.model=="dlow":prediction=sample_dlow(model,data)[0].detach().cpu().numpy()*2.0
            else:
                model.eval();model.set_data(data);prediction=model.inference("infer",20)[0][0].detach().cpu().numpy()*2.0
            past=np.stack([x.numpy() for x in data["pre_motion_3D"]],axis=1)*2.0;future=np.stack([x.numpy() for x in data["fut_motion_3D"]],axis=1)*2.0
            xy=np.concatenate([past,future]);number=len(visuals)+1;stem=a.run_dir/"visuals"/f"scene_{number:02d}"
            visual_metrics.append(render_scene(xy,np.ones(xy.shape[:2],bool),prediction,stem,f"{a.fold.upper()} held-out scene {number}"));visuals.append(stem.with_suffix(".png"))
    contact_sheet(visuals,a.run_dir/"visuals/contact_sheet.png")
    if not ades:raise RuntimeError("AgentFormer evaluation population is empty")
    result={"k":20,"n_predictions":len(ades),"topk_ade":float(np.mean(ades)),"topk_fde":float(np.mean(fdes)),"visual_scenes":len(visuals)}
    atomic_json(a.run_dir/"visuals/metrics.json",{"scenes":visual_metrics})
    atomic_json(a.run_dir/"evaluation.json",result);run.metric({"stage":"test",**result});run.update(status="complete",stage="evaluation",**result)

if __name__=="__main__":main()
