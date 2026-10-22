from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.agentformer import UPSTREAM_COMMIT,build_dlow,identity_collate,topk_dlow
from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.status import RunStatus,checkpoint_metadata
from baselines.provenance import snapshot_dataset,snapshot_split_inputs
from baselines.registry import resolve
from baselines.train_agentformer import sources
from baselines.preprocessing_options import add_preprocessing_arguments,config_preprocessing,handle_preprocessing_introspection
from forecasting_config import config_as_dict


def save(path,model,opt,scheduler,iteration,stage,predictor_checkpoint):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp")
    torch.save({"upstream_commit":UPSTREAM_COMMIT,"stage":stage,"iteration":iteration,"model":model.state_dict(),
                "predictor_checkpoint":str(predictor_checkpoint),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
                **checkpoint_metadata(path.parents[1])},tmp);os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=["pretrain","scratch","finetune"])
    p.add_argument("--fold",choices=["eth","hotel","univ","zara1","zara2"]);p.add_argument("--predictor-checkpoint",required=True)
    p.add_argument("--checkpoint",help="PTDPES DLow checkpoint or model ID for finetuning")
    p.add_argument("--run-dir",required=True,type=Path);p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--iterations",type=int);p.add_argument("--max-scenes",type=int);p.add_argument("--status-every",type=int,default=100);p.add_argument("--patience",type=int,default=10,help="Stop after this many validation epochs without improvement; 0 disables");p.add_argument("--min-delta",type=float,default=1e-4);p.add_argument("--seed",type=int,default=1);p.add_argument("--data-seed",type=int,default=42);p.add_argument("--pretrain-dataset",choices=["atc","ptdpes","steps"],default="ptdpes");p.add_argument("--validation-mode",choices=["none","grouped"],default="none",help="ETH/UCY only: all four non-test folds, or reserve one grouped sequence");p.add_argument("--max-val-batches",type=int,default=10,help="Cap validation batches during training; final test evaluation remains exhaustive");add_preprocessing_arguments(p);handle_preprocessing_introspection();a=p.parse_args()
    if a.stage!="pretrain" and not a.fold:p.error("--fold required")
    if a.stage=="finetune" and not a.checkpoint:p.error("--checkpoint required for DLow finetuning")
    predictor_checkpoint=resolve(a.predictor_checkpoint);checkpoint=resolve(a.checkpoint) if a.checkpoint else None
    preprocessing=config_preprocessing(a);forecasting=preprocessing["forecasting"]
    device=training_device(a.device);torch.manual_seed(a.seed);np.random.seed(a.seed)
    run=RunStatus(a.run_dir,{**vars(a),"run_dir":str(a.run_dir),"checkpoint":str(checkpoint) if checkpoint else None,"checkpoint_reference":a.checkpoint,
                               "predictor_checkpoint":str(predictor_checkpoint),"predictor_checkpoint_reference":a.predictor_checkpoint,"model":"agentformer_dlow","k":20,"device_realized":str(device),
                               **forecasting.flat_provenance(),"forecasting_config":config_as_dict(forecasting)})
    run.update(status="running",stage="loading_datasets",iteration=0)
    train,val,*_=sources(a.stage,a.fold,a.max_scenes,a.data_seed,a.seed,a.pretrain_dataset,preprocessing,a.validation_mode);tl=DataLoader(train,batch_size=1,shuffle=True,collate_fn=identity_collate);vl=DataLoader(val,batch_size=1,collate_fn=identity_collate) if val is not None else None
    cfg,model=build_dlow(device,predictor_checkpoint,a.fold or "eth")
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"])
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr);scheduler=torch.optim.lr_scheduler.StepLR(opt,cfg.decay_step,gamma=cfg.decay_gamma)
    run.update(status="running",stage="saving_dataset_manifests",n_train=len(train),n_val=len(val) if val is not None else 0)
    snapshot_split_inputs(a.run_dir);snapshot_dataset(a.run_dir,"train",train)
    if val is not None:snapshot_dataset(a.run_dir,"val",val)
    iteration=0;stopper=EarlyStopping(a.patience,a.min_delta);run.update(status="running",stage="training",n_train=len(train),n_val=len(val) if val is not None else 0)
    for epoch in range(1,a.epochs+1):
        model.train()
        batches=run.progress(tl,desc=f"train epoch {epoch}/{a.epochs}")
        for data in batches:
            model.set_data(data);model();loss,weighted,raw=model.compute_loss();opt.zero_grad();loss.backward();opt.step();iteration+=1
            batches.set_postfix(loss=f"{float(loss.detach()):.4f}",iteration=iteration)
            if iteration%a.status_every==0:
                run.metric({"stage":"train","epoch":epoch,"iteration":iteration,"loss":float(loss.detach()),**raw});run.update(epoch=epoch,iteration=iteration,loss=float(loss.detach()))
                save(a.run_dir/"checkpoints/latest.pt",model,opt,scheduler,iteration,a.stage,predictor_checkpoint)
            if a.iterations is not None and iteration>=a.iterations:break
        scheduler.step()
        if vl is not None:
            ades=[];fdes=[]
            for number,data in enumerate(vl):
                ade,fde=topk_dlow(model,data);ades.append(ade);fdes.append(fde)
                if a.max_val_batches is not None and number+1>=a.max_val_batches:break
            metric=float(np.mean(ades));run.metric({"stage":"val","epoch":epoch,"iteration":iteration,"top20_ade":metric,"top20_fde":float(np.mean(fdes))})
            improved,stopped=stopper.update(metric)
            if improved:save(a.run_dir/"checkpoints/best.pt",model,opt,scheduler,iteration,a.stage,predictor_checkpoint)
        else:
            stopped=False;save(a.run_dir/"checkpoints/best.pt",model,opt,scheduler,iteration,a.stage,predictor_checkpoint)
        save(a.run_dir/"checkpoints/latest.pt",model,opt,scheduler,iteration,a.stage,predictor_checkpoint)
        run.update(epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best if vl is not None else None,early_stopping_bad_epochs=stopper.bad_epochs if vl is not None else 0,early_stopping_patience=a.patience if vl is not None else 0)
        if stopped:
            run.log("early stopping",epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best,patience=a.patience);break
        if a.iterations is not None and iteration>=a.iterations:break
    run.update(status="complete",stage="training",epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best if vl is not None else None,early_stopped=stopped)

if __name__=="__main__":main()
