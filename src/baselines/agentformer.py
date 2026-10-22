from __future__ import annotations

import contextlib
import sys
import types
import builtins
from typing import Optional,Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.modules.linear as torch_linear
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[2];REFERENCE=ROOT/"references/agentformer"
if str(REFERENCE) not in sys.path:sys.path.insert(0,str(REFERENCE))


@contextlib.contextmanager
def _scoped_reference_namespace(reference:Path,package_names:tuple[str,...]=("model","utils")):
    """Register agentformer's synthetic ``model``/``utils`` packages for one import only.

    ``references/trajectron-plus-plus`` also vendors a top-level ``model`` package. Whichever
    reference checkout's ``model``/``utils`` namespace lands in ``sys.modules`` first would
    otherwise permanently shadow the other's `from model.X import ...` imports in any process
    that uses both (see E-005). Scoping registration to the duration of agentformer's own
    ``model`` import, and restoring whatever was previously registered afterward, keeps the two
    reference packages mutually import-safe regardless of import order.
    """
    saved={name:sys.modules.get(name) for name in package_names}
    for name in package_names:
        package=types.ModuleType(name);package.__path__=[str(reference/name)];sys.modules[name]=package
    try:
        yield
    finally:
        # Only evict the exact synthetic packages and their submodules populated during this
        # scope (keys equal to or prefixed with a package name). A blanket before/after
        # sys.modules diff would also evict unrelated third-party modules (e.g. scipy internals)
        # transitively imported while agentformer's model package was loading.
        for key in [k for k in sys.modules if any(k==name or k.startswith(name+".") for name in package_names)]:
            del sys.modules[key]
        for name,module in saved.items():
            if module is not None:sys.modules[name]=module


# PyTorch 1.8 exposed this private alias; the pinned model only uses it as a
# base-class name, for which modern Linear is equivalent.
if not hasattr(torch_linear,"_LinearWithBias"):
    torch_linear._LinearWithBias=torch_linear.Linear
# The reference copied PyTorch 1.8's transformer implementation, whose wildcard
# imports happened to expose these annotation names. Modern torch does not.
builtins.Tensor=torch.Tensor;builtins.Optional=Optional;builtins.Tuple=Tuple
with _scoped_reference_namespace(REFERENCE):
    from model.model_lib import model_dict  # noqa:E402
    from model.common.mlp import MLP  # noqa:E402
AgentFormer=model_dict["agentformer"]
DLow=model_dict["dlow"]

UPSTREAM_COMMIT="e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5"


def _easy(value):
    if isinstance(value,dict):return EasyDict({key:_easy(item) for key,item in value.items()})
    if isinstance(value,list):return [_easy(item) for item in value]
    return value


def configuration(fold:str="eth"):
    # A single pretrained model must have one architecture across all folds.
    # Upstream varies architecture and traj_scale by held-out location, so use
    # its ETH architecture as the frozen cross-dataset contract; `fold` selects
    # data only in our pipeline.
    path=REFERENCE/"cfg/eth_ucy/eth/eth_agentformer_pre.yml"
    cfg=_easy(yaml.safe_load(path.read_text()));cfg.rand_rot_scene=True
    return cfg


class AgentFormerSceneDataset(Dataset):
    def __init__(self,source,limit:int|None=None):self.source=source;self.limit=len(source) if limit is None else min(len(source),limit)
    def __len__(self):return self.limit
    def __getitem__(self,index):
        scene=self.source[index];complete=np.asarray(scene.valid).all(axis=0);complete[0]=True
        xy=np.asarray(scene.xy[:,complete],dtype=np.float32)/2.0
        return {"pre_motion_3D":[torch.from_numpy(xy[:8,i].copy()) for i in range(xy.shape[1])],
                "fut_motion_3D":[torch.from_numpy(xy[8:,i].copy()) for i in range(xy.shape[1])],
                "pre_motion_mask":[torch.ones(8) for _ in range(xy.shape[1])],
                "fut_motion_mask":[torch.ones(12) for _ in range(xy.shape[1])],
                "heading":None,"seq":scene.recording,"frame":scene.start_step,
                "traj_scale":2.0,"scene_map":None}


def identity_collate(batch):
    if len(batch)!=1:raise ValueError("AgentFormer reference model consumes one multi-agent scene per batch")
    return batch[0]


def build(device:torch.device,fold:str="eth"):
    cfg=configuration(fold);model=AgentFormer(cfg);model.set_device(device);return cfg,model


def dlow_configuration(fold:str="eth"):
    path=REFERENCE/"cfg/eth_ucy/eth/eth_agentformer.yml"
    return _easy(yaml.safe_load(path.read_text()))


class InjectedDLow(DLow):
    """Official DLow architecture with an explicitly supplied predictor checkpoint."""
    def __init__(self,cfg,predictor:AgentFormer):
        torch.nn.Module.__init__(self);self.device=torch.device("cpu");self.cfg=cfg
        self.nk=cfg.sample_k;self.nz=cfg.nz;self.share_eps=cfg.get("share_eps",True)
        self.train_w_mean=cfg.get("train_w_mean",False);self.loss_cfg=cfg.loss_cfg;self.loss_names=list(cfg.loss_cfg)
        predictor.eval();self.pred_model_dim=predictor.cfg.tf_model_dim;self.pred_model=[predictor]
        self.qnet_mlp=cfg.get("qnet_mlp",[512,256]);self.q_mlp=MLP(self.pred_model_dim,self.qnet_mlp)
        self.q_A=torch.nn.Linear(self.q_mlp.out_dim,self.nk*self.nz);self.q_b=torch.nn.Linear(self.q_mlp.out_dim,self.nk*self.nz)


def build_dlow(device:torch.device,predictor_checkpoint:Path,fold:str="eth"):
    _,predictor=build(device,fold);payload=torch.load(predictor_checkpoint,map_location=device,weights_only=False)
    predictor.load_state_dict(payload["model"]);cfg=dlow_configuration(fold);model=InjectedDLow(cfg,predictor);model.set_device(device)
    return cfg,model


@torch.no_grad()
def topk_dlow(model:InjectedDLow,data:dict):
    prediction=sample_dlow(model,data)
    pred=prediction[0]*2.0;gt=torch.stack(data["fut_motion_3D"])[0].to(pred.device)*2.0
    distance=torch.linalg.norm(pred-gt[None],dim=-1)
    return float(distance.mean(-1).min()),float(distance[:,-1].min())


@torch.no_grad()
def sample_dlow(model:InjectedDLow,data:dict):
    model.eval();model.set_data(data);prediction,_=model.inference("infer",model.nk);return prediction


@torch.no_grad()
def topk(model:AgentFormer,data:dict,k:int=20):
    model.eval();model.set_data(data);prediction,_=model.inference(mode="infer",sample_num=k)
    # Reference output: agents x samples x future x xy, coordinates divided by traj_scale.
    pred=prediction[0]*2.0;gt=torch.stack(data["fut_motion_3D"])[0].to(pred.device)*2.0
    distance=torch.linalg.norm(pred-gt[None],dim=-1)
    return float(distance.mean(-1).min()),float(distance[:,-1].min())
