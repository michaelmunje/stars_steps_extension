from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import dill
import numpy as np
import pyarrow.parquet as pq
import zstandard as zstd
from filelock import FileLock
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

from baselines.trajectron_data import CHUNK_ADVANCE, ROOT, environment, recording_environment

CACHE_VERSION=7
SCENES_PER_SHARD=2
# Trajectron++ builds a dense N x N scene graph.  A few long MADRAS clips
# contain tens of thousands of short-lived tracker nodes; letting those reach
# the reference graph constructor can request tens of GiB.  Keep the most
# persistent nodes in such pathological chunks so caching remains bounded.
MAX_GRAPH_NODES=2048


@dataclass(frozen=True)
class Shard:
    dataset_dir:Path
    recording:str
    role:str
    path:Path
    budget:int


def _digest(dataset_dir:Path,recording:str,role:str,data_seed:int,hp:dict,part_start:int,part_stop:int)->str:
    inputs=[]
    for path in (dataset_dir/"converted"/f"{recording}.traj.parquet",dataset_dir/"indices"/f"scenes-seed{data_seed}.parquet"):
        stat=path.stat();inputs.append((str(path.relative_to(ROOT)),stat.st_size,stat.st_mtime_ns))
    payload={"version":CACHE_VERSION,"recording":recording,"role":role,"data_seed":data_seed,"part_start":part_start,"part_stop":part_stop,
             "edge_addition_filter":hp["edge_addition_filter"],"edge_removal_filter":hp["edge_removal_filter"],
             "forecast_preprocessing_hash":hp.get("steps_forecast_preprocessing_hash"),"inputs":inputs}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16]


def shard_path(dataset_dir:Path,recording:str,role:str,data_seed:int,hp:dict,part_start:int,part_stop:int)->Path:
    digest=_digest(dataset_dir,recording,role,data_seed,hp,part_start,part_stop)
    return dataset_dir/".cache/trajectron"/f"seed{data_seed}"/role/f"{recording}-part{part_start:04d}-{part_stop:04d}-{digest}.dill.zst"


def load_index(dataset_dir:Path,data_seed:int,*,recording:str,role:str):
    """Load one recording/split slice instead of materializing a 15M-row index.

    PTDPES's full index expands to roughly 10 GiB as a pandas object. Cache
    preparation only needs one recording at a time, so predicate pushdown keeps
    both peak RAM and concurrent-seed contention bounded.
    """
    path=dataset_dir/"indices"/f"scenes-seed{data_seed}.parquet"
    columns=["recording","split","start_step","accepted","track_kept","eval_kept"]
    available=set(pq.ParquetFile(path).schema.names)
    return pq.read_table(
        path,
        columns=[column for column in columns if column in available],
        filters=[("recording","=",recording),("split","=",role)],
    ).to_pandas()


def accepted_rows(index,recording:str,role:str):
    rows=index[(index.recording==recording)&(index.split==role)]
    if "accepted" in rows:rows=rows[rows.accepted]
    if "track_kept" in rows:rows=rows[rows.track_kept]
    if role!="train" and "eval_kept" in rows:rows=rows[rows.eval_kept]
    return rows


def build_shards(dataset_dir:Path,recording:str,role:str,data_seed:int,hp:dict,index,run=None)->list[Shard]:
    rows=accepted_rows(index,recording,role)
    if rows.empty:return []
    all_rows=index[(index.recording==recording)&(index.split==role)] if role!="test" else index[index.recording==recording]
    first=int(all_rows.start_step.min());last=int(all_rows.start_step.max())+19
    total_parts=max(1,max(0,last-first)//CHUNK_ADVANCE+1);result=[]
    for part_start in range(0,total_parts,SCENES_PER_SHARD):
        part_stop=min(total_parts,part_start+SCENES_PER_SHARD)
        assigned=((rows.start_step.astype(np.int64)-first)//CHUNK_ADVANCE).clip(0,total_parts-1)
        budget=int(((assigned>=part_start)&(assigned<part_stop)).sum())
        if budget==0:continue
        path=shard_path(dataset_dir,recording,role,data_seed,hp,part_start,part_stop);path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(path)+".lock"):
            if not path.exists():
                env=recording_environment(dataset_dir,recording,role,index,part_start=part_start,part_stop=part_stop)
                if env is None or not env.scenes:continue
                for scene in env.scenes:
                    if len(scene.nodes)>MAX_GRAPH_NODES:
                        original_nodes=len(scene.nodes)
                        scene.nodes=sorted(scene.nodes,key=lambda node:(-node.timesteps,str(node.id)))[:MAX_GRAPH_NODES]
                        if run:run.log("bounded oversized Trajectron scene",recording=recording,role=role,
                                        nodes_dropped=original_nodes-MAX_GRAPH_NODES,max_nodes=MAX_GRAPH_NODES)
                    scene.calculate_scene_graph(env.attention_radius,hp["edge_addition_filter"],hp["edge_removal_filter"])
                temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
                with temporary.open("wb") as compressed:
                    with zstd.ZstdCompressor(level=1).stream_writer(compressed) as stream:dill.dump(env,stream,protocol=dill.HIGHEST_PROTOCOL)
                os.replace(temporary,path)
                if run:run.log("Trajectron cache shard written",recording=recording,role=role,part_start=part_start,part_stop=part_stop,path=str(path),size_bytes=path.stat().st_size)
                del env;gc.collect()
        result.append(Shard(dataset_dir,f"{recording}:part{part_start}-{part_stop}",role,path,budget))
    return result


def load_shard(path:Path):
    with path.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as stream:return dill.load(stream)


def recording_names(dataset_dir:Path,role:str|None=None,data_seed:int=42)->list[str]:
    if role is None:
        return sorted(path.name.removesuffix(".traj.parquet") for path in (dataset_dir/"converted").glob("*.traj.parquet") if "Test_set" not in path.name)
    path=dataset_dir/"indices"/f"scenes-seed{data_seed}.parquet"
    table=pq.read_table(path,columns=["recording"],filters=[("split","=",role)])
    return sorted(str(value) for value in table["recording"].unique().to_pylist() if "Test_set" not in str(value))


def prepare_shards(dataset_dir:Path,role:str,data_seed:int,hp:dict,run=None,recordings:list[str]|None=None)->list[Shard]:
    result=[]
    selected=recordings or recording_names(dataset_dir,role,data_seed)
    for recording in tqdm(selected,desc=f"caching {dataset_dir.name} {role}",unit="recording",dynamic_ncols=True):
        index=load_index(dataset_dir,data_seed,recording=recording,role=role)
        result.extend(build_shards(dataset_dir,recording,role,data_seed,hp,index,run))
    return result


def _normalize_types(env,template_env):
    old=env.NodeType.PEDESTRIAN;new=template_env.NodeType.PEDESTRIAN
    for scene in env.scenes:
        for node in scene.nodes:node.type=new
    env.NodeType=template_env.NodeType;env.attention_radius={(new,new):3.0}
    return env


class ShardedNodeDataset(IterableDataset):
    def __init__(self,shards:list[Shard],hp:dict,template_env,seed:int,shuffle:bool,augment:bool):
        self.shards,self.hp,self.template_env,self.seed,self.shuffle,self.augment=shards,hp,template_env,seed,shuffle,augment
        self.epoch=0
    def __len__(self):return sum(shard.budget for shard in self.shards)
    def __iter__(self):
        from model.dataset import EnvironmentDataset
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]));order=np.arange(len(self.shards))
        if self.shuffle:rng.shuffle(order)
        self.epoch+=1
        for position in order:
            shard=self.shards[int(position)];env=_normalize_types(load_shard(shard.path),self.template_env)
            dataset=EnvironmentDataset(env,self.hp["state"],self.hp["pred_state"],node_freq_mult=False,scene_freq_mult=False,
                hyperparams=self.hp,min_history_timesteps=self.hp["minimum_history_length"],min_future_timesteps=self.hp["prediction_horizon"],return_robot=True)
            nodes=dataset.node_type_datasets[0];nodes.augment=self.augment
            count=min(shard.budget,len(nodes));indices=np.arange(len(nodes))
            shard_rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(hashlib.sha256(shard.recording.encode()).hexdigest()[:8],16)]))
            if count<len(indices):indices=shard_rng.choice(indices,count,replace=False)
            if self.shuffle:shard_rng.shuffle(indices)
            for index in indices:yield nodes[int(index)]
            del nodes,dataset,env;gc.collect()
