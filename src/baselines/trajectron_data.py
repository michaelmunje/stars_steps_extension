from __future__ import annotations

import json
import sys
import collections
import collections.abc
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from baselines.eth_ucy import fold_recording_partitions
from forecasting_config import ForecastingConfig

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references/trajectron-plus-plus/trajectron"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

# The pinned 2020 code uses aliases removed in NumPy 2. Keep this compatibility
# local to the adapter; the reference checkout remains byte-for-byte unchanged.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence  # type: ignore[attr-defined]

from environment import Environment, Node, Scene, derivative_of  # noqa: E402

STANDARDIZATION = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}
DATA_COLUMNS = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)
CHUNK_SPAN=100-1
CHUNK_ADVANCE=100-20


def validate_trajectron_preprocessing(config: ForecastingConfig) -> None:
    """Reject resolved options the pinned upstream adapter cannot honor.

    The wrapper builds one upstream Environment per source recording, so
    window-local filtering or alternate feature layouts cannot be applied
    without changing the reference model's data contract. Rotation may be
    disabled as a controlled ablation; every other compatibility field is
    checked explicitly to prevent misleading run metadata.
    """
    expected = {
        "window": (8, 12, 1),
        "signal": ("raw", "raw"),
        "coordinates": "scene_mean_center",
        "features": ("position_velocity_acceleration", "physical_derivative_dt"),
        "standardization": "trajectron_pp",
        "membership": ("target_complete", 1),
        "rotation": ("discrete", 15.0),
    }
    actual = {
        "window": (
            config.window.observation_steps, config.window.prediction_steps,
            config.window.stride,
        ),
        "signal": (config.signal.input_mode, config.signal.target_mode),
        "coordinates": config.coordinates.frame,
        "features": (config.features.representation, config.features.derivative_mode),
        "standardization": config.standardization.mode,
        "membership": (config.scene_membership, config.min_complete_agents),
        "rotation": (
            config.augmentation.random_rotation.mode,
            config.augmentation.random_rotation.step_degrees,
        ),
    }
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if config.augmentation.random_flip.enabled or config.augmentation.random_scale.enabled:
        mismatches.append("flip/scale augmentation")
    if mismatches:
        details = ", ".join(dict.fromkeys(mismatches))
        raise ValueError(
            "The pinned Trajectron++ adapter cannot faithfully apply these resolved "
            f"preprocessing overrides: {details}. Use the exact trajectron_pp preset "
            "(optionally disabling random rotation) rather than recording an override "
            "that the upstream Environment would ignore."
        )


def augment_scene(scene: Scene) -> Scene:
    """Reference 15-degree train rotation, generated on demand to bound RAM."""
    angle=float(np.random.choice(np.arange(0,360,15)))*np.pi/180.0
    matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    augmented=Scene(timesteps=scene.timesteps,dt=scene.dt,name=scene.name)
    for source in scene.nodes:
        xy=np.column_stack([source.data.position.x,source.data.position.y])@matrix.T
        vx,vy=derivative_of(xy[:,0],scene.dt),derivative_of(xy[:,1],scene.dt)
        ax,ay=derivative_of(vx,scene.dt),derivative_of(vy,scene.dt)
        data=pd.DataFrame(np.column_stack([xy[:,0],xy[:,1],vx,vy,ax,ay]),columns=DATA_COLUMNS)
        augmented.nodes.append(Node(source.type,source.id,data,first_timestep=source.first_timestep))
    augmented.temporal_scene_graph=scene.temporal_scene_graph
    return augmented


def reference_hyperparams() -> dict:
    path = REFERENCE.parent / "experiments/pedestrians/models/eth_vel/config.json"
    return json.loads(path.read_text())


def _segments(table: pd.DataFrame, first: int, last: int):
    table = table[(table.step >= first) & (table.step <= last)].sort_values(["track_id", "step"])
    for track_id, track in table.groupby("track_id", sort=True):
        steps = track.step.to_numpy(dtype=np.int64)
        xy = track[["x", "y"]].to_numpy(dtype=np.float64)
        cuts = np.flatnonzero(np.diff(steps) != 1) + 1
        for part, (part_steps, part_xy) in enumerate(zip(np.split(steps, cuts), np.split(xy, cuts))):
            if len(part_steps) >= 2:
                yield f"{track_id}:{part}", part_steps, part_xy


def recording_scene(frame: pd.DataFrame, first_step: int, last_step: int, name: str, *, source_scene_center: np.ndarray | None = None) -> Scene | None:
    pieces = list(_segments(frame, first_step, last_step))
    if not pieces:
        return None
    origin = min(int(piece[1][0]) for piece in pieces)
    final = max(int(piece[1][-1]) for piece in pieces)
    if source_scene_center is None:
        raise ValueError("Trajectron++ source/file mean center must be supplied explicitly")
    center = np.asarray(source_scene_center)
    scene = Scene(timesteps=final - origin + 1, dt=0.4, name=name)
    for node_id, steps, xy in pieces:
        xy = xy - center
        vx, vy = derivative_of(xy[:, 0], 0.4), derivative_of(xy[:, 1], 0.4)
        ax, ay = derivative_of(vx, 0.4), derivative_of(vy, 0.4)
        values = np.column_stack([xy[:, 0], xy[:, 1], vx, vy, ax, ay])
        data = pd.DataFrame(values, columns=DATA_COLUMNS)
        node = Node(node_type=None, node_id=node_id, data=data, first_timestep=int(steps[0]) - origin)
        scene.nodes.append(node)
    return scene


@lru_cache(maxsize=1024)
def _recording_center(path_text: str, size: int, mtime_ns: int) -> tuple[float, float]:
    table=pq.read_table(Path(path_text),columns=["x","y"])
    return float(np.mean(table["x"].to_numpy())),float(np.mean(table["y"].to_numpy()))


def recording_environment(dataset_dir:Path,recording:str,role:str,index:pd.DataFrame,*,part_start:int=0,part_stop:int|None=None):
    """Build a bounded temporal portion of one recording's environment."""
    rows=index[(index.recording==recording)&(index.split==role)] if role!="test" else index[index.recording==recording]
    if rows.empty:return None
    first=int(rows.start_step.min());last=int(rows.start_step.max())+19
    total_parts=max(1,max(0,last-first)//CHUNK_ADVANCE+1)
    stop=min(total_parts,part_stop if part_stop is not None else total_parts)
    if part_start>=stop:return None
    read_first=first+part_start*CHUNK_ADVANCE
    read_last=min(last,first+(stop-1)*CHUNK_ADVANCE+CHUNK_SPAN)
    path=dataset_dir/"converted"/f"{recording}.traj.parquet"
    stat=path.stat();source_center=np.asarray(_recording_center(str(path),stat.st_size,stat.st_mtime_ns))
    frame=pq.read_table(path,columns=["track_id","step","x","y"],filters=[("step",">=",read_first),("step","<=",read_last)]).to_pandas()
    env=Environment(node_type_list=["PEDESTRIAN"],standardization=STANDARDIZATION)
    env.attention_radius={(env.NodeType.PEDESTRIAN,env.NodeType.PEDESTRIAN):3.0};scenes=[]
    for part in range(part_start,stop):
        chunk_first=first+part*CHUNK_ADVANCE;chunk_last=min(last,chunk_first+CHUNK_SPAN)
        scene=recording_scene(frame,chunk_first,chunk_last,f"{dataset_dir.name}:{recording}:{role}:{part}",source_scene_center=source_center)
        if scene is not None:
            for node in scene.nodes:node.type=env.NodeType.PEDESTRIAN
            if role=="train":scene.aug_func=augment_scene
            scenes.append(scene)
    env.scenes=scenes
    return env


def environment(dataset_recordings: dict[Path, set[str]], role: str, *, max_recordings: int | None = None,data_seed:int=42,indices:dict[Path,pd.DataFrame]|None=None):
    """Create an official Trajectron Environment from our indexed split rows."""
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0}
    scenes = [];recording_count=0
    for dataset_dir, recordings in dataset_recordings.items():
        index = indices[dataset_dir] if indices is not None and dataset_dir in indices else pq.read_table(dataset_dir / f"indices/scenes-seed{data_seed}.parquet").to_pandas()
        index = index[index.recording.isin(recordings)]
        if role != "test":
            index = index[index.split == role]
        selected_recordings=sorted(recordings)
        progress_total=min(len(selected_recordings),max_recordings) if max_recordings is not None else len(selected_recordings)
        for recording in tqdm(selected_recordings,desc=f"loading {dataset_dir.name} {role} recordings",total=progress_total,unit="recording",dynamic_ncols=True):
            rows = index[index.recording == recording]
            if rows.empty:
                continue
            recording_env=recording_environment(dataset_dir,recording,role,index)
            if recording_env is not None:scenes.extend(recording_env.scenes)
            recording_count+=1
            if max_recordings is not None and recording_count>=max_recordings:break
        if max_recordings is not None and recording_count>=max_recordings:
            break
    env.scenes = scenes
    return env


def fold_environment(fold: str, role: str, *, max_recordings: int | None = None,data_seed:int=42,validation_mode:str="none"):
    if role not in {"train","val","test"}:raise ValueError(f"Unknown fold role {role!r}")
    selected=fold_recording_partitions(fold,data_seed,validation_mode)[role]
    if not selected:return None
    sources = {
        ROOT / "datasets/eth": {x for x in selected if x.startswith("seq_")},
        ROOT / "datasets/ucy": {x for x in selected if not x.startswith("seq_")},
    }
    sources={path:names for path,names in sources.items() if names}
    indices={}
    for path,names in sources.items():
        index=pq.read_table(path/f"indices/scenes-seed{data_seed}.parquet").to_pandas()
        index=index[index.recording.isin(names)].copy()
        index["split"]=role
        indices[path]=index
    return environment(sources,role,max_recordings=max_recordings,data_seed=data_seed,indices=indices)
