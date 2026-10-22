from __future__ import annotations

import argparse, hashlib, json, math, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
from artifact_provenance import git_provenance

# D-005-scope: PTDPES wrote no run-provenance sidecar at all. Reuse
# artifact_provenance's git_provenance() the same way atc/steps do
# (resolved once per run, degrades to {"git_commit": "unknown", "git_dirty": None}
# rather than raising) and add a minimal `.provenance.json` sibling per converted
# output. Parquet schema metadata is deliberately untouched.
CONVERTER_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_provenance_sidecar(output: Path, provenance: dict[str, object]) -> None:
    payload = {
        "converter_version": CONVERTER_VERSION,
        "invocation": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")

SOURCE_CLOCK_HZ=10
NATIVE_HZ=2.0
OUTPUT_HZ=2.5
OUTPUT_STRIDE=int(SOURCE_CLOCK_HZ/OUTPUT_HZ)
MAX_PEDESTRIAN_SPEED_MPS=10.0
WEB_MERCATOR_RADIUS_M=6_378_137.0
REQUIRED=("tracker_id","time","real_world_sw_x","real_world_sw_y")
SCENE_NAME_ALIASES={
    "bieslko_biala":"bielsko_biala",
    "novisad":"novi_sad",
}

def _normalize_scene_name(value:str)->str:
    ascii_name=unicodedata.normalize("NFKD",value).encode("ascii","ignore").decode()
    normalized=re.sub(r"[^a-z0-9]+","_",ascii_name.lower()).strip("_")
    return SCENE_NAME_ALIASES.get(normalized,normalized)

def source_files(dataset_dir:Path)->list[Path]:
    root=dataset_dir/"raw/01.Trajectories/02.Trajectories_modified"
    files=sorted(p for p in root.rglob("*.csv") if "__MACOSX" not in p.parts and "Test_set" not in p.parts)
    if not files: raise FileNotFoundError(f"No final PTDPES CSV files under {root}")
    return files

def recording_name(path:Path,root:Path)->str:
    relative=path.relative_to(root); stem=path.stem.removesuffix("_speed")
    readable=re.sub(r"[^A-Za-z0-9_.-]+","_","__".join((*relative.parts[:-1],stem))).strip("_")
    return f"{readable}__{hashlib.sha1(str(relative).encode()).hexdigest()[:8]}"

def scene_id_from_path(path:Path,root:Path)->str:
    """Return the physical square, never the comparative/seasonal collection."""
    relative=path.relative_to(root)
    for index,part in enumerate(relative.parts[:-1]):
        if "Trajectories_season" in part and index+1<len(relative.parts)-1:
            return _normalize_scene_name(relative.parts[index+1])
    match=re.search(
        r"(?:morning|noon|evening|saturday)_(.+?)(?:_15(?:fps)?_|_yolo|$)",
        path.stem,
        flags=re.IGNORECASE,
    )
    if match:return _normalize_scene_name(match.group(1))
    for index,part in enumerate(relative.parts[:-1]):
        if "Trajectories_comparative" in part and index+1<len(relative.parts)-1:
            candidate=relative.parts[index+1]
            if candidate.lower() not in {"csv","speed"}:return _normalize_scene_name(candidate)
    raise ValueError(f"Cannot identify the PTDPES physical square from {relative}")

def _headings(ids,steps,x,y):
    heading=np.full(len(ids),np.nan,dtype=np.float32)
    # Group rows by track with a single O(N log N) sort instead of one full-array
    # `ids==track_id` scan per track (that was O(rows x tracks)). _resample_tracks emits
    # rows sorted by (step, track_id), so tracks are not contiguous on entry. lexsort is
    # stable and orders each track's rows by step, matching the previous per-track argsort
    # (steps are unique within a track after _resample_tracks' (track_id, step) dedup).
    order=np.lexsort((steps,ids))
    boundaries=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1,len(order)]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        indices=order[first:last]
        for position,index in enumerate(indices):
            left=max(0,position-1);right=min(len(indices)-1,position+1)
            if left==right or steps[indices[right]]-steps[indices[left]]>10:continue
            dx=x[indices[right]]-x[indices[left]];dy=y[indices[right]]-y[indices[left]]
            if math.hypot(dx,dy)>1e-8:heading[index]=math.atan2(dy,dx)
    return heading

def _resample_tracks(frame:pd.DataFrame)->pd.DataFrame:
    """Linearly resample contiguous published 2 Hz track segments to 2.5 Hz."""
    rows=[]
    for track_id,track in frame.groupby("track_id",sort=False):
        track=track.sort_values("step",kind="stable")
        native_steps=track.step.to_numpy(dtype=np.int64); xy=track[["x","y"]].to_numpy(dtype=np.float64)
        boundaries=np.r_[0,np.flatnonzero(np.diff(native_steps)!=int(SOURCE_CLOCK_HZ/NATIVE_HZ))+1,len(track)]
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            source_steps=native_steps[left:right]
            if len(source_steps)<2: continue
            first=int(math.ceil(source_steps[0]/OUTPUT_STRIDE)*OUTPUT_STRIDE)
            last=int(math.floor(source_steps[-1]/OUTPUT_STRIDE)*OUTPUT_STRIDE)
            if first>last: continue
            target=np.arange(first,last+1,OUTPUT_STRIDE,dtype=np.int64)
            rows.append(pd.DataFrame({"track_id":int(track_id),"step":target//OUTPUT_STRIDE,
                "x":np.interp(target,source_steps,xy[left:right,0]),
                "y":np.interp(target,source_steps,xy[left:right,1])}))
    if not rows:return pd.DataFrame(columns=("track_id","step","x","y"))
    return pd.concat(rows,ignore_index=True).drop_duplicates(["track_id","step"]).sort_values(["step","track_id"],kind="stable")

def convert_file(path:Path,output:Path,root:Path,provenance:dict[str,object]|None=None)->Path:
    frame=pd.read_csv(path,usecols=lambda c:c in REQUIRED);missing=set(REQUIRED)-set(frame.columns)
    if missing:raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame=frame.loc[:,REQUIRED].dropna()
    for column in REQUIRED:frame[column]=pd.to_numeric(frame[column],errors="raise")
    if frame.empty:raise ValueError(f"{path}: no valid final trajectory rows")
    if not np.allclose(frame.tracker_id,np.rint(frame.tracker_id)):raise ValueError(f"{path}: non-integral tracker_id")
    time_origin=float(frame.time.min());x_origin=float(frame.real_world_sw_x.iloc[0]);y_origin=float(frame.real_world_sw_y.iloc[0])
    latitude_rad=float(np.median(np.arctan(np.sinh(frame.real_world_sw_y.to_numpy(dtype=float)/WEB_MERCATOR_RADIUS_M))))
    web_mercator_scale=float(math.cos(latitude_rad))
    frame["track_id"]=np.rint(frame.tracker_id).astype(np.int64);frame["step"]=np.rint((frame.time-time_origin)*SOURCE_CLOCK_HZ).astype(np.int64)
    if ((frame.step%5)!=0).mean()>.001:raise ValueError(f"{path}: timestamps are not on the published 2 Hz grid")
    frame["x"]=(frame.real_world_sw_x-x_origin)*web_mercator_scale;frame["y"]=(frame.real_world_sw_y-y_origin)*web_mercator_scale
    duplicates=int(frame.duplicated(["track_id","step"]).sum());frame=frame.groupby(["track_id","step"],as_index=False,sort=False).agg(x=("x","mean"),y=("y","mean"))
    if frame.track_id.min()<np.iinfo(np.int32).min or frame.track_id.max()>np.iinfo(np.int32).max:raise ValueError(f"{path}: tracker ID exceeds int32")
    frame=frame.sort_values(["track_id","step"],kind="stable");ids=frame.track_id.to_numpy();steps=frame.step.to_numpy();x=frame.x.to_numpy();y=frame.y.to_numpy()
    dt=np.diff(steps)/SOURCE_CLOCK_HZ;same=ids[1:]==ids[:-1];adjacent=same&(dt>0)&(dt<=1.0)
    speed=np.linalg.norm(np.diff(np.c_[x,y],axis=0),axis=1)/np.where(dt>0,dt,1.0)
    jump_tracks=np.unique(ids[:-1][adjacent&(speed>MAX_PEDESTRIAN_SPEED_MPS)])
    frame=_resample_tracks(frame)
    if frame.empty:raise ValueError(f"{path}: no tracks remain after resampling")
    ids=frame.track_id.to_numpy();steps=frame.step.to_numpy();x=frame.x.to_numpy();y=frame.y.to_numpy()
    if not np.isfinite(np.c_[x,y]).all():raise ValueError(f"{path}: non-finite local coordinates")
    heading=_headings(ids,steps,x,y);transform=[[web_mercator_scale,0.,-web_mercator_scale*x_origin],[0.,web_mercator_scale,-web_mercator_scale*y_origin],[0.,0.,1.]]
    recording=recording_name(path,root); relative=path.relative_to(root); scene_id=scene_id_from_path(path,root)
    metadata=canonical_metadata(dataset="ptdpes",scene_id=scene_id,scene_sequence=recording,recording=recording,extra={"source_file":str(relative),"source_to_local_3x3":json.dumps(transform),"heading_transform":"Heading is derived after the isotropic local Web Mercator scale correction; any future planar rotation must also rotate heading.","coordinate_source":"real_world_sw_x/y smoothed EPSG:3857 coordinates, translated and corrected to local ground-distance metres by cos(latitude)","web_mercator_latitude_deg":str(math.degrees(latitude_rad)),"web_mercator_ground_scale":str(web_mercator_scale),"native_hz":str(NATIVE_HZ),"output_hz":str(OUTPUT_HZ),"resampling":"linear interpolation within contiguous native 2 Hz segments; gaps are never bridged","time_origin_s":str(time_origin),"duplicates_averaged":str(duplicates),"jump_audit_threshold_mps":str(MAX_PEDESTRIAN_SPEED_MPS),"tracks_above_jump_audit_threshold":str(len(jump_tracks)),"tracks_filtered":str(0),"processing_provenance":"Published final modified trajectories are preserved; conversion corrects EPSG:3857 scale, localizes coordinates, and linearly resamples contiguous 2 Hz segments to the shared 2.5 Hz grid"})
    schema=schema_with_metadata(metadata);table=pa.Table.from_arrays([pa.array(ids,pa.int32()),pa.array(steps,pa.int32()),pa.array(x,pa.float32()),pa.array(y,pa.float32()),pa.array(heading,pa.float32(),from_pandas=True)],schema=schema)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(".tmp.parquet");pq.write_table(table,temporary,compression="zstd");temporary.replace(output)
    _write_provenance_sidecar(output,provenance if provenance is not None else git_provenance(REPO_ROOT))
    return output

def convert_dataset(dataset_dir:Path)->list[Path]:
    files=source_files(dataset_dir);root=dataset_dir/"raw/01.Trajectories/02.Trajectories_modified";outputs=[]
    # Resolved once per conversion run so every file's provenance sidecar records the
    # same commit/dirty snapshot, rather than one git call per file.
    provenance=git_provenance(REPO_ROOT)
    for number,path in enumerate(files,1):
        outputs.append(convert_file(path,dataset_dir/"converted"/f"{recording_name(path,root)}.traj.parquet",root,provenance));print(f"converted {number}/{len(files)}: {path.name}",flush=True)
    return outputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument("dataset_dir",type=Path,nargs="?",default=Path(__file__).resolve().parents[3]/"datasets/PTDPES");args=parser.parse_args();convert_dataset(args.dataset_dir)
if __name__=="__main__":main()
