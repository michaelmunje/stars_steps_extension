from __future__ import annotations

import argparse
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm

from canonical import CANONICAL_HZ
from config import Config,DatasetSpec,SeedConfig
from dataset_paths import PAPER_STATS_ROOT, metadata_path
from data import _read_json,_recordings,_scene_arrays
from models import constant_velocity_prediction

ROOT=Path(__file__).resolve().parents[1]
STATS_SCHEMA_VERSION=14
AREA_CELL_SIZE_M=.25
AREA_MASS_FRACTION=.75
AREA_MIN_CELLS=20


def cv_fde(xy:np.ndarray,obs:int=8,pred:int=12)->float:
    prediction=constant_velocity_prediction(xy[:obs],pred)
    return float(np.linalg.norm(prediction[-1]-xy[obs+pred-1]))


def path_efficiency(xy:np.ndarray)->float:
    length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
    return float(np.linalg.norm(xy[-1]-xy[0])/length) if length>1e-9 else 0.0


def heading_entropy(xy:np.ndarray)->float:
    delta=np.diff(xy,axis=0); hist,_=np.histogram(np.arctan2(delta[:,1],delta[:,0]),bins=36,range=(-np.pi,np.pi))
    if not hist.sum(): return 0.0
    p=hist/hist.sum(); p=p[p>0]
    return float(-(p*np.log(p)).sum()/np.log(36))


def rms_jerk(xy:np.ndarray,dt:float=.4)->float:
    jerk=np.diff(np.diff(np.diff(xy,axis=0)/dt,axis=0)/dt,axis=0)/dt
    return float(np.sqrt(np.mean(np.sum(jerk*jerk,axis=1)))) if len(jerk) else float("nan")


MIN_TCA_CONTACT_RADIUS_M=.33  # OpenTraj's pedestrian radius (references/opentraj/.../indicators/collision_energy.py:53:
                              # "# assume pedestrians radius is 0.33"), a bare in-line assumption upstream with no
                              # further citation of its own. STEPS previously used 0.30 m here, a value with no
                              # provenance anywhere in references/; K-001 replaced it with OpenTraj's 0.33 m so the
                              # two TTC-family quantities reported in the paper share one contact radius.
MIN_TCA_DEFINITION=(
    "min_time_to_closest_approach_scene reports the time (relative to the last observed step) at which "
    "two agents' constant-velocity extrapolated tracks reach minimum separation, gated on that minimum "
    "being under 2*radius (a contact/collision event), following the OpenTraj TTC/DCA convention "
    "(references/opentraj/.../indicators/collision_energy.py). This is time to *closest approach*, not "
    "time to *first contact* (the earlier moment the gap first reaches 2*radius) -- OpenTraj itself calls "
    "this quantity \"TTC\", so the name is an inherited misnomer rather than a STEPS one; A2-016 renamed "
    "the STEPS field to state the definition instead. Three implementation details diverge from the "
    "OpenTraj port and are deliberately left unchanged: (1) times>=0 is required here where upstream "
    "requires times>0, (2) the relative-velocity denominator is regularised with >=1e-9 here versus "
    "upstream's +1e-6 additive term, and (3) velocity is derived from a 2-step finite difference of "
    "positions here rather than from stored vx,vy as upstream does."
)


def min_time_to_closest_approach_scene(xy:np.ndarray,valid:np.ndarray,obs:int=8,dt:float=.4,radius:float=MIN_TCA_CONTACT_RADIUS_M)->float:
    if xy.shape[1]<=1:return float("inf")
    usable=valid[obs-2:obs,1:].all(axis=0)
    if not usable.any():return float("inf")
    primary_v=(xy[obs-1,0]-xy[obs-2,0])/dt
    dp=xy[obs-1,1:][usable]-xy[obs-1,0]
    dv=(xy[obs-1,1:][usable]-xy[obs-2,1:][usable])/dt-primary_v
    denom=np.sum(dv*dv,axis=1);moving=denom>=1e-9
    if not moving.any():return float("inf")
    dp,dv,denom=dp[moving],dv[moving],denom[moving]
    times=-np.sum(dp*dv,axis=1)/denom
    collision_course=(times>=0)&(np.linalg.norm(dp+times[:,None]*dv,axis=1)<2*radius)
    return float(times[collision_course].min()) if collision_course.any() else float("inf")


def _area(points:np.ndarray,source:dict)->tuple[float,float,str]:
    if source.get("area_m2"):
        return float(source["area_m2"]),float(source["area_m2"]),source.get("area_source","documented")
    if len(points)<3:return float("nan"),float("nan"),"unavailable"
    area=float(ConvexHull(points).volume)
    lo,hi=np.quantile(points,[.025,.975],axis=0); trimmed=points[((points>=lo)&(points<=hi)).all(axis=1)]
    trimmed_area=float(ConvexHull(trimmed).volume) if len(trimmed)>=3 else float("nan")
    return area,trimmed_area,"estimated convex hull of trajectory points"


def _geographic_polygon_area(coordinates:np.ndarray)->float:
    """Small-area equirectangular projection followed by the shoelace formula."""
    # Some PTDPES GeoJSON files declare CRS84 but still contain EPSG:3857
    # coordinates. Detect those by range and perform the analytic inverse.
    if np.max(np.abs(coordinates[:,0]))>180 or np.max(np.abs(coordinates[:,1]))>90:
        web_radius=6_378_137.0
        coordinates=np.c_[np.rad2deg(coordinates[:,0]/web_radius),np.rad2deg(np.arctan(np.sinh(coordinates[:,1]/web_radius)))]
    longitude=np.deg2rad(coordinates[:,0]);latitude=np.deg2rad(coordinates[:,1])
    radius=6_371_008.8;latitude0=float(latitude.mean())
    x=radius*(longitude-longitude.mean())*np.cos(latitude0);y=radius*(latitude-latitude.mean())
    return .5*abs(float(x@np.roll(y,1)-y@np.roll(x,1)))


def _ptdpes_polygon_areas(dataset_dir:Path)->dict[str,float]:
    areas={}
    for path in sorted((dataset_dir/"raw"/"02.Resources"/"02.Place_polygons"/"geojsons").glob("*.geojson")):
        match=re.match(r"(\d+)_",path.name)
        if not match: continue
        payload=_read_json(path)
        coordinates=np.asarray([feature["geometry"]["coordinates"] for feature in payload.get("features",[]) if feature.get("geometry",{}).get("type")=="Point"],float)
        if len(coordinates)>=3: areas[match.group(1).lstrip("0") or "0"]=_geographic_polygon_area(coordinates)
    return areas


def _grid_cell_counts(x:np.ndarray,y:np.ndarray,cell_size:float=AREA_CELL_SIZE_M)->Counter:
    cells=np.stack((np.floor(x/cell_size).astype(np.int64),np.floor(y/cell_size).astype(np.int64)),axis=1)
    unique,counts=np.unique(cells,axis=0,return_counts=True)
    return Counter({tuple(cell):int(count) for cell,count in zip(unique,counts)})


def _strided_step_occupancy(unique_steps:np.ndarray,counts:np.ndarray,stride:int)->np.ndarray:
    """Per-slot agent counts over ``range(min_step, max_step+1, stride)``.

    ``unique_steps``/``counts`` are the output of ``np.unique(steps,return_counts=True)``
    and must be non-empty. Steps that do not land on the stride grid contribute
    nothing; slots with no step are zero. Returns float64, one value per grid slot.
    """
    assert len(unique_steps),"occupancy requires at least one step"
    offsets=unique_steps-unique_steps.min();on_grid=offsets%stride==0
    grid_length=-(-int(offsets.max()+1)//stride)  # ceil(span/stride) == len(range(min,max+1,stride))
    return np.bincount(offsets[on_grid]//stride,weights=counts[on_grid],minlength=grid_length)


def _effective_grid_area_from_counts(counts:Counter,cell_size:float=AREA_CELL_SIZE_M,mass_fraction:float=AREA_MASS_FRACTION,min_cells:int=AREA_MIN_CELLS)->float:
    if not counts: return float("nan")
    descending=np.sort(np.fromiter(counts.values(),dtype=np.int64))[::-1]
    occupied=int(np.searchsorted(np.cumsum(descending),mass_fraction*descending.sum(),side="left")+1)
    return cell_size*cell_size*occupied if occupied>=min_cells else float("nan")


def _effective_grid_area(x:np.ndarray,y:np.ndarray,cell_size:float=AREA_CELL_SIZE_M,mass_fraction:float=AREA_MASS_FRACTION,min_cells:int=AREA_MIN_CELLS)->float:
    """Area of the most-visited grid cells containing ``mass_fraction`` of samples.

    Unlike an absolute points-per-cell cutoff, this definition does not change
    merely because a recording is longer.  It also rejects the sparsely visited
    tail in a directly interpretable way.
    """
    if not len(x): return float("nan")
    return _effective_grid_area_from_counts(_grid_cell_counts(x,y,cell_size),cell_size,mass_fraction,min_cells)


def analyze(spec:DatasetSpec,cfg:Config,max_scenes:int|None=None)->dict:
    task=_read_json(cfg.task_path)
    storage_hz=CANONICAL_HZ; dataset_task=task.get("datasets",{}).get(spec.name,{})
    stride=int(dataset_task.get("canonical_stride",task["canonical_stride"])); dt=stride/storage_hz
    sizing_path=PAPER_STATS_ROOT/f"{spec.name}_sizing.json"
    sizing=_read_json(sizing_path) if sizing_path.exists() else {}
    index_path=spec.path/"indices"/f"scenes-seed{cfg.seeds.data_seed}.parquet"
    index=pq.read_table(index_path).to_pylist();scene_population=len(index);full_type_counts=Counter(int(row["scene_tag"]) for row in index)
    if max_scenes is not None and len(index)>max_scenes:
        selected=np.sort(np.random.default_rng(cfg.seeds.data_seed).choice(len(index),max_scenes,replace=False))
        index=[index[int(position)] for position in selected]
    by_recording=defaultdict(list)
    for row in index: by_recording[row["recording"]].append(row)
    rng=np.random.default_rng(cfg.seeds.data_seed);occupancy=[];speed_samples=[];track_spans=[];all_segment_lengths=[];scene_values=defaultdict(list);type_counts=Counter()
    tracks_with_gaps=0;extra_segments=0
    ptdpes_areas=_ptdpes_polygon_areas(spec.path) if spec.name.lower().startswith("ptdpes") else {}
    source=_read_json(metadata_path(spec.path,"source.json"));recording_densities=[];recording_areas=[];grid_areas=[];area_omissions=[];polygon_densities=[];polygon_areas=[];documented_densities=[];grid_polygon_ratios=[];grid_polygon_ratios_by_square=defaultdict(list);neighbors_within_5m=[];neighbors_within_2m=[]
    area_aggregation=source.get("effective_area_aggregation","per_recording")
    if area_aggregation not in {"per_recording","union_across_recordings"}:
        raise ValueError(f"Unsupported effective_area_aggregation: {area_aggregation}")
    shared_grid_counts=Counter();shared_area_recordings=0
    recording_paths=_recordings(spec.path)
    scene_progress=tqdm(total=len(index),desc=f"statistics scenes: {spec.name}",unit="scene",mininterval=1.0)
    for path in tqdm(recording_paths,desc=f"statistics recordings: {spec.name}",unit="recording"):
        table=pq.read_table(path,columns=["track_id","step","x","y"]); arrays=[table[c].combine_chunks().to_numpy() for c in ("track_id","step","x","y")]
        ids,steps,x,y=arrays; unique,counts=np.unique(steps,return_counts=True)
        order=np.lexsort((steps,ids));ordered_ids,ordered_steps=ids[order],steps[order];ordered_xy=np.c_[x[order],y[order]];delta_steps=np.diff(ordered_steps)
        track_boundaries=np.r_[0,np.flatnonzero(np.diff(ordered_ids))+1,len(ordered_ids)]
        track_spans.extend((ordered_steps[track_boundaries[1:]-1]-ordered_steps[track_boundaries[:-1]])/storage_hz+dt)
        for left,right in zip(track_boundaries[:-1],track_boundaries[1:]):
            track_steps=ordered_steps[left:right]
            track_steps=track_steps[track_steps%stride==0]
            if not len(track_steps):continue
            boundaries=np.flatnonzero(np.diff(track_steps)!=stride)+1
            if len(boundaries):
                tracks_with_gaps+=1;extra_segments+=len(boundaries)
            all_segment_lengths.extend(len(segment) for segment in np.split(track_steps,boundaries) if len(segment))
        task_indices=np.flatnonzero(ordered_steps%stride==0)
        task_ids=ordered_ids[task_indices];task_steps=ordered_steps[task_indices];task_xy=ordered_xy[task_indices]
        task_delta=np.diff(task_steps);adjacent=(task_ids[1:]==task_ids[:-1])&(task_delta==stride);pair=np.flatnonzero(adjacent)
        # LargeView was manually tracked at about 2 Hz and interpolated in the
        # release.  Keep it in the dataset, but do not let interpolation knots
        # masquerade as native velocity/jerk evidence in the statistics table.
        velocity_statistics_allowed=not path.stem.startswith("LargeView_")
        if len(pair) and velocity_statistics_allowed:
            speeds=np.linalg.norm(task_xy[pair+1]-task_xy[pair],axis=1)/dt
            if len(speeds)>50000:speeds=rng.choice(speeds,50000,replace=False)
            speed_samples.append(speeds)
        if len(unique):
            file_occupancy=_strided_step_occupancy(unique,counts,stride)
            occupancy.extend(file_occupancy)
            place_codes=re.findall(r"__(\d+)_",path.stem)
            place_code=(place_codes[-1].lstrip("0") or "0") if place_codes else None
            cell_counts=_grid_cell_counts(x,y)
            grid_area=_effective_grid_area_from_counts(cell_counts)
            if area_aggregation=="union_across_recordings" and cell_counts:
                shared_grid_counts.update(cell_counts);shared_area_recordings+=1
            if np.isfinite(grid_area) and grid_area>0:
                grid_areas.append(grid_area)
            # Main-table density always uses the same effective walked-area
            # denominator.  Documented/polygon coverage areas are retained as
            # separate appendix values and are never mixed into this column.
            recording_area=grid_area
            if np.isfinite(recording_area) and recording_area>0:
                recording_areas.append(recording_area)
                if area_aggregation=="per_recording": recording_densities.extend(file_occupancy/recording_area)
            elif area_aggregation=="per_recording":
                area_omissions.append(path.name.removesuffix(".traj.parquet"))
            if source.get("area_m2"):
                documented_densities.extend(file_occupancy/float(source["area_m2"]))
            if place_code in ptdpes_areas:
                polygon_area=ptdpes_areas[place_code];polygon_areas.append(polygon_area);polygon_densities.extend(file_occupancy/polygon_area)
                if np.isfinite(grid_area) and grid_area>0:
                    ratio=grid_area/polygon_area;grid_polygon_ratios.append(ratio);grid_polygon_ratios_by_square[place_code].append(ratio)
        rows=by_recording.get(path.name.removesuffix(".traj.parquet"),[])
        for row in rows:
            final=int(row["start_step"])+(int(task["history_steps"])+int(task["future_steps"])-1)*stride
            lo=np.searchsorted(steps,int(row["start_step"]),side="left");hi=np.searchsorted(steps,final,side="right")
            scene=_scene_arrays(ids[lo:hi],steps[lo:hi],x[lo:hi],y[lo:hi],int(row["start_step"]),int(task["history_steps"]),int(task["future_steps"]),stride,int(row["primary_track_id"]))
            primary=scene["xy"][:,0]; type_counts[int(row["scene_tag"])]+=1
            present=scene["valid"][int(task["history_steps"])-1,1:]
            neighbor_distance=np.linalg.norm(scene["xy"][int(task["history_steps"])-1,1:][present]-primary[int(task["history_steps"])-1],axis=1)
            neighbors_within_5m.append(int(np.count_nonzero(neighbor_distance<5.0)))
            neighbors_within_2m.append(int(np.count_nonzero(neighbor_distance<2.0)))
            scene_values["cv_fde"].append(cv_fde(primary,obs=int(task["history_steps"]),pred=int(task["future_steps"])));scene_values["path_efficiency"].append(path_efficiency(primary))
            scene_values["heading_entropy"].append(heading_entropy(primary))
            if velocity_statistics_allowed: scene_values["rms_jerk"].append(rms_jerk(primary,dt))
            scene_values["min_tca"].append(min_time_to_closest_approach_scene(scene["xy"],scene["valid"],dt=dt))
            scene_progress.update()
    scene_progress.close()
    occ=np.asarray(occupancy,float)
    mean_recording_area=float(np.mean(recording_areas)) if recording_areas else float("nan")
    median_recording_area=float(np.median(recording_areas)) if recording_areas else float("nan")
    if area_aggregation=="union_across_recordings":
        area=_effective_grid_area_from_counts(shared_grid_counts)
        density=occ/area if np.isfinite(area) and area>0 else np.array([])
        area_method="occupancy_grid_75pct_mass_union_same_scene"
        area_source=f"shared-site union across recordings on a {AREA_CELL_SIZE_M:g} m occupancy grid; smallest cells containing {100*AREA_MASS_FRACTION:g}% of observations"
    else:
        area=median_recording_area
        density=np.asarray(recording_densities,float)
        area_method="occupancy_grid_75pct_mass"
        area_source=f"per-recording {AREA_CELL_SIZE_M:g} m occupancy grid; smallest cells containing {100*AREA_MASS_FRACTION:g}% of observations; recordings require >={AREA_MIN_CELLS} cells"
    speeds=np.concatenate(speed_samples) if speed_samples else np.array([])
    finite_tca=np.asarray(scene_values["min_tca"]);finite_tca=finite_tca[np.isfinite(finite_tca)]
    result={"stats_schema_version":STATS_SCHEMA_VERSION,"dataset":spec.name,"storage_hz":storage_hz,"task_hz":1/dt,"hours":sizing.get("hours"),"n_trajectories_raw":sizing.get("T_d_raw"),"n_trajectories_usable":sizing.get("T_d_usable"),
      "track_length_mean_s":sizing.get("L_mean",float("nan"))*dt,"track_length_median_s":sizing.get("L_med",float("nan"))*dt,"all_segment_length_median_s":float(np.median(all_segment_lengths)*dt),"track_span_mean_s":float(np.mean(track_spans)),"track_span_median_s":float(np.median(track_spans)),"track_length_definition":"track_length is conditioned on >=20 samples; all_segment_length includes every contiguous segment; track span is first-to-last converted sample plus one sample interval",
      "n_tracks_with_gaps":tracks_with_gaps,"n_extra_segments":extra_segments,"segments_per_gapped_track":float(1+extra_segments/tracks_with_gaps) if tracks_with_gaps else 1.0,
      "area_m2":area,"area_mean_recording_m2":mean_recording_area,"area_median_recording_m2":median_recording_area,"area_method":area_method,"area_source":area_source,"area_recordings_used":shared_area_recordings if area_aggregation=="union_across_recordings" else len(recording_areas),"area_recordings_omitted":area_omissions,"occupancy_mean":float(occ.mean()) if len(occ) else float("nan"),"occupancy_p95":float(np.quantile(occ,.95)) if len(occ) else float("nan"),"occupancy_max":float(occ.max()) if len(occ) else float("nan"),
      "density_mean_ped_m2":float(density.mean()) if len(density) else float("nan"),"density_p95_ped_m2":float(np.quantile(density,.95)) if len(density) else float("nan"),"density_max_ped_m2":float(density.max()) if len(density) else float("nan"),"mean_speed_mps":float(speeds.mean()) if len(speeds) else float("nan"),"p999_speed_mps":float(np.quantile(speeds,.999)) if len(speeds) else float("nan"),"type_counts":dict(full_type_counts),"type3_fraction":full_type_counts[3]/scene_population if scene_population else float("nan"),"mean_neighbors_within_5m":float(np.mean(neighbors_within_5m)) if len(neighbors_within_5m) else float("nan"),"mean_neighbors_within_2m":float(np.mean(neighbors_within_2m)) if len(neighbors_within_2m) else float("nan"),"scene_filter_survival":sizing.get("scene_filter_survival"),
      "cv_fde_mean":float(np.mean(scene_values["cv_fde"])) if len(scene_values["cv_fde"]) else float("nan"),"cv_definition":"last observed displacement extrapolated one canonical timestep at a time (Schöller et al. convention)","path_efficiency_mean":float(np.mean(scene_values["path_efficiency"])) if len(scene_values["path_efficiency"]) else float("nan"),"heading_entropy_mean":float(np.mean(scene_values["heading_entropy"])) if len(scene_values["heading_entropy"]) else float("nan"),"rms_jerk_mean":float(np.nanmean(scene_values["rms_jerk"])) if scene_values["rms_jerk"] else float("nan"),"min_tca_p05_steps_r033_s":float(np.quantile(finite_tca,.05)) if len(finite_tca) else float("inf"),"min_tca_definition":MIN_TCA_DEFINITION,"min_tca_contact_radius_m":MIN_TCA_CONTACT_RADIUS_M,"scene_population":scene_population,"scenes_analyzed":sum(type_counts.values()),"scene_sample_fraction":sum(type_counts.values())/scene_population if scene_population else float("nan"),"scene_limit":max_scenes,"scene_sampling":"all indexed scenes" if max_scenes is None or scene_population<=max_scenes else f"deterministic uniform sample without replacement (data_seed={cfg.seeds.data_seed})","velocity_stats_exclusions":"MADRAS LargeView recordings (manually tracked at ~2 Hz and interpolated)" if spec.name.lower()=="madras" else "none"}
    analysis_summary_path=spec.path/"analysis/opentraj_summary.json"
    analysis_summary=_read_json(analysis_summary_path) if analysis_summary_path.exists() else {}
    result.update({
        "valid_opentraj_analysis_trajlets":analysis_summary.get("valid_opentraj_trajlets"),
        "opentraj_static_filtered_trajlets":analysis_summary.get("opentraj_static_filtered_trajlets"),
        "retained_opentraj_analysis_trajlets":analysis_summary.get("retained_analysis_trajlets"),
        "valid_forecast_windows":scene_population,
        "sample_representations":"OpenTraj analysis trajlets and 8+12 forecast windows are independent indices",
    })
    if grid_areas:
        result["grid_area_mean_m2"]=float(np.mean(grid_areas))
        result["grid_area_median_m2"]=float(np.median(grid_areas))
        if source.get("area_m2"):
            result["grid_to_documented_area_ratio_mean"]=float(np.mean(grid_areas)/float(source["area_m2"]))
    if documented_densities:
        documented=np.asarray(documented_densities);result.update({"documented_area_m2":float(source["area_m2"]),"documented_area_source":source.get("area_source","documented"),"documented_density_mean_ped_m2":float(documented.mean()),"documented_density_interpretation":"coverage-area density; reported separately from the comparable effective walked-area density"})
    if polygon_densities:
        polygon_density=np.asarray(polygon_densities);result.update({"polygon_area_method":"supplied_polygon","polygon_area_mean_m2":float(np.mean(polygon_areas)),"polygon_density_mean_ped_m2":float(polygon_density.mean()),"polygon_density_p95_ped_m2":float(np.quantile(polygon_density,.95)),"polygon_density_max_ped_m2":float(polygon_density.max()),"effective_to_polygon_area_ratio_mean":float(np.mean(grid_polygon_ratios)),"effective_to_polygon_area_ratio_median":float(np.median(grid_polygon_ratios)),"grid_to_polygon_area_ratio_by_square":{key:float(np.mean(value)) for key,value in sorted(grid_polygon_ratios_by_square.items())},"polygon_density_interpretation":"lower bound on density using the full supplied place polygon"})
    if spec.name.lower() in {"eth","ucy"}:
        measured=result["density_mean_ped_m2"]
        if not .1<=measured<=.5:
            raise AssertionError(f"{spec.name} density {measured:.4f} ped/m2 is outside the MADRAS comparison range [0.1, 0.5]")
        result["density_validation"]="within MADRAS-published ETH/UCY comparison range [0.1, 0.5] ped/m2"
    return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("dataset_dir",type=Path);parser.add_argument("--name");parser.add_argument("--data-seed",type=int,default=42);parser.add_argument("--max-scenes",type=int)
    args=parser.parse_args();spec=DatasetSpec(args.name or args.dataset_dir.name,args.dataset_dir);result=analyze(spec,Config(seeds=SeedConfig(args.data_seed,42)),args.max_scenes)
    prefix="OK" if any(word in result["area_source"] for word in ("documented","supplied")) else "EST";print(f"[{prefix}] {spec.name:8s} area = {result['area_m2']:.1f} m2 ({result['area_source']}; median recording {result['area_median_recording_m2']:.1f})")
    speed_level="OK" if 1.0<=result["mean_speed_mps"]<=1.8 else "WARN";print(f"[{speed_level}] {spec.name:8s} mean speed = {result['mean_speed_mps']:.3f} m/s; p999 = {result['p999_speed_mps']:.3f}")
    if result["scenes_analyzed"]<100:print(f"[WARN] {spec.name:8s} only {result['scenes_analyzed']} scenes contributed Table 2 metrics")
    output=PAPER_STATS_ROOT/f"{spec.name}_paper_stats.json";output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+"\n");print(output)


if __name__=="__main__":main()
