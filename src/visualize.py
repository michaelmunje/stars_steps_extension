from __future__ import annotations

import argparse
import logging
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from config import Config, DatasetSpec, SeedConfig
from data import Scene, SceneDataset, _read_json
from models import ConstantVelocityModel
from normalization import inverse_scene

logger = logging.getLogger(__name__)


def _recording_metadata(path: Path) -> dict[str, str]:
    metadata = pq.ParquetFile(path).metadata.metadata or {}
    return {key.decode(): value.decode() for key, value in metadata.items()}


def load_track_diagnostic_window(
    dataset_dir: Path,
    recording: str,
    track_ids: list[int],
    anchor_seconds: float,
    history_steps: int,
    future_steps: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Load only requested tracks, with ``anchor_seconds`` at the last observed step."""
    path = dataset_dir / "converted" / f"{recording}.traj.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical recording not found: {path}")
    if not track_ids:
        raise ValueError("At least one track ID is required")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Track IDs must be unique")

    metadata = _recording_metadata(path)
    canonical_hz = float(metadata.get("canonical_hz", metadata.get("output_hz", "2.5")))
    anchor_step = int(round(anchor_seconds * canonical_hz))
    start_step = anchor_step - history_steps + 1
    end_step = anchor_step + future_steps
    if start_step < 0:
        raise ValueError(
            f"Requested time {anchor_seconds:g}s is too early for {history_steps} observations"
        )

    table = pq.read_table(
        path,
        columns=["track_id", "step", "x", "y"],
        filters=[
            ("track_id", "in", track_ids),
            ("step", ">=", start_step),
            ("step", "<=", end_step),
        ],
    )
    frame = table.to_pandas()
    duplicate = frame.duplicated(["track_id", "step"])
    if duplicate.any():
        raise ValueError(f"Duplicate (track_id, step) rows in {path}")

    steps = np.arange(start_step, end_step + 1, dtype=np.int32)
    xy = np.full((len(steps), len(track_ids), 2), np.nan, dtype=np.float32)
    index_by_step = {int(step): index for index, step in enumerate(steps)}
    index_by_track = {track_id: index for index, track_id in enumerate(track_ids)}
    for row in frame.itertuples(index=False):
        xy[index_by_step[int(row.step)], index_by_track[int(row.track_id)]] = (row.x, row.y)

    missing = [track_id for i, track_id in enumerate(track_ids) if not np.isfinite(xy[:, i]).any()]
    if missing:
        raise ValueError(
            f"Tracks {missing} have no samples between steps {start_step} and {end_step}"
        )
    return steps, xy, anchor_step, canonical_hz


def render_track_diagnostic(
    dataset_dir: Path,
    recording: str,
    track_ids: list[int],
    anchor_seconds: float,
    output: Path,
    history_steps: int,
    future_steps: int,
) -> Path:
    """Render an ID-restricted BEV GIF around an exact recording-relative time."""
    steps, xy, anchor_step, canonical_hz = load_track_diagnostic_window(
        dataset_dir, recording, track_ids, anchor_seconds, history_steps, future_steps
    )
    finite = np.isfinite(xy).all(axis=-1)
    bounds = xy[finite]
    padding = max(1.0, 0.08 * float(np.ptp(bounds, axis=0).max()))
    palette = plt.get_cmap("tab10").colors
    colors = [palette[index % len(palette)] for index in range(len(track_ids))]

    output.parent.mkdir(parents=True, exist_ok=True)
    final_image = output.with_name(f"{output.stem}_final.png")
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(bounds[:, 0].min() - padding), float(bounds[:, 0].max() + padding))
    axis.set_ylim(float(bounds[:, 1].min() - padding), float(bounds[:, 1].max() + padding))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(f"Track diagnostic: {recording}")
    axis.grid(alpha=0.2)
    lines = [
        axis.plot([], [], color=color, linewidth=2.0, label=f"Track {track_id}")[0]
        for track_id, color in zip(track_ids, colors)
    ]
    markers = [axis.scatter([], [], color=color, s=42, zorder=4) for color in colors]
    anchor_markers = [
        axis.scatter([], [], facecolors="none", edgecolors=color, s=150, linewidths=2.0, zorder=5)
        for color in colors
    ]
    time_label = axis.text(0.02, 0.98, "", transform=axis.transAxes, va="top")
    axis.legend(loc="best")

    def update(frame_index: int):
        for agent, (line, marker, anchor_marker) in enumerate(
            zip(lines, markers, anchor_markers)
        ):
            # Retaining NaNs deliberately breaks the line across missing annotations.
            path = xy[: frame_index + 1, agent]
            line.set_data(path[:, 0], path[:, 1])
            current = xy[frame_index, agent]
            marker.set_offsets(current[None, :] if np.isfinite(current).all() else np.empty((0, 2)))
            anchor_position = xy[history_steps - 1, agent]
            if frame_index >= history_steps - 1 and np.isfinite(anchor_position).all():
                anchor_marker.set_offsets(anchor_position[None, :])
            else:
                anchor_marker.set_offsets(np.empty((0, 2)))
        step = int(steps[frame_index])
        elapsed = step / canonical_hz
        minutes, seconds = divmod(elapsed, 60.0)
        suffix = "  (requested time)" if step == anchor_step else ""
        time_label.set_text(
            f"canonical timestep {step}  |  t={int(minutes):02d}:{seconds:04.1f}{suffix}"
        )
        return [*lines, *markers, *anchor_markers, time_label]

    animation = FuncAnimation(figure, update, frames=len(steps), interval=250, blit=True)
    animation.save(output, writer=PillowWriter(fps=4))
    update(len(steps) - 1)
    figure.savefig(final_image, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.info("Recording-relative time: %.1fs -> canonical timestep %d", anchor_seconds, anchor_step)
    logger.info(
        "Window: steps %d..%d (%.1fs..%.1fs)",
        steps[0], steps[-1], steps[0] / canonical_hz, steps[-1] / canonical_hz,
    )
    for agent, track_id in enumerate(track_ids):
        present_steps = steps[finite[:, agent]]
        logger.info(
            "Track %d: %d/%d samples; steps %d..%d",
            track_id, len(present_steps), len(steps), present_steps[0], present_steps[-1],
        )
    logger.info("GIF: %s\nFinal image: %s", output, final_image)
    return output


def _absolute_scene(scene: Scene) -> np.ndarray:
    """Undo only the SE(2) scene transform (rotation/center), not any augmentation.

    ``prepare_forecast_representation`` (src/forecasting_representation.py) may apply
    random rotation/flip/scale *after* that transform when the scene comes from a
    ``training=True`` split; that augmentation is not inverted here. Plotting such a
    scene under real-world axis labels and a title naming the recording/step would
    silently mislabel augmented coordinates as raw world coordinates. ``scene.augmentation``
    is always populated by ``SceneDataset.__getitem__`` with an ``"applied"`` bool (see
    ``apply_training_augmentation``); a hand-built ``Scene`` that never went through that
    pipeline leaves it at the dataclass default of ``None``, which we treat as "unknown,
    trust it" rather than fail on, since that is how this module's own tests build scenes.
    """
    # A plain ``assert`` would be stripped under ``python -O``, disabling exactly the
    # guard whose purpose is to make a silent mislabel loud. Raise explicitly instead.
    if scene.augmentation is not None and scene.augmentation.get("applied") is not False:
        raise ValueError(
            "_absolute_scene only inverts the scene's rotation/center; refusing to plot a "
            f"scene whose forecasting representation also applied augmentation ({scene.augmentation!r}), "
            "which would mislabel augmented coordinates as raw world coordinates."
        )
    return inverse_scene(scene.xy, scene.rotation, scene.center)


def render_storyboard(scene: Scene, output: Path, task: dict, show_title: bool = True,
                      show_legend: bool = True) -> Path:
    """Render four vector snapshots across a scene for paper/LaTeX use."""
    history_steps, future_steps = int(task["history_steps"]), int(task["future_steps"])
    total = history_steps + future_steps
    dataset_task = task.get("datasets", {}).get(scene.dataset, {})
    sample_dt = 1.0 / float(dataset_task.get("sample_hz", task["sample_hz"]))
    absolute = _absolute_scene(scene); valid = scene.valid
    prediction_centered = ConstantVelocityModel(history_steps, future_steps).predict(scene)[0]
    prediction = inverse_scene(prediction_centered[:, None, :], scene.rotation, scene.center)[:, 0]
    snapshot_interval=max(1,int(round(1.6/sample_dt)))
    frames = list(range(snapshot_interval-1,total,snapshot_interval))
    if frames[-1] != total - 1:
        frames.append(total - 1)
    bounds = np.concatenate((absolute[valid], prediction), axis=0); padding = 1.0
    xlim = (float(bounds[:, 0].min() - padding), float(bounds[:, 0].max() + padding))
    ylim = (float(bounds[:, 1].min() - padding), float(bounds[:, 1].max() + padding))
    observed_color = "tab:blue"; future_color = "tab:orange"
    context_color = (0.62, 0.62, 0.62, 0.65)
    figure, axes = plt.subplots(1, len(frames), figsize=(17.5, 4.5), sharex=True, sharey=True)
    for axis, frame in zip(axes, frames):
        for agent in range(absolute.shape[1]):
            if agent == 0:
                shown = valid[:frame + 1, agent]; path = absolute[:frame + 1, agent][shown]
                if not len(path): continue
                observed_end = min(frame, history_steps - 1)
                observed = absolute[:observed_end + 1, 0][valid[:observed_end + 1, 0]]
                axis.plot(observed[:, 0], observed[:, 1], color=observed_color, linewidth=2.0)
                if frame >= history_steps:
                    future_valid = valid[history_steps - 1:frame + 1, 0]
                    future = absolute[history_steps - 1:frame + 1, 0][future_valid]
                    axis.plot(future[:, 0], future[:, 1], color=future_color, linewidth=2.0)
                marker_color = future_color if frame >= history_steps else observed_color
                axis.scatter(path[-1, 0], path[-1, 1], color=marker_color, s=20, zorder=4)
            else:
                # Retaining NaNs deliberately breaks the line across missing annotations
                # (matches the GIF writer's convention above); do not compact by `valid`.
                if not valid[:frame + 1, agent].any(): continue
                path = absolute[:frame + 1, agent]
                axis.plot(path[:, 0], path[:, 1], color=context_color, linewidth=0.9)
                # Only draw the "current position" marker when the agent actually has a
                # valid sample at this exact frame; a gap or an ended track draws no dot.
                if valid[frame, agent]:
                    axis.scatter(path[-1, 0], path[-1, 1], color=context_color, s=20, zorder=4)
        if frame >= history_steps:
            count = frame - history_steps + 1
            forecast_path = np.vstack((absolute[history_steps - 1, 0], prediction[:count]))
            axis.plot(forecast_path[:, 0], forecast_path[:, 1], color=observed_color, linestyle="--", linewidth=1.3)
        axis.set(xlim=xlim, ylim=ylim, aspect="equal")
        axis.grid(alpha=0.2)
        axis.set_title(f"$t={(frame + 1) * sample_dt:.1f}\\,s$")
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    dataset_name = scene.dataset.removesuffix("_visual_sample").upper()
    if show_title:
        figure.suptitle(f"Trajectory Example of {dataset_name}")
    observation_boundary = history_steps * sample_dt
    if show_legend:
        figure.legend(handles=[
            Line2D([0], [0], color=observed_color,
                   label=f"Past observation ($t \\leq {observation_boundary:g}$ s)"),
            Line2D([0], [0], color=context_color, label="Context Pedestrian Trajectories"),
            Line2D([0], [0], color=future_color, linewidth=2.0,
                   label=f"Future ground truth ($t > {observation_boundary:g}$ s)"),
            Line2D([0], [0], color=observed_color, linestyle="--", label="Constant Velocity"),
        ], loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
    top = 0.92 if show_title else 0.98
    bottom = 0.16 if show_legend else 0.04
    figure.tight_layout(rect=(0, bottom, 1, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output


def visualize(scene: Scene, output: Path, task: dict, storyboard_output: Path | None = None,
              storyboard_title: bool = True, storyboard_legend: bool = True) -> None:
    history_steps, future_steps = int(task["history_steps"]), int(task["future_steps"])
    dataset_task=task.get("datasets",{}).get(scene.dataset,{})
    sample_dt = 1.0 / float(dataset_task.get("sample_hz",task["sample_hz"]))
    absolute = _absolute_scene(scene)
    valid = scene.valid
    prediction_centered = ConstantVelocityModel(history_steps, future_steps).predict(scene)[0]
    prediction = inverse_scene(prediction_centered[:, None, :], scene.rotation, scene.center)[:, 0]
    output.parent.mkdir(parents=True, exist_ok=True)
    final_image = output.with_name(f"{output.stem}_final.png")

    points = [absolute[valid], prediction]
    bounds = np.concatenate(points)
    padding = 1.0
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(bounds[:, 0].min()-padding), float(bounds[:, 0].max()+padding))
    axis.set_ylim(float(bounds[:, 1].min()-padding), float(bounds[:, 1].max()+padding))
    axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)")
    axis.set_title(f"{scene.dataset} / {scene.recording} / step {scene.start_step} / type {scene.scene_tag}")
    axis.grid(alpha=0.2)
    primary_color="tab:blue"; context_color=(0.55,0.55,0.55,0.75)
    colors=[primary_color]+[context_color]*(absolute.shape[1]-1)
    lines=[axis.plot([],[],color=color,linewidth=1.7 if i==0 else 1.2)[0] for i,color in enumerate(colors)]
    forecast=axis.plot([],[],color=primary_color,linestyle="--",linewidth=1.3)[0]
    markers=axis.scatter(absolute[0,:,0],absolute[0,:,1],c=colors,s=24)
    endings=axis.scatter([],[],marker="x",c=[context_color],s=55,linewidths=1.8,zorder=5)
    time_label=axis.text(0.02,0.98,"",transform=axis.transAxes,va="top")
    axis.legend(handles=[Line2D([0],[0],color=primary_color,label="Primary / ground truth"),Line2D([0],[0],color=context_color,label="Context Pedestrian Trajectories"),Line2D([0],[0],color=primary_color,linestyle="--",label="Constant Velocity"),Line2D([0],[0],color=context_color,marker="x",linestyle="None",label="Annotation gap or end")],loc="lower left",fontsize="small")

    def update(frame:int):
        positions=[]; terminated=[]
        for agent,line in enumerate(lines):
            shown=valid[:frame+1,agent]
            path=absolute[:frame+1,agent][shown]
            line.set_data(path[:,0],path[:,1]) if len(path) else line.set_data([],[])
            positions.append(path[-1] if len(path) else np.array([np.nan,np.nan]))
            if agent>0 and frame>=history_steps and not valid[frame,agent] and valid[:frame,agent].any():
                terminated.append(positions[-1])
        if frame>=history_steps:
            count=frame-history_steps+1; path=np.vstack((absolute[history_steps-1,0],prediction[:count]))
            forecast.set_data(path[:,0],path[:,1])
        else: forecast.set_data([],[])
        markers.set_offsets(np.asarray(positions)); endings.set_offsets(np.asarray(terminated).reshape(-1,2))
        time_label.set_text(f"timestep {frame+1}/{history_steps+future_steps}  t={frame*sample_dt:.1f}s")
        return [*lines,forecast,markers,endings,time_label]

    animation=FuncAnimation(figure,update,frames=history_steps+future_steps,interval=100,blit=True)
    animation.save(output,writer=PillowWriter(fps=5)); update(history_steps+future_steps-1)
    figure.savefig(final_image,dpi=150,bbox_inches="tight"); plt.close(figure)
    dataset_key = scene.dataset.removesuffix("_visual_sample").lower()
    storyboard = render_storyboard(
        scene, storyboard_output or output.parent / f"{dataset_key}_traj_example.pdf", task,
        show_title=storyboard_title, show_legend=storyboard_legend
    )
    logger.info(
        "GIF: %s\nFinal image: %s\nStoryboard: %s\nPedestrians: %d  Type: %s",
        output, final_image, storyboard, absolute.shape[1], scene.scene_tag,
    )


def _select_indices(dataset: SceneDataset,count:int,selection:str,seed:int)->list[int]:
    count=min(count,len(dataset))
    if selection=="random": return random.Random(seed).sample(range(len(dataset)),count)
    if selection=="dynamic-crowded":
        candidates=[i for i,row in enumerate(dataset.rows) if int(row["scene_tag"])!=1]
    else:
        candidates=range(len(dataset))
    # Only the agent count is needed to rank candidates; recording/start_step come from the
    # index row, which is exactly what SceneDataset stores on the Scene (data.py builds the
    # Scene with row["recording"] and int(row["start_step"])). `SceneDataset.agent_count`
    # reads that count without materializing a full Scene (no `prepare_forecast_representation`,
    # no SE(2) transform/features/augmentation, no array clones) -- see data.py for why that
    # matters at dataset scale.
    #
    # Ranking used to carry `recording` (a Python string) and `start_step` in every tuple, but
    # `index` is unique across candidates so a tie on (agent_count, index) is impossible: tuple
    # comparison never reaches a third element. Rank on (agent_count, index) alone -- a plain
    # list of 2-tuples, not a numpy array, because argsort's tie-break behavior is easy to get
    # subtly wrong for a descending, unique-key sort, and `sorted(..., reverse=True)` is already
    # exactly correct here -- and fetch recording/start_step from `dataset.rows[index]` only for
    # the candidates the selection loop below actually examines.
    ranked=sorted(
        (
            (dataset.agent_count(i),i)
            for i in tqdm(candidates,desc=f"ranking {selection} candidates",unit="scene")
        ),
        reverse=True,
    )
    selected=[]; windows=[]; separation=dataset.total_steps*dataset.canonical_stride
    for _,index in ranked:
        recording=dataset.rows[index]["recording"]; start=int(dataset.rows[index]["start_step"])
        if all(r!=recording or abs(start-s)>=separation for r,s in windows):
            selected.append(index); windows.append((recording,start))
            if len(selected)==count: break
    return selected


def main()->None:
    parser=argparse.ArgumentParser(description="Visualize primary-agent trajectory scenes")
    parser.add_argument("dataset_dir",type=Path); parser.add_argument("--split",choices=("train","val","test"),default="val")
    parser.add_argument("--count",type=int,default=1); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--selection",choices=("random","crowded","dynamic-crowded"),default="random"); parser.add_argument("--output-dir",type=Path,default=Path("paper/figures/trajectory_examples"))
    parser.add_argument("--no-storyboard-title",action="store_true"); parser.add_argument("--no-storyboard-legend",action="store_true")
    parser.add_argument("--recording", help="Canonical recording name for direct track diagnostics")
    parser.add_argument("--track-ids", type=int, nargs="+", help="Show only these track IDs")
    parser.add_argument("--time-seconds", type=float, help="Recording-relative time placed at the last observed step")
    args=parser.parse_args()
    # `train_rotation` is a deprecated flat field that `Config.__post_init__`
    # (src/config.py:120-127) translates into `augmentation.random_rotation.enabled`
    # while warning on every run. Resolve the forecasting config the normal way first,
    # then express the same "never plot rotated coordinates" intent directly on it,
    # so no deprecated field is passed and no warning fires.
    resolved_forecasting=Config(seeds=SeedConfig(args.seed,args.seed)).forecasting
    if resolved_forecasting.augmentation.random_rotation.enabled:
        resolved_forecasting=replace(resolved_forecasting,augmentation=replace(
            resolved_forecasting.augmentation,
            random_rotation=replace(resolved_forecasting.augmentation.random_rotation,enabled=False),
        ))
    cfg=Config(seeds=SeedConfig(args.seed,args.seed),forecasting=resolved_forecasting)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"visualize-{datetime.now():%Y%m%dT%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger.info("Logging to %s", log_path)
    diagnostic_values = (args.recording, args.track_ids, args.time_seconds)
    if any(value is not None for value in diagnostic_values):
        if not all(value is not None for value in diagnostic_values):
            parser.error("--recording, --track-ids, and --time-seconds must be supplied together")
        task = _read_json(cfg.task_path)
        history_steps, future_steps = int(task["history_steps"]), int(task["future_steps"])
        time_token = f"{args.time_seconds:g}s".replace(".", "p")
        id_token = "-".join(str(track_id) for track_id in args.track_ids)
        output = args.output_dir / f"{args.dataset_dir.name}-{args.recording}-t{time_token}-tracks-{id_token}.gif"
        render_track_diagnostic(
            args.dataset_dir, args.recording, args.track_ids, args.time_seconds, output,
            history_steps, future_steps,
        )
        return
    dataset=SceneDataset(DatasetSpec(args.dataset_dir.name,args.dataset_dir),args.split,cfg)
    if not dataset: raise ValueError(f"Split {args.split!r} has no samples")
    task=_read_json(cfg.task_path)
    for number,index in enumerate(_select_indices(dataset,args.count,args.selection,args.seed),1):
        scene=dataset[index]; output=args.output_dir/f"{args.dataset_dir.name}-{args.split}-{number:02d}-{scene.recording}-{scene.start_step}.gif"
        dataset_key=scene.dataset.removesuffix("_visual_sample").lower()
        pdf=args.output_dir/(f"{dataset_key}_traj_example.pdf" if args.count==1 else f"{dataset_key}_traj_example_{number:02d}.pdf")
        visualize(scene,output,task,pdf,not args.no_storyboard_title,not args.no_storyboard_legend)


if __name__=="__main__": main()
