from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
from tqdm.auto import tqdm

from canonical import CANONICAL_DT, CANONICAL_HZ
from config import Config, DatasetSpec, SeedConfig
from data import SPLITS, SceneDataset


def _scene_features(scene, obs_len: int) -> tuple[int, float, float]:
    primary = scene.xy[:, 0]
    velocity = np.diff(primary, axis=0) / CANONICAL_DT
    speed = float(np.mean(np.linalg.norm(velocity, axis=1)))
    theta = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turning = float(np.sum(np.abs(np.diff(theta)))) if len(theta) > 1 else 0.0
    density = int(scene.valid[obs_len - 1].sum())
    return density, speed, turning


def analyze_dataset(dataset_dir: Path, seed: int = 42, max_scenes_per_split: int = 1000) -> dict:
    cfg = Config(seeds=SeedConfig(seed, seed), train_rotation=False)
    spec = DatasetSpec(dataset_dir.name, dataset_dir)
    rng = np.random.default_rng(seed)
    result = {"dataset": spec.name, "splits": {}}
    for split in SPLITS:
        dataset = SceneDataset(spec, split, cfg, include_rejected=True)
        indices = np.arange(len(dataset))
        if len(indices) > max_scenes_per_split:
            indices = np.sort(rng.choice(indices, max_scenes_per_split, replace=False))
        types, subtypes, density, speed, turning = Counter(), Counter(), [], [], []
        for index in tqdm(indices, total=len(indices), desc=f"diversity scenes: {spec.name}/{split}", unit="scene"):
            scene = dataset[int(index)]
            types[scene.scene_tag] += 1
            if scene.scene_tag == 3:
                subtypes.update(scene.sub_tags)
            d, s, t = _scene_features(scene, dataset.history_steps)
            density.append(d); speed.append(s); turning.append(t)
        result["splits"][split] = {
            "sampled_scenes": len(indices), "type_counts": dict(types), "subtype_counts": dict(subtypes),
            "density": density, "speed_mps": speed, "turning_radians": turning,
        }
    durations = []
    storage_hz=CANONICAL_HZ
    for path in sorted((dataset_dir / "converted").glob("*.traj.parquet")):
        table = pq.read_table(path, columns=["track_id", "step"])
        ids = table["track_id"].combine_chunks().to_numpy(); steps = table["step"].combine_chunks().to_numpy()
        order = np.lexsort((steps, ids)); ordered_ids, ordered_steps = ids[order], steps[order]
        for indices in np.split(np.arange(len(order)), np.flatnonzero(np.diff(ordered_ids)) + 1):
            values = ordered_steps[indices]
            durations.append(float((values.max() - values.min()) / storage_hz))
        if len(durations) >= 100_000: break
    result["track_duration_s"] = durations[:100_000]
    return result


def plot_diversity(result: dict, output_dir: Path) -> list[Path]:
    """Render the type-mix/density and speed/curvature figures. Each figure is written as a
    `.pdf` (primary vector output) and a `.png` (raster preview); all paths are returned."""
    output_dir.mkdir(parents=True, exist_ok=True); dataset=result["dataset"]; outputs=[]
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    bottoms=np.zeros(3); labels=list(SPLITS)
    for tag in range(1,5):
        values=np.array([result["splits"][s]["type_counts"].get(tag,0) for s in SPLITS],float)
        totals=np.array([max(1,result["splits"][s]["sampled_scenes"]) for s in SPLITS]); fraction=values/totals
        axes[0].bar(labels,fraction,bottom=bottoms,label=f"Type {tag}"); bottoms+=fraction
    axes[0].set_ylabel("Fraction of scenes"); axes[0].set_title("Trajectory type mix"); axes[0].legend()
    for split in SPLITS: sns.kdeplot(result["splits"][split]["density"],ax=axes[1],label=split,cut=0)
    axes[1].set_title("Crowd density by temporal split"); axes[1].set_xlabel("Pedestrians at final observation"); axes[1].legend()
    stem=output_dir/f"{dataset}_types_density"; figure.tight_layout()
    figure.savefig(stem.with_suffix(".pdf")); outputs.append(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"),dpi=160); outputs.append(stem.with_suffix(".png"))
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for split in SPLITS:
        sns.kdeplot(result["splits"][split]["speed_mps"],ax=axes[0],label=split,cut=0)
        sns.kdeplot(result["splits"][split]["turning_radians"],ax=axes[1],label=split,cut=0)
    axes[0].set_xlabel("Mean speed (m/s)"); axes[1].set_xlabel("Total turning angle (rad)")
    axes[0].legend(); axes[1].legend(); stem=output_dir/f"{dataset}_speed_curvature"
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".pdf")); outputs.append(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"),dpi=160); outputs.append(stem.with_suffix(".png"))
    plt.close(figure)
    return outputs


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset_dir",type=Path); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--max-scenes",type=int,default=1000)
    parser.add_argument("--output-dir",type=Path,default=Path("paper/figures/diversity"),help="Directory for per-figure outputs: each figure is written as .pdf (primary) and .png (preview)")
    args=parser.parse_args(); result=analyze_dataset(args.dataset_dir,args.seed,args.max_scenes)
    for path in plot_diversity(result,args.output_dir): print(path)


if __name__=="__main__": main()
