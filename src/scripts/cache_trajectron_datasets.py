from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
SOURCE_ROOT=PROJECT_ROOT/"src"
if str(SOURCE_ROOT) not in sys.path:sys.path.insert(0,str(SOURCE_ROOT))

from tqdm.auto import tqdm

from baselines.trajectron_cache import prepare_shards,recording_names
from baselines.trajectron_data import ROOT,reference_hyperparams
from forecasting_config import resolve_forecasting_config


def main():
    parser=argparse.ArgumentParser(description="Build resumable bounded temporal Trajectron++ cache shards")
    parser.add_argument("--datasets",default="all",help="Comma-separated dataset names, or all")
    parser.add_argument("--data-seed",type=int,default=42)
    parser.add_argument("--roles",default="train,val")
    args=parser.parse_args();hp=reference_hyperparams();root=ROOT/"datasets"
    hp["steps_forecast_preprocessing_hash"] = resolve_forecasting_config(
        preset="trajectron_pp"
    ).forecast_preprocessing_hash
    if args.datasets=="all":
        names=[path.name for path in sorted(root.iterdir()) if path.is_dir() and (path/"converted").exists() and (path/"indices"/f"scenes-seed{args.data_seed}.parquet").exists()]
    else:names=[name.strip() for name in args.datasets.split(",") if name.strip()]
    for name in names:
        dataset_dir=root/name;recordings=recording_names(dataset_dir)
        for role in [item.strip() for item in args.roles.split(",") if item.strip()]:
            tqdm.write(f"Caching {name} {role}: {len(recordings)} recordings")
            prepare_shards(dataset_dir,role,args.data_seed,hp,recordings=recordings)
    print("Trajectron++ cache preparation complete",flush=True)


if __name__=="__main__":main()
