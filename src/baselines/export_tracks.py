"""Export unified metric tracks to the Social-GAN four-column interchange format."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data import _recordings

ROOT=Path(__file__).resolve().parents[2]


def export_recording(path:Path,output:Path,canonical_stride:int=1)->None:
    table=pq.read_table(path,columns=["track_id","step","x","y"])
    track=table["track_id"].combine_chunks().to_numpy()
    step=table["step"].combine_chunks().to_numpy()
    x=table["x"].combine_chunks().to_numpy();y=table["y"].combine_chunks().to_numpy()
    # Upstream ETH/UCY readers divide frame numbers by 10. Emit that convention
    # while retaining our exact 2.5 Hz sample index.
    frame=(step//canonical_stride)*10
    values=np.c_[frame,track,x,y]
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(output,values,delimiter="\t",fmt=("%d","%d","%.6f","%.6f"))


def export_dataset(dataset_dir:Path,output_dir:Path)->list[Path]:
    outputs=[]
    for path in _recordings(dataset_dir):
        output=output_dir/f"{path.name.removesuffix('.traj.parquet')}.txt"
        export_recording(path,output);outputs.append(output)
    return outputs


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset_dir",type=Path);parser.add_argument("output_dir",type=Path)
    args=parser.parse_args()
    for path in export_dataset(args.dataset_dir,args.output_dir):print(path)


if __name__=="__main__":main()
