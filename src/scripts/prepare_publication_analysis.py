from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT_DIR / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from analysis_pipeline import build_publication_analysis_dataset
from opentraj_report import compute_opentraj_report, plot_opentraj_compatible_figures
from paper_dataset_metrics import (
    ROOT,
    _artifact_paths,
    load_paper_datasets,
    select_canonical_files,
)


def main() -> None:
    parser=argparse.ArgumentParser(description="Build exact OpenTraj artifacts for explicit paper dataset membership")
    parser.add_argument("--datasets",nargs="+",help="Canonical dataset keys; default is every publication source")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--allow-missing",action="store_true")
    parser.add_argument(
        "--plot-dir-root",
        type=Path,
        default=ROOT / "paper" / "figures" / "opentraj",
        help="Root for publication-analysis figures (default: paper/figures/opentraj)",
    )
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    requested=set(args.datasets or [])
    sources=[]
    for row in load_paper_datasets():
        for source in row.sources:
            if (not requested or source.canonical_dataset in requested) and source not in sources:
                sources.append(source)
    errors=[]
    for source in sources:
        try:
            _,sequences=select_canonical_files(source)
            dataset_dir=ROOT/"datasets"/source.canonical_dataset
            index,_,summary=build_publication_analysis_dataset(
                dataset_dir,scene_sequences=set(sequences),artifact_variant=source.analysis_variant,
                workers=max(1,args.workers),
            )
            _,report_dir_index=_artifact_paths(source)
            report_dir=report_dir_index.parent
            metrics,_=compute_opentraj_report(
                dataset_dir,index_path=index,sequences=set(sequences),output_dir=report_dir,
                report_name=source.canonical_dataset,
            )
            if not args.no_plots:
                plot_opentraj_compatible_figures(
                    [metrics], args.plot_dir_root / source.canonical_dataset.lower()
                )
            print(f"{source.canonical_dataset}: candidates={summary['valid_opentraj_trajlets']} retained={summary['retained_analysis_trajlets']} PASS")
        except FileNotFoundError as error:
            errors.append(f"{source.canonical_dataset}: {type(error).__name__}: {error}")
            print(f"{source.canonical_dataset}: MISSING ({type(error).__name__})")
    if errors and not args.allow_missing:
        raise RuntimeError("Publication analysis incomplete:\n  - "+"\n  - ".join(errors))


if __name__=="__main__":main()
