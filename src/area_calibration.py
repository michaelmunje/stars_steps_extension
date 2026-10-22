"""Calibrate occupancy-grid area against ATC's documented monitored area."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data import _read_json, _recordings
from dataset_paths import PAPER_STATS_ROOT, metadata_path
from stats import _effective_grid_area_from_counts

ROOT = Path(__file__).resolve().parents[1]

# Published verbatim as "tolerance_fraction" in the calibration artifact below; this is
# the single definition shared by calibration_status's default and the emitted field, so
# the artifact cannot describe a threshold different from the one actually applied.
CALIBRATION_TOLERANCE_FRACTION = 0.15


def calibration_status(measured_m2: float, documented_m2: float, tolerance: float = CALIBRATION_TOLERANCE_FRACTION) -> str:
    ratio = measured_m2 / documented_m2
    return "calibrated" if 1.0 - tolerance <= ratio <= 1.0 + tolerance else "outside_tolerance"


def calibrate_atc(dataset_dir: Path, cell_sizes=(0.25, 0.5, 1.0), mass_fractions=(0.75,0.90,0.95,0.975,0.99)) -> dict:
    # Read the documented area (and its provenance) from source.json, the same
    # project-owned record stats.analyze() reads via metadata_path -- so this
    # artifact and analyze()'s documented_area_m2 cannot drift apart the way a
    # hardcoded literal would. A dataset_dir with no source.json / area_m2 has
    # no documented area to calibrate against, so this raises rather than
    # falling back to a stale constant.
    source = _read_json(metadata_path(dataset_dir, "source.json"))
    if not source.get("area_m2"):
        raise KeyError(f"{dataset_dir}: source.json has no area_m2; calibrate_atc requires a documented area to calibrate against")
    documented_area_m2 = float(source["area_m2"])
    documented_area_source = source.get("area_source", "documented")
    dataset_name = dataset_dir.name
    # ATC is one fixed monitored scene across many dates. Accumulate the union
    # over recordings; averaging per-date footprints would measure visitation,
    # not the documented camera coverage.
    finest=float(min(cell_sizes))
    finest_counts: Counter[int] = Counter()
    per_recording_counts: list[Counter[int]]=[]
    files = _recordings(dataset_dir)
    if not files:
        raise FileNotFoundError(f"{dataset_dir}: no canonical recordings under converted/; calibrate_atc has nothing to measure")
    for number, path in enumerate(files, 1):
        recording_counts: Counter[int]=Counter()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["x", "y"], batch_size=1_000_000):
            x = batch.column(0).to_numpy()
            y = batch.column(1).to_numpy()
            ix=np.floor(x/finest).astype(np.int32);iy=np.floor(y/finest).astype(np.int32)
            packed=(ix.astype(np.uint32).astype(np.uint64)<<np.uint64(32))|iy.astype(np.uint32).astype(np.uint64)
            unique,multiplicity=np.unique(packed,return_counts=True)
            additions={int(cell):int(n) for cell,n in zip(unique,multiplicity)}
            finest_counts.update(additions);recording_counts.update(additions)
        per_recording_counts.append(recording_counts)
        print(f"calibrated {number}/{len(files)}: {path.name}", flush=True)
    def coarsen(finest_values:Counter[int],size:float)->Counter:
        """Aggregate finest-grid counts (keyed by the packed uint64 cell) onto a
        ``size``-metre grid, keyed by ``(ix,iy)``.

        The returned counter is consumed **only** through ``.values()`` (by
        ``_effective_grid_area_from_counts`` below), so the ``factor==1`` case
        returns the counts still keyed by the packed cell instead of paying for a
        pure re-keying pass. A future caller that needs the ``(ix,iy)`` keys must
        re-key that branch rather than rely on this shortcut.
        """
        factor=int(round(float(size)/finest))
        assert factor>=1,f"coarse cell {size} is finer than the base grid {finest}"
        if not finest_values or factor==1: return Counter(finest_values)
        packed=np.fromiter(finest_values.keys(),dtype=np.uint64,count=len(finest_values))
        occurrences=np.fromiter(finest_values.values(),dtype=np.int64,count=len(finest_values))
        # NumPy integer // floors toward -inf exactly as Python's does, so negative
        # cell indices coarsen identically to the scalar loop this replaces.
        ix=(packed>>np.uint64(32)).astype(np.uint32).view(np.int32)//factor
        iy=(packed&np.uint64(0xffffffff)).astype(np.uint32).view(np.int32)//factor
        coarse=(ix.astype(np.uint32).astype(np.uint64)<<np.uint64(32))|iy.astype(np.uint32).astype(np.uint64)
        unique,inverse=np.unique(coarse,return_inverse=True)
        # bincount(weights=...) accumulates in float64; counts are far below 2**53,
        # so the cast back to int64 is exact and the assertion pins that.
        totals=np.bincount(inverse.ravel(),weights=occurrences,minlength=len(unique)).astype(np.int64)
        assert int(totals.sum())==int(occurrences.sum()),"coarsening changed the total observation count"
        keys_x=(unique>>np.uint64(32)).astype(np.uint32).view(np.int32)
        keys_y=(unique&np.uint64(0xffffffff)).astype(np.uint32).view(np.int32)
        return Counter(dict(zip(zip(keys_x.tolist(),keys_y.tolist()),totals.tolist())))
    sweep = []
    counts={float(size):coarsen(finest_counts,float(size)) for size in cell_sizes}
    recording_counts_by_size={float(size):[coarsen(value,float(size)) for value in per_recording_counts] for size in cell_sizes}
    for size, counter in counts.items():
        for mass_fraction in mass_fractions:
            # min_cells=0: the calibration sweep must report the raw mass area at every
            # swept cell size. stats.AREA_MIN_CELLS (the production default) rejects
            # recordings occupying fewer cells by returning nan, which would erase most
            # coarse-grid rows of this sweep rather than calibrate them.
            union_area=_effective_grid_area_from_counts(counter,size,mass_fraction,min_cells=0)
            recording_areas=np.asarray([_effective_grid_area_from_counts(value,size,mass_fraction,min_cells=0) for value in recording_counts_by_size[size]])
            median=float(np.median(recording_areas))
            sweep.append({"cell_size_m":size,"observation_mass_fraction":mass_fraction,"mean_recording_area_m2":float(recording_areas.mean()),"median_recording_area_m2":median,"union_area_m2":union_area,"ratio_to_documented":median/documented_area_m2})
    # cell_size_m=0.25, observation_mass_fraction=0.75 matches stats.py's
    # AREA_CELL_SIZE_M / AREA_MASS_FRACTION, the parameters actually used by
    # analyze() to compute every dataset's reported area_m2/density_*. Keep
    # the full sweep in the artifact so the choice is visible and other
    # mass fractions can be compared against it.
    default = next(row for row in sweep if row["cell_size_m"] == 0.25 and row["observation_mass_fraction"] == 0.75)
    if not np.isfinite(default["ratio_to_documented"]):
        # A non-empty file list can still yield no measurable observations (e.g. every
        # recording empty after the mass-fraction cut), which produces the same
        # undefined-verdict problem the empty-``files`` guard above catches: NaN inputs
        # would otherwise be reported as a definite status/bias_direction/closest-setting.
        raise ValueError(f"{dataset_dir}: default sweep setting (cell_size_m=0.25, observation_mass_fraction=0.75) has no measurable recording area; calibrate_atc has nothing to report a verdict against")
    status = calibration_status(default["median_recording_area_m2"], documented_area_m2)
    closest = min(sweep, key=lambda row: abs(row["ratio_to_documented"] - 1.0))
    return {
        "dataset": dataset_name,
        "documented_area_m2": documented_area_m2,
        "documented_area_source": documented_area_source,
        "default": default,
        "status": status,
        "tolerance_fraction": CALIBRATION_TOLERANCE_FRACTION,
        "bias_direction": "underestimate" if default["ratio_to_documented"] < 1 else "overestimate",
        "closest_swept_setting": closest,
        "selection_policy":"Production parameters are the scale-free swept setting that passes the documented ATC area check; the full sweep is retained.",
        "sweep": sweep,
        "interpretation": (
            "The scale-free grid reports the smallest cells containing a fixed fraction of observations. "
            f"It measures effective pedestrian-visited area, whereas {documented_area_m2:g} m2 is the full documented monitored area; "
            "a mismatch is reported rather than silently tuning the main-table method."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=ROOT / "datasets/atc")
    args = parser.parse_args()
    result = calibrate_atc(args.dataset_dir)
    output = PAPER_STATS_ROOT / "atc_area_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
