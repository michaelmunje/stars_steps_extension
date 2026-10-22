# ATC

Source: https://dil.atr.jp/ISL/crest2010_HRI/ATC_dataset/

Convert millimetres to metres, centre the fixed ATC map frame using the center
measured across the first official archive, resample timestamps directly to 2.5 Hz, and
retain angle of motion as heading. Reusing the fixed transform keeps every date
in the same coordinate system without scanning the full collection twice.
Ignore height and body-facing angle.

From the repository root, run:

```bash
python -m datasets_preprocessing.atc.convert datasets/atc --workers 4
```

If the CSV files are stored elsewhere, avoid copying them by passing their
directory directly:

```bash
python -m datasets_preprocessing.atc.convert datasets/atc --raw-dir ../..
```

This writes one `converted/*.traj.parquet` file per raw recording and stores the map-centering transform in the parquet metadata.

The ATC converter produces the parquet files consumed by the shared loader. Visualize validation samples with a constant-velocity linear baseline from the repository root:

```bash
python -m visualize datasets/atc --split val --count 10 --output-dir paper/figures/trajectory_debug/atc-val
```

Each sample writes a GIF and a `_final.png` image. Solid lines are loader-provided observed or actual trajectories; dashed lines are the baseline forecast.
