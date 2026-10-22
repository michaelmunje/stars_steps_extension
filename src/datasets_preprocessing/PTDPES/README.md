# PTDPES adapter

The adapter reads the 391 published final modified trajectory CSVs. These are the
article's filtered, homography-transformed, smoothed, boundary-clipped trajectories,
downsampled to 2 Hz. It translates `real_world_sw_x/y` into a recording-local metric
frame, then linearly resamples each contiguous track segment onto the project's shared
2.5 Hz grid. Missing source intervals are never interpolated across.

The source audit found large jumps associated with the article's documented ID-merge
and georeferencing limitations. They are retained so the published postprocessed data
is preserved. Metadata records the number of tracks exceeding the 10 m/s audit
threshold, but the converter does not filter them.

```bash
python -m datasets_preprocessing.PTDPES.convert datasets/PTDPES
python -m preprocess datasets/PTDPES --name ptdpes --data-seed 42
```

To reproduce the retained jump diagnostic and its GIF:

```bash
python -m datasets_preprocessing.PTDPES.make_jump_example
python -m visualize datasets/PTDPES_jump_example --split test --output-dir paper/figures/trajectory_debug/ptdpes_jump
```
