# STEPS adapter

Place each cleaned tracker export below `datasets/steps/raw/`; nested recording and
`*_bag` directories are supported as long as each contains `frames.cleaned.jsonl` (the
upstream cleanup pass's output; see `cleaned_source` in [`source.json`](source.json)).

The source frames are approximately 20 Hz. The converter streams the JSONL and stores
every eighth source frame directly on the canonical 2.5 Hz timeline.

Only confirmed, directly observed `global_kalman_posterior` states are written to the
trajectory Parquet. Tracker predictions remain in the quality sidecar and cannot become
training targets. Frames finalized by `end_of_stream_flush` are also omitted because
they do not have the configured fixed-lag evidence from the final second.

```bash
python -m datasets_preprocessing.steps.convert datasets/steps
python -m preprocess datasets/steps --name steps --data-seed 42
```

For a source stored somewhere else, use a repeatable explicit argument:

```bash
python -m datasets_preprocessing.steps.convert datasets/steps --source /path/to/recording_bag/frames.cleaned.jsonl
```

Outputs per recording:

- `converted/<recording>.traj.parquet`: unified training trajectories
- `quality/<recording>.quality.parquet`: all sampled tracker states and inclusion flags
- `quality/<recording>.frames.parquet`: timestamps, camera availability, synchronization,
  and fixed-lag finalization details
- `quality/<recording>.summary.json`: conversion counts and measured source rate
