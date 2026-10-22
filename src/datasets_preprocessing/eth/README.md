# ETH / BIWI Walking Pedestrians

Source: https://vision.ee.ethz.ch/datsets/index.en.html

The light EWAP archive contains the `seq_eth` and `seq_hotel` recordings. Its
`obsmat.txt` rows provide ground-plane position and velocity in metres at 2.5 Hz.
The source frame numbering is video-specific, so the converter infers and
validates each recording's native frame interval and maps every annotation
interval to one step on the canonical 2.5 Hz timeline. Zero-velocity observations
have a null heading; all others use `atan2(v_y, v_x)`.

From the repository root, run:

```bash
python -m datasets_preprocessing.eth.convert datasets/eth
```

Each recording is an indivisible `scene_sequence`. Grouped manifests never split
overlapping windows from one recording across train/validation/test. Because ETH has
only two sequences, ordinary three-way grouped splitting necessarily has an empty
partition; use the explicit ETH/UCY five-location protocol for model evaluation.
Visualize loader samples with:

```bash
python -m visualize datasets/eth --split test --selection crowded --count 10 --output-dir paper/figures/trajectory_debug/eth-test
```

Each sample writes a GIF and a `_final.png` image. Solid lines are loader-provided
observed or actual trajectories; dashed lines are the constant-velocity baseline.
