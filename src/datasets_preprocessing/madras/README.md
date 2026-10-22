# MADRAS

Source: https://doi.org/10.5281/zenodo.13830435

The converter uses the published camera-local metric coordinates. TopView recordings
are downsampled from 30 Hz directly to the canonical 2.5 Hz grid and use `id_global`;
the three time-varying LargeView products are downsampled from 10 Hz and use `id`. Motion heading is
derived from consecutive metric positions. `LargeView_density.txt` is excluded because
it contains only a single frame.

```bash
python -m datasets_preprocessing.madras.convert datasets/madras
python -m preprocess datasets/madras --name madras --data-seed 42
python -m visualize datasets/madras --split test --selection crowded --count 5
```
