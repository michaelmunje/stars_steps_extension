# UCY Crowds-by-Example adapter

`convert.py` extracts the supplied RAR archives and converts all seven canonical UCY
recordings: `arxiepiskopi1`, `crowds_zara01`, `crowds_zara02`, `crowds_zara03`,
`students001`, `students003`, and `uni_examples`. All seven are canonical and are
included in publication and statistics characterization. Only four of them —
`crowds_zara01`, `crowds_zara02`, `students001`, and `students003` — form the
standard Social-GAN five-location forecasting benchmark; `arxiepiskopi1`,
`crowds_zara03`, and `uni_examples` are excluded solely from that forecasting
benchmark's training, validation, and evaluation.

Conversion follows the supplied toolkit's linear interpolation contract, sampling
every ten 25 Hz video frames, and applies the metric homographies from the pinned
OpenTraj release. Linear interpolation also avoids cubic overshoot across long gaps
between control points.

The archives supplied here do not contain the optional homography matrices, so the
pinned lightweight OpenTraj matrices are committed in `homographies.json`. This is
recorded in both `source.json` and every Parquet file's metadata.

```bash
python -m datasets_preprocessing.ucy.convert datasets/ucy
python -m preprocess datasets/ucy --name ucy --data-seed 42
python -m visualize datasets/ucy --split test --selection crowded --count 5
```
