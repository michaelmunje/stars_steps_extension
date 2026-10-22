# Grand Central preprocessing

Place the original annotation text files under either
`datasets/grand_central/raw/Annotation/` or
`datasets/grand_central/raw/extracted/Annotation/`, then run:

```bash
python -m datasets_preprocessing.grand_central.convert
```

To regenerate both canonical and publication-analysis artifacts:

```bash
./scripts/regenerate_canonical_dataset.sh grand_central
python -m scripts.prepare_publication_analysis --datasets grand_central
```

The converter intentionally follows the pinned OpenTraj `loader_gcs.py`: large-gap splitting, track-local linear interpolation, homography, and 0.8 world scaling.
