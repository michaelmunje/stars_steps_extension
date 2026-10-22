# STEPS — full metric set (all 198 sessions)

Generated 2026-09-16 from the complete rebuild (convert → index → analysis_pipeline →
opentraj_report → stats), all stages `rc=0`. Data seed 42.

Sources:
- `paper/stats/steps_paper_stats.json`
- `datasets/steps/analysis/opentraj_reports/summary.json`
- `datasets/steps/indices/scenes-seed42.parquet`

## Scale

| Metric | STEPS |
| --- | ---: |
| Recordings | 198 |
| Recorded duration (h) | 31.87 |
| Annotated frames | 275,918 |
| Trajectories (raw) | 25,997 |
| Trajectories (usable, >=20 samples) | 16,844 |
| Candidate forecast windows | 3,528,070 |
| Retained trajlets (OpenTraj) | 48,511 |
| Tracks with gaps | 8,888 |
| Extra segments from gaps | 57,450 |
| Track length mean / median (s) | 58.98 / 16.40 |
| Track span median (s) | 12.0 |

## Space and crowding

| Metric | STEPS |
| --- | ---: |
| Area (m^2, union) | 37.50 |
| Occupancy mean / p95 / max | 15.20 / 28 / 51 |
| Density mean (ped/m^2) | 0.4053 |
| Density p95 / max | 0.7467 / 1.3600 |
| Mean neighbors within 5 m | 7.148 |
| Mean neighbors within 2 m | 2.032 |
| Global density mean (OpenTraj) | 0.0469 |
| Local density mean (filtered) | 0.6244 |
| Local density, duplicates removed | 0.6227 |

## Speed — three denominators

The three figures below are NOT alternative estimates of the same quantity; they differ in
what they average over. Pick deliberately when building a table.

| Definition | STEPS | Population |
| --- | ---: | --- |
| Filtered trajlet-level (OpenTraj, static removed) | 0.965 m/s | 48,511 trajlets |
| Unfiltered track-level | 0.736 m/s | 24,801 tracks (>=3 points) |
| Unfiltered observation-level (`mean_speed_mps`) | 0.175 m/s | every scene observation |

The observation-level value is far lower because it weights by dwell time: an agent standing
still for ten minutes contributes ~1,500 near-zero observations but only one track. Use it for
claims about how static the dataset is; use the track-level figure for claims about how fast
pedestrians move. The 0.736 track-level number is the like-for-like unfiltered counterpart to
the 0.965 currently reported in the paper table.

Supporting distribution (unfiltered, track-level): median 0.870, p90 1.407 m/s;
38.9% of tracks below 0.35 m/s mean, 18.8% below 0.1 m/s.
p99.9 observation speed: 1.883 m/s.

## Motion and interaction

| Metric | STEPS |
| --- | ---: |
| Path efficiency (all scenes) | 0.2063 |
| Path efficiency (filtered trajlets) | 0.9615 |
| Heading entropy | 0.6994 |
| RMS jerk | 1.3971 |
| CV FDE mean (m) | 0.4111 |
| min-TCA p05 (r=0.33 m, s) | 1.1820 |
| TTC median (s) | 3.165 |
| TTC < 3 s | 47.2% |
| TTC < 1 s | 16.6% |
| DCA mean / median (m) | 0.570 / 0.142 |
| Collision energy mean / median | 1.995 / 0.217 |
| Acceleration mean-abs / max-abs | 0.117 / 0.273 |
| Angular deviation std, 1.6 / 2.4 / 4.8 s (deg) | 23.82 / 19.65 / 23.40 |

TTC, DCA and collision energy are computed on the retained (non-static) trajlets only.

## Duplicate-annotation filter

0.05 m separation, minimum 3 frames, union-find grouping, lowest track id survives.

| Quantity | STEPS |
| --- | ---: |
| Duplicate pairs | 61 |
| Tracks excluded | 61 |
| Recordings affected | 28 / 197 |

Effect on local density: 0.6244 -> 0.6227 (-0.3%). Negligible at this scale.

## TrajNet category distribution

Candidate windows, before any acceptance filtering:

| Category | Count | Share |
| --- | ---: | ---: |
| static | 3,290,262 | 93.26% |
| interacting | 148,664 | 4.21% |
| non-linear | 65,872 | 1.87% |
| linear | 23,272 | 0.66% |

## Split composition (seed 42)

| Split | Recordings | Candidates | Retained | Tracks | Static | Linear | Interacting | Non-linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 155 | 2,830,340 | 101,032 | 2,919 | 59.23% | 4.18% | 24.89% | 11.70% |
| val | 20 | 399,613 | 12,500 | 1,460 | 90.77% | 1.04% | 5.65% | 2.54% |
| test | 18 | 298,117 | 12,500 | 1,444 | 88.47% | 0.98% | 7.02% | 3.52% |
| all | 198 | 3,528,070 | 126,032 | 5,823 | 65.26% | 3.55% | 21.21% | 9.98% |

Train retains 101,032 windows, essentially on the `n_target: 100000` cap, so it stays
comparable to PTDPES (101,413).

### Train and eval are deliberately different distributions

Category acceptance `(0.1, 1.0, 1.0, 1.0)` and the track-level Bernoulli subsample apply to
**train only** (train: 16.0% accepted, 10.0% for static; val/test: 100% accepted). Val and test
are thinned purely by `eval_stride`, a uniform temporal stride keeping 3.1% and 4.2%
respectively, which preserves the natural distribution.

Consequence: models train on a de-staticized mix (59% static) but are scored against a ~90%
static evaluation set, so minADE/minFDE on STEPS is dominated by near-stationary agents.
This is by design, not drift, but it should be stated wherever those numbers are reported.

## Which metrics are static-filtered

Two independent pipelines produce the numbers above, and they use different populations.

`opentraj_report.py` applies the OpenTraj static filter once, at line 119
(`index = index[index.retained_after_static_filter]`), which gates **every** indicator it
emits. The criterion is start-to-end displacement > 1.0 m over the trajlet
(`paper_dataset_metrics.py:370`).

| | STEPS | Share |
| --- | ---: | ---: |
| Candidate trajlets | 551,529 | 100% |
| Dropped as static | 503,018 | 91.2% |
| Retained | 48,511 | 8.8% |

So the filtered indicators describe 8.8% of the data.

**Filtered (computed on 48,511 retained trajlets):**
speed_mean, speed_range, acceleration mean/max-abs, path_efficiency (0.9615),
local_density_max, local_density_max_deduplicated, ttc_min, dca_min, collision_energy,
angular_deviation.

**Unfiltered (computed on all 3,528,070 indexed scenes, `scene_sample_fraction` = 1.0):**
everything from `stats.py` — cv_fde_mean, path_efficiency (0.2063), heading_entropy,
rms_jerk, mean_speed_mps, min_tca_p05, occupancy, density, neighbor counts, type_counts.

Note the same name appears in both columns with very different values: path efficiency is
0.9615 filtered and 0.2063 unfiltered. Check which pipeline a number came from before putting
it in a table.

CV FDE (0.4111 m) needs no unfiltered counterpart — it is already computed over every scene.
TTC, DCA and collision energy have no unfiltered counterpart yet; producing one requires
re-running `opentraj_report` with the line-119 filter lifted, roughly 11.4x the trajlet volume
(~1 h of CPU for this dataset).

## Notes

- The calibration bag `atrium_20260824_120001` is blacklisted via `excluded_recordings` in
  `source.json` and is absent from all figures above. 197 recordings appear in
  recording-level statistics; 198 converted files exist.
- `scene_filter_survival` = 0.9988, `scene_sample_fraction` = 1.0 (no subsampling).
- Duration cross-check: the pipeline reports 31.87 h; an independent sum of per-recording
  first-to-last step spans gives 32.06 h, a 0.6% difference attributable to gap accounting.
