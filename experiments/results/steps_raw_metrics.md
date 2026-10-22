# STEPS — raw (unfiltered) metrics

All 198 sessions, data seed 42. **No static filter anywhere.** Every indicator below is computed
over all 551,529 candidate trajlets / 3,528,070 indexed scenes.

## Scale

| Metric | Value |
| --- | ---: |
| Recordings | 198 |
| Recorded duration (h) | 31.87 |
| Annotated frames | 275,918 |
| Trajectories (raw) | 25,997 |
| Trajectories (usable, >=20 samples) | 16,844 |
| Tracks with gaps | 8,888 |
| Extra segments from gaps | 57,450 |
| Track length mean / median (s) | 58.98 / 16.40 |
| Track span median (s) | 12.0 |
| Candidate forecast windows | 3,528,070 |
| Candidate trajlets | 551,529 |

## Headline

| Metric | Value |
| --- | ---: |
| **Path efficiency** (mean) | **0.4138** |
| Path efficiency (median) | 0.3426 |
| **TTC < 1 s** | **12.28%** |
| **TTC < 3 s** | **24.07%** |
| TTC median (s) | 9.762 |
| TTC mean (s) | 45.281 |

TTC percentages are over the 531,907 trajlets with a finite TTC (96.4% of all). As a share of
all 551,529 trajlets they are 11.85% and 23.22%.

## Category distribution

Candidate windows, no acceptance filtering:

| Category | Count | Share |
| --- | ---: | ---: |
| static | 3,290,262 | 93.26% |
| interacting | 148,664 | 4.21% |
| non-linear | 65,872 | 1.87% |
| linear | 23,272 | 0.66% |

Split composition (retained after acceptance + eval stride):

| Split | Recordings | Candidates | Retained | Tracks | Static | Linear | Interacting | Non-linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 155 | 2,830,340 | 101,032 | 2,919 | 59.23% | 4.18% | 24.89% | 11.70% |
| val | 20 | 399,613 | 12,500 | 1,460 | 90.77% | 1.04% | 5.65% | 2.54% |
| test | 18 | 298,117 | 12,500 | 1,444 | 88.47% | 0.98% | 7.02% | 3.52% |
| all | 198 | 3,528,070 | 126,032 | 5,823 | 65.26% | 3.55% | 21.21% | 9.98% |

## Motion

| Metric | Mean | Median |
| --- | ---: | ---: |
| Speed (m/s) | 0.1106 | 0.0169 |
| Speed range (m/s) | 0.0934 | 0.0361 |
| Acceleration mean-abs (m/s^2) | 0.0412 | 0.0224 |
| Acceleration max-abs (m/s^2) | 0.1054 | 0.0580 |
| Path efficiency | 0.4138 | 0.3426 |

Angular deviation std: 97.55 deg @1.6 s, 96.96 @2.4 s, 96.65 @4.8 s.

Other speed denominators, for reference: track-level mean 0.736 m/s; observation-level
`mean_speed_mps` 0.175 m/s; p99.9 observation speed 1.883 m/s.

## Density and interaction

| Metric | Mean | Median |
| --- | ---: | ---: |
| Local density (ped/m^2) | 1.5989 | 0.4308 |
| Local density, duplicates removed | 1.2652 | 0.4283 |
| DCA min (m) | 0.1431 | 0.0540 |
| Collision energy | 3.0820 | 0.5637 |
| TTC min (s) | 45.281 | 9.762 |
| Global density (ped/m^2) | 0.0469 | 0.0474 |

Scene-level (all indexed scenes):

| Metric | Value |
| --- | ---: |
| Area (m^2, union) | 37.50 |
| Occupancy mean / p95 / max | 15.20 / 28 / 51 |
| Density mean / p95 / max (ped/m^2) | 0.4053 / 0.7467 / 1.3600 |
| Mean neighbors within 5 m | 7.148 |
| Mean neighbors within 2 m | 2.032 |
| CV FDE mean (m) | 0.4111 |
| Heading entropy | 0.6994 |
| RMS jerk | 1.3971 |
| min-TCA p05 (r=0.33 m, s) | 1.1820 |

## Duplicate-annotation filter

0.05 m separation, min 3 frames: 61 pairs, 61 tracks excluded, 28 / 197 recordings.
Effect on local density is large once static agents are included: 1.5989 -> 1.2652 (-20.9%),
against -0.3% on the static-filtered set.

## Provenance

Unfiltered indicators produced by re-running `compute_opentraj_report` with the static filter at
`src/opentraj_report.py:119` lifted (single-line patch; `INDICATOR_IMPLEMENTATION_HASH` verified
identical to the original module). Output written to
`datasets/steps/analysis/opentraj_reports_unfiltered/`; the published filtered
artifacts in `opentraj_reports/` were not modified. Scene-level rows come from
`paper/stats/steps_paper_stats.json`, which was never static-filtered.
