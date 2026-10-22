# OpenTraj Table 1 reproduction log

Statistics are audited sequentially against OpenTraj commit `97ec1d5e1b579f26febeaa9083705952deac338a`.

## ETH Univ

Included scene sequences: `seq_eth`.

| Metric | Ours | OpenTraj | Difference | Status |
|---|---:|---:|---:|---|
| Pedestrians / trajectories | 360 | 360 | +0 | PASS |
| Recording duration (min) | 12.8933 | 13.0 | -0.1067 | PASS (reported precision) |
| Total trajectory duration (h) | 0.9498 | 1.0 | -0.0502 | PASS (source parity; table approximate) |
| 4.8 s trajlets | 823 | 823 | +0 | PASS |
| Non-static trajlets (%) | 93.5601 | 93.0 | +0.5601 | PASS (reported precision) |

Non-static count: **770 / 823** (93.5601%).
Tracks containing canonical timestep gaps: **0**.

Discrepancy notes:

- Pinned OpenTraj source computes 3419.2 s (56m59.2s, 0.94978 h), exactly matching ours. The paper's 1 h cell is therefore a coarse display value, not a strict one-decimal rounding of its implementation output.
- OpenTraj's benchmark loader sets use_kalman=True for ETH-Univ. This gives 770 non-static trajlets; unsmoothed source positions give 768. Our analysis branch correctly uses the OpenTraj Kalman-smoothed positions.
- Our canonical recording span is 773.6 s versus 773.4 s in the pinned loader. ETH contains a video-frame phase irregularity, and an integer 2.5 Hz grid must quantize this half-timestep endpoint. The 0.2 s difference does not change the displayed 13 min value, trajectory count, duration sum, or trajlet counts.

## ETH Hotel

Included scene sequences: `seq_hotel`.

| Metric | Ours | OpenTraj | Difference | Status |
|---|---:|---:|---:|---|
| Pedestrians / trajectories | 390 | 390 | +0 | PASS table / FLAG source count |
| Recording duration (min) | 12.0400 | 12.0 | +0.0400 | PASS (reported precision) |
| Total trajectory duration (h) | 0.6838 | 0.7 | -0.0162 | PASS (reported precision) |
| 4.8 s trajlets | 484 | 484 | +0 | PASS |
| Non-static trajlets (%) | 66.9421 | 66.0 | +0.9421 | PASS (reported precision) |

Non-static count: **324 / 484** (66.9421%).
Tracks containing canonical timestep gaps: **0**.

Discrepancy notes:

- The original obsmat.txt and our lossless canonical file contain 390 unique pedestrian IDs, matching Table 1. Track 314 has only one sample (frame 13161), so OpenTraj postprocess removes it and general_stats reports 389. The paper therefore counts raw IDs while its released statistics function counts postprocessed trajectories. We preserve the source track and flag this one-pedestrian table-vs-code inconsistency.
- OpenTraj's benchmark does not enable Kalman smoothing for ETH-Hotel. Both the original positions and our OpenTraj-smoothed analysis view nevertheless produce the same 324/484 non-static result, so this does not affect Table 1.

## UCY Zara

Included scene sequences: `crowds_zara01, crowds_zara02, crowds_zara03`.

| Metric | Ours | OpenTraj | Difference | Status |
|---|---:|---:|---:|---|
| Pedestrians / trajectories | 489 | 489 | +0 | PASS |
| Recording duration (min) | 18.0067 | 18.0 | +0.0067 | PASS (reported precision) |
| Total trajectory duration (h) | 2.0941 | 2.1 | -0.0059 | PASS (reported precision) |
| 4.8 s trajlets | 2130 | 2130 | +0 | PASS |
| Non-static trajlets (%) | 75.3991 | 75.0 | +0.3991 | PASS (reported precision) |

Non-static count: **1606 / 2130** (75.3991%).
Tracks containing canonical timestep gaps: **0**.

Discrepancy notes:

- OpenTraj's UCY-Zara row combines Zara01, Zara02, and Zara03. The standard five-fold forecasting corpus uses Zara01 and Zara02 only, so this audit reads the archived lossless seven-sequence unified conversion. Zara03 is not reintroduced into forecasting training or evaluation.
- A direct run of the pinned OpenTraj VSP loader, homographies, Kalman smoother, and trajlet splitter matches every unrounded value exactly.

## UCY Students

Included scene sequences: `students001, students003, uni_examples`.

| Metric | Ours | OpenTraj | Difference | Status |
|---|---:|---:|---:|---|
| Pedestrians / trajectories | 967 | 967 | +0 | PASS |
| Recording duration (min) | 11.4733 | 11.5 | -0.0267 | PASS (reported precision) |
| Total trajectory duration (h) | 4.4950 | 4.5 | -0.0050 | PASS (reported precision) |
| 4.8 s trajlets | 4702 | 4702 | +0 | PASS |
| Non-static trajlets (%) | 79.8171 | 96.0 | -16.1829 | FLAG paper/source inconsistency |

Non-static count: **3753 / 4702** (79.8171%).
Tracks containing canonical timestep gaps: **0**.

Discrepancy notes:

- OpenTraj's UCY-Students row combines students001, students003, and uni_examples. The five-fold forecasting corpus uses students001 and students003 only, so this audit reads the archived lossless seven-sequence conversion without adding uni_examples to model training or evaluation.
- A direct run of the pinned OpenTraj loader, students03 homography, Kalman smoother, and exact splitter produces 3753/4702 = 79.8171%; its own general_stats.py prints 79%. This exactly matches our unified pipeline.
- The released trajlet code has used a strict >1.0 m endpoint-displacement criterion throughout its repository history. On the current source data, approximately 96% would require a threshold near 0.08 m, which is not the documented or implemented protocol and would invalidate the other rows. The paper's 96% cell is therefore irreproducible from its released code and data and is retained as an explicitly flagged paper inconsistency.

