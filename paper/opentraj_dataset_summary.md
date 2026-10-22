# OpenTraj-compatible dataset summary

Pinned reference: `crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a`.

| Dataset | Recording duration (h) | Total trajectory duration (h) | Raw trajectories | 4.8 s trajlets | Mean peak local density (ped/m²) | Median peak local density (ped/m²) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| eth | 0.416 | 1.634 | 750 | 1307 | 0.627 | 0.472 | complete |
| ucy | 0.500 | 6.737 | 1480 | 6996 | 1.641 | 1.079 | complete |
| atc | — | — | — | — | — | — | unavailable: no canonical converted recordings |
| madras | 0.439 | 30.089 | 7010 | 30904 | 5.358 | 2.323 | complete |
| grand_central | 1.333 | 95.105 | 17862 | 103249 | 2.654 | 0.524 | complete |
| ptdpes | 184.889 | 2909.115 | 344056 | 2633942 | 1.813 | 0.436 | complete |

## Definitions

- **recording duration:** OpenTraj general_stats.dataset_duration: sum(max(timestamp)-min(timestamp)) over source recordings/scene_sequences.
- **total trajectory duration:** OpenTraj general_stats.total_trajectory_duration: sum(max(timestamp)-min(timestamp)) over (scene_sequence, agent_id) tracks; gaps remain inside the span.
- **raw trajectories:** OpenTraj general_stats.num_pedestrians: distinct (scene_sequence, agent_id) identities in the loaded canonical data, before trajlet/window filters (not raw detector observations).
- **trajlets 4 8s:** OpenTraj split_trajectories candidates: 12 returned samples, a 13th static endpoint, starts every 7 samples at 2.5 Hz; count is before the >1 m filter.
- **local density:** Per-trajlet PEAK local density over the 4.8 s window, not a mean over frames: OpenTraj takes the per-frame maximum (crowd_density.py:64), where per-frame density is the a=1 adaptive nearest-neighbour Gaussian estimator, not a fixed-radius count. Reported as both the mean and the median over retained trajlets; the mean of maxima is dominated by its upper tail, so the two differ substantially in crowded scenes.
- **deduplicated local density:** The same quantity with duplicate pedestrian annotations excluded: a track identity held within 0.05 m of another identity for at least 3 canonical frames is one person annotated twice, and is removed from every frame before the density kernel runs. The filter flags nothing at all in ETH or UCY, and the per-frame kernel itself is unchanged and still bit-exact against pinned upstream. The published table reports the de-duplicated mean and median; average_local_density_ped_m2 and median_local_density_ped_m2 retain the unfiltered values for comparison.

ETH/UCY reproduce the pinned source exactly. For datasets with temporal gaps, our trajlet extractor intentionally rejects candidates crossing a gap or scene_sequence boundary; upstream split_trajectories assumes already-continuous trajectories.
