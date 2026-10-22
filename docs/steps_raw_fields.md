# STEPS — raw field reference and anonymity review

Reviewed 2026-09-18 against the release at `datasets/steps/raw`, a symlink to
`/media/mike/Expansion/robodata/ICRA_GDC_cleaned` (749 GB, 198 recording directories).

Each recording holds one bag directory with three files:

| File | Size | Contents |
| --- | --- | --- |
| `frames.cleaned.jsonl` | ~600 MB | one JSON object per 20 Hz frame |
| `cleanup.report.json` | ~2.4 KB | upstream cleanup statistics and configuration |
| `cleanup.receipt.json` | ~0.8 KB | input/output file stats and run id |

Schema below was derived by unioning every key path across 4,000 frames of
`atrium_20260818_094501`, cross-checked against three recordings.

---

## Frame-level fields

| Field | Type | Notes |
| --- | --- | --- |
| `frame_index` | int | 0-based index within the recording |
| `timestamp_ns` | int | **monotonic clock, not wall-clock** (see anonymity note A4) |
| `timestamp_s` | float | same clock in seconds |
| `source_group_index` | int | index of the synchronised source group |
| `sync_skew_ns` | int | spread between camera timestamps in the group |
| `output_fixed_lag_seconds` | float | configured finalisation lag (1.0) |
| `present_cameras` | list[int] | camera indices 0–5 |
| `fresh_cameras` | list[int] | cameras with a new measurement this frame |
| `stale_cameras` | list[int] | cameras whose last measurement aged out |
| `missing_cameras` | list[int] | cameras absent from this group |
| `source_timestamp_ns_by_camera` | dict[str→int] | per-camera capture timestamps, keys `"0"`–`"5"` |
| `source_timestamp_offset_ns_by_camera` | dict[str→int] | per-camera offset from the group reference |
| `output_finalization.mode` | str | `fixed_lag` |
| `output_finalization.reason` | str | `age_cutoff`, or `end_of_stream_flush` (excluded downstream) |
| `output_finalization.actual_lag_seconds` | float | realised lag |
| `output_finalization.configured_delay_seconds` | float | configured lag |
| `output_finalization.finalized_at_timestamp_ns` | int | finalisation time |

## `tracked_people[]` — the global tracker output

This is the array the forecasting pipeline consumes (`source.json` maps `track_id`,
`position_ground_xy`, `yaw_ground_rad` from here).

| Field | Type | Notes |
| --- | --- | --- |
| `track_id` | int | per-recording track identity |
| `position_ground_xy` | list[float] | metres, scene-global frame |
| `velocity_ground_xy_mps` | list[float] | metres/second |
| `position_covariance` | list[list[float]] | 2×2 Kalman covariance |
| `yaw_ground_rad` | float | measured body yaw, **not** direction of travel |
| `yaw_ground_rad_original` | float | yaw before flip correction |
| `yaw_flipped` | bool | whether the 180° flip correction was applied |
| `yaw_concentration` | float | circular concentration of the yaw estimate |
| `confirmed` | bool | track passed confirmation |
| `observed_this_frame` | bool | measurement present this frame |
| `stale` | bool | no recent measurement |
| `state_source` | str | `global_kalman_posterior` / `global_kalman_prediction` |
| `age_steps`, `hit_count`, `consecutive_hit_count` | int | track lifetime counters |
| `consecutive_observed_seconds`, `missed_seconds` | float | continuity timers |
| `missed_steps` | int | consecutive missed updates |
| `spread_m` | float | cross-camera spread of the fused measurement |
| `last_observed_position_ground_xy` | list[float] | last measured position |
| `last_observed_timestamp_ns` | int | time of last measurement |
| `last_observed_camera_indices` | list[int] | cameras contributing that measurement |

## `fused_people[]` — multi-camera fused observations

| Field | Type | Notes |
| --- | --- | --- |
| `track_id`, `observation_index` | int | identity and ordinal |
| `position_ground_xy` | list[float] | fused position, metres |
| `position_raw_mean_ground_xy` | list[float] | unweighted mean of members |
| `position_smoothed_xy` | list[float] \| null | smoothed estimate |
| `roi_position_ground_xy` | list[float] | position clipped to the region of interest |
| `member_positions_ground_xy` | list[list[float]] | each contributing camera's position |
| `member_detection_ids` | list[str] | `"<timestamp_ns>:cam<N>:det<M>"` |
| `camera_indices` | list[int] | contributing cameras |
| `fusion_weights` | list[float] | per-camera weights |
| `measurement_covariance` | list[list[float]] | 2×2 |
| `max_spread_m`, `rms_spread_m` | float | agreement between cameras |
| `yaw_ground_rad`, `yaw_ground_rad_original`, `yaw_flipped`, `yaw_concentration` | | as above |
| `global_birth_retroactively_labeled` | bool | track birth relabelled after the fact |
| `global_birth_was_pending_causally` | bool | birth was pending at the causal time |
| `global_reassociation_update_only` | bool | update came from re-association |
| `global_track_bootstrap_only` | bool | bootstrap-only observation |
| `track_prediction_guided` | bool | association used the predicted state |

## `projected_detections[]` — per-camera detections

| Field | Type | Notes |
| --- | --- | --- |
| `detection_id` | str | `"<timestamp_ns>:cam<N>:det<M>"` |
| `duplicate_of_detection_id` | str \| null | set when suppressed as a duplicate |
| `suppressed_duplicate` | bool | |
| `camera_index` | int | 0–5 |
| `timestamp_ns` | int | camera capture time |
| `confidence` | float | detector confidence |
| `bbox_xyxy` | list[float] | **pixel bounding box** |
| `foot_pixel_uv` | list[float] | **pixel foot location** |
| `image_width_px`, `image_height_px` | int | 1440 × 1080 |
| `foot_source` | str | `toe_heel_midpoint` / `ankles` |
| `foot_quality` | float | |
| `position_ground_xy`, `roi_position_ground_xy` | list[float] | projected ground position |
| `camera_position_smoothed_xy` | list[float] \| null | per-camera smoothed position |
| `camera_velocity_ground_xy_mps` | list[float] \| null | per-camera velocity |
| `camera_position_covariance`, `camera_measurement_covariance` | list[list[float]] \| null | 2×2 |
| `yaw_ground_rad` | float | per-camera yaw |
| `camera_track_id` | int \| null | per-camera track identity |
| `camera_track_confirmed` | bool \| null | |
| `camera_track_observed` | bool | |
| `camera_track_consecutive_hits` | int | |
| `camera_track_consistent_seconds` | float | |
| `track_id` | int \| null | global track this was associated to |
| `association_eligible` | bool | |
| `association_rejection_reason` | str \| null | `camera_track_unconfirmed`, `same_camera_duplicate` |
| `cluster_index` | int \| null | |
| `global_birth_retroactively_labeled`, `global_birth_was_pending_causally`, `global_reassociation_update_only` | bool | as above |

### `projected_detections[].sam3d` — body pose estimates

| Field | Type | Notes |
| --- | --- | --- |
| `bbox_xyxy` | list[float] | pixel box |
| `confidence` | float | |
| `keypoints_2d` | list[list[float]] | **2D body keypoints, image space** |
| `keypoints_3d` | list[list[float]] | **3D body keypoints** |
| `global_rot_zyx` | list[float] | body global rotation |
| `model_fields.keypoint_schema` | str | `mhr70` (70 keypoints) |
| `model_fields.focal_length` | float | camera focal length |
| `model_fields.pred_cam_t` | list[float] | predicted camera translation |
| `model_fields.foot_contact.contact_source` | str | `toe_heel_midpoint` |
| `model_fields.foot_contact.contact_quality` | float | |
| `model_fields.foot_contact.fallback_source` | str | `ankles` |
| `model_fields.foot_contact.fallback_quality` | float | |
| `model_fields.foot_contact.left_indices` / `right_indices` / `pooled_indices` / `fallback_indices` | list[int] | keypoint indices used |

## `cleanup.report.json` / `cleanup.receipt.json`

Counts (`input_fused_people`, `output_track_records`, `removed_track_records`, `frames`,
`empty_frames`, `flipped_output_records`, …), throughput statistics, the cleanup `config`
thresholds, `input_sha256`, `report_sha256`, `run_id`, `orientation_events[]` (per-track yaw-flip
events with frame indices and timestamps), and **absolute source/output paths** — see A1/A2 below.

---

# Anonymity review

## Findings requiring action

**A1 — Building name in every sidecar (396 files).**
Both `cleanup.report.json` and `cleanup.receipt.json` in all 198 recordings embed the original
absolute paths:

```
/robodata/gdc_atrium/ICRA_GDC/gdc_<YYYYMMDD>_<HHMMSS>/gdc_<YYYYMMDD>_<HHMMSS>_bag/frames.jsonl
/robodata/gdc_atrium/ICRA_GDC_cleaned/gdc_<...>/gdc_<...>_bag/frames.cleaned.jsonl
```

This exposes the building (GDC), the original collection name (`ICRA_GDC`), and the original
per-recording identifiers (`gdc_<date>_<time>`). The local copies were renamed to `atrium_*`, but
the sidecars still carry the original names.

**A2 — Personal username in the calibration bag.**
Three occurrences of `/home/zhengpengen/gdc_atrium/investigations/clean_tracks_<date>_<time>/…`
appear in `atrium_20260824_120001/atrium_20260824_120001_bag/cleanup.{receipt,report}.json`.
That recording is the excluded tracking-validation bag, but its sidecars ship with the release.

**A3 — Release directory name.**
The on-disk release is `ICRA_GDC_cleaned`, and `datasets/steps/raw` is a symlink
pointing at it, so the name appears in the repository as well.

**A4 — Wall-clock exposure.**
Frame `timestamp_ns` values are monotonic (≈66 days from an arbitrary epoch), so the frame stream
itself does not reveal collection time. However recording directory names encode local date and
time (`atrium_20260818_094501`), and `cleanup.receipt.json` stores real `mtime_ns` values
(e.g. 2026-09-01 09:09:19). Combined with a known building this narrows who was present.

**A5 — Body pose data.**
`projected_detections[].sam3d` contains 2D and 3D body keypoints (70-point `mhr70` schema), pixel
bounding boxes, and foot pixel coordinates. No images or faces are stored, but 3D body keypoints
support gait-based re-identification and are more sensitive than the trajectories alone. Decide
explicitly whether `sam3d` ships in a public release.

## Verified clean

**B1 — Frame payloads contain no identifiable text.** Across 9,000 sampled frames from three
recordings, the only distinct string values are machine-generated detection ids
(`<timestamp>:cam<N>:det<M>`) and nine enumerated technical constants: `toe_heel_midpoint`,
`ankles`, `mhr70`, `global_kalman_posterior`, `global_kalman_prediction`, `fixed_lag`,
`age_cutoff`, `same_camera_duplicate`, `camera_track_unconfirmed`. A full grep of one complete
602 MB `frames.cleaned.jsonl` for `gdc|speedway|austin|utexas|/home/|/robodata` returned zero
matches.

**B2 — No street or city identifiers anywhere.** `speedway`, `austin`, `texas`, and `utexas` do not
appear in any raw file, sidecar, or derived artifact.

**B3 — Derived artifacts are clean.** Parquet metadata in `converted/` carries only dataset,
scene, schema, and repo-relative `source_file` paths. A byte scan flagged 54 parquet files, but
every match is random compressed data spelling `gDc`/`GDC` with non-printable surroundings — all
false positives, confirmed by inspecting the surrounding bytes.

**B4 — Camera identifiers are generic.** Cameras are referenced only as indices `0`–`5` and
`cam0`–`cam5`.

## Note on the repository's own description

`datasets/steps/source.json` records `cleaned_source.release` as
`/robodata/atrium/ICRA_atrium_cleaned`, while the data's own sidecars record
`/robodata/gdc_atrium/ICRA_GDC_cleaned`. The repository metadata has been sanitised; the data has
not. Worth reconciling so the recorded provenance matches reality.

## Suggested remediation before any release

1. Rewrite the `source`, `input`, and `output` path strings in all 396 sidecars to the anonymised
   recording names — note this changes `report_sha256` in each receipt, so regenerate receipts
   together with reports.
2. Remove the three `/home/zhengpengen/...` paths from the calibration bag's sidecars.
3. Rename the release directory and repoint the `raw` symlink.
4. Decide whether recording identifiers keep real dates, or move to a sequential scheme with the
   real timestamps held privately.
5. Decide whether `sam3d` keypoints are part of the public release.
6. Re-run this scan afterwards to confirm.
