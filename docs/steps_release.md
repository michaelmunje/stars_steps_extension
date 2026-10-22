# STEPS release — contents and build

Built 2026-09-18 from the cleaned recordings with
`scripts/build_anonymous_release.py`. Output: `/home/mike/STEPS_release`, 743 GB.

| | |
| --- | --- |
| Recordings | 197 |
| Frames | 2,358,311 |
| Duration | 32.8 hours |
| Days covered | 19 (08-18 to 09-11) |
| Sampling | 20.0 Hz |
| Cameras | 6 |

## Layout

```text
Atrium_Data/
├── manifest.json
└── MM/DD/STEPS_MM_DD_HHMMSS_trajectory_data.jsonl
Cleanup_Reports/
└── MM/DD/STEPS_MM_DD_HHMMSS_cleanup_report.json
```

`MM` and `DD` are month and day; `HHMMSS` is the 24-hour local recording time. Trajectory data and
cleanup provenance share the same `MM/DD` structure under separate roots, so the data tree contains
only data.

## Frame record

Each line of a `.jsonl` file is one synchronised frame.

| Field | Meaning |
| --- | --- |
| `frame_index` | 0-based index within the recording |
| `timestamp_ns` | see [Timestamps](#timestamps) — **not** Unix epoch |
| `timestamp_s` | the same clock in seconds |
| `present_cameras` | camera indices contributing to this frame |
| `fresh_cameras` / `stale_cameras` / `missing_cameras` | camera liveness for this frame |
| `sync_skew_ns` | spread between camera timestamps in the group |
| `source_timestamp_ns_by_camera` | per-camera capture times, keys `"0"`–`"5"` |
| `source_timestamp_offset_ns_by_camera` | per-camera offset from the group reference |
| `source_group_index` | index of the synchronised source group |
| `output_fixed_lag_seconds` | configured finalisation lag (1.0 s) |
| `output_finalization` | `mode`, `reason`, `actual_lag_seconds`, `configured_delay_seconds`, `finalized_at_timestamp_ns` |

Four arrays carry the content:

### `model_detections_by_camera` — per-camera detections

A mapping from camera id to that camera's detections. Everything expressed in the camera's own
frame of reference lives here, including body pose.

| Field | Meaning |
| --- | --- |
| `detection_id` | `"<timestamp_ns>:cam<N>:det<M>"`, unique across the recording |
| `camera_index` | which camera produced it |
| `bbox_xyxy` | pixel bounding box |
| `foot_pixel_uv`, `foot_quality`, `foot_source` | ground-contact estimate in image space |
| `image_width_px`, `image_height_px` | 1440 × 1080 |
| `confidence` | detector confidence |
| `camera_track_id`, `camera_track_confirmed`, `camera_track_observed`, `camera_track_consecutive_hits`, `camera_track_consistent_seconds` | per-camera tracking state |
| `camera_position_smoothed_xy`, `camera_position_covariance`, `camera_measurement_covariance`, `camera_velocity_ground_xy_mps` | per-camera ground estimate |
| `suppressed_duplicate`, `duplicate_of_detection_id` | same-camera duplicate suppression |
| `sam3d` | body pose, below |

`sam3d` holds `keypoints_2d` (70 × 2, image space), `keypoints_3d` (70 × 3), `bbox_xyxy`,
`confidence`, `global_rot_zyx`, and `model_fields` (`keypoint_schema` = `mhr70`, `focal_length`,
`pred_cam_t`, and a `foot_contact` block). Source imagery is **not** part of this release.

**The key set varies by frame.** Only cameras present in that frame appear, so
`frame["model_detections_by_camera"]["3"]` raises `KeyError` when camera 3 was down. Iterate over
the keys, or use `.get(cam, [])`.

### `projected_detections` — global view

A flat list with one world-frame record per detection, linked to its camera record by
`detection_id`.

| Field | Meaning |
| --- | --- |
| `detection_id` | links to `model_detections_by_camera` |
| `timestamp_ns` | camera capture time for this detection |
| `position_ground_xy` | ground-plane position, metres |
| `roi_position_ground_xy` | position clipped to the region of interest |
| `yaw_ground_rad` | body yaw over ground |
| `track_id` | global track it was associated to |
| `cluster_index` | multi-view cluster |
| `association_eligible`, `association_rejection_reason` | association outcome |
| `global_birth_retroactively_labeled`, `global_birth_was_pending_causally`, `global_reassociation_update_only` | track-birth bookkeeping |

### `fused_people` — multi-view fusion

One entry per fused observation: `track_id`, `position_ground_xy`, `position_raw_mean_ground_xy`,
`position_smoothed_xy`, `roi_position_ground_xy`, `member_positions_ground_xy`,
`member_detection_ids`, `camera_indices`, `fusion_weights`, `measurement_covariance`,
`max_spread_m`, `rms_spread_m`, the yaw group (`yaw_ground_rad`, `yaw_ground_rad_original`,
`yaw_flipped`, `yaw_concentration`), and track-birth flags.

### `tracked_people` — final tracks

The array downstream work consumes. Track identifiers, positions, and body headings for both
dataset characterization and forecasting come from here.

| Field | Meaning |
| --- | --- |
| `track_id` | per-recording track identity |
| `position_ground_xy`, `velocity_ground_xy_mps`, `position_covariance` | Kalman state |
| `yaw_ground_rad` | measured body yaw — **not** direction of travel |
| `yaw_ground_rad_original`, `yaw_flipped`, `yaw_concentration` | yaw provenance |
| `confirmed`, `observed_this_frame`, `stale` | track status |
| `state_source` | `global_kalman_posterior` or `global_kalman_prediction` |
| `age_steps`, `hit_count`, `consecutive_hit_count`, `missed_steps`, `missed_seconds`, `consecutive_observed_seconds` | lifetime counters |
| `spread_m` | cross-camera agreement |
| `last_observed_position_ground_xy`, `last_observed_timestamp_ns`, `last_observed_camera_indices` | last measurement |

## Timestamps

`timestamp_ns` is nanoseconds from an arbitrary **monotonic** origin, not the Unix epoch — parsing
it as epoch time lands in 1970. The origin is shared across every recording, so differences between
recordings are meaningful and reconstruct the full collection schedule. Wall-clock date and time
are in the filename.

## Cleanup reports

`Cleanup_Reports/` documents what the upstream cleanup pass did before the data reached this
release. It is provenance, not data: nothing in it is needed to load or use a recording.

Each report carries the filter thresholds (`config`: bounds, `min_speed` 0.35 m/s, `min_duration`
1.5 s, yaw and gap limits), input-versus-output counts (`input_track_records` versus
`output_track_records`, and similarly for fused people and projected detections),
`removed_track_records`, yaw-flip corrections (`flipped_records_by_track`, `orientation_events`),
and `recording_seconds` / `frames`.

The `min_speed` filter removes stationary and slow pedestrians, so density and interaction
statistics from this release are **not** directly comparable to an unfiltered build of the same
scene.

## Anonymity

Verified twice: every line is scanned during conversion — a recording fails rather than shipping if
a pattern matches — and an independent post-hoc scan re-read all 743 GB.

```text
files with any identifier match: 0   (394 files, 16-way parallel, 3m56s)
patterns: building name, street, city, institution, usernames, /home/, /robodata, /Users/
```

Absolute source paths and a personal home directory that appeared in the original sidecars are gone
by construction: released reports are rebuilt from an allow-list of fields, so no path-valued field
can survive, rather than being pattern-scrubbed. Recording identifiers were renamed, and camera
identity is exposed only as indices `0`–`5`.

**Excluded:** `atrium_20260824_120001`, the tracking-validation calibration bag, is not in the
release in any form.

## Rebuilding

```bash
python3 scripts/build_anonymous_release.py \
  --source <cleaned recordings> --dest <output> --workers 16
```

The calibration bag is excluded by default; `--exclude` adds more. `--pose-layout duplicate` keeps
`sam3d` inside `projected_detections` as well, at roughly +76% size for no extra information.

Build performance is bounded by disk, not CPU. Reading and writing on one drive gave 59 MB/s;
reading from the source drive while writing to a separate NVMe gave 134 MB/s and finished 198
recordings in 100 minutes.
