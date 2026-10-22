# STEPS — category distribution restricted to long tracks

Tests whether track fragmentation explains the high static fraction. Trajectories above a
duration floor were, by construction, not fragmented. If fragmentation were inflating the static
count, the static fraction should fall as the floor rises.

**It does not fall. It rises, monotonically, under both duration definitions.**

All 198 sessions, data seed 42, candidate forecast windows before any acceptance filtering.

## Floor on observed duration (n samples x 0.4 s)

| Floor | Windows | Ego tracks | static | linear | interacting | non-linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 3,528,070 | 16,844 | 93.26% | 0.66% | 4.21% | 1.87% |
| >= 30 s | 3,403,266 | 7,142 | 96.02% | 0.22% | 2.98% | 0.78% |
| >= 60 s | 3,295,682 | 5,517 | 96.82% | 0.18% | 2.45% | 0.56% |
| >= 120 s | 3,042,357 | 3,918 | 97.59% | 0.14% | 1.87% | 0.40% |

## Floor on first-to-last span

| Floor | Windows | Ego tracks | static | linear | interacting | non-linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 3,528,070 | 16,844 | 93.26% | 0.66% | 4.21% | 1.87% |
| >= 30 s | 3,408,802 | 7,433 | 95.91% | 0.23% | 3.04% | 0.82% |
| >= 60 s | 3,302,867 | 5,692 | 96.75% | 0.18% | 2.49% | 0.58% |
| >= 120 s | 3,060,891 | 4,082 | 97.51% | 0.14% | 1.93% | 0.42% |

The two definitions agree to within 0.15 percentage points at every floor, so the result does not
depend on how duration is measured.

## Why the unrestricted row shows 16,844 and not 26,226 tracks

The "Ego tracks" column counts tracks that are the ego of at least one forecasting window, not
all annotated tracks. A window needs 8 observed + 12 predicted *consecutive* samples, so a track
without a 20-step contiguous run never becomes an ego.

| | Tracks |
| --- | ---: |
| All tracks in converted files | 26,226 |
| less the blacklisted calibration recording (0 ever ego) | -229 |
| = tracks outside the blacklist (`n_trajectories_raw`) | 25,997 |
| less tracks whose longest contiguous run is < 20 steps | -9,112 |
| = tracks with a >=20-step run (`n_trajectories_usable`) | 16,885 |
| less tracks never selected as ego | -41 |
| = egos in the index | 16,844 |

The 9,112 dominate: a track with 25 total samples split by a gap into runs of 12 and 13 has no
20-step run and yields no window, even though it is a real track in the 26,226.

This does not affect the category percentages, which are computed over windows.

## Track counts behind the floors

| Floor | Tracks (observed) | Tracks (span) |
| --- | ---: | ---: |
| >= 30 s | 7,267 (27.71%) | 7,674 (29.26%) |
| >= 60 s | 5,589 (21.31%) | 5,802 (22.12%) |
| >= 120 s | 3,972 (15.15%) | 4,159 (15.86%) |

Note that window count barely falls even as track count drops by two thirds: 3,528,070 -> 3,295,682
at the 60 s floor, a 6.6% reduction from 67% fewer tracks. Long-lived tracks generate the
overwhelming majority of forecasting windows.

## Reading

Fragmentation cannot explain the static fraction. Restricting to trajectories that were never
fragmented leaves the static share **higher**, not lower, at every threshold and under both
definitions.

One honest caveat on the direction of the effect: the rise is not itself evidence of additional
stasis, because the selection cuts that way anyway -- a person standing still stays in view and
accumulates a long track, while a person walking through leaves the scene and cannot. So the
monotone increase is expected. The load-bearing observation is the negative one: the static
fraction never drops below its unrestricted value of 93.26%, which is what the fragmentation
hypothesis requires.

The remaining escape is an identity swap between two stationary people. That would misattribute
which person was standing where, but both were standing, so the behavioral claim is unaffected.

Generated from `datasets/steps/indices/scenes-seed42.parquet` joined to per-track
durations from `datasets/steps/converted/*.traj.parquet`; 0 of 3,528,070 windows had an
unmatched primary track.
