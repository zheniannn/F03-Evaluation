# Stage 08 Kalman Detect-Then-Track Baseline

## Method

- Constant-velocity Kalman filter per track (6-state position +
  velocity, discrete white-noise acceleration process model).
- Greedy gated nearest-neighbor association on predicted position
  (Euclidean gate 2500 m).
- Tracks confirm after 3 hits and delete after
  3 consecutive misses.
- The tracker never sees truth labels; is_target / trajectory_id are
  used only in this evaluation.
- A TRUE track is confirmed with purity >= 0.5
  and dominant-trajectory fraction >= 0.5;
  coverage denominators are trajectories with >=
  5 target detections in the evaluated frames.
- Baseline simplifications (stage 9+ targets): isotropic measurement
  noise, Euclidean (not Mahalanobis) gating, radial velocity unused.

## Results

| date | threshold_db | frames | detections_in | tracks_confirmed | true_tracks | false_tracks | trajectory_coverage | fragmentation | mean_true_track_purity | target_det_absorption | clutter_det_absorption |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | 6 | 200 | 262,335 | 25,717 | 5,260 | 20,457 | 0.9280 | 10.7347 | 0.8400 | 0.9761 | 0.8698 |

## Interpretation

The Kalman baseline turns point detections into persistent objects:
true tracks follow real aircraft across frames, while clutter-born
tracks rarely survive confirmation because false alarms are spatially
independent between scans. Compare trajectory coverage here against
the stage-7 single-frame Pd at the same threshold to quantify what
temporal integration buys before any ML model is introduced.
