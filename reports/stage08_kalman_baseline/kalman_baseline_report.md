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
| 2022-06-06 | -5 | 8,639 | 2,221,123 | 31,626 | 25,501 | 6,125 | 0.9854 | 1.4045 | 0.9980 | 0.9904 | 0.0688 |
| 2022-06-06 | 0 | 8,639 | 1,705,595 | 32,158 | 30,578 | 1,580 | 0.9862 | 1.6828 | 0.9987 | 0.9819 | 0.0400 |
| 2022-06-06 | 3 | 8,639 | 1,417,466 | 33,826 | 33,174 | 652 | 0.9771 | 1.8463 | 0.9990 | 0.9702 | 0.0285 |
| 2022-06-06 | 6 | 8,639 | 1,147,885 | 31,927 | 31,620 | 307 | 0.9496 | 1.8785 | 0.9991 | 0.9536 | 0.0199 |

## Interpretation

The Kalman baseline turns point detections into persistent objects:
true tracks follow real aircraft across frames, while clutter-born
tracks rarely survive confirmation because false alarms are spatially
independent between scans. Compare trajectory coverage here against
the stage-7 single-frame Pd at the same threshold to quantify what
temporal integration buys before any ML model is introduced.
