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
| 2022-06-06 | 9 | 8,639 | 908,053 | 27,767 | 27,601 | 166 | 0.9178 | 1.8790 | 0.9993 | 0.9372 | 0.0138 |
| 2022-06-06 | 12 | 8,639 | 705,685 | 22,890 | 22,766 | 124 | 0.8935 | 1.8366 | 0.9993 | 0.9219 | 0.0120 |
| 2022-06-13 | -5 | 8,639 | 2,144,026 | 30,746 | 24,571 | 6,175 | 0.9839 | 1.3807 | 0.9980 | 0.9903 | 0.0692 |
| 2022-06-13 | 0 | 8,639 | 1,650,230 | 30,581 | 29,040 | 1,541 | 0.9850 | 1.6299 | 0.9987 | 0.9823 | 0.0395 |
| 2022-06-13 | 3 | 8,639 | 1,375,010 | 32,207 | 31,574 | 633 | 0.9778 | 1.7902 | 0.9989 | 0.9713 | 0.0272 |
| 2022-06-13 | 6 | 8,639 | 1,117,523 | 30,713 | 30,444 | 269 | 0.9485 | 1.8417 | 0.9990 | 0.9552 | 0.0182 |
| 2022-06-20 | -5 | 8,639 | 2,695,017 | 37,800 | 31,685 | 6,115 | 0.9811 | 1.3973 | 0.9980 | 0.9904 | 0.0705 |
| 2022-06-20 | 0 | 8,639 | 2,106,885 | 39,628 | 38,058 | 1,570 | 0.9812 | 1.6782 | 0.9986 | 0.9821 | 0.0416 |
| 2022-06-20 | 3 | 8,639 | 1,762,214 | 42,253 | 41,542 | 711 | 0.9759 | 1.8448 | 0.9989 | 0.9709 | 0.0304 |
| 2022-06-20 | 6 | 8,639 | 1,434,749 | 39,979 | 39,610 | 369 | 0.9450 | 1.8821 | 0.9991 | 0.9545 | 0.0209 |
| 2022-06-27 | -5 | 8,639 | 2,189,127 | 31,292 | 25,087 | 6,205 | 0.9863 | 1.4214 | 0.9980 | 0.9902 | 0.0700 |
| 2022-06-27 | 0 | 8,639 | 1,674,332 | 31,667 | 30,082 | 1,585 | 0.9868 | 1.7037 | 0.9990 | 0.9816 | 0.0400 |
| 2022-06-27 | 3 | 8,639 | 1,390,514 | 33,191 | 32,552 | 639 | 0.9814 | 1.8570 | 0.9988 | 0.9698 | 0.0278 |
| 2022-06-27 | 6 | 8,639 | 1,124,167 | 31,174 | 30,877 | 297 | 0.9477 | 1.8890 | 0.9990 | 0.9532 | 0.0200 |

## Interpretation

The Kalman baseline turns point detections into persistent objects:
true tracks follow real aircraft across frames, while clutter-born
tracks rarely survive confirmation because false alarms are spatially
independent between scans. Compare trajectory coverage here against
the stage-7 single-frame Pd at the same threshold to quantify what
temporal integration buys before any ML model is introduced.
