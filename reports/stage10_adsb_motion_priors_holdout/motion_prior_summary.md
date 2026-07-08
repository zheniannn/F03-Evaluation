# Stage 10 Empirical ADS-B Motion Priors

## Purpose

- Stage 10 learns **empirical motion priors** from the F01 stage-4
  ADS-B trajectories: histogram densities and quantiles for speed,
  |acceleration|, vector acceleration, |turn rate|, and |vertical
  speed| of real fixed-wing GA flight.
- These priors are **not yet applied to tracks** -- nothing in stages
  7-9 changes.
- **Stage 11** will use them to replace or augment the stage-9
  hand-designed penalties with data-derived likelihoods. Nothing here
  is a neural network: the priors are transparent empirical
  histograms.

## Input data

- Input directory: `data/active/trajectories_10s`
- Training dates: 2022-06-06, 2022-06-13, 2022-06-20
- Holdout dates: 2022-06-27
- Rows read: 9,676,882; trajectories: 77,520
- Fitting filters (disclosed, applied only for prior fitting): speed in
  [5, 160] m/s, |accel| <= 15 m/s^2,
  |turn rate| <= 30 deg/s, |vertical speed| <= 40 m/s.

## Fitted features

- **speed_mps** -- stage-4 ground speed on the 10 s grid.
- **accel_abs_mps2** -- absolute longitudinal acceleration.
- **accel_vector_mps2** -- vector (centripetal-inclusive) acceleration (available).
- **turn_rate_abs_deg_s** -- absolute turn rate.
- **vertical_speed_abs_mps** -- |d(alt)/dt| per trajectory
  (np.gradient over the trajectory's timestamps).

## Quantile summary

| feature | units | n_samples_total | n_samples_used | n_samples_dropped | p001 | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p995 | p999 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| speed_mps | m/s | 7,411,366 | 7,408,630 | 2,736 | 18.9195 | 29.0939 | 36.2834 | 40.3556 | 47.4332 | 56.2231 | 71.0779 | 85.4623 | 92.4695 | 106.1723 | 112.7187 | 128.7680 |
| accel_abs_mps2 | m/s^2 | 7,292,114 | 7,292,113 | 1 | 0 | 0.0008 | 0.0072 | 0.0157 | 0.0423 | 0.0964 | 0.1860 | 0.3179 | 0.4370 | 0.7796 | 0.9659 | 1.5869 |
| accel_vector_mps2 | m/s^2 | 7,292,114 | 7,292,088 | 26 | 0 | 0.0085 | 0.0316 | 0.0499 | 0.0961 | 0.1858 | 0.3786 | 1.0961 | 2.1121 | 3.7159 | 4.3529 | 6.3342 |
| turn_rate_abs_deg_s | deg/s | 7,292,114 | 7,292,114 | 0 | 0 | 0.0004 | 0.0045 | 0.0099 | 0.0297 | 0.0968 | 0.2988 | 1.1943 | 2.4892 | 4.6004 | 5.4286 | 8.4430 |
| vertical_speed_abs_mps | m/s | 7,411,366 | 7,409,558 | 1,808 | 0 | 0 | 0 | 0 | 0.0569 | 0.4664 | 1.8808 | 3.2436 | 4.0521 | 5.7958 | 6.7087 | 10.2406 |

## Comparison to Stage 09 hand thresholds

| stage 9 knee | hand-tuned | empirical ADS-B p95 |
|---:|---:|---:|
| speed_mps p95 | good <= 110, bad >= 160 m/s | 92.47 |
| accel p95 | good <= 3.0, bad >= 8.0 m/s^2 | 0.44 |
| turn rate p95 | good <= 5, bad >= 15 deg/s | 2.49 |
| vertical speed p95 | good <= 10, bad >= 25 m/s | 4.05 |

Reading: where the empirical p95 sits well BELOW a stage-9 'good' knee,
the hand threshold was conservative (it under-penalizes); where the
empirical p95 approaches the knee, the hand threshold was aggressive.
Stage 11 should score with the full histogram likelihoods instead of
knees, making this comparison moot.

## Day-to-day consistency

| date | holdout | rows_read | trajectories | speed_p50 | speed_p95 | accel_abs_p95 | turn_abs_p95 | vertical_speed_abs_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | False | 2,304,562 | 18,426 | 56.1790 | 92.8690 | 0.4360 | 2.4950 | 4.0260 |
| 2022-06-13 | False | 2,213,629 | 18,088 | 56.6220 | 94.0430 | 0.4470 | 2.4980 | 4.0370 |
| 2022-06-20 | False | 2,893,175 | 23,112 | 56.0500 | 91.0180 | 0.4310 | 2.4580 | 4.0480 |
| 2022-06-27 | True | 2,265,516 | 17,894 | 56.3560 | 92.9530 | 0.4220 | 2.4860 | 4.0790 |

## Model files

- `acceleration_prior.json`
- `joint_motion_prior.json`
- `motion_prior_config.json`
- `motion_prior_manifest.json`
- `speed_prior.json`
- `turn_rate_prior.json`
- `vector_acceleration_prior.json`
- `vertical_speed_prior.json`

## Validation

See the validation gate output in the run log: prior files present,
monotonic quantiles, valid histogram shapes, plausibility anchors
(speed p50 in 20-90 m/s), and the report-only stage-9 comparison.

## Next stage

Stage 11 will apply these empirical priors to Stage 08 tracks and
compare against Stage 09 hand-designed physics scoring.
