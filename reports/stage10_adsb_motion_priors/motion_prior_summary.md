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
- Training dates: 2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27
- Holdout dates: none
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
| speed_mps | m/s | 9,676,882 | 9,673,597 | 3,285 | 18.5830 | 28.8771 | 36.2579 | 40.3312 | 47.4116 | 56.2400 | 71.0699 | 85.4806 | 92.6515 | 106.5297 | 113.0562 | 129.3372 |
| accel_abs_mps2 | m/s^2 | 9,521,842 | 9,521,841 | 1 | 0 | 0.0008 | 0.0073 | 0.0157 | 0.0422 | 0.0960 | 0.1851 | 0.3162 | 0.4339 | 0.7734 | 0.9585 | 1.5638 |
| accel_vector_mps2 | m/s^2 | 9,521,842 | 9,521,813 | 29 | 0 | 0.0084 | 0.0315 | 0.0497 | 0.0958 | 0.1849 | 0.3764 | 1.0928 | 2.1110 | 3.7213 | 4.3595 | 6.3180 |
| turn_rate_abs_deg_s | deg/s | 9,521,842 | 9,521,842 | 0 | 0 | 0.0004 | 0.0044 | 0.0098 | 0.0296 | 0.0961 | 0.2960 | 1.1879 | 2.4845 | 4.6201 | 5.4509 | 8.5457 |
| vertical_speed_abs_mps | m/s | 9,676,882 | 9,674,628 | 2,254 | 0 | 0 | 0 | 0 | 0.0547 | 0.4634 | 1.8862 | 3.2560 | 4.0598 | 5.7944 | 6.7327 | 10.2870 |

## Comparison to Stage 09 hand thresholds

| stage 9 knee | hand-tuned | empirical ADS-B p95 |
|---:|---:|---:|
| speed_mps p95 | good <= 110, bad >= 160 m/s | 92.65 |
| accel p95 | good <= 3.0, bad >= 8.0 m/s^2 | 0.43 |
| turn rate p95 | good <= 5, bad >= 15 deg/s | 2.48 |
| vertical speed p95 | good <= 10, bad >= 25 m/s | 4.06 |

Reading: where the empirical p95 sits well BELOW a stage-9 'good' knee,
the hand threshold was conservative (it under-penalizes); where the
empirical p95 approaches the knee, the hand threshold was aggressive.
Stage 11 should score with the full histogram likelihoods instead of
knees, making this comparison moot.

## Day-to-day consistency

| date | holdout | rows_read | trajectories | speed_p50 | speed_p95 | accel_abs_p95 | turn_abs_p95 | vertical_speed_abs_p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | False | 2,304,562 | 18,426 | 56.1560 | 92.7900 | 0.4320 | 2.4910 | 4.0430 |
| 2022-06-13 | False | 2,213,629 | 18,088 | 56.6020 | 94.1290 | 0.4430 | 2.4810 | 4.0490 |
| 2022-06-20 | False | 2,893,175 | 23,112 | 56.0280 | 91.0480 | 0.4300 | 2.4490 | 3.9910 |
| 2022-06-27 | False | 2,265,516 | 17,894 | 56.2990 | 93.0380 | 0.4230 | 2.4910 | 4.0960 |

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
