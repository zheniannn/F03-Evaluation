# Stage 11 Empirical ADS-B-Prior Track Scoring

## Status

- Stage 11 applies the **Stage 10 empirical ADS-B priors** to the
  Stage 08 Kalman tracks.
- These are one-day results for 2022-06-06.
- This is **not VAE**, diffusion, or neural ML -- the priors are
  transparent empirical histograms and quantiles.
- **Truth labels are used only after scoring**, for evaluation.

## Purpose

- Stage 09 used hand-designed physics thresholds (soft knees).
- Stage 10 learned empirical GA motion priors from 9.68M ADS-B
  samples.
- Stage 11 tests whether those data-derived priors suppress false
  tracks better than the hand knees while retaining true tracks.

## Priors used

| prior | feature | units | p95 | p99 | p999 |
|---:|---:|---:|---:|---:|---:|
| speed_prior.json | speed_mps | m/s | 92.6520 | 106.5300 | 129.3370 |
| acceleration_prior.json | accel_abs_mps2 | m/s^2 | 0.4340 | 0.7730 | 1.5640 |
| vector_acceleration_prior.json | accel_vector_mps2 | m/s^2 | 2.1110 | 3.7210 | 6.3180 |
| turn_rate_prior.json | turn_rate_abs_deg_s | deg/s | 2.4850 | 4.6200 | 8.5460 |
| vertical_speed_prior.json | vertical_speed_abs_mps | m/s | 4.0600 | 5.7940 | 10.2870 |

- Joint prior: loaded and used (weight 0.5).
- SNR auxiliary term: available.

## Scoring model

- **Empirical quantile-exceedance penalties** per feature, from the
  prior JSON quantiles (nothing hand-coded): raw = frac(>p99) +
  2*frac(>p995) + 4*frac(>p999), normalized and inverted to a [0,1]
  score. Speed also penalizes the LOW side (frac(<p01) + 4*frac(<p001))
  because implausibly slow tracks are as suspicious as fast ones.
- **Mean histogram log-densities** per feature are carried as
  diagnostic columns (speed/accel/vector/turn/vertical _mean_logpdf),
  not mixed into the score.
- **Joint prior score**: robust Mahalanobis-like distance of the
  track's median log1p feature vector to the stage-10 joint median
  under the log1p covariance, score = exp(-0.5 * min(d2, 50)). The
  track median (not p95) is compared because the joint prior
  describes per-sample values.
- **Weak auxiliary terms**: continuity = hit rate (w=0.5), SNR mapped over (-10.0, 3.0) dB (w=0.25).
- Final score = weighted mean of available components, clamped to
  [0,1]; keep threshold **0.5**.

## Overall results

| threshold_db | stage08_confirmed_tracks | stage08_true_tracks | stage08_false_tracks | stage09_kept_true_tracks | stage09_kept_false_tracks | stage09_true_track_retention | stage09_false_track_reduction | stage09_precision | stage11_score_threshold | stage11_kept_true_tracks | stage11_kept_false_tracks | stage11_true_track_retention | stage11_false_track_reduction | stage11_precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 31,626 | 25,488 | 6,138 | 24,812 | 1,302 | 0.9735 | 0.7879 | 0.9501 | 0.5000 | 25,488 | 6,098 | 1 | 0.0065 | 0.8069 |
| 0 | 32,158 | 30,554 | 1,604 | 30,108 | 583 | 0.9854 | 0.6365 | 0.9810 | 0.5000 | 30,553 | 1,603 | 1 | 0.0006 | 0.9501 |
| 3 | 33,826 | 33,167 | 659 | 32,648 | 295 | 0.9844 | 0.5524 | 0.9910 | 0.5000 | 33,167 | 659 | 1 | 0 | 0.9805 |
| 6 | 31,927 | 31,653 | 274 | 31,078 | 143 | 0.9818 | 0.4781 | 0.9954 | 0.5000 | 31,653 | 274 | 1 | 0 | 0.9914 |
| 9 | 27,767 | 27,663 | 104 | 27,009 | 60 | 0.9764 | 0.4231 | 0.9978 | 0.5000 | 27,662 | 104 | 1 | 0 | 0.9963 |
| 12 | 22,890 | 22,841 | 49 | 22,184 | 30 | 0.9712 | 0.3878 | 0.9986 | 0.5000 | 22,841 | 49 | 1 | 0 | 0.9979 |

## Comparison with Stage 09

- Mean false-track reduction: stage 9 0.544 vs stage 11 0.001 -- hand knees removed more; empirical marginal priors alone are not sufficient here.
- Mean true-track retention: stage 9 0.979 vs stage 11 1.000.
- Where stage 11 is stricter, it is because the real GA priors are
  much tighter than the stage-9 knees (empirical p95 speed 92.7 vs
  knee 110 m/s; |accel| 0.43 vs 3.0 m/s^2; turn 2.5 vs 5 deg/s;
  vertical 4.1 vs 10 m/s) -- clutter chains that slipped under the
  generous knees now sit in the far tail of the data distribution.

### Score-threshold calibration (pooled over detection thresholds)

The exceedance penalties are normalized by their theoretical maximum,
so scores cluster near 1 and the nominal 0.5 keep-threshold is nearly
inert. The pooled sweep below shows the score's actual operating
curve -- pick the keep-threshold from here, not from the 0.5 default:

| score_threshold | true_retention | false_reduction |
|---:|---:|---:|
| 0.1000 | 1 | 0 |
| 0.2000 | 1 | 0 |
| 0.3000 | 1 | 0 |
| 0.4000 | 1 | 0 |
| 0.5000 | 1 | 0.0050 |
| 0.6000 | 0.9980 | 0.2540 |
| 0.7000 | 0.9620 | 0.7880 |
| 0.8000 | 0.7150 | 0.9900 |
| 0.9000 | 0.0190 | 1 |

## Range-bin behavior

| range_bin | tracks | true_retention | false_reduction |
|---:|---:|---:|---:|
| 0-50 km | 83,975 | 1 | 0.0012 |
| 50-100 km | 88,393 | 1 | 0 |
| 100-200 km | 7,821 | 1 | nan |
| >200 km | 5 | 1 | nan |

## Score separability

Median ADS-B prior score: true tracks 0.836, false tracks 0.646.
True and false tracks separate under the empirical priors.

## Failure modes

- Real but unusual maneuvers (aerobatics, tight pattern work) sit in
  the empirical tails and get penalized like clutter.
- Smooth clutter chains whose kinematics land inside the priors can
  still survive.
- Histogram priors treat samples independently -- they ignore the
  temporal SHAPE of the trajectory, which is exactly the information
  a sequence model can use.
- These motivate Stage 12 learned sequence priors.

## Recommended next stage

Stage 12 should test learned trajectory-window models:

- VAE trajectory prior
- denoising autoencoder
- TCN/GRU autoencoder
- diffusion denoiser later
