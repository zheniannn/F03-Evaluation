# Stage 12 Learned Sequence-Prior Track Scoring

## Status

- Stage 12 trains trajectory-window autoencoders from clean
  ADS-B/radar truth windows and scores Stage 08 Kalman tracks by
  reconstruction plausibility.
- These are one-day scoring results for 2022-06-06.
- This is **not VAE** (stage 13), **not diffusion** (stage 14), and
  not the full model zoo.
- **Truth labels are used only after scoring**, for evaluation.

## Motivation

- Stage 09 used hand-designed feature penalties; stage 11 used
  empirical marginal ADS-B priors. Both score features mostly
  independently and miss the temporal SHAPE of a trajectory.
- Stage 12 learns sequence shape directly: real GA windows should
  reconstruct well; clutter-born tracks should not.

## Training data

- Source: `data/active/radar_truth_relocated`
- Train dates: 2022-06-06, 2022-06-13, 2022-06-20; holdout dates: 2022-06-27
- Train windows: 200,000; validation windows: 60,000
- Features: dx, dy, dz, vx, vy, vz, speed, vertical_speed, turn_rate
- Window length 20 @ stride 5 (10 s grid); windows are origin-shifted and heading-rotated before
  standardization.

## Models

- **mlp_dae** -- flattened-window MLP denoising autoencoder.
- **gru_ae** -- GRU encoder to latent, latent-driven GRU decoder.
- **tcn_ae** -- 1D-convolutional encoder/decoder over time.

All trained with denoising MSE (Gaussian input noise).

## Validation reconstruction

| model | n_val_windows | error_p50 | error_p75 | error_p90 | error_p95 | error_p99 | error_mean | error_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_dae | 60,000 | 0.0071 | 0.0236 | 0.0628 | 0.1004 | 0.2827 | 0.0328 | 0.4679 |
| gru_ae | 60,000 | 0.0131 | 0.0385 | 0.0977 | 0.1604 | 0.4954 | 0.1307 | 4.5134 |
| tcn_ae | 60,000 | 0.0005 | 0.0012 | 0.0023 | 0.0032 | 0.0078 | 0.0013 | 0.0142 |

## Scoring model

- Per track: build the same normalized windows from the posterior
  states; track_error = **median** per-window reconstruction error.
- Calibration against clean holdout windows: score 1 at/below the
  validation p50 error, 0 at/above the p99, linear between.
- Keep threshold: **0.5**. Tracks with fewer than
  window_len points cannot be windowed and are excluded from
  filtering metrics (NaN score; 9,861 such track-rows here).

## Overall results

| threshold_db | stage08_true_tracks | stage08_false_tracks | stage09_true_retention | stage09_false_reduction | stage11_true_retention | stage11_false_reduction | model | stage12_calibration_mode | stage12_true_retention | stage12_false_reduction | stage12_precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1,962 | 5,141 | 0.9818 | 0.4781 | 1 | 0 | mlp_dae | track_purity | 0.8858 | 0.2680 | 0.3159 |

## Model comparison

- **mlp_dae**: mean false reduction 0.268, mean true retention 0.886.

## Range-bin behavior

| model | range_bin | tracks | true_retention | false_reduction | median_score |
|---:|---:|---:|---:|---:|---:|
| mlp_dae | 0-50 km | 4,003 | 0.8884 | 0.2904 | 0.7384 |
| mlp_dae | 50-100 km | 3,100 | 0.8629 | 0.2508 | 0.7491 |

## Score separability

Median sequence score (pooled models): true 0.877 vs false 0.698.
Sequence reconstruction separates true from false tracks.

## Score-threshold calibration

Raw reconstruction errors vs the clean-holdout calibration band
(score 1 at/below val p50, 0 at/above val p99):

| model | true_median_error | false_median_error | calibration_p50 | calibration_p99 |
|---:|---:|---:|---:|---:|
| mlp_dae | 0.8823 | 1.8854 | 0.1935 | 5.7986 |

- Reconstruction error itself is strongly separable: false-track
  median errors sit roughly an order of magnitude above true-track
  median errors for every model.
- But the calibration band comes from CLEAN truth windows, while
  scored windows come from Kalman posteriors over noisy stage-6
  measurements -- so typical TRUE tracks land at or above the
  clean-holdout p99 and are compressed toward score 0.
- Consequence at every swept score threshold: false-track
  reduction is total (precision after filtering = 1.0) while true
  retention is far too low to use as a filter at 0.5.
- The discrimination is real; the score MAPPING is miscalibrated.
  Calibrating against noisy (measurement-matched or
  Kalman-filtered) truth windows, or matching the training noise
  to the stage-6 measurement noise, is the concrete fix -- a
  natural part of the stage 13 probabilistic treatment.

## Failure modes

- Short tracks (< window_len points) cannot be windowed at all.
- Autoencoders can reconstruct some smooth false tracks -- smoothness
  is exactly what they learn.
- Reconstruction error is not a calibrated likelihood; the p50/p99
  mapping is a pragmatic surrogate.
- Unusual but valid maneuvers may reconstruct poorly and be rejected.
- Probabilistic sequence models are the natural next step.

## Recommended next stage

Stage 13 should implement a **VAE trajectory prior** over the same
normalized windows, giving a proper likelihood-based score.

## Stage 12.5 Noise-Matched Calibration

The original stage-12 score mapping used **clean truth** validation
windows for its p50->1 / p99->0 band. Scoring noisy stage-8 Kalman
tracks against that clean band is a domain shift: genuine tracks
reconstruct worse than clean truth and collapse toward score 0.
Stage 12.5 recalibrates the error->score band using **high-purity
stage-8 true tracks** instead, so the 0.5 threshold is meaningful in
the noisy-track domain.

- The **autoencoder weights are unchanged** -- nothing is retrained;
  only the reconstruction-error quantiles that define the score band
  are replaced.
- **Truth labels are used only to select calibration tracks** and to
  evaluate metrics; they never enter the score itself.
- This is still **not VAE** (stage 13) and **not diffusion** (stage 14).

Calibration tracks: dates 2022-06-06; thresholds 6 dB; eligibility target_fraction >= 0.95 and purity >= 0.95 (higher thresholds are used for calibration because they yield cleaner, more reliable high-purity true tracks).

### Calibration error quantiles (per model)

| model | calibration | error_p50 | error_p90 | error_p99 | n_calibration_tracks | n_calibration_windows |
|---:|---:|---:|---:|---:|---:|---:|
| mlp_dae | clean_truth | 0.0071 | 0.0628 | 0.2827 | nan | 60,000 |
| mlp_dae | track_purity | 0.1935 | 2.5063 | 5.7986 | 833 | 3,109 |

The primary scores in this run use the **track_purity** calibration.
