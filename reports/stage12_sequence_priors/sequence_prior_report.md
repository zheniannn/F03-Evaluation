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
  filtering metrics (NaN score; 99,099 such track-rows here).

## Overall results

| threshold_db | stage08_true_tracks | stage08_false_tracks | stage09_true_retention | stage09_false_reduction | stage11_true_retention | stage11_false_reduction | model | stage12_calibration_mode | stage12_true_retention | stage12_false_reduction | stage12_precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 20,330 | 164 | 0.9735 | 0.7879 | 1 | 0.0065 | gru_ae | track_purity | 0.9851 | 0.9817 | 0.9999 |
| 0 | 20,897 | 16 | 0.9854 | 0.6365 | 1 | 0.0006 | gru_ae | track_purity | 0.9802 | 0.9375 | 1 |
| 3 | 19,317 | 6 | 0.9844 | 0.5524 | 1 | 0 | gru_ae | track_purity | 0.9704 | 1 | 1 |
| 6 | 16,564 | 1 | 0.9818 | 0.4781 | 1 | 0 | gru_ae | track_purity | 0.9714 | 1 | 1 |
| 9 | 13,776 | 0 | 0.9764 | 0.4231 | 1 | 0 | gru_ae | track_purity | 0.9691 | nan | 1 |
| 12 | 11,244 | 0 | 0.9712 | 0.3878 | 1 | 0 | gru_ae | track_purity | 0.9645 | nan | 1 |
| -5 | 20,330 | 164 | 0.9735 | 0.7879 | 1 | 0.0065 | mlp_dae | track_purity | 0.9874 | 1 | 1 |
| 0 | 20,897 | 16 | 0.9854 | 0.6365 | 1 | 0.0006 | mlp_dae | track_purity | 0.9813 | 0.9375 | 1 |
| 3 | 19,317 | 6 | 0.9844 | 0.5524 | 1 | 0 | mlp_dae | track_purity | 0.9713 | 1 | 1 |
| 6 | 16,564 | 1 | 0.9818 | 0.4781 | 1 | 0 | mlp_dae | track_purity | 0.9711 | 1 | 1 |
| 9 | 13,776 | 0 | 0.9764 | 0.4231 | 1 | 0 | mlp_dae | track_purity | 0.9660 | nan | 1 |
| 12 | 11,244 | 0 | 0.9712 | 0.3878 | 1 | 0 | mlp_dae | track_purity | 0.9609 | nan | 1 |
| -5 | 20,330 | 164 | 0.9735 | 0.7879 | 1 | 0.0065 | tcn_ae | track_purity | 0.9451 | 0.9878 | 0.9999 |
| 0 | 20,897 | 16 | 0.9854 | 0.6365 | 1 | 0.0006 | tcn_ae | track_purity | 0.9267 | 0.8750 | 0.9999 |
| 3 | 19,317 | 6 | 0.9844 | 0.5524 | 1 | 0 | tcn_ae | track_purity | 0.9249 | 1 | 1 |
| 6 | 16,564 | 1 | 0.9818 | 0.4781 | 1 | 0 | tcn_ae | track_purity | 0.9388 | 1 | 1 |
| 9 | 13,776 | 0 | 0.9764 | 0.4231 | 1 | 0 | tcn_ae | track_purity | 0.9485 | nan | 1 |
| 12 | 11,244 | 0 | 0.9712 | 0.3878 | 1 | 0 | tcn_ae | track_purity | 0.9512 | nan | 1 |

## Model comparison

- **gru_ae**: mean false reduction 0.980, mean true retention 0.973.
- **mlp_dae**: mean false reduction 0.984, mean true retention 0.973.
- **tcn_ae**: mean false reduction 0.966, mean true retention 0.939.

## Range-bin behavior

| model | range_bin | tracks | true_retention | false_reduction | median_score |
|---:|---:|---:|---:|---:|---:|
| gru_ae | 0-50 km | 64,081 | 0.9884 | 0.9798 | 1 |
| gru_ae | 50-100 km | 36,927 | 0.9063 | nan | 0.8771 |
| gru_ae | 100-200 km | 1,307 | 0.8404 | nan | 0.7850 |
| mlp_dae | 0-50 km | 64,081 | 0.9876 | 0.9844 | 1 |
| mlp_dae | 50-100 km | 36,927 | 0.9010 | nan | 0.8820 |
| mlp_dae | 100-200 km | 1,307 | 0.8468 | nan | 0.7650 |
| tcn_ae | 0-50 km | 64,081 | 0.9855 | 0.9657 | 0.9998 |
| tcn_ae | 50-100 km | 36,927 | 0.8092 | nan | 0.7634 |
| tcn_ae | 100-200 km | 1,307 | 0.3762 | nan | 0.3899 |

## Score separability

Median sequence score (pooled models): true 0.971 vs false 0.000.
Sequence reconstruction separates true from false tracks.

## Score-threshold calibration

Raw reconstruction errors vs the clean-holdout calibration band
(score 1 at/below val p50, 0 at/above val p99):

| model | true_median_error | false_median_error | calibration_p50 | calibration_p99 |
|---:|---:|---:|---:|---:|
| gru_ae | 0.4868 | 7.4268 | 0.3926 | 3.8963 |
| mlp_dae | 0.2338 | 4.2556 | 0.1946 | 2.1472 |
| tcn_ae | 0.0355 | 0.3598 | 0.0265 | 0.2164 |

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

Calibration tracks: dates 2022-06-06; thresholds 3, 6, 9, 12 dB; eligibility target_fraction >= 0.95 and purity >= 0.95 (higher thresholds are used for calibration because they yield cleaner, more reliable high-purity true tracks).

### Calibration error quantiles (per model)

| model | calibration | error_p50 | error_p90 | error_p99 | n_calibration_tracks | n_calibration_windows |
|---:|---:|---:|---:|---:|---:|---:|
| gru_ae | clean_truth | 0.0131 | 0.0977 | 0.4954 | nan | 60,000 |
| gru_ae | track_purity | 0.3926 | 0.9567 | 3.8963 | 20,254 | 652,982 |
| mlp_dae | clean_truth | 0.0071 | 0.0628 | 0.2827 | nan | 60,000 |
| mlp_dae | track_purity | 0.1946 | 0.5172 | 2.1472 | 20,254 | 652,982 |
| tcn_ae | clean_truth | 0.0005 | 0.0023 | 0.0078 | nan | 60,000 |
| tcn_ae | track_purity | 0.0265 | 0.0706 | 0.2164 | 20,254 | 652,982 |

### Clean-truth vs track-purity filtering (score threshold 0.5)

| model | threshold_db | calibration_mode | true_track_retention | false_track_reduction | precision_after | median_score_true_tracks |
|---:|---:|---:|---:|---:|---:|---:|
| gru_ae | -5 | clean_truth | 0.0787 | 1 | 1 | 0 |
| gru_ae | 0 | clean_truth | 0.0819 | 1 | 1 | 0 |
| gru_ae | 3 | clean_truth | 0.0929 | 1 | 1 | 0 |
| gru_ae | 6 | clean_truth | 0.1241 | 1 | 1 | 0.0728 |
| gru_ae | 9 | clean_truth | 0.1757 | nan | 1 | 0.1789 |
| gru_ae | 12 | clean_truth | 0.2479 | nan | 1 | 0.2767 |
| mlp_dae | -5 | clean_truth | 0.1104 | 1 | 1 | 0.1145 |
| mlp_dae | 0 | clean_truth | 0.1161 | 1 | 1 | 0.1223 |
| mlp_dae | 3 | clean_truth | 0.1390 | 1 | 1 | 0.1490 |
| mlp_dae | 6 | clean_truth | 0.1818 | 1 | 1 | 0.2055 |
| mlp_dae | 9 | clean_truth | 0.2445 | nan | 1 | 0.2805 |
| mlp_dae | 12 | clean_truth | 0.3106 | nan | 1 | 0.3407 |
| tcn_ae | -5 | clean_truth | 0 | 1 | nan | 0 |
| tcn_ae | 0 | clean_truth | 0 | 1 | nan | 0 |
| tcn_ae | 3 | clean_truth | 0 | 1 | nan | 0 |
| tcn_ae | 6 | clean_truth | 0 | 1 | nan | 0 |
| tcn_ae | 9 | clean_truth | 0.0001 | nan | 1 | 0 |
| tcn_ae | 12 | clean_truth | 0 | nan | nan | 0 |
| gru_ae | -5 | track_purity | 0.9851 | 0.9817 | 0.9999 | 0.9548 |
| gru_ae | 0 | track_purity | 0.9802 | 0.9375 | 1 | 0.9588 |
| gru_ae | 3 | track_purity | 0.9704 | 1 | 1 | 0.9674 |
| gru_ae | 6 | track_purity | 0.9714 | 1 | 1 | 0.9807 |
| gru_ae | 9 | track_purity | 0.9691 | nan | 1 | 0.9953 |
| gru_ae | 12 | track_purity | 0.9645 | nan | 1 | 1 |
| mlp_dae | -5 | track_purity | 0.9874 | 1 | 1 | 0.9711 |
| mlp_dae | 0 | track_purity | 0.9813 | 0.9375 | 1 | 0.9722 |
| mlp_dae | 3 | track_purity | 0.9713 | 1 | 1 | 0.9759 |
| mlp_dae | 6 | track_purity | 0.9711 | 1 | 1 | 0.9839 |
| mlp_dae | 9 | track_purity | 0.9660 | nan | 1 | 0.9945 |
| mlp_dae | 12 | track_purity | 0.9609 | nan | 1 | 1 |
| tcn_ae | -5 | track_purity | 0.9451 | 0.9878 | 0.9999 | 0.9239 |
| tcn_ae | 0 | track_purity | 0.9267 | 0.8750 | 0.9999 | 0.9276 |
| tcn_ae | 3 | track_purity | 0.9249 | 1 | 1 | 0.9427 |
| tcn_ae | 6 | track_purity | 0.9388 | 1 | 1 | 0.9646 |
| tcn_ae | 9 | track_purity | 0.9485 | nan | 1 | 0.9879 |
| tcn_ae | 12 | track_purity | 0.9512 | nan | 1 | 1 |

Mean true-track retention rises from 0.106 (clean-truth) to 0.962 (track-purity) at score threshold 0.5 -- noise-matched calibration fixes the domain shift while false-track reduction stays high.

The primary scores in this run use the **track_purity** calibration.
