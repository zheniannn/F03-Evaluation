# Stage 13 VAE Trajectory Prior

## Status

- Stage 13 trains a VAE over normalized trajectory windows.
- Stage 13 scores Stage 08 Kalman tracks using reconstruction and
  ELBO-like anomaly scores.
- Score calibration uses high-purity noisy Stage 08 tracks by default.
- These are one-day results for 2022-06-06.
- This is **not diffusion** and not the full model zoo.
- Truth labels are used only for calibration track selection and evaluation.

## Motivation

- Stage 12.5 deterministic autoencoders worked very well after
  noise-matched calibration (mean true retention ~0.96 with high
  false-track reduction).
- Stage 13 tests whether a probabilistic latent trajectory model (a
  VAE) adds value: a reconstruction score, KL/latent diagnostics, and
  latent motion structure.

## Training data

- Source: `data/active/radar_truth_relocated`
- Train dates: 2022-06-06, 2022-06-13, 2022-06-20; holdout dates: 2022-06-27
- Train windows: 200,000; validation windows: 60,000
- Features: dx, dy, dz, vx, vy, vz, speed, vertical_speed, turn_rate
- Window length 20 @ stride 5 (origin-shifted, heading-rotated, standardized).
- Normalizer: stage12:models/sequence_priors/normalizer.json

## Model

- Fully-connected sequence VAE; latent dim 32, hidden dim 256.
- Encoder: flatten window -> 2x Linear+ReLU -> (mu, logvar).
- Decoder: latent -> 2x Linear+ReLU -> Linear -> window.
- Loss: MSE reconstruction + beta * KL to N(0, I); beta 0.001, KL annealed over 5 epochs.

## Validation

| n_val_windows | error_p50 | error_p75 | error_p90 | error_p95 | error_p99 | error_mean | error_std | kl_p50 | kl_p90 | kl_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 0.0099 | 0.0279 | 0.0653 | 0.1028 | 0.2971 | 0.0474 | 1.4391 | 89.2362 | 155.7959 | 110.2829 |

## Calibration

- Noise-matched **track-purity** calibration (the stage-12.5 lesson):
  anomaly quantiles come from high-purity stage-8 true tracks, not
  clean truth. Score 1 at/below p50, 0 at/above p99.
- Calibration dates 2022-06-06; thresholds 3, 6, 9, 12 dB; eligibility target_fraction >= 0.95 and purity >= 0.95.
- Two variants: **reconstruction** (recon error only) and **elbo** (recon + beta_score * KL, beta_score 0.001).

| variant | error_p50 | error_p99 | n_calibration_tracks | n_calibration_windows |
|---:|---:|---:|---:|---:|
| reconstruction | 0.3207 | 2.9482 | 60,763 | 652,982 |
| elbo | 0.4435 | 3.2735 | 60,763 | 652,982 |

## Results

Best stage-12 model for comparison: **mlp_dae** (max mean false-reduction subject to mean true-retention >= 0.95).

| threshold_db | stage08_true_tracks | stage08_false_tracks | stage09_true_retention | stage09_false_reduction | stage11_true_retention | stage11_false_reduction | stage12_model | stage12_calibration_mode | stage12_true_retention | stage12_false_reduction | stage13_variant | stage13_true_retention | stage13_false_reduction | stage13_precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 20,330 | 164 | 0.9735 | 0.7879 | 1 | 0.0065 | mlp_dae | track_purity | 0.9874 | 1 | elbo | 0.9830 | 0.9817 | 0.9998 |
| 0 | 20,897 | 16 | 0.9854 | 0.6365 | 1 | 0.0006 | mlp_dae | track_purity | 0.9813 | 0.9375 | elbo | 0.9768 | 0.9375 | 1 |
| 3 | 19,317 | 6 | 0.9844 | 0.5524 | 1 | 0 | mlp_dae | track_purity | 0.9713 | 1 | elbo | 0.9666 | 1 | 1 |
| 6 | 16,564 | 1 | 0.9818 | 0.4781 | 1 | 0 | mlp_dae | track_purity | 0.9711 | 1 | elbo | 0.9690 | 1 | 1 |
| 9 | 13,776 | 0 | 0.9764 | 0.4231 | 1 | 0 | mlp_dae | track_purity | 0.9660 | nan | elbo | 0.9659 | nan | 1 |
| 12 | 11,244 | 0 | 0.9712 | 0.3878 | 1 | 0 | mlp_dae | track_purity | 0.9609 | nan | elbo | 0.9618 | nan | 1 |
| -5 | 20,330 | 164 | 0.9735 | 0.7879 | 1 | 0.0065 | mlp_dae | track_purity | 0.9874 | 1 | reconstruction | 0.9830 | 0.9817 | 0.9998 |
| 0 | 20,897 | 16 | 0.9854 | 0.6365 | 1 | 0.0006 | mlp_dae | track_purity | 0.9813 | 0.9375 | reconstruction | 0.9762 | 0.9375 | 1 |
| 3 | 19,317 | 6 | 0.9844 | 0.5524 | 1 | 0 | mlp_dae | track_purity | 0.9713 | 1 | reconstruction | 0.9667 | 1 | 1 |
| 6 | 16,564 | 1 | 0.9818 | 0.4781 | 1 | 0 | mlp_dae | track_purity | 0.9711 | 1 | reconstruction | 0.9684 | 1 | 1 |
| 9 | 13,776 | 0 | 0.9764 | 0.4231 | 1 | 0 | mlp_dae | track_purity | 0.9660 | nan | reconstruction | 0.9657 | nan | 1 |
| 12 | 11,244 | 0 | 0.9712 | 0.3878 | 1 | 0 | mlp_dae | track_purity | 0.9609 | nan | reconstruction | 0.9624 | nan | 1 |

## VAE vs deterministic autoencoders

- **reconstruction**: mean false reduction 0.980, mean true retention 0.970.
- **elbo**: mean false reduction 0.980, mean true retention 0.971.
- The VAE does not beat stage 12 (mlp_dae: mean reduction 0.984, retention 0.973); **deterministic autoencoders remain the stronger baseline.**
- ELBO vs reconstruction-only: compare the two variant rows above -- when KL adds no separation, the reconstruction term dominates and the variants are near-identical.

## Range-bin behavior

| variant | range_bin | tracks | true_retention | false_reduction | median_vae_score |
|---:|---:|---:|---:|---:|---:|
| elbo | 0-50 km | 64,081 | 0.9879 | 0.9798 | 1 |
| elbo | 50-100 km | 36,927 | 0.8943 | nan | 0.8704 |
| elbo | 100-200 km | 1,307 | 0.7965 | nan | 0.7523 |
| reconstruction | 0-50 km | 64,081 | 0.9880 | 0.9798 | 1 |
| reconstruction | 50-100 km | 36,927 | 0.8974 | nan | 0.8722 |
| reconstruction | 100-200 km | 1,307 | 0.7969 | nan | 0.7535 |

## Latent diagnostics

| variant | threshold_db | is_true_track | n_windows | mu_norm_p50 | mu_norm_p90 | mu_dim_std_mean | kl_p50 | kl_p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elbo | -5 | False | 232 | 20.9852 | 35.3268 | 0.6631 | 404.9543 | 866.3610 |
| elbo | -5 | True | 330,669 | 5.6618 | 9.7417 | 1.1500 | 126.9707 | 173.3099 |
| elbo | 0 | False | 22 | 18.6522 | 34.6172 | 0.2229 | 312.9001 | 757.6285 |
| elbo | 0 | True | 273,217 | 5.9329 | 10.3294 | 1.0549 | 127.9519 | 180.0948 |
| elbo | 3 | False | 7 | 21.1989 | 47.8740 | 0.4363 | 461.1920 | 1307.1326 |
| elbo | 3 | True | 226,710 | 6.0650 | 10.6879 | 1.0191 | 127.4623 | 183.5335 |
| elbo | 6 | False | 1 | 24.0104 | 24.0104 | 0 | 432.2496 | 432.2496 |
| elbo | 6 | True | 180,403 | 6.0731 | 10.7219 | 0.9700 | 125.1394 | 181.4381 |
| elbo | 9 | True | 139,850 | 6.0279 | 10.6676 | 0.9230 | 122.1475 | 179.1686 |
| elbo | 12 | True | 106,331 | 6.0479 | 10.8595 | 0.9176 | 119.7472 | 180.9487 |
| reconstruction | -5 | False | 232 | 20.9852 | 35.3268 | 0.6631 | 404.9543 | 866.3610 |
| reconstruction | -5 | True | 330,669 | 5.6618 | 9.7417 | 1.1500 | 126.9707 | 173.3099 |
| reconstruction | 0 | False | 22 | 18.6522 | 34.6172 | 0.2229 | 312.9001 | 757.6285 |
| reconstruction | 0 | True | 273,217 | 5.9329 | 10.3294 | 1.0549 | 127.9519 | 180.0948 |
| reconstruction | 3 | False | 7 | 21.1989 | 47.8740 | 0.4363 | 461.1920 | 1307.1326 |
| reconstruction | 3 | True | 226,710 | 6.0650 | 10.6879 | 1.0191 | 127.4623 | 183.5335 |
| reconstruction | 6 | False | 1 | 24.0104 | 24.0104 | 0 | 432.2496 | 432.2496 |
| reconstruction | 6 | True | 180,403 | 6.0731 | 10.7219 | 0.9700 | 125.1394 | 181.4381 |
| reconstruction | 9 | True | 139,850 | 6.0279 | 10.6676 | 0.9230 | 122.1475 | 179.1686 |
| reconstruction | 12 | True | 106,331 | 6.0479 | 10.8595 | 0.9176 | 119.7472 | 180.9487 |

## Failure modes

- Posterior collapse is possible (KL -> 0); then the VAE degenerates
  toward a plain autoencoder and the latent carries no structure.
- KL may not improve anomaly scoring over reconstruction error alone.
- Reconstruction error may still dominate the ELBO-like score.
- The VAE may smooth rare but valid maneuvers and reject them.
- Calibration remains essential: clean-truth quantiles under-retain
  noisy tracks exactly as in stage 12.

## Recommended next stage

Stage 14 should test diffusion or a broader model-zoo benchmark,
depending on whether the VAE adds value over Stage 12.5.
