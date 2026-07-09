# Stage 15 Diffusion Trajectory Denoising Study

## Status

- Stage 15 trains a lightweight diffusion denoiser over normalized
  trajectory windows.
- Stage 15 is primarily a denoising/gap-filling study, **not a new
  primary false-track classifier**.
- Results are for 2022-06-06 and thresholds -5, 0, 3, 6 unless more data are processed.
- **Stage 12.5 remains the current best false-track suppression method
  unless diffusion clearly beats it.**
- This is not the full model zoo.

## Motivation

- Stage 12.5 already nearly solves false-track suppression (a single
  false track survived it in the stage-14 benchmark).
- Diffusion may still help with noisy-track regularization, gap filling,
  and rare-maneuver reconstruction -- so it is evaluated on those tasks,
  with keep/reject classification kept explicitly secondary.

## Training data

- Source: `data/active/radar_truth_relocated`
- Train dates: 2022-06-06, 2022-06-13, 2022-06-20; holdout: 2022-06-27
- Train windows: 200,000; validation windows: 60,000
- Window length 20 @ stride 5; features dx, dy, dz, vx, vy, vz, speed, vertical_speed, turn_rate.
- Normalizer: stage12:models/sequence_priors/normalizer.json

## Model

- DDPM-style **noise prediction**: corrupt a clean window at a random
  timestep, predict the injected noise, MSE loss.
- Temporal 1D-conv residual denoiser: hidden 128, 4 residual blocks, sinusoidal timestep embedding.
- Noise schedule: 100 steps, linear beta 0.0001..0.02.
- Evaluation uses single-step denoising (Mode A) at denoise_t = 20.

## Validation denoising

Synthetic-noise recovery on clean holdout windows (improvement_ratio = 1 - denoised_mse / noisy_mse):

| denoise_t | n_val_windows | noisy_mse | denoised_mse | improvement_ratio |
|---:|---:|---:|---:|---:|
| 5 | 60,000 | 0.0036 | 0.0009 | 0.7438 |
| 10 | 60,000 | 0.0121 | 0.0023 | 0.8087 |
| 20 | 60,000 | 0.0438 | 0.0060 | 0.8620 |
| 50 | 60,000 | 0.2443 | 0.0222 | 0.9093 |

## True-track denoising

Normalized-space metrics (per-window meter-space truth alignment is out of
scope for v1). RMSE columns measure recovery of synthetic corruption of the
Kalman windows; smoothness columns measure direct regularization of the
actual Kalman posterior windows.

| date | threshold_db | n_tracks | n_windows | kalman_position_rmse_norm | diffusion_position_rmse_norm | rmse_improvement_ratio | kalman_velocity_rmse_norm | diffusion_velocity_rmse_norm | velocity_improvement_ratio | smoothness_before | smoothness_after | smoothness_improvement_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | -5 | 22,796 | 20,000 | 0.2101 | 0.2662 | -0.2673 | 0.2101 | 0.1724 | 0.1794 | 0.3776 | 0.1822 | 0.5174 |
| 2022-06-06 | 0 | 25,790 | 20,000 | 0.2100 | 0.2494 | -0.1876 | 0.2102 | 0.1706 | 0.1881 | 0.3189 | 0.1632 | 0.4882 |
| 2022-06-06 | 3 | 25,879 | 20,000 | 0.2100 | 0.2354 | -0.1212 | 0.2100 | 0.1696 | 0.1925 | 0.2746 | 0.1469 | 0.4653 |
| 2022-06-06 | 6 | 23,310 | 20,000 | 0.2100 | 0.2190 | -0.0431 | 0.2100 | 0.1678 | 0.2009 | 0.2309 | 0.1302 | 0.4362 |

- Mean RMSE improvement ratio: -0.1548 -- diffusion does NOT reduce corruption RMSE on these windows.
- Mean smoothness improvement ratio: 0.4768 (positive = diffusion smooths the Kalman track).

## Gap-filling

Mask timesteps, linearly interpolate, then refine with diffusion; MSE at
the masked steps (normalized space):

| date | threshold_db | gap_mode | gap_fraction | n_windows | interp_mse | diffusion_mse | improvement_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | -5 | block | 0.1500 | 20,000 | 0.9630 | 0.8630 | 0.1038 |
| 2022-06-06 | -5 | random | 0.2000 | 20,000 | 0.8101 | 0.7543 | 0.0689 |
| 2022-06-06 | 0 | block | 0.1500 | 20,000 | 0.9350 | 0.8385 | 0.1032 |
| 2022-06-06 | 0 | random | 0.2000 | 20,000 | 0.7733 | 0.7190 | 0.0702 |
| 2022-06-06 | 3 | block | 0.1500 | 20,000 | 0.9779 | 0.8894 | 0.0906 |
| 2022-06-06 | 3 | random | 0.2000 | 20,000 | 0.8808 | 0.8284 | 0.0595 |
| 2022-06-06 | 6 | block | 0.1500 | 20,000 | 0.8995 | 0.8116 | 0.0977 |
| 2022-06-06 | 6 | random | 0.2000 | 20,000 | 0.7522 | 0.7018 | 0.0670 |

- Mean gap-fill improvement ratio: 0.0826 -- diffusion refinement beats linear interpolation at masked steps.

## Residual score as classifier

Calibration: track-purity on high-purity true tracks (dates 2022-06-06; thresholds 3, 6, 9, 12 dB); p50 0.02814 -> score 1, p99 0.08397 -> score 0.
- Median residual score: true 0.894 vs false 0.550 -- the diffusion residual separates true and false tracks.
- **This is a secondary diagnostic**, not the primary filter.

| threshold_db | stage08_confirmed_tracks | stage08_true_tracks | stage08_false_tracks | kept_tracks | kept_true_tracks | kept_false_tracks | true_track_retention | false_track_reduction | precision_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 20,494 | 20,330 | 164 | 17,948 | 17,853 | 95 | 0.8782 | 0.4207 | 0.9947 |
| 0 | 20,913 | 20,897 | 16 | 17,847 | 17,841 | 6 | 0.8538 | 0.6250 | 0.9997 |
| 3 | 19,323 | 19,317 | 6 | 16,910 | 16,906 | 4 | 0.8752 | 0.3333 | 0.9998 |
| 6 | 16,565 | 16,564 | 1 | 15,027 | 15,027 | 0 | 0.9072 | 1 | 1 |

## Comparison with Stage 12.5 and Stage 13

| threshold_db | stage12_best_model | stage12_true_retention | stage12_false_reduction | stage13_best_variant | stage13_true_retention | stage13_false_reduction | stage15_diffusion_true_retention | stage15_diffusion_false_reduction | stage15_denoise_rmse_improvement | stage15_gap_fill_improvement | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | mlp_dae | 0.9874 | 1 | elbo | 0.9830 | 0.9817 | 0.8782 | 0.4207 | -0.2673 | 0.0863 | diffusion residual is a SECONDARY score |
| 0 | mlp_dae | 0.9813 | 0.9375 | elbo | 0.9768 | 0.9375 | 0.8538 | 0.6250 | -0.1876 | 0.0867 | diffusion residual is a SECONDARY score |
| 3 | mlp_dae | 0.9713 | 1 | elbo | 0.9666 | 1 | 0.8752 | 0.3333 | -0.1212 | 0.0750 | diffusion residual is a SECONDARY score |
| 6 | mlp_dae | 0.9711 | 1 | elbo | 0.9690 | 1 | 0.9072 | 1 | -0.0431 | 0.0824 | diffusion residual is a SECONDARY score |

- As a classifier the diffusion residual (mean false reduction 0.595) does NOT beat the best stage-12.5 model (mean 0.984).

## Runtime and complexity

| component | seconds | notes |
|---:|---:|---:|
| training | nan | see manifest / run log |
| evaluation | nan | single-step Mode A inference |

- A DDPM denoiser is heavier than a deterministic autoencoder (extra
  timestep conditioning and, if used, iterative sampling). Single-step
  Mode A keeps inference cheap.

## Conclusion

- Diffusion is useful diagnostically (it recovers synthetic corruption and can smooth tracks) but does NOT improve enough over the deterministic stage-12.5 autoencoders to replace them as the primary false-track filter.

## Recommended next stage

Stage 16 should be either:
1. a compact **model-zoo benchmark** including only promising families, or
2. a **robustness/ablation study** across all four days and clutter/noise
   levels.
