# Stage 14 Unified Method Benchmark

## Status

- **Stage 14 does not introduce a new model** and retrains nothing;
  it consolidates Stages 07, 08, 09, 11, 12.5, and 13.
- Results are for 2022-06-06 unless more dates are processed.
- The goal is **operating-point selection** before diffusion or
  broader model-zoo work (Stage 15).
- 11 track-level methods (operating points across score thresholds) were unified.

## Methods compared

- threshold-only (Stage 07) -- frame-level context only, not a track filter
- Kalman only (Stage 08) -- no-filter track baseline
- hand physics (Stage 09)
- ADS-B marginal priors (Stage 11)
- deterministic sequence autoencoders (Stage 12.5)
- VAE prior (Stage 13)

## Input inventory

| stage | method_family | method_id | source_file | available | rows_loaded | notes |
|---:|---:|---:|---:|---:|---:|---:|
| 07 | threshold_only | threshold_only | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage07_threshold_only/threshold_overall.csv | True | 6 | frame-level context only; not a track filter |
| 08 | kalman_only | kalman_only | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage08_kalman_baseline/kalman_metrics_by_day.csv | True | 6 | no-filter baseline |
| 09 | hand_physics | stage09_hand_physics | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage09_physics_scoring/physics_filter_sweep.csv | True | 54 |  |
| 11 | adsb_marginal_prior | stage11_adsb_marginal | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage11_adsb_prior_scoring/adsb_prior_filter_sweep.csv | True | 54 |  |
| 12.5 | sequence_autoencoder | stage12_sequence_autoencoders | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage12_sequence_priors/sequence_filter_sweep.csv | True | 162 |  |
| 12.5 | sequence_autoencoder | stage12_clean_truth | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage12_sequence_priors/calibration/sequence_calibration_comparison.csv | True | 36 | clean-truth calibration (for contrast) |
| 13 | vae_prior | stage13_vae_prior | /home/tzhen/projects/PLSWORK/F03-Evaluation/reports/stage13_vae_prior/vae_filter_sweep.csv | True | 108 |  |

## Unified operating curves

Per-method true-track retention, false-track reduction, and precision
after filtering are in `unified_method_metrics.csv` (402 operating-point rows).

## Best method by threshold

| threshold_db | best_method_id | best_method_family | score_threshold | true_track_retention | false_track_reduction | precision_after | kept_true_tracks | kept_false_tracks | selection_rule |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.7000 | 0.9627 | 1 | 1 | 19,572 | 0 | retention>=0.95, max false_reduction |
| 0 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9941 | 0.9375 | 1 | 20,773 | 1 | retention>=0.95, max false_reduction |
| 3 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9885 | 1 | 1 | 19,094 | 0 | retention>=0.95, max false_reduction |
| 6 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9867 | 1 | 1 | 16,343 | 0 | retention>=0.95, max false_reduction |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.7000 | 0.9608 | 0.5577 | 0.9983 | 26,579 | 46 | retention>=0.95, max false_reduction |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.7000 | 0.9582 | 0.4898 | 0.9989 | 21,886 | 25 | retention>=0.95, max false_reduction |

> **Denominator caveat.** Each stage reports its own stage-8 true/false-track
> counts, and the learned stages (12.5, 13) only score tracks long enough to
> window (>= window_len points, >= 5 hits), so at high detection thresholds
> (9/12 dB) their windowable false-track count is ~0 while the physics/ADS-B
> stages still count short false tracks. Cross-stage false-reduction is
> therefore only apples-to-apples where every method sees a comparable false
> population -- i.e. the informative low/mid thresholds (-5, 0, 3, 6 dB). At
> 9/12 dB the 'best method' collapses to whichever stage still has false tracks
> to remove; read those rows with care.

- **Best overall method:** stage12_mlp_dae_track_calibrated
- **Best low-threshold method:** stage12_gru_ae_track_calibrated
- **Best interpretable method:** stage09_hand_physics
- **Best learned method:** stage12_mlp_dae_track_calibrated

## Matched-retention comparison

At similar true-track retention (target 0.97), which method removes the
most false tracks?

| threshold_db | method_id | method_family | score_threshold | target_retention | true_track_retention | false_track_reduction | precision_after | retention_gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9985 | 0.2980 | 0.8552 | 0.0285 |
| -5 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9735 | 0.7879 | 0.9501 | 0.0035 |
| -5 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.8063 | 0.0300 |
| -5 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.0787 | 1 | 1 | -0.8913 |
| -5 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.6000 | 0.9700 | 0.9777 | 0.9878 | 0.9999 | 0.0077 |
| -5 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1104 | 1 | 1 | -0.8596 |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.6000 | 0.9700 | 0.9810 | 1 | 1 | 0.0110 |
| -5 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0 | 1 | nan | -0.9700 |
| -5 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9700 | 0.9733 | 0.9268 | 0.9994 | 0.0033 |
| -5 | stage13_vae_elbo | vae_prior | 0.6000 | 0.9700 | 0.9752 | 1 | 1 | 0.0052 |
| -5 | stage13_vae_reconstruction | vae_prior | 0.6000 | 0.9700 | 0.9748 | 1 | 1 | 0.0048 |
| 0 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9991 | 0.1758 | 0.9585 | 0.0291 |
| 0 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9854 | 0.6365 | 0.9810 | 0.0154 |
| 0 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.9509 | 0.0300 |
| 0 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.0819 | 1 | 1 | -0.8881 |
| 0 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9700 | 0.9802 | 0.9375 | 1 | 0.0102 |
| 0 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1161 | 1 | 1 | -0.8539 |
| 0 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.6000 | 0.9700 | 0.9701 | 0.9375 | 1 | 0.0001 |
| 0 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0 | 1 | nan | -0.9700 |
| 0 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9700 | 0.9714 | 0.6875 | 0.9998 | 0.0014 |
| 0 | stage13_vae_elbo | vae_prior | 0.5000 | 0.9700 | 0.9768 | 0.9375 | 1 | 0.0068 |
| 0 | stage13_vae_reconstruction | vae_prior | 0.5000 | 0.9700 | 0.9762 | 0.9375 | 1 | 0.0062 |
| 3 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9987 | 0.1244 | 0.9829 | 0.0287 |
| 3 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9844 | 0.5524 | 0.9910 | 0.0144 |
| 3 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.9807 | 0.0300 |
| 3 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.0929 | 1 | 1 | -0.8771 |
| 3 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9700 | 0.9704 | 1 | 1 | 0.0004 |
| 3 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1390 | 1 | 1 | -0.8310 |
| 3 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9700 | 0.9713 | 1 | 1 | 0.0013 |
| 3 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0 | 1 | nan | -0.9700 |
| 3 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9700 | 0.9748 | 0.6667 | 0.9999 | 0.0048 |
| 3 | stage13_vae_elbo | vae_prior | 0.4000 | 0.9700 | 0.9768 | 1 | 1 | 0.0068 |
| 3 | stage13_vae_reconstruction | vae_prior | 0.4000 | 0.9700 | 0.9766 | 1 | 1 | 0.0066 |
| 6 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9979 | 0.1277 | 0.9925 | 0.0279 |
| 6 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9818 | 0.4781 | 0.9954 | 0.0118 |
| 6 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.9904 | 0.0300 |
| 6 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1241 | 1 | 1 | -0.8459 |
| 6 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9700 | 0.9714 | 1 | 1 | 0.0014 |
| 6 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1818 | 1 | 1 | -0.7882 |
| 6 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9700 | 0.9711 | 1 | 1 | 0.0011 |
| 6 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0 | 1 | nan | -0.9700 |
| 6 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9700 | 0.9744 | 0 | 0.9999 | 0.0044 |
| 6 | stage13_vae_elbo | vae_prior | 0.4000 | 0.9700 | 0.9762 | 1 | 1 | 0.0062 |
| 6 | stage13_vae_reconstruction | vae_prior | 0.4000 | 0.9700 | 0.9762 | 1 | 1 | 0.0062 |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9969 | 0.1250 | 0.9967 | 0.0269 |
| 9 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9764 | 0.4231 | 0.9978 | 0.0064 |
| 9 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.9940 | 0.0300 |
| 9 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.1757 | nan | 1 | -0.7943 |
| 9 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.4000 | 0.9700 | 0.9758 | nan | 1 | 0.0058 |
| 9 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.2445 | nan | 1 | -0.7255 |
| 9 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.4000 | 0.9700 | 0.9744 | nan | 1 | 0.0044 |
| 9 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.0001 | nan | 1 | -0.9699 |
| 9 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9700 | 0.9704 | nan | 1 | 0.0004 |
| 9 | stage13_vae_elbo | vae_prior | 0.4000 | 0.9700 | 0.9735 | nan | 1 | 0.0035 |
| 9 | stage13_vae_reconstruction | vae_prior | 0.4000 | 0.9700 | 0.9729 | nan | 1 | 0.0029 |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9700 | 0.9958 | 0.0408 | 0.9979 | 0.0258 |
| 12 | stage09_hand_physics | hand_physics | 0.5000 | 0.9700 | 0.9712 | 0.3878 | 0.9986 | 0.0012 |
| 12 | kalman_only | kalman_only | nan | 0.9700 | 1 | 0 | 0.9946 | 0.0300 |
| 12 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.2479 | nan | 1 | -0.7221 |
| 12 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.4000 | 0.9700 | 0.9705 | nan | 1 | 0.0005 |
| 12 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0.3106 | nan | 1 | -0.6594 |
| 12 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9700 | 0.9747 | nan | 1 | 0.0047 |
| 12 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9700 | 0 | nan | nan | -0.9700 |
| 12 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9700 | 0.9764 | nan | 1 | 0.0064 |
| 12 | stage13_vae_elbo | vae_prior | 0.3000 | 0.9700 | 0.9748 | nan | 1 | 0.0048 |
| 12 | stage13_vae_reconstruction | vae_prior | 0.3000 | 0.9700 | 0.9747 | nan | 1 | 0.0047 |

## Matched-false-reduction comparison

At similar false-track reduction (target 0.95), which method retains the
most true tracks?

| threshold_db | method_id | method_family | score_threshold | target_false_reduction | true_track_retention | false_track_reduction | precision_after | false_reduction_gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.7770 | 0.9963 | 0.9988 | 0.0463 |
| -5 | stage09_hand_physics | hand_physics | 0.7000 | 0.9500 | 0.8355 | 0.9770 | 0.9934 | 0.0270 |
| -5 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.8063 | -0.9500 |
| -5 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.0787 | 1 | 1 | 0.0500 |
| -5 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9500 | 0.9942 | 0.9512 | 0.9996 | 0.0012 |
| -5 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.1104 | 1 | 1 | 0.0500 |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9500 | 0.9965 | 0.9573 | 0.9997 | 0.0073 |
| -5 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0 | 1 | nan | 0.0500 |
| -5 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.4000 | 0.9500 | 0.9586 | 0.9756 | 0.9998 | 0.0256 |
| -5 | stage13_vae_elbo | vae_prior | 0.2000 | 0.9500 | 0.9934 | 0.9573 | 0.9997 | 0.0073 |
| -5 | stage13_vae_reconstruction | vae_prior | 0.2000 | 0.9500 | 0.9932 | 0.9573 | 0.9997 | 0.0073 |
| 0 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.7293 | 0.9800 | 0.9986 | 0.0300 |
| 0 | stage09_hand_physics | hand_physics | 0.8000 | 0.9500 | 0.6513 | 0.9882 | 0.9990 | 0.0382 |
| 0 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.9509 | -0.9500 |
| 0 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.0819 | 1 | 1 | 0.0500 |
| 0 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9500 | 0.9901 | 0.9375 | 1 | -0.0125 |
| 0 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.1161 | 1 | 1 | 0.0500 |
| 0 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9500 | 0.9941 | 0.9375 | 1 | -0.0125 |
| 0 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0 | 1 | nan | 0.0500 |
| 0 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.6000 | 0.9500 | 0.8961 | 0.9375 | 0.9999 | -0.0125 |
| 0 | stage13_vae_elbo | vae_prior | 0.4000 | 0.9500 | 0.9845 | 0.9375 | 1 | -0.0125 |
| 0 | stage13_vae_reconstruction | vae_prior | 0.4000 | 0.9500 | 0.9843 | 0.9375 | 1 | -0.0125 |
| 3 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.6934 | 0.9712 | 0.9992 | 0.0212 |
| 3 | stage09_hand_physics | hand_physics | 0.8000 | 0.9500 | 0.5980 | 0.9788 | 0.9993 | 0.0288 |
| 3 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.9807 | -0.9500 |
| 3 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.0929 | 1 | 1 | 0.0500 |
| 3 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9500 | 0.9885 | 1 | 1 | 0.0500 |
| 3 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.1390 | 1 | 1 | 0.0500 |
| 3 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9500 | 0.9841 | 1 | 1 | 0.0500 |
| 3 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0 | 1 | nan | 0.0500 |
| 3 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9500 | 0.9682 | 1 | 1 | 0.0500 |
| 3 | stage13_vae_elbo | vae_prior | 0.3000 | 0.9500 | 0.9819 | 1 | 1 | 0.0500 |
| 3 | stage13_vae_reconstruction | vae_prior | 0.4000 | 0.9500 | 0.9766 | 1 | 1 | 0.0500 |
| 6 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.6887 | 0.9599 | 0.9995 | 0.0099 |
| 6 | stage09_hand_physics | hand_physics | 0.8000 | 0.9500 | 0.5832 | 0.9672 | 0.9995 | 0.0172 |
| 6 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.9904 | -0.9500 |
| 6 | stage12_gru_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.1241 | 1 | 1 | 0.0500 |
| 6 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9500 | 0.9867 | 1 | 1 | 0.0500 |
| 6 | stage12_mlp_dae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0.1818 | 1 | 1 | 0.0500 |
| 6 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9500 | 0.9896 | 1 | 1 | 0.0500 |
| 6 | stage12_tcn_ae_clean_truth | sequence_autoencoder | 0.5000 | 0.9500 | 0 | 1 | nan | 0.0500 |
| 6 | stage12_tcn_ae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9500 | 0.9666 | 1 | 1 | 0.0500 |
| 6 | stage13_vae_elbo | vae_prior | 0.1000 | 0.9500 | 0.9891 | 1 | 1 | 0.0500 |
| 6 | stage13_vae_reconstruction | vae_prior | 0.1000 | 0.9500 | 0.9890 | 1 | 1 | 0.0500 |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.6965 | 0.9519 | 0.9997 | 0.0019 |
| 9 | stage09_hand_physics | hand_physics | 0.8000 | 0.9500 | 0.5750 | 0.9615 | 0.9997 | 0.0115 |
| 9 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.9940 | -0.9500 |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.9500 | 0.7161 | 0.9592 | 0.9999 | 0.0092 |
| 12 | stage09_hand_physics | hand_physics | 0.8000 | 0.9500 | 0.5863 | 0.9796 | 0.9999 | 0.0296 |
| 12 | kalman_only | kalman_only | nan | 0.9500 | 1 | 0 | 0.9946 | -0.9500 |

## Pareto frontier

Non-dominated operating points (a point is dominated if another has >=
retention AND >= false reduction, with at least one strictly greater):

| threshold_db | method_id | method_family | score_threshold | true_track_retention | false_track_reduction | precision_after | is_pareto |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.5000 | 0.9874 | 1 | 1 | True |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.4000 | 0.9915 | 0.9939 | 1 | True |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.3000 | 0.9939 | 0.9817 | 0.9999 | True |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9955 | 0.9695 | 0.9998 | True |
| -5 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9965 | 0.9573 | 0.9997 | True |
| -5 | stage09_hand_physics | hand_physics | 0.4000 | 0.9985 | 0.5469 | 0.9015 | True |
| -5 | stage09_hand_physics | hand_physics | 0.3000 | 0.9999 | 0.2872 | 0.8535 | True |
| -5 | stage09_hand_physics | hand_physics | 0.2000 | 1 | 0.0249 | 0.8098 | True |
| 0 | stage09_hand_physics | hand_physics | 0.9000 | 0.4935 | 1 | 1 | True |
| 0 | stage09_hand_physics | hand_physics | 0.8000 | 0.6513 | 0.9882 | 0.9990 | True |
| 0 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.7293 | 0.9800 | 0.9986 | True |
| 0 | stage09_hand_physics | hand_physics | 0.7000 | 0.8329 | 0.9377 | 0.9961 | True |
| 0 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.2000 | 0.9941 | 0.9375 | 1 | True |
| 0 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9955 | 0.8750 | 0.9999 | True |
| 0 | stage09_hand_physics | hand_physics | 0.4000 | 0.9991 | 0.3797 | 0.9684 | True |
| 0 | stage09_hand_physics | hand_physics | 0.3000 | 0.9999 | 0.1565 | 0.9576 | True |
| 0 | stage09_hand_physics | hand_physics | 0.2000 | 1 | 0.0069 | 0.9504 | True |
| 3 | stage12_gru_ae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9885 | 1 | 1 | True |
| 3 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9895 | 0.8333 | 0.9999 | True |
| 3 | stage09_hand_physics | hand_physics | 0.4000 | 0.9982 | 0.2959 | 0.9862 | True |
| 3 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9987 | 0.1244 | 0.9829 | True |
| 3 | stage09_hand_physics | hand_physics | 0.3000 | 0.9998 | 0.1168 | 0.9828 | True |
| 3 | stage09_hand_physics | hand_physics | 0.2000 | 1 | 0.0015 | 0.9805 | True |
| 6 | stage12_mlp_dae_track_calibrated | sequence_autoencoder | 0.1000 | 0.9896 | 1 | 1 | True |
| 6 | stage09_hand_physics | hand_physics | 0.4000 | 0.9968 | 0.2263 | 0.9933 | True |
| 6 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9979 | 0.1277 | 0.9925 | True |
| 6 | stage09_hand_physics | hand_physics | 0.3000 | 0.9997 | 0.0730 | 0.9920 | True |
| 6 | stage09_hand_physics | hand_physics | 0.2000 | 1 | 0.0073 | 0.9915 | True |
| 9 | stage09_hand_physics | hand_physics | 0.9000 | 0.3427 | 1 | 1 | True |
| 9 | stage09_hand_physics | hand_physics | 0.8000 | 0.5750 | 0.9615 | 0.9997 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.6965 | 0.9519 | 0.9997 | True |
| 9 | stage09_hand_physics | hand_physics | 0.7000 | 0.8185 | 0.8654 | 0.9994 | True |
| 9 | stage09_hand_physics | hand_physics | 0.6000 | 0.9212 | 0.7115 | 0.9988 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.7000 | 0.9608 | 0.5577 | 0.9983 | True |
| 9 | stage09_hand_physics | hand_physics | 0.5000 | 0.9764 | 0.4231 | 0.9978 | True |
| 9 | stage09_hand_physics | hand_physics | 0.4000 | 0.9952 | 0.2404 | 0.9971 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.6000 | 0.9969 | 0.1250 | 0.9967 | True |
| 9 | stage09_hand_physics | hand_physics | 0.3000 | 0.9993 | 0.0673 | 0.9965 | True |
| 9 | kalman_only | kalman_only | nan | 1 | 0 | 0.9940 | True |
| 9 | stage09_hand_physics | hand_physics | 0.1000 | 1 | 0 | 0.9963 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.1000 | 1 | 0 | 0.9963 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.2000 | 1 | 0 | 0.9963 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.3000 | 1 | 0 | 0.9963 | True |
| 9 | stage11_adsb_marginal | adsb_marginal_prior | 0.4000 | 1 | 0 | 0.9963 | True |
| 12 | stage09_hand_physics | hand_physics | 0.9000 | 0.3413 | 1 | 1 | True |
| 12 | stage09_hand_physics | hand_physics | 0.8000 | 0.5863 | 0.9796 | 0.9999 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.8000 | 0.7161 | 0.9592 | 0.9999 | True |
| 12 | stage09_hand_physics | hand_physics | 0.7000 | 0.8166 | 0.8776 | 0.9997 | True |
| 12 | stage09_hand_physics | hand_physics | 0.6000 | 0.9153 | 0.6122 | 0.9991 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.7000 | 0.9582 | 0.4898 | 0.9989 | True |
| 12 | stage09_hand_physics | hand_physics | 0.5000 | 0.9712 | 0.3878 | 0.9986 | True |
| 12 | stage09_hand_physics | hand_physics | 0.4000 | 0.9924 | 0.1224 | 0.9981 | True |
| 12 | stage09_hand_physics | hand_physics | 0.3000 | 0.9992 | 0.0408 | 0.9979 | True |
| 12 | kalman_only | kalman_only | nan | 1 | 0 | 0.9946 | True |
| 12 | stage09_hand_physics | hand_physics | 0.1000 | 1 | 0 | 0.9979 | True |
| 12 | stage09_hand_physics | hand_physics | 0.2000 | 1 | 0 | 0.9979 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.1000 | 1 | 0 | 0.9979 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.2000 | 1 | 0 | 0.9979 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.3000 | 1 | 0 | 0.9979 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.4000 | 1 | 0 | 0.9979 | True |
| 12 | stage11_adsb_marginal | adsb_marginal_prior | 0.5000 | 1 | 0 | 0.9979 | True |

## Rankings

Descriptive utility = 0.5*retention + 0.5*false_reduction (plus recall- and
precision-weighted variants). **These utility scores are descriptive, not
absolute truth -- operational priorities determine the final method choice.**
Top operating point per threshold by balanced utility:

| threshold_db | method_id | true_track_retention | false_track_reduction | utility |
|---:|---:|---:|---:|---:|
| -5 | stage12_mlp_dae_track_calibrated | 0.9874 | 1 | 0.9937 |
| 0 | stage12_mlp_dae_track_calibrated | 0.9941 | 0.9375 | 0.9658 |
| 3 | stage12_gru_ae_track_calibrated | 0.9885 | 1 | 0.9942 |
| 6 | stage12_mlp_dae_track_calibrated | 0.9896 | 1 | 0.9948 |
| 9 | stage09_hand_physics | 0.8185 | 0.8654 | 0.8419 |
| 12 | stage09_hand_physics | 0.8166 | 0.8776 | 0.8471 |

## Runtime inventory

| stage | method_family | runtime_reported | notes |
|---:|---:|---:|---:|
| 08 | kalman_only | 2m29s; 11m04s; 2m29s | parsed from run_log.md |
| 09 | hand_physics | 3m37s; 5m41s | parsed from run_log.md |
| 11 | adsb_marginal_prior | unknown | no run log parsed |
| 12.5 | sequence_autoencoder | unknown | no run log parsed |
| 13 | vae_prior | unknown | vae_prior_report.md present, no runtime pattern |

## Interpretation

- **Stage 12.5 deterministic sequence autoencoders are currently the
  strongest learned method** for keep/reject scoring.
- **Stage 13 VAE does not improve over the deterministic autoencoders**
  for keep/reject scoring, though its latent diagnostics separate true and
  false tracks.
- **Stage 09 hand physics remains the strongest simple/interpretable
  baseline** and is a close, transparent fallback.
- **Stage 11 empirical marginal priors** are useful evidence but not enough
  alone -- they do not clearly beat Stage 09.
- Future diffusion or model-zoo work should target remaining gaps, not
  repeat solved comparisons.

## Recommended operating point

- **Best conservative / overall method:** stage12_mlp_dae_track_calibrated (highest balanced utility at retention >= 0.95).
- **Best low-threshold method:** stage12_gru_ae_track_calibrated (worst clutter regime, -5 dB).
- **Best interpretable method:** stage09_hand_physics (transparent, no training).
- **Best learned method:** stage12_mlp_dae_track_calibrated.

## Failure cases to inspect

Candidate disagreement cases are in `failure_case_candidates.csv` (false tracks surviving each filter, true tracks wrongly rejected, and VAE latent outliers).

## Next stage

Stage 15 should either:
1. test **diffusion** specifically for denoising / gap filling, or
2. build a broader **model-zoo** benchmark only if there is a clearly defined
   remaining gap.
