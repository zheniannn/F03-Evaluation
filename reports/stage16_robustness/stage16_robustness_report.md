# Stage 16 Robustness and Ablation Study

## Status

- **Stage 16 adds no new model** and retrains nothing; it consolidates
  robustness and ablation evidence for the current best method.
- It focuses on **Stage 12.5** deterministic sequence autoencoders with
  noise-matched (track-purity) calibration.
- It uses available compact reports and only runs missing scoring if
  explicitly requested (`--run-missing`).
- Results are limited to the dates/thresholds available: dates 2022-06-06; thresholds -5, 0, 3, 6.

## Input coverage

| source | date | threshold_db | method_or_file | available | rows | notes |
|---:|---:|---:|---:|---:|---:|---:|
| Stage 08 metrics | 2022-06-06 | -5,0,3,6,9,12 | kalman_metrics_by_day.csv | True | 6 |  |
| Stage 09 metrics | - | -5,0,3,6,9,12 | physics_metrics_by_threshold.csv | True | 6 |  |
| Stage 12 metrics | - | -5,0,3,6,9,12 | sequence_metrics_by_model_threshold.csv | True | 18 |  |
| Stage 12 calibration comparison | - | -5,0,3,6,9,12 | sequence_calibration_comparison.csv | True | 36 |  |
| Stage 12 range bins | - | -5,0,3,6,9,12 | sequence_range_bin_metrics.csv | True | 45 |  |
| Stage 12 filter sweep | - | -5,0,3,6,9,12 | sequence_filter_sweep.csv | True | 162 |  |
| Stage 14 benchmark | - | -5,0,3,6,9,12 | best_method_by_threshold.csv | True | 6 |  |
| Stage 14 failure cases | 2022-06-06 | -5,0 | failure_case_candidates.csv | True | 81 |  |
| Stage 08 large track files | - | - | /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/tracks_kalman | True | 6 | git-ignored; presence only |

- Dates available: **2022-06-06**.
- **Limitation:** stage-12.5 scores currently cover a single day (2022-06-06); multi-day robustness is not yet established.

## Robustness by day

| date | model | calibration_mode | thresholds_included | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | true_track_retention | false_track_reduction | precision_before | precision_after | n_thresholds | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | mlp_dae | track_purity | -5,0,3,6 | 77,108 | 187 | 75,429 | 1 | 0.9782 | 0.9947 | 0.9976 | 1 | 4 | SINGLE-DAY evidence only -- robustness across days is not yet established |

This is **single-day evidence**; the conclusion below is strong for 2022-06-06 but multi-day confirmation is still required.

## Robustness by threshold

Informative thresholds (-5/0/3/6 dB); 9/12 dB carry the windowability
caveat (see the audit).

| threshold_db | model | calibration_mode | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | true_track_retention | false_track_reduction | precision_before | precision_after | windowability_caveat | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | mlp_dae | track_purity | 20,330 | 164 | 20,074 | 0 | 0.9874 | 1 | 0.9920 | 1 | False |  |
| 0 | mlp_dae | track_purity | 20,897 | 16 | 20,506 | 1 | 0.9813 | 0.9375 | 0.9992 | 1 | False |  |
| 3 | mlp_dae | track_purity | 19,317 | 6 | 18,763 | 0 | 0.9713 | 1 | 0.9997 | 1 | False |  |
| 6 | mlp_dae | track_purity | 16,564 | 1 | 16,086 | 0 | 0.9711 | 1 | 0.9999 | 1 | False |  |

## Model ablation: MLP vs GRU vs TCN

| row_type | model | calibration_mode | threshold_db | true_track_retention | false_track_reduction | precision_after | kept_true_tracks | kept_false_tracks | rank_at_threshold | mean_true_retention | mean_false_reduction | mean_precision_after | overall_rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregate | mlp_dae | track_purity | nan | nan | nan | nan | nan | nan | nan | 0.9778 | 0.9844 | 1 | 1 |
| aggregate | gru_ae | track_purity | nan | nan | nan | nan | nan | nan | nan | 0.9768 | 0.9798 | 1 | 2 |
| aggregate | tcn_ae | track_purity | nan | nan | nan | nan | nan | nan | nan | 0.9339 | 0.9657 | 0.9999 | 3 |

- **mlp_dae** ranks best by mean false reduction; MLP and GRU are close and strong, TCN retains fewer true tracks. Differences between MLP and GRU are small (both are defensible primary choices).

## Calibration ablation

The key stage-12.5 finding, preserved:

| model | threshold_db | calibration_mode | true_track_retention | false_track_reduction | precision_after | median_score_true_tracks | median_score_false_tracks | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru_ae | -5 | clean_truth | 0.0787 | 1 | 1 | 0 | 0 | clean-truth under-retains noisy true tracks |
| gru_ae | -5 | track_purity | 0.9851 | 0.9817 | 0.9999 | 0.9548 | 0 | track-purity fixes the noisy-track domain shift |
| gru_ae | 0 | clean_truth | 0.0819 | 1 | 1 | 0 | 0 | clean-truth under-retains noisy true tracks |
| gru_ae | 0 | track_purity | 0.9802 | 0.9375 | 1 | 0.9588 | 0 | track-purity fixes the noisy-track domain shift |
| gru_ae | 3 | clean_truth | 0.0929 | 1 | 1 | 0 | 0 | clean-truth under-retains noisy true tracks |
| gru_ae | 3 | track_purity | 0.9704 | 1 | 1 | 0.9674 | 0 | track-purity fixes the noisy-track domain shift |
| gru_ae | 6 | clean_truth | 0.1241 | 1 | 1 | 0.0728 | 0 | clean-truth under-retains noisy true tracks |
| gru_ae | 6 | track_purity | 0.9714 | 1 | 1 | 0.9807 | 0.1031 | track-purity fixes the noisy-track domain shift |
| mlp_dae | -5 | clean_truth | 0.1104 | 1 | 1 | 0.1145 | 0 | clean-truth under-retains noisy true tracks |
| mlp_dae | -5 | track_purity | 0.9874 | 1 | 1 | 0.9711 | 0 | track-purity fixes the noisy-track domain shift |
| mlp_dae | 0 | clean_truth | 0.1161 | 1 | 1 | 0.1223 | 0 | clean-truth under-retains noisy true tracks |
| mlp_dae | 0 | track_purity | 0.9813 | 0.9375 | 1 | 0.9722 | 0 | track-purity fixes the noisy-track domain shift |
| mlp_dae | 3 | clean_truth | 0.1390 | 1 | 1 | 0.1490 | 0 | clean-truth under-retains noisy true tracks |
| mlp_dae | 3 | track_purity | 0.9713 | 1 | 1 | 0.9759 | 0 | track-purity fixes the noisy-track domain shift |
| mlp_dae | 6 | clean_truth | 0.1818 | 1 | 1 | 0.2055 | 0 | clean-truth under-retains noisy true tracks |
| mlp_dae | 6 | track_purity | 0.9711 | 1 | 1 | 0.9839 | 0 | track-purity fixes the noisy-track domain shift |
| tcn_ae | -5 | clean_truth | 0 | 1 | nan | 0 | 0 | clean-truth under-retains noisy true tracks |
| tcn_ae | -5 | track_purity | 0.9451 | 0.9878 | 0.9999 | 0.9239 | 0 | track-purity fixes the noisy-track domain shift |
| tcn_ae | 0 | clean_truth | 0 | 1 | nan | 0 | 0 | clean-truth under-retains noisy true tracks |
| tcn_ae | 0 | track_purity | 0.9267 | 0.8750 | 0.9999 | 0.9276 | 0 | track-purity fixes the noisy-track domain shift |
| tcn_ae | 3 | clean_truth | 0 | 1 | nan | 0 | 0 | clean-truth under-retains noisy true tracks |
| tcn_ae | 3 | track_purity | 0.9249 | 1 | 1 | 0.9427 | 0.0038 | track-purity fixes the noisy-track domain shift |
| tcn_ae | 6 | clean_truth | 0 | 1 | nan | 0 | 0 | clean-truth under-retains noisy true tracks |
| tcn_ae | 6 | track_purity | 0.9388 | 1 | 1 | 0.9646 | 0.2473 | track-purity fixes the noisy-track domain shift |

- Mean true-track retention rises from **0.077** (clean-truth) to **0.963** (track-purity) at score threshold 0.5: **clean-truth calibration
  is mismatched; track-purity calibration fixes the noisy-track domain shift.**

## Score-threshold sensitivity

Recommended operating points are flagged in `score_threshold_sensitivity.csv` (target-retention 0.97; max false reduction
at retention >= 0.95; max balanced utility).

| threshold_db | score_threshold | true_track_retention | false_track_reduction | precision_after | is_target_retention_point | is_max_false_reduction_at_retention_floor | is_max_utility |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 0.5000 | 0.9874 | 1 | 1 | False | True | True |
| -5 | 0.6000 | 0.9810 | 1 | 1 | True | False | False |
| 0 | 0.2000 | 0.9941 | 0.9375 | 1 | False | True | True |
| 0 | 0.6000 | 0.9701 | 0.9375 | 1 | True | False | False |
| 3 | 0.3000 | 0.9841 | 1 | 1 | False | True | True |
| 3 | 0.5000 | 0.9713 | 1 | 1 | True | False | False |
| 6 | 0.1000 | 0.9896 | 1 | 1 | False | True | True |
| 6 | 0.5000 | 0.9711 | 1 | 1 | True | False | False |

## Windowability audit

Sequence methods only score tracks long enough to window (>= window_len
points, >= 5 hits). This is why high-threshold false-reduction comparisons
are not apples-to-apples with shorter-track methods (the stage-14 caveat).

| threshold_db | stage08_confirmed_tracks | stage08_true_tracks | stage08_false_tracks | windowable_tracks | windowable_true_tracks | windowable_false_tracks | windowable_fraction_all | windowable_fraction_true | windowable_fraction_false | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 31,626 | 22,796 | 1,545 | 20,494 | 20,330 | 164 | 0.8420 | 0.8918 | 0.1061 |  |
| 0 | 32,158 | 25,790 | 252 | 20,913 | 20,897 | 16 | 0.8030 | 0.8103 | 0.0635 |  |
| 3 | 33,826 | 25,879 | 86 | 19,323 | 19,317 | 6 | 0.7442 | 0.7464 | 0.0698 |  |
| 6 | 31,927 | 23,310 | 11 | 16,565 | 16,564 | 1 | 0.7103 | 0.7106 | 0.0909 |  |
| 9 | 27,767 | 19,645 | 1 | 13,776 | 13,776 | 0 | 0.7012 | 0.7012 | 0 | high threshold: windowable false-track count near zero |
| 12 | 22,890 | 16,033 | 0 | 11,244 | 11,244 | 0 | 0.7013 | 0.7013 | nan | high threshold: windowable false-track count near zero |

## Range-bin robustness

| model | threshold_db | range_bin | stage08_tracks | stage08_true_tracks | stage08_false_tracks | stage12_kept_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | true_track_retention | false_track_reduction | precision_after | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_dae | -5 | 0-50 km | 9,070 | 8,906 | 164 | 8,868 | 8,868 | 0 | 0.9957 | 1 | 1 |  |
| mlp_dae | -5 | 100-200 km | 1,257 | 1,257 | 0 | 1,098 | 1,098 | 0 | 0.8735 | nan | 1 |  |
| mlp_dae | -5 | 50-100 km | 10,167 | 10,167 | 0 | 10,108 | 10,108 | 0 | 0.9942 | nan | 1 |  |
| mlp_dae | 0 | 0-50 km | 9,631 | 9,615 | 16 | 9,587 | 9,586 | 1 | 0.9970 | 0.9375 | 0.9999 |  |
| mlp_dae | 0 | 100-200 km | 50 | 50 | 0 | 41 | 41 | 0 | 0.8200 | nan | 1 |  |
| mlp_dae | 0 | 50-100 km | 11,232 | 11,232 | 0 | 10,879 | 10,879 | 0 | 0.9686 | nan | 1 |  |
| mlp_dae | 3 | 0-50 km | 10,377 | 10,371 | 6 | 10,322 | 10,322 | 0 | 0.9953 | 1 | 1 |  |
| mlp_dae | 3 | 100-200 km | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan | nan |  |
| mlp_dae | 3 | 50-100 km | 8,946 | 8,946 | 0 | 8,441 | 8,441 | 0 | 0.9436 | nan | 1 |  |
| mlp_dae | 6 | 0-50 km | 11,522 | 11,521 | 1 | 11,458 | 11,458 | 0 | 0.9945 | 1 | 1 |  |
| mlp_dae | 6 | 50-100 km | 5,043 | 5,043 | 0 | 4,628 | 4,628 | 0 | 0.9177 | nan | 1 |  |

## Failure modes

| case_type | threshold_db | count | median_range_m | median_purity | median_target_fraction | notes |
|---:|---:|---:|---:|---:|---:|---:|
| false_survives_s09_rejected_s12 | -5 | 15 | 1674.6000 | 0 | 0 |  |
| false_survives_s09_rejected_s12 | 0 | 5 | 1161.5000 | 0 | 0 |  |
| false_survives_s12 | 0 | 1 | 5792.4000 | 0.7838 | 0.7838 |  |
| true_kept_s12_rejected_vae | -5 | 20 | 110,860 | 1 | 1 |  |
| true_rejected_s12 | -5 | 20 | 113,705 | 1 | 1 |  |
| vae_latent_outlier_true | -5 | 20 | 108,780 | 1 | 1 |  |

- False tracks that survive Stage 12.5 are **rare but important** to inspect.
- True tracks rejected by Stage 12.5 should be reviewed before deployment.
- Failures tend to be unusual maneuvers, short tracks, or calibration edge cases.

## Consolidated conclusion

- **Stage 12.5 remains the recommended primary false-track filter.**
- **Stage 09 hand physics remains the recommended interpretable fallback.**
- **Stage 15 diffusion** is useful for regularization / gap filling but **not**
  the primary false-track filter.
- More model work should be justified by a **remaining gap**, not by adding
  complexity.

## Recommended next stage

1. **run the same robustness study across all four days (single-day evidence today)**, or
2. perform final report / paper packaging, or
3. run a compact model-zoo only for specific remaining gaps.
