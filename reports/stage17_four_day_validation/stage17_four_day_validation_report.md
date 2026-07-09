# Stage 17 Four-Day Validation

## Status

- **Stage 17 adds no new model** and retrains nothing.
- It validates the selected **Stage 12.5** method (deterministic sequence
  autoencoder, track-purity calibration) across all available ADS-B days.
- It focuses on the **four-day** validation at thresholds -5/0/3/6 dB.
- 9/12 dB remain **audit-only** due to the windowability caveat.
- It runs missing stages only if explicitly requested (`--run-missing-*`).

## Input availability

Expected four days: 2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27. Days with complete stage-08/09/12 results: **2022-06-06**.

| date | cells | complete |
|---:|---:|---:|
| 2022-06-06 | 4 | 4 |
| 2022-06-13 | 4 | 0 |
| 2022-06-20 | 4 | 0 |
| 2022-06-27 | 4 | 0 |

**Not all four days are available** -- see the run plan for the exact commands to generate the missing outputs.

## Run plan

9 missing stage output(s). Exact commands to generate them (not run unless `--run-missing-*` is passed):

| date | threshold_db | missing_stage | command | will_run | notes |
|---:|---:|---:|---:|---:|---:|
| 2022-06-13 | -5 0 3 6 | stage08 | python scripts/08_run_kalman_baseline.py --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --truth-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/radar_truth_relocated --tracks-dir data/active/tracks_kalman --report-dir reports/stage08_kalman_baseline --threshold-db -5 0 3 6 --date 2022-06-13 --overwrite | False | needs stage-6 detections for this day |
| 2022-06-13 | -5 0 3 6 | stage09 | python scripts/09_score_tracks_physics.py --tracks-dir data/active/tracks_kalman --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --report-dir reports/stage09_physics_scoring --threshold-db -5 0 3 6 --date 2022-06-13 --overwrite | False | needs stage-08 tracks for this day |
| 2022-06-13 | -5 0 3 6 | stage12 | python scripts/12_score_tracks_sequence_prior.py --tracks-dir data/active/tracks_kalman --models-dir models/sequence_priors --report-dir reports/stage12_sequence_priors --threshold-db -5 0 3 6 --date 2022-06-13 --calibration-mode track_purity --calibration-threshold-db 3 6 9 12 --score-threshold 0.5 --overwrite | False | needs stage-08 tracks for this day |
| 2022-06-20 | -5 0 3 6 | stage08 | python scripts/08_run_kalman_baseline.py --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --truth-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/radar_truth_relocated --tracks-dir data/active/tracks_kalman --report-dir reports/stage08_kalman_baseline --threshold-db -5 0 3 6 --date 2022-06-20 --overwrite | False | needs stage-6 detections for this day |
| 2022-06-20 | -5 0 3 6 | stage09 | python scripts/09_score_tracks_physics.py --tracks-dir data/active/tracks_kalman --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --report-dir reports/stage09_physics_scoring --threshold-db -5 0 3 6 --date 2022-06-20 --overwrite | False | needs stage-08 tracks for this day |
| 2022-06-20 | -5 0 3 6 | stage12 | python scripts/12_score_tracks_sequence_prior.py --tracks-dir data/active/tracks_kalman --models-dir models/sequence_priors --report-dir reports/stage12_sequence_priors --threshold-db -5 0 3 6 --date 2022-06-20 --calibration-mode track_purity --calibration-threshold-db 3 6 9 12 --score-threshold 0.5 --overwrite | False | needs stage-08 tracks for this day |
| 2022-06-27 | -5 0 3 6 | stage08 | python scripts/08_run_kalman_baseline.py --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --truth-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/radar_truth_relocated --tracks-dir data/active/tracks_kalman --report-dir reports/stage08_kalman_baseline --threshold-db -5 0 3 6 --date 2022-06-27 --overwrite | False | needs stage-6 detections for this day |
| 2022-06-27 | -5 0 3 6 | stage09 | python scripts/09_score_tracks_physics.py --tracks-dir data/active/tracks_kalman --detections-dir /home/tzhen/projects/PLSWORK/F03-Evaluation/data/active/sim_detections_relocated --report-dir reports/stage09_physics_scoring --threshold-db -5 0 3 6 --date 2022-06-27 --overwrite | False | needs stage-08 tracks for this day |
| 2022-06-27 | -5 0 3 6 | stage12 | python scripts/12_score_tracks_sequence_prior.py --tracks-dir data/active/tracks_kalman --models-dir models/sequence_priors --report-dir reports/stage12_sequence_priors --threshold-db -5 0 3 6 --date 2022-06-27 --calibration-mode track_purity --calibration-threshold-db 3 6 9 12 --score-threshold 0.5 --overwrite | False | needs stage-08 tracks for this day |

## Four-day Stage 08 context

| date | stage08_true_tracks | stage08_false_tracks | stage08_confirmed_tracks |
|---:|---:|---:|---:|
| 2022-06-06 | 120,873 | 8,664 | 129,537 |

## Stage 12.5 four-day validation

| date | model | n_thresholds | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | gru_ae | 4 | 77,108 | 187 | 75,347 | 4 | 0.9768 | 0.9798 | 1 | 0.9772 | 0.9786 | 0.9999 | single-day cell |
| 2022-06-06 | mlp_dae | 4 | 77,108 | 187 | 75,429 | 1 | 0.9778 | 0.9844 | 1 | 0.9782 | 0.9947 | 1 | single-day cell |

### By threshold (pooled across days)

| threshold_db | model | n_dates | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | gru_ae | 1 | 20,330 | 164 | 20,028 | 3 | 0.9851 | 0.9817 | 0.9999 | 0.9851 | 0.9817 | 0.9999 |  |
| 0 | gru_ae | 1 | 20,897 | 16 | 20,484 | 1 | 0.9802 | 0.9375 | 1 | 0.9802 | 0.9375 | 1 |  |
| 3 | gru_ae | 1 | 19,317 | 6 | 18,745 | 0 | 0.9704 | 1 | 1 | 0.9704 | 1 | 1 |  |
| 6 | gru_ae | 1 | 16,564 | 1 | 16,090 | 0 | 0.9714 | 1 | 1 | 0.9714 | 1 | 1 |  |
| -5 | mlp_dae | 1 | 20,330 | 164 | 20,074 | 0 | 0.9874 | 1 | 1 | 0.9874 | 1 | 1 |  |
| 0 | mlp_dae | 1 | 20,897 | 16 | 20,506 | 1 | 0.9813 | 0.9375 | 1 | 0.9813 | 0.9375 | 1 |  |
| 3 | mlp_dae | 1 | 19,317 | 6 | 18,763 | 0 | 0.9713 | 1 | 1 | 0.9713 | 1 | 1 |  |
| 6 | mlp_dae | 1 | 16,564 | 1 | 16,086 | 0 | 0.9711 | 1 | 1 | 0.9711 | 1 | 1 |  |

### Overall

| model | dates_included | thresholds_included | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | recommended | recommendation_reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru_ae | 2022-06-06 | -5,0,3,6 | 77,108 | 187 | 75,347 | 4 | 0.9768 | 0.9798 | 1 | 0.9772 | 0.9786 | 0.9999 | False |  |
| mlp_dae | 2022-06-06 | -5,0,3,6 | 77,108 | 187 | 75,429 | 1 | 0.9778 | 0.9844 | 1 | 0.9782 | 0.9947 | 1 | True | best mean false reduction at >=0.95 retention; SINGLE/PARTIAL-day evidence |

## MLP vs GRU

| date | threshold_db | metric | mlp_dae | gru_ae | winner | difference | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | nan | true_track_retention | 0.9778 | 0.9768 | mlp_dae | 0.0010 | aggregate mean |
| ALL | nan | false_track_reduction | 0.9844 | 0.9798 | mlp_dae | 0.0046 | aggregate mean |
| ALL | nan | precision_after | 1 | 1 | mlp_dae | 0 | aggregate mean |
| ALL | nan | stage12_kept_false_tracks | 0.2500 | 1 | mlp_dae | -0.7500 | aggregate mean |
| ALL | nan | stage12_kept_true_tracks | 18857.2500 | 18836.7500 | mlp_dae | 20.5000 | aggregate mean |

MLP remains the best overall model; GRU stays the strong low-threshold alternative (differences are small).

## Stage 09 interpretable fallback comparison

| date | threshold_db | stage09_true_retention | stage09_false_reduction | stage09_precision_after | best_stage12_model | stage12_true_retention | stage12_false_reduction | stage12_precision_after | stage12_minus_stage09_true_retention | stage12_minus_stage09_false_reduction | stage12_minus_stage09_precision | interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | -5 | 0.9735 | 0.7879 | 0.9501 | mlp_dae | 0.9874 | 1 | 1 | 0.0139 | 0.2121 | 0.0499 | stage-12 wins on false reduction |
| 2022-06-06 | 0 | 0.9854 | 0.6365 | 0.9810 | mlp_dae | 0.9813 | 0.9375 | 1 | -0.0041 | 0.3010 | 0.0189 | stage-12 wins on false reduction |
| 2022-06-06 | 3 | 0.9844 | 0.5524 | 0.9910 | mlp_dae | 0.9713 | 1 | 1 | -0.0130 | 0.4476 | 0.0090 | stage-12 wins on false reduction |
| 2022-06-06 | 6 | 0.9818 | 0.4781 | 0.9954 | mlp_dae | 0.9711 | 1 | 1 | -0.0107 | 0.5219 | 0.0046 | stage-12 wins on false reduction |

- Stage 12 removes more false tracks than Stage 09 in **4/4** cells; **Stage 09 hand physics remains the recommended
  interpretable fallback** where transparency matters more than peak reduction.

## Windowability caveat

Sequence methods only score windowable tracks (>= window_len points, >= 5 hits),
so 9/12 dB comparisons are not apples-to-apples (few/no windowable false tracks).
High thresholds are kept audit-only.

| date | threshold_db | stage08_confirmed_tracks | stage08_true_tracks | stage08_false_tracks | windowable_tracks | windowable_true_tracks | windowable_false_tracks | windowable_fraction_all | windowable_fraction_true | windowable_fraction_false | notes | high_threshold_caveat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | -5 | 31,626 | 22,796 | 1,545 | 20,494 | 20,330 | 164 | 0.8420 | 0.8918 | 0.1061 |  | False |
| 2022-06-06 | 0 | 32,158 | 25,790 | 252 | 20,913 | 20,897 | 16 | 0.8030 | 0.8103 | 0.0635 |  | False |
| 2022-06-06 | 3 | 33,826 | 25,879 | 86 | 19,323 | 19,317 | 6 | 0.7442 | 0.7464 | 0.0698 |  | False |
| 2022-06-06 | 6 | 31,927 | 23,310 | 11 | 16,565 | 16,564 | 1 | 0.7103 | 0.7106 | 0.0909 |  | False |

## Failure cases

| case_type | date | threshold_db | model | count | median_range_m | median_score | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| false_survives_s12 | 2022-06-06 | 0 | mlp_dae | 1 | 5792.4000 | 1 |  |
| true_rejected_s12 | 2022-06-06 | -5 | mlp_dae | 20 | 113,705 | 0.3597 |  |

- Surviving false tracks after Stage 12.5 are rare but worth inspecting.
- True tracks rejected by Stage 12.5 should be reviewed before deployment.

## Consolidated conclusion

- **The Stage 12.5 conclusion is only partially validated because some required outputs are missing.**
- Stage 12.5 remains the recommended primary false-track filter.
- **Stage 09 interpretable fallback** remains recommended where transparency wins.
- The single-day limitation from Stage 16 is **still open**; run the missing stages (see run plan) to close it.

## Recommended next stage

1. **Run the missing Stage 08/09/12.5 outputs for all four days first** (see `run_plan.csv`; the stage-6 detections and stage-12 checkpoints exist
   locally, so `--run-missing-stage08/09/12` can generate them), then re-run
   this validation.
