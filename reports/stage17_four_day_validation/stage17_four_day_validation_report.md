# Stage 17 Four-Day Validation

## Status

- **Stage 17 adds no new model** and retrains nothing.
- It validates the selected **Stage 12.5** method (deterministic sequence
  autoencoder, track-purity calibration) across all available ADS-B days.
- It focuses on the **four-day** validation at thresholds -5/0/3/6 dB.
- 9/12 dB remain **audit-only** due to the windowability caveat.
- It runs missing stages only if explicitly requested (`--run-missing-*`).

## Input availability

Expected four days: 2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27. Days with complete stage-08/09/12 results: **2022-06-06, 2022-06-13, 2022-06-20, 2022-06-27**.

| date | cells | complete |
|---:|---:|---:|
| 2022-06-06 | 4 | 4 |
| 2022-06-13 | 4 | 4 |
| 2022-06-20 | 4 | 4 |
| 2022-06-27 | 4 | 4 |

**All four days are available** -- the single-day limitation from Stage 16 is addressed below.

## Run plan

All required stage-08/09/12.5 outputs are present; **no expensive reruns needed.**


## Four-day Stage 08 context

| date | stage08_true_tracks | stage08_false_tracks | stage08_confirmed_tracks |
|---:|---:|---:|---:|
| 2022-06-06 | 120,873 | 8,664 | 129,537 |
| 2022-06-13 | 115,629 | 8,618 | 124,247 |
| 2022-06-20 | 150,895 | 8,765 | 159,660 |
| 2022-06-27 | 118,598 | 8,726 | 127,324 |

## Stage 12.5 four-day validation

| date | model | n_thresholds | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | gru_ae | 4 | 77,108 | 187 | 75,347 | 4 | 0.9768 | 0.9798 | 1 | 0.9772 | 0.9786 | 0.9999 | single-day cell |
| 2022-06-13 | gru_ae | 4 | 75,166 | 203 | 72,875 | 2 | 0.9690 | 0.9842 | 1 | 0.9695 | 0.9901 | 1 | single-day cell |
| 2022-06-20 | gru_ae | 4 | 96,315 | 216 | 93,894 | 8 | 0.9743 | 0.9629 | 0.9999 | 0.9749 | 0.9630 | 0.9999 | single-day cell |
| 2022-06-27 | gru_ae | 4 | 75,219 | 209 | 72,699 | 3 | 0.9660 | 0.9789 | 1 | 0.9665 | 0.9856 | 1 | single-day cell |
| 2022-06-06 | mlp_dae | 4 | 77,108 | 187 | 75,429 | 1 | 0.9778 | 0.9844 | 1 | 0.9782 | 0.9947 | 1 | single-day cell |
| 2022-06-13 | mlp_dae | 4 | 75,166 | 203 | 73,043 | 1 | 0.9712 | 0.9981 | 1 | 0.9718 | 0.9951 | 1 | single-day cell |
| 2022-06-20 | mlp_dae | 4 | 96,315 | 216 | 93,850 | 0 | 0.9737 | 1 | 1 | 0.9744 | 1 | 1 | single-day cell |
| 2022-06-27 | mlp_dae | 4 | 75,219 | 209 | 72,760 | 3 | 0.9666 | 0.9947 | 1 | 0.9673 | 0.9856 | 1 | single-day cell |

### By threshold (pooled across days)

| threshold_db | model | n_dates | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | gru_ae | 4 | 85,496 | 719 | 84,049 | 12 | 0.9827 | 0.9836 | 0.9999 | 0.9831 | 0.9833 | 0.9999 |  |
| 0 | gru_ae | 4 | 87,370 | 76 | 85,130 | 5 | 0.9742 | 0.9314 | 0.9999 | 0.9744 | 0.9342 | 0.9999 |  |
| 3 | gru_ae | 4 | 81,133 | 18 | 78,384 | 0 | 0.9658 | 1 | 1 | 0.9661 | 1 | 1 |  |
| 6 | gru_ae | 4 | 69,809 | 2 | 67,252 | 0 | 0.9633 | 1 | 1 | 0.9634 | 1 | 1 |  |
| -5 | mlp_dae | 4 | 85,496 | 719 | 84,252 | 4 | 0.9851 | 0.9946 | 0.9999 | 0.9854 | 0.9944 | 1 |  |
| 0 | mlp_dae | 4 | 87,370 | 76 | 85,308 | 1 | 0.9763 | 0.9844 | 1 | 0.9764 | 0.9868 | 1 |  |
| 3 | mlp_dae | 4 | 81,133 | 18 | 78,383 | 0 | 0.9660 | 1 | 1 | 0.9661 | 1 | 1 |  |
| 6 | mlp_dae | 4 | 69,809 | 2 | 67,139 | 0 | 0.9618 | 1 | 1 | 0.9618 | 1 | 1 |  |

### Overall

| model | dates_included | thresholds_included | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | mean_true_retention | mean_false_reduction | mean_precision_after | pooled_true_retention | pooled_false_reduction | pooled_precision_after | recommended | recommendation_reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru_ae | 2022-06-06,2022-06-13,2022-06-20,2022-06-27 | -5,0,3,6 | 323,808 | 815 | 314,815 | 17 | 0.9715 | 0.9757 | 1 | 0.9722 | 0.9791 | 0.9999 | False |  |
| mlp_dae | 2022-06-06,2022-06-13,2022-06-20,2022-06-27 | -5,0,3,6 | 323,808 | 815 | 315,082 | 5 | 0.9723 | 0.9940 | 1 | 0.9731 | 0.9939 | 1 | True | best mean false reduction at >=0.95 retention |

## MLP vs GRU

| date | threshold_db | metric | mlp_dae | gru_ae | winner | difference | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | nan | true_track_retention | 0.9723 | 0.9715 | mlp_dae | 0.0008 | aggregate mean |
| ALL | nan | false_track_reduction | 0.9940 | 0.9757 | mlp_dae | 0.0183 | aggregate mean |
| ALL | nan | precision_after | 1 | 1 | mlp_dae | 0 | aggregate mean |
| ALL | nan | stage12_kept_false_tracks | 0.3125 | 1.0625 | mlp_dae | -0.7500 | aggregate mean |
| ALL | nan | stage12_kept_true_tracks | 19692.6250 | 19675.9375 | mlp_dae | 16.6875 | aggregate mean |

MLP remains the best overall model; GRU stays the strong low-threshold alternative (differences are small).

## Stage 09 interpretable fallback comparison

| date | threshold_db | stage09_true_retention | stage09_false_reduction | stage09_precision_after | best_stage12_model | stage12_true_retention | stage12_false_reduction | stage12_precision_after | stage12_minus_stage09_true_retention | stage12_minus_stage09_false_reduction | stage12_minus_stage09_precision | interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | -5 | 0.9735 | 0.7879 | 0.9501 | mlp_dae | 0.9874 | 1 | 1 | 0.0139 | 0.2121 | 0.0499 | stage-12 wins on false reduction |
| 2022-06-06 | 0 | 0.9854 | 0.6365 | 0.9810 | mlp_dae | 0.9813 | 0.9375 | 1 | -0.0041 | 0.3010 | 0.0189 | stage-12 wins on false reduction |
| 2022-06-06 | 3 | 0.9844 | 0.5524 | 0.9910 | mlp_dae | 0.9713 | 1 | 1 | -0.0130 | 0.4476 | 0.0090 | stage-12 wins on false reduction |
| 2022-06-06 | 6 | 0.9818 | 0.4781 | 0.9954 | mlp_dae | 0.9711 | 1 | 1 | -0.0107 | 0.5219 | 0.0046 | stage-12 wins on false reduction |
| 2022-06-13 | -5 | 0.9731 | 0.7865 | 0.9476 | mlp_dae | 0.9833 | 0.9943 | 0.9999 | 0.0102 | 0.2077 | 0.0524 | stage-12 wins on false reduction |
| 2022-06-13 | 0 | 0.9852 | 0.6394 | 0.9808 | mlp_dae | 0.9757 | 1 | 1 | -0.0095 | 0.3606 | 0.0192 | stage-12 wins on false reduction |
| 2022-06-13 | 3 | 0.9865 | 0.5383 | 0.9906 | mlp_dae | 0.9649 | 1 | 1 | -0.0216 | 0.4617 | 0.0094 | stage-12 wins on false reduction |
| 2022-06-13 | 6 | 0.9841 | 0.4745 | 0.9955 | mlp_dae | 0.9607 | nan | 1 | -0.0234 | nan | 0.0045 | undefined: no windowable false tracks for stage-12 in this cell |
| 2022-06-20 | -5 | 0.9731 | 0.7889 | 0.9597 | mlp_dae | 0.9896 | 1 | 1 | 0.0165 | 0.2111 | 0.0403 | stage-12 wins on false reduction |
| 2022-06-20 | 0 | 0.9826 | 0.6301 | 0.9846 | mlp_dae | 0.9766 | 1 | 1 | -0.0060 | 0.3699 | 0.0154 | stage-12 wins on false reduction |
| 2022-06-20 | 3 | 0.9826 | 0.5733 | 0.9929 | mlp_dae | 0.9676 | 1 | 1 | -0.0150 | 0.4267 | 0.0071 | stage-12 wins on false reduction |
| 2022-06-20 | 6 | 0.9781 | 0.4525 | 0.9963 | mlp_dae | 0.9609 | 1 | 1 | -0.0172 | 0.5475 | 0.0037 | stage-12 wins on false reduction |
| 2022-06-27 | -5 | 0.9729 | 0.7890 | 0.9490 | mlp_dae | 0.9803 | 0.9840 | 0.9998 | 0.0074 | 0.1949 | 0.0509 | stage-12 wins on false reduction |
| 2022-06-27 | 0 | 0.9835 | 0.6471 | 0.9815 | mlp_dae | 0.9718 | 1 | 1 | -0.0118 | 0.3529 | 0.0185 | stage-12 wins on false reduction |
| 2022-06-27 | 3 | 0.9868 | 0.5416 | 0.9906 | mlp_dae | 0.9600 | 1 | 1 | -0.0268 | 0.4584 | 0.0094 | stage-12 wins on false reduction |
| 2022-06-27 | 6 | 0.9841 | 0.4982 | 0.9955 | mlp_dae | 0.9543 | nan | 1 | -0.0298 | nan | 0.0045 | undefined: no windowable false tracks for stage-12 in this cell |

- Stage 12 removes more false tracks than Stage 09 in **14/16** cells; **Stage 09 hand physics remains the recommended
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
| 2022-06-13 | -5 | 30,746 | 22,023 | 1,557 | 20,020 | 19,846 | 174 | 0.8490 | 0.9011 | 0.1118 |  | False |
| 2022-06-13 | 0 | 30,581 | 24,568 | 253 | 20,241 | 20,217 | 24 | 0.8155 | 0.8229 | 0.0949 |  | False |
| 2022-06-13 | 3 | 32,207 | 24,905 | 59 | 18,842 | 18,837 | 5 | 0.7548 | 0.7564 | 0.0847 |  | False |
| 2022-06-13 | 6 | 30,713 | 22,510 | 8 | 16,266 | 16,266 | 0 | 0.7224 | 0.7226 | 0 |  | False |
| 2022-06-20 | -5 | 37,800 | 28,417 | 1,590 | 25,644 | 25,450 | 194 | 0.8546 | 0.8956 | 0.1220 |  | False |
| 2022-06-20 | 0 | 39,628 | 31,936 | 263 | 26,012 | 25,995 | 17 | 0.8079 | 0.8140 | 0.0646 |  | False |
| 2022-06-20 | 3 | 42,253 | 32,506 | 72 | 24,109 | 24,105 | 4 | 0.7400 | 0.7416 | 0.0556 |  | False |
| 2022-06-20 | 6 | 39,979 | 29,154 | 14 | 20,766 | 20,765 | 1 | 0.7119 | 0.7123 | 0.0714 |  | False |
| 2022-06-27 | -5 | 31,292 | 22,294 | 1,586 | 20,057 | 19,870 | 187 | 0.8399 | 0.8913 | 0.1179 |  | False |
| 2022-06-27 | 0 | 31,667 | 25,134 | 272 | 20,280 | 20,261 | 19 | 0.7982 | 0.8061 | 0.0699 |  | False |
| 2022-06-27 | 3 | 33,191 | 25,266 | 76 | 18,877 | 18,874 | 3 | 0.7449 | 0.7470 | 0.0395 |  | False |
| 2022-06-27 | 6 | 31,174 | 22,688 | 15 | 16,214 | 16,214 | 0 | 0.7142 | 0.7147 | 0 |  | False |

## Failure cases

| case_type | date | threshold_db | model | count | median_range_m | median_score | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| false_survives_s12 | 2022-06-06 | 0 | mlp_dae | 1 | 5792.4000 | 1 |  |
| true_rejected_s12 | 2022-06-06 | -5 | mlp_dae | 20 | 113,705 | 0.3597 |  |

- Surviving false tracks after Stage 12.5 are rare but worth inspecting.
- True tracks rejected by Stage 12.5 should be reviewed before deployment.

## Consolidated conclusion

- **The Stage 12.5 conclusion generalizes across all four days.**
- Stage 12.5 remains the recommended primary false-track filter.
- **Stage 09 interpretable fallback** remains recommended where transparency wins.
- The single-day limitation from Stage 16 is **closed**.

## Recommended next stage

1. **Final report / paper packaging** of the validated result, or
2. an optional compact model-zoo benchmark only if a specific gap remains, or
3. an optional deployment-style runtime / operating-point study.
