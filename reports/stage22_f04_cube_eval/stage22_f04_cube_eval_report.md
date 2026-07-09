# Stage 22 Evaluation of F04 Cube-Derived CFAR Detections in F03

## Status

- Stage 22 evaluates **F04 cube-derived CFAR detections** using the existing F03 tracking/scoring pipeline.
- This is a small **200-frame, single-day, single-operating-point stress test**.
- It **adds no new model** and retrains nothing; stage-08 and stage-12.5 code is unchanged.
- **F04 detections come from a synthetic range-Doppler-azimuth intensity cube. This is not raw RF/IQ, not a full radar waveform simulation, and not measured radar data. Target/clutter labels are evaluation-only.**
- **F04 CFAR scale is not the same as F03 threshold dB.** Scale 6 is a linear multiplier on the local CFAR noise estimate, i.e. 7.78 dB above it -- *not* a 6 dB SNR threshold.

## Input operating point

| field | value |
|---|---|
| date | 2022-06-06 |
| frames | 200 |
| cfar_type | `ca` |
| threshold_scale | 6 (= 7.78 dB above local noise) |
| cap (`max_detections_per_frame`) | 2000 |
| Stage 20.5 recommendation | `B_balanced`, the only candidate with `stage21_recommended = True`; its cap does not bind |
| detections imported | 262,335 (102,086 target / 160,249 clutter) |

The cap not binding matters: a binding cap truncates each frame strongest-first and turns Pd and false-alarm rate into lower bounds rather than measurements. Stage 21 verified the observed maximum was 1,496 detections/frame against the cap of 2000.

## Import and aliasing

- **Original F04 export** (never modified): `/home/tzhen/projects/PLSWORK/F04-RADAR-CUBE/data/active/f03_exports/f04_cube_detections_2022-06-06_cfar_ca_scale_6p0_cap_2000.csv`
- **Imported into F03 as**: `f04_cube_detections_2022-06-06_cfar_ca_scale_6p0_cap_2000.csv`
- **Compatibility alias**: `detections_2022-06-06_thr_6p0dB.csv`
- **Alias metadata**: `f04_cube_alias_metadata_2022-06-06.json`

> **thr_6p0dB is a compatibility filename token only; it represents F04 CFAR scale 6, not a 6 dB SNR threshold. CFAR scale 6 is 7.78 dB above the local noise estimate.**

F03's stage-08 discovers detection files only via `detections_<date>_thr_<token>dB.csv` and filters them with a numeric `--threshold-db`. The alias exists purely to satisfy that parser. Nothing in F03 interprets the token beyond matching it.

### Adaptations applied on import

1. reconstructed `timestamp` as frame_id * 10s: the F04 export carries only 2 distinct timestamps for 200 frames (6-significant-digit float_format rounding). Stage 12 window features differentiate position w.r.t. timestamp, so the original column would corrupt every velocity feature

The clock repair deserves emphasis. The F04 Stage 21 writer serializes with `float_format="%.6g"`, which rounds epoch seconds (~1.65e9) to six significant digits: the export carries only **2 distinct timestamps for 200 frames**. Stage 08 is immune (it takes `dt` from `--frame-period-s`), but stage 12's window features differentiate position with respect to `timestamp`, so the original column would have silently corrupted every velocity feature. Stage 22 reconstructs `timestamp = frame_id * 10s` -- F04's cube simulator defines `frame_id = floor(timestamp / scan_period_s)` -- and verifies the reconstruction rounds to the same six-significant-digit value the export carries (agreement: True). **This is a real bug in F04's Stage 21 writer and should be fixed there** (exclude `timestamp` from the float format, or write it as an integer). Stage 22 works around it without modifying F04.

## Stage 08 Kalman tracking on F04 detections

Stage 08 ran: **yes**.

| date | alias_token | cfar_type | threshold_scale | cap | confirmed_tracks | true_tracks | false_tracks | track_detection_rate | target_assignment_rate | clutter_assignment_rate | mean_position_rmse_m | median_position_rmse_m |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | 6p0 | ca | 6 | 2,000 | 25,717 | 5,260 | 20,457 | 0.9280 | 0.9761 | 0.8698 | 2665.1781 | 2508.2277 |

Notes: alias thr 6p0dB = F04 CFAR scale 6 (7.78 dB), not a dB SNR threshold

## Stage 12.5 sequence-prior scoring on F04 tracks

Stage 12.5 ran: **yes**. Calibration mode: **`track_purity`**.

The noise-matched (`track_purity`) calibration was built from the **F04 cube tracks themselves**, by pinning `--calibration-tracks-dir` at the F04 track directory. This is not a detail: that flag defaults to the canonical F02 `data/active/tracks_kalman`, which happens to contain a `thr_6p0dB` file too, so omitting it would have quietly calibrated on F02 point-detection tracks while scoring F04 cube-CFAR tracks -- exactly the noise mismatch stage 12.5 exists to remove. The calibration JSON is written inside this stage's report directory; the canonical stage-12 calibration is untouched.

| date | alias_token | cfar_type | threshold_scale | cap | model | calibration_mode | score_threshold | stage08_true_tracks | stage08_false_tracks | stage12_kept_true_tracks | stage12_kept_false_tracks | true_track_retention | false_track_reduction | precision_before | precision_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-06-06 | 6p0 | ca | 6 | 2,000 | mlp_dae | track_purity | 0.5000 | 1,962 | 5,141 | 1,738 | 3,763 | 0.8858 | 0.2680 | 0.2762 | 0.3159 |

### Which false tracks? (the denominator matters here)

Stage 12 scores only tracks that yield at least one window. Its `stage08_true_tracks` / `stage08_false_tracks` columns are therefore the **windowable subset**, not what stage 08 confirmed. On F04 cube tracks that gap is not a footnote:

- stage 08 confirmed **20,457** false tracks
- only **5,141** of them are windowable, so **15,316 false tracks are never scored by the filter at all**

The headline reduction is over the windowable subset. The pipeline-level number depends on a policy stage 12.5 does not define -- what happens to a confirmed track the filter never scored -- so both branches are reported:

| unscored_true_tracks | unscored_false_tracks | denominator | true_tracks | false_tracks | kept_true_tracks | kept_false_tracks | true_track_retention | false_track_reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,298 | 15,316 | stage12_windowable_only | 1,962 | 5,141 | 1,738 | 3,763 | 0.8858 | 0.2680 |
| 3,298 | 15,316 | stage08_all_unscored_kept | 5,260 | 20,457 | 5,036 | 19,079 | 0.9574 | 0.0674 |
| 3,298 | 15,316 | stage08_all_unscored_dropped | 5,260 | 20,457 | 1,738 | 3,763 | 0.3304 | 0.8161 |

Retaining unscored tracks gives a pipeline false-track reduction of **6.7%** at **95.7%** true retention. Dropping them gives **81.6%** reduction but collapses true retention to **33.0%** -- it buys suppression by throwing away most real targets. Neither branch reproduces the F02 result.

The F02 stage-12/17 figures use the same windowable denominator, so the `stage12_windowable_only` row is the like-for-like comparison. What differs is how *much* of the track population is windowable: cube-CFAR clutter produces a large population of short, unwindowable false tracks that the F02 point-detection simulator did not.

## Comparison with original F02 point-detection experiment

The two experiments differ in **detection source, frame count, day count and threshold semantics simultaneously**, so no single difference between them is attributable to any one cause.

- The F02/F03 headline result is a **four-day, multi-threshold point-detection** experiment, where the threshold really is an SNR threshold in dB.
- The F04 result is a **200-frame, one-day, one-operating-point structured cube-CFAR stress test**, where the `6` in the filename is a linear CFAR scale.

**The F04 cube-derived result is a small structured-clutter stress test, not a direct replacement for the four-day F02 result.**

| comparison_scope | method | source | date | threshold_or_operating_point | true_retention | false_reduction | false_tracks_kept | true_tracks | false_tracks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f04_cube_cfar_200_frames_1_day | stage12.5 mlp_dae | F04 synthetic radar cube -> CA-CFAR | 2022-06-06 | cfar=ca scale=6 (7.78 dB) cap=2000 | 0.8858 | 0.2680 | 3,763 | 1,962 | 5,141 |
| f02_point_detection_1_day_thr_6dB | stage12.5 mlp_dae | F02 point-detection simulator | 2022-06-06 | threshold_db=6.0 (a real 6 dB SNR threshold) | 0.9711 | 1 | 0 | 16,564 | 1 |
| f02_point_detection_4_day_pooled | stage12.5 mlp_dae | F02 point-detection simulator | 2022-06-06,2022-06-13,2022-06-20,2022-06-27 | -5,0,3,6 | 0.9731 | 0.9939 | 5 | 323,808 | 815 |

## Interpretation

- **Did Stage 08 run?** Yes.
- **Did Stage 12.5 run?** Yes.
- **Did Stage 12.5 suppress F04 false tracks?** stage 12.5 removes only 26.8% of false tracks -- structured CFAR clutter is substantially harder than F02 point clutter (denominator: windowable scored tracks only)
  Measured against **all** stage-08 false tracks (unscored ones passing through), the suppression is only **6.7%**.
- **Were true tracks preserved?** true-track retention is only 88.6%; the filter is losing real targets
- **Are F04 CFAR false tracks harder or easier?** F04 false_reduction 0.2680 vs F02 four-day pooled 0.9939 (delta -0.7258): cube-CFAR false tracks look harder. Different detection source, frame count and threshold semantics -- suggestive, not a controlled comparison

**Answer to the research question.** The stage-12.5 sequence-prior filter does **not** carry over. On F02 point detections it removed 99.4% of false tracks at 97.3% true-track retention; on cube-derived CA-CFAR detections at the same windowable denominator it removes 26.8% at 88.6% retention, and most cube-CFAR false tracks are too short to be scored at all. Structured CFAR clutter produces false tracks that both evade the filter's window requirement and, when scored, look far more target-like than the F02 simulator's independent point clutter. This is a negative result for transfer, not a bug: nothing was retrained, and the filter is being asked to generalize across detection statistics it never saw.

## Limitations

- Only **200 frames**, one day.
- A **single CFAR operating point**; no sweep.
- **Synthetic cube abstraction**: injected point-target energy in a range-Doppler-azimuth intensity cube.
- **No raw RF/IQ**, no waveform or PRF modelling.
- **No elevation grid** in F04: clutter carries `meas_elevation_rad = 0` and labelled targets borrow the nearest truth elevation, so elevation is not measured.
- F03 was built around **F02 point-detection filenames** and required an alias to ingest F04 output at all.
- The `thr_6p0dB` alias token is a **compatibility label**, not a dB threshold.
- Target/clutter labels are **evaluation-only**; neither the tracker nor the scorer sees them.

## Recommended next stage

Stage 08 and Stage 12.5 both ran end-to-end on cube-derived detections, so the adapter is sound. **Stage 23 should run an F04 operating-point sweep through F03**, using 2-3 of the CFAR operating points Stage 20.5 recommended (`A_high_pd` scale 4 / cap 10000, `B_balanced` scale 6 / cap 2000, `C_conservative` scale 12 / cap 2000), so false-track difficulty can be traced against detector aggressiveness rather than inferred from one point.

Two specific questions Stage 23 should answer, both raised by this run:

1. **Does the unwindowable-false-track population shrink at a stricter CFAR threshold?** If most cube-CFAR false tracks are short because clutter detections are dense and transient, `C_conservative` should thin them out. If it does not, the tracker's confirmation logic -- not the filter -- is what needs attention.
2. **Is the transfer gap a calibration gap or a training-distribution gap?** The filter here was calibrated on F04 tracks but *trained* on F02 trajectories. A stage-23 run that retrains the sequence prior on cube-derived tracks would separate the two. That is a training stage and is deliberately out of Stage 22's scope.

## Validation gate

- import summary exists and is nonempty: OK
- alias metadata JSON exists: OK
- alias warning present in metadata: OK
- imported detection CSV exists: OK
- compatibility alias CSV exists: OK
- required F03 stage-08 input columns present: OK
- reconstructed timestamps agree with the rounded original: OK
- reconstructed clock has one timestamp per frame: OK -- 200 unique vs 200 frames
- stage 08 report/metrics exist (or failure documented): OK
- stage 12 report/metrics exist (or failure documented): OK
- report contains the required statements: OK
- canonical stage-12 calibration file unchanged: OK
- no large CSVs staged for commit: OK

## Plots

- `plots/f04_detection_counts.png`
- `plots/f04_track_counts.png`
- `plots/f04_stage12_filter_effect.png`
- `plots/f04_vs_f02_comparison.png`

---

> F04 detections come from a synthetic range-Doppler-azimuth intensity cube. This is not raw RF/IQ, not a full radar waveform simulation, and not measured radar data. Target/clutter labels are evaluation-only.

> thr_6p0dB is a compatibility filename token only; it represents F04 CFAR scale 6, not a 6 dB SNR threshold. CFAR scale 6 is 7.78 dB above the local noise estimate.

