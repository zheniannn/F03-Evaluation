# Stage 09 Physics-Guided Track Scoring

## Status

- These are one-day results for 2022-06-06.
- **Stage 10 is intentionally deferred.**
- Stage 09 remains **hand-designed physics scoring**, not empirical
  ADS-B prior learning and not VAE.

## Experiment definition

- Input is **Stage 08 confirmed Kalman tracks**; Stage 09 does not
  change the tracker in any way.
- Stage 09 computes a **rule-based physics plausibility score** in
  [0, 1] per confirmed track, from posterior-state kinematics and
  measurement features only.
- **Truth labels are used only after scoring**, to evaluate retention
  and reduction. The score itself never sees is_target,
  trajectory_id, purity, target_fraction, or truth positions.
- This is **not ML** and not a VAE — nothing is trained; every
  penalty is a transparent soft threshold.
- True-track definition in this stage (target_fraction >= 0.8 and
  purity >= 0.8) is stricter than the stage-8 report's 0.5-based
  definition, so the stage-8 baseline counts here are recomputed
  under the stricter rule and will not match the stage-8 report
  numbers exactly.
- Channels unavailable in these stage-8 outputs and excluded from the weighting: association (Mahalanobis d2).

## Scoring model

Soft penalties (0 inside the good knee, 1 beyond the bad knee,
linear between, NaN -> 0.5), combined as a weighted mean and
inverted: score = 1 - normalized penalty.

- speed: p95 speed good <= 110 m/s, bad >= 160 m/s (w=1)
- acceleration: p95 good <= 3.0 m/s^2, bad >= 8.0 m/s^2 (w=1)
- turn rate: p95 |turn| good <= 5 deg/s, bad >= 15 deg/s (w=1)
- vertical speed: p95 |vz| good <= 10 m/s, bad >= 25 m/s (w=1)
- continuity: hit rate good >= 0.8, bad <= 0.4 (reversed; w=1)
- SNR: median good >= 3 dB, bad <= -10 dB (reversed; lower weight w=0.5 because weak real targets are valid)
- association consistency: p95 Mahalanobis d2 good <= 10, bad >= 25 (w=1)

Keep/reject threshold used: **physics_score >= 0.5**.

## Overall results

| threshold_db | stage08_confirmed_tracks | stage08_true_tracks | stage08_false_tracks | stage09_score_threshold | stage09_kept_tracks | stage09_kept_true_tracks | stage09_kept_false_tracks | stage09_true_track_retention | stage09_false_track_reduction | stage08_precision | stage09_precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | 31,626 | 25,488 | 6,138 | 0.5000 | 26,114 | 24,812 | 1,302 | 0.9735 | 0.7879 | 0.8059 | 0.9501 |
| 0 | 32,158 | 30,554 | 1,604 | 0.5000 | 30,691 | 30,108 | 583 | 0.9854 | 0.6365 | 0.9501 | 0.9810 |
| 3 | 33,826 | 33,167 | 659 | 0.5000 | 32,943 | 32,648 | 295 | 0.9844 | 0.5524 | 0.9805 | 0.9910 |
| 6 | 31,927 | 31,653 | 274 | 0.5000 | 31,221 | 31,078 | 143 | 0.9818 | 0.4781 | 0.9914 | 0.9954 |
| 9 | 27,767 | 27,663 | 104 | 0.5000 | 27,069 | 27,009 | 60 | 0.9764 | 0.4231 | 0.9963 | 0.9978 |
| 12 | 22,890 | 22,841 | 49 | 0.5000 | 22,214 | 22,184 | 30 | 0.9712 | 0.3878 | 0.9979 | 0.9986 |

coverage_proxy columns are NaN: this stage evaluates **track-level
retention**, not full trajectory coverage — kept-track trajectory
coverage belongs to a stage-8-style re-evaluation over the filtered
track set.

## Threshold sweep

How false-track reduction trades against true-track retention as the
score threshold moves (per detection threshold):

False-track reduction:

| score_threshold | -5.0 | 0.0 | 3.0 | 6.0 | 9.0 | 12.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1000 | 0.0010 | 0 | 0 | 0 | 0 | 0 |
| 0.2000 | 0.0250 | 0.0070 | 0.0020 | 0.0070 | 0 | 0 |
| 0.3000 | 0.2870 | 0.1560 | 0.1170 | 0.0730 | 0.0670 | 0.0410 |
| 0.4000 | 0.5470 | 0.3800 | 0.2960 | 0.2260 | 0.2400 | 0.1220 |
| 0.5000 | 0.7880 | 0.6370 | 0.5520 | 0.4780 | 0.4230 | 0.3880 |
| 0.6000 | 0.9180 | 0.8400 | 0.7660 | 0.7300 | 0.7120 | 0.6120 |
| 0.7000 | 0.9770 | 0.9380 | 0.9030 | 0.8650 | 0.8650 | 0.8780 |
| 0.8000 | 0.9980 | 0.9880 | 0.9790 | 0.9670 | 0.9620 | 0.9800 |
| 0.9000 | 1 | 1 | 1 | 1 | 1 | 1 |

True-track retention:

| score_threshold | -5.0 | 0.0 | 3.0 | 6.0 | 9.0 | 12.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1000 | 1 | 1 | 1 | 1 | 1 | 1 |
| 0.2000 | 1 | 1 | 1 | 1 | 1 | 1 |
| 0.3000 | 1 | 1 | 1 | 1 | 0.9990 | 0.9990 |
| 0.4000 | 0.9990 | 0.9990 | 0.9980 | 0.9970 | 0.9950 | 0.9920 |
| 0.5000 | 0.9730 | 0.9850 | 0.9840 | 0.9820 | 0.9760 | 0.9710 |
| 0.6000 | 0.9180 | 0.9330 | 0.9270 | 0.9250 | 0.9210 | 0.9150 |
| 0.7000 | 0.8360 | 0.8330 | 0.8190 | 0.8170 | 0.8180 | 0.8170 |
| 0.8000 | 0.7430 | 0.6510 | 0.5980 | 0.5830 | 0.5750 | 0.5860 |
| 0.9000 | 0.6600 | 0.4930 | 0.3950 | 0.3530 | 0.3430 | 0.3410 |

## Score separability

Median physics score: true tracks 0.851, false tracks 0.407.
True and false tracks separate in score — the physics plausibility features carry real signal.

## Failure modes

- False tracks that happen to look kinematically plausible (clutter
  chains that mimic straight flight) survive the filter.
- Real weak/noisy tracks can be rejected when the continuity/SNR
  penalties are too harsh for genuinely low-SNR targets.
- Hand-tuned knees are brittle across scenarios — this motivates
  empirical ADS-B priors and later VAE/learned motion priors.

## Recommended next stage

Stage 10 should learn empirical motion-prior distributions (speed,
acceleration, turn rate, vertical speed) from ADS-B/Stage 4 data or
from true tracks, and replace these hand-tuned penalties with
data-derived likelihoods.
