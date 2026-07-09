# Stage 09 Physics-Guided Track Scoring

## Status

- These are one-day results for 2022-06-13.
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
| -5 | 30,746 | 24,553 | 6,193 | 0.5000 | 25,215 | 23,893 | 1,322 | 0.9731 | 0.7865 | 0.7986 | 0.9476 |
| 0 | 30,581 | 29,031 | 1,550 | 0.5000 | 29,160 | 28,601 | 559 | 0.9852 | 0.6394 | 0.9493 | 0.9808 |
| 3 | 32,207 | 31,568 | 639 | 0.5000 | 31,437 | 31,142 | 295 | 0.9865 | 0.5383 | 0.9802 | 0.9906 |
| 6 | 30,713 | 30,458 | 255 | 0.5000 | 30,107 | 29,973 | 134 | 0.9841 | 0.4745 | 0.9917 | 0.9955 |

coverage_proxy columns are NaN: this stage evaluates **track-level
retention**, not full trajectory coverage — kept-track trajectory
coverage belongs to a stage-8-style re-evaluation over the filtered
track set.

## Threshold sweep

How false-track reduction trades against true-track retention as the
score threshold moves (per detection threshold):

False-track reduction:

| score_threshold | -5.0 | 0.0 | 3.0 | 6.0 |
|---:|---:|---:|---:|---:|
| 0.1000 | 0.0010 | 0.0010 | 0.0020 | 0 |
| 0.2000 | 0.0240 | 0.0120 | 0.0080 | 0 |
| 0.3000 | 0.2850 | 0.1530 | 0.1240 | 0.0860 |
| 0.4000 | 0.5360 | 0.3760 | 0.3100 | 0.2470 |
| 0.5000 | 0.7870 | 0.6390 | 0.5380 | 0.4750 |
| 0.6000 | 0.9190 | 0.8300 | 0.7840 | 0.7140 |
| 0.7000 | 0.9760 | 0.9230 | 0.8950 | 0.8670 |
| 0.8000 | 0.9980 | 0.9890 | 0.9780 | 0.9760 |
| 0.9000 | 1 | 1 | 1 | 1 |

True-track retention:

| score_threshold | -5.0 | 0.0 | 3.0 | 6.0 |
|---:|---:|---:|---:|---:|
| 0.1000 | 1 | 1 | 1 | 1 |
| 0.2000 | 1 | 1 | 1 | 1 |
| 0.3000 | 1 | 1 | 1 | 1 |
| 0.4000 | 0.9980 | 0.9990 | 0.9980 | 0.9970 |
| 0.5000 | 0.9730 | 0.9850 | 0.9870 | 0.9840 |
| 0.6000 | 0.9190 | 0.9340 | 0.9340 | 0.9310 |
| 0.7000 | 0.8430 | 0.8400 | 0.8310 | 0.8260 |
| 0.8000 | 0.7570 | 0.6680 | 0.6160 | 0.5930 |
| 0.9000 | 0.6770 | 0.5140 | 0.4130 | 0.3660 |

## Score separability

Median physics score: true tracks 0.887, false tracks 0.407.
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
