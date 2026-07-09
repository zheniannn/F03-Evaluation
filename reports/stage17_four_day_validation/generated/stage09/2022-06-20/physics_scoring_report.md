# Stage 09 Physics-Guided Track Scoring

## Status

- These are one-day results for 2022-06-20.
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
| -5 | 37,800 | 31,671 | 6,129 | 0.5000 | 32,112 | 30,818 | 1,294 | 0.9731 | 0.7889 | 0.8379 | 0.9597 |
| 0 | 39,628 | 38,052 | 1,576 | 0.5000 | 37,974 | 37,391 | 583 | 0.9826 | 0.6301 | 0.9602 | 0.9846 |
| 3 | 42,253 | 41,564 | 689 | 0.5000 | 41,135 | 40,841 | 294 | 0.9826 | 0.5733 | 0.9837 | 0.9929 |
| 6 | 39,979 | 39,716 | 263 | 0.5000 | 38,990 | 38,846 | 144 | 0.9781 | 0.4525 | 0.9934 | 0.9963 |

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
| 0.1000 | 0.0010 | 0 | 0 | 0 |
| 0.2000 | 0.0200 | 0.0080 | 0.0070 | 0.0080 |
| 0.3000 | 0.2830 | 0.1430 | 0.1250 | 0.0650 |
| 0.4000 | 0.5350 | 0.3730 | 0.3020 | 0.2090 |
| 0.5000 | 0.7890 | 0.6300 | 0.5730 | 0.4520 |
| 0.6000 | 0.9210 | 0.8050 | 0.7690 | 0.6540 |
| 0.7000 | 0.9740 | 0.9220 | 0.8930 | 0.8400 |
| 0.8000 | 0.9980 | 0.9800 | 0.9750 | 0.9850 |
| 0.9000 | 1 | 1 | 1 | 1 |

True-track retention:

| score_threshold | -5.0 | 0.0 | 3.0 | 6.0 |
|---:|---:|---:|---:|---:|
| 0.1000 | 1 | 1 | 1 | 1 |
| 0.2000 | 1 | 1 | 1 | 1 |
| 0.3000 | 1 | 1 | 1 | 0.9990 |
| 0.4000 | 0.9980 | 0.9990 | 0.9980 | 0.9960 |
| 0.5000 | 0.9730 | 0.9830 | 0.9830 | 0.9780 |
| 0.6000 | 0.9180 | 0.9250 | 0.9220 | 0.9130 |
| 0.7000 | 0.8380 | 0.8230 | 0.8100 | 0.7990 |
| 0.8000 | 0.7490 | 0.6400 | 0.5910 | 0.5710 |
| 0.9000 | 0.6590 | 0.4840 | 0.3960 | 0.3520 |

## Score separability

Median physics score: true tracks 0.873, false tracks 0.409.
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
