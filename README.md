# F03-Evaluation

> Part of the PLSWORK light-GA radar pipeline: [F01-Preprocessing](https://github.com/zheniannn/F01-Preprocessing) (stages 1-4) → [F02-Radar](https://github.com/zheniannn/F02-Radar) (stages 5-6) → [F03-Evaluation](https://github.com/zheniannn/F03-Evaluation) (stages 7-9).

Evaluation stages of the light-GA radar pipeline. **F03-Evaluation consumes
outputs from F02-Radar** (which ends at stage 6 synthetic detections):

- **Stage 07** — threshold-only baseline evaluation (no tracking)
- **Stage 08** — constant-velocity Kalman detect-then-track baseline
- **Stage 09+** (future) — physics-guided / advanced tracking evaluation

Upstream: F01-Preprocessing owns stages 1–4 (ADS-B → uniform trajectories),
F02-Radar owns stages 5–6 (radar truth → synthetic detections).

## Structure

```
F03-Evaluation/
├── scripts/
│   ├── 06_audit_relocated_experiment.py   # read-only audit of the F02 experiment
│   ├── 07_evaluate_threshold_only.py      # stage 7: threshold-only baseline
│   └── 08_run_kalman_baseline.py          # stage 8: CV-Kalman detect-then-track
├── utils/
│   ├── common.py                          # shared repo-root/parsing/markdown helpers
│   ├── threshold_eval.py                  # stage 7 logic
│   ├── kalman_tracker.py                  # stage 8 tracker (CV-KF + greedy gated NN)
│   ├── track_eval.py                      # stage 8 evaluation + report
│   └── track_physics_score.py             # stage 9 physics scoring
├── data/active/
│   ├── radar_truth_relocated/             # INPUT from F02 (git-ignored)
│   ├── sim_detections_relocated/          # INPUT from F02 (git-ignored)
│   └── tracks_kalman/                     # stage-8 track output (git-ignored)
└── reports/
    ├── relocated_experiment_audit.md
    ├── stage07_threshold_only/            # committed compact tables/plots/report
    └── stage08_kalman_baseline/           # stage-8 summary + report
```

## Input contract from F02-Radar

Required stage-5 truth inputs:

```
data/active/radar_truth_relocated/radar_truth_YYYY-MM-DD.csv
```

Required stage-6 detection inputs:

```
data/active/sim_detections_relocated/detections_YYYY-MM-DD_thr_<THRESHOLD>dB.csv
data/active/sim_detections_relocated/sim_detection_summary.csv
```

Detections must carry the stage-6 columns (frame_id, timestamp,
detection_id, is_target, trajectory_id, meas_range/azimuth/elevation/
radial-velocity, per-component errors, snr_db, …). The `is_target` /
`trajectory_id` labels are used **only for evaluation**, never by the
tracker.

## How to copy inputs from F02-Radar

```bash
cp ../F02-Radar/data/active/radar_truth_relocated/radar_truth_*.csv \
   data/active/radar_truth_relocated/

cp ../F02-Radar/data/active/sim_detections_relocated/detections_*.csv \
   data/active/sim_detections_relocated/

cp ../F02-Radar/data/active/sim_detections_relocated/sim_detection_summary.csv \
   data/active/sim_detections_relocated/
```

These CSVs are large (~24 GB total) and **git-ignored** — only the
directory structure (`.gitkeep`) is committed. Symlinking instead of
copying works too.

## Stage 07 — threshold-only baseline

Evaluates the stage-6 threshold sweep frame-by-frame against truth: overall
and per-day operating curves, Pd by range bin, clutter by range, SNR and
measurement-error summaries. **No tracking.** Outputs (committed, compact)
land in `reports/stage07_threshold_only/`.

```bash
python scripts/07_evaluate_threshold_only.py \
  --truth-dir data/active/radar_truth_relocated \
  --detections-dir data/active/sim_detections_relocated \
  --output-dir reports/stage07_threshold_only \
  --coverage-range-m 100000 \
  --range-bins-m 0,50000,100000,200000,inf \
  --chunksize 1000000 \
  --overwrite
```

`--self-test` runs without real data.

## Stage 08 — Kalman detect-then-track baseline

Classical multi-target tracking over one stage-6 detection stream:
constant-velocity Kalman filter (6-state), greedy gated nearest-neighbor
association on predicted position, confirm-after-3-hits /
delete-after-3-misses track management. The tracker **never sees truth
labels** — truth enters only in the post-hoc evaluation (track purity,
dominant trajectory, coverage, fragmentation, clutter absorption).

```bash
python scripts/08_run_kalman_baseline.py \
  --detections-dir data/active/sim_detections_relocated \
  --truth-dir data/active/radar_truth_relocated \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage08_kalman_baseline \
  --threshold-db 0 \
  --date 2022-06-06 \
  --max-frames 1000 \
  --overwrite
```

Track CSVs go to `data/active/tracks_kalman/` (git-ignored); the metrics
CSV (`kalman_metrics_by_day.csv`, accumulated across runs) and Markdown
report go to `reports/stage08_kalman_baseline/`. `--self-test` runs
without real data. Documented baseline simplifications (stage 9+ targets):
isotropic measurement noise, Euclidean rather than Mahalanobis gating,
radial velocity unused.

### Recommended evaluation sequence

Run stage 8 in controlled increments — do not jump to the full 4-day ×
6-threshold sweep:

1. **One full day at 0 dB:**

```bash
python scripts/08_run_kalman_baseline.py \
  --detections-dir data/active/sim_detections_relocated \
  --truth-dir data/active/radar_truth_relocated \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage08_kalman_baseline \
  --threshold-db 0 \
  --date 2022-06-06 \
  --overwrite
```

2. **One-day threshold sweep at −5, 0, 3, 6 dB:**

```bash
python scripts/08_run_kalman_baseline.py \
  --detections-dir data/active/sim_detections_relocated \
  --truth-dir data/active/radar_truth_relocated \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage08_kalman_baseline \
  --threshold-db -5 0 3 6 \
  --date 2022-06-06 \
  --overwrite
```

3. **Compare stage-7 frame-level metrics to stage-8 track-level metrics**
   (they measure different things — never equate frame Pd with track
   detection rate):

```bash
python scripts/08_compare_stage07_stage08.py \
  --stage07 reports/stage07_threshold_only/threshold_by_day.csv \
  --stage08 reports/stage08_kalman_baseline/kalman_metrics_by_day.csv \
  --date 2022-06-06 \
  --output-prefix reports/stage08_kalman_baseline/stage07_vs_stage08_2022-06-06
```

4. **Only then** expand to all days and thresholds.

## Stage 09 — Physics-guided track scoring

Scores every stage-8 confirmed track with a **transparent, rule-based
physics plausibility score** in [0, 1] (soft penalties on p95 speed,
acceleration, turn rate, vertical speed, hit-rate continuity, median SNR,
and association consistency, weighted and inverted), then filters at a
score threshold and evaluates false-track reduction vs true-track
retention. **This is post-track scoring — the Kalman tracker is not
modified**, and **truth labels are evaluation-only**: the score never sees
`is_target`, `trajectory_id`, purity, or truth positions. Nothing is
trained (no ML, no VAE). **Stage 10 will replace the hand-tuned penalty
knees with empirical ADS-B motion priors** learned from stage-4 data or
true tracks.

- **Inputs:** stage-8 track files in `data/active/tracks_kalman/`
  (both the `tracks_*` baseline schema and a richer
  `track_points_*`/`track_summary_*` schema are accepted; for the baseline
  schema, per-detection `snr_db` is recovered from the stage-6 detections
  and unavailable channels are excluded from the score weighting).
- **Outputs:** compact tables, plots, and `physics_scoring_report.md` in
  `reports/stage09_physics_scoring/`; large per-track scored CSVs in
  `data/active/tracks_scored_physics/` (git-ignored).

```bash
python scripts/09_score_tracks_physics.py \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage09_physics_scoring \
  --threshold-db -5 0 3 6 \
  --date 2022-06-06 \
  --score-threshold 0.5 \
  --overwrite
```

`--self-test` runs without real data; `--sweep-thresholds` controls the
retention-vs-reduction sweep; `--w-*` flags adjust penalty weights.

**Current status:** Stage 09 has been run on the full one-day threshold
sweep for 2022-06-06 (−5, 0, 3, 6, 9, 12 dB). **Stage 10 empirical ADS-B
prior learning is intentionally deferred.**

## Stage 10 — Empirical ADS-B motion priors

Learns **data-derived motion plausibility distributions** from the F01
stage-4 fixed-wing GA trajectories: transparent histogram priors (density +
quantiles, JSON) for speed, |acceleration|, vector acceleration,
|turn rate|, and |vertical speed|, plus a simple joint summary
(correlations, log1p covariance, median/MAD). **Stage 10 only learns
priors** — it does not score tracks; **stage 11 will apply these priors to
stage-8 tracks** and compare against stage-9's hand-designed penalties. No
neural networks, no VAE.

- **Input contract (from F01 stage 4):**
  `states_YYYY-MM-DD_conventionalGA_trajectories_10s.csv` in
  `data/active/trajectories_10s/` with at least `trajectory_id`,
  `timestamp`, an altitude column (`alt_smooth` preferred, `alt_interp`
  fallback), and `speed_mps`; `accel_mps2` / `accel_vector_mps2` /
  `turn_rate_deg_s` are used when present. Copy or symlink them in:

```bash
cp ../F01-Preprocessing/data/active/trajectories_10s/states_*_trajectories_10s.csv \
   data/active/trajectories_10s/
```

- **Outputs:** prior JSONs in `models/motion_priors/` (committed) and
  compact tables/plots/report in `reports/stage10_adsb_motion_priors/`
  (committed). Counts and histograms are exact; quantiles come from a
  seeded streaming reservoir (disclosed in the manifest). Fitting filters
  (speed 5–160 m/s, |accel| ≤ 15 m/s², |turn| ≤ 30 °/s, |vz| ≤ 40 m/s)
  drop noise/outliers from the fit and are disclosed in every prior file.

```bash
python scripts/10_learn_adsb_motion_priors.py \
  --input-dir data/active/trajectories_10s \
  --models-dir models/motion_priors \
  --report-dir reports/stage10_adsb_motion_priors \
  --overwrite
```

Train/holdout variant (holdout day reported but excluded from the fit;
note the separate models dir so the canonical all-days priors are not
overwritten):

```bash
python scripts/10_learn_adsb_motion_priors.py \
  --input-dir data/active/trajectories_10s \
  --models-dir models/motion_priors_holdout \
  --report-dir reports/stage10_adsb_motion_priors_holdout \
  --holdout-date 2022-06-27 \
  --overwrite
```

`--self-test` runs without real data.

## Audit

`scripts/06_audit_relocated_experiment.py` is the read-only audit of the
relocated wide-area experiment (copied here from F02 because it is
evaluation, not simulation); its report lives at
`reports/relocated_experiment_audit.md`.
