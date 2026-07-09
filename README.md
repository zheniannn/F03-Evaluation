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

## Stage 11 — Empirical ADS-B-prior track scoring

Applies the **stage-10 empirical ADS-B priors** to the stage-8 confirmed
Kalman tracks: per-feature quantile-exceedance penalties (from the prior
JSON quantiles — nothing hand-coded), histogram log-density diagnostics,
and a joint-prior Mahalanobis score, combined into a weighted
`adsb_prior_score` in [0, 1]. Results are **compared against stage-9
hand-designed physics scoring** on the same tracks with the same strict
true-track definition. Truth labels are evaluation-only; **still no
VAE/diffusion/ML model zoo** — that is stage 12.

- **Inputs:** stage-8 tracks (`data/active/tracks_kalman/`, both schemas),
  stage-10 priors (`models/motion_priors/`), and — when present — the
  stage-9 comparison CSV for the three-way table.
- **Outputs:** compact tables, plots, and `adsb_prior_scoring_report.md`
  in `reports/stage11_adsb_prior_scoring/` (committed).

```bash
python scripts/11_score_tracks_adsb_prior.py \
  --tracks-dir data/active/tracks_kalman \
  --priors-dir models/motion_priors \
  --stage09-dir reports/stage09_physics_scoring \
  --report-dir reports/stage11_adsb_prior_scoring \
  --threshold-db -5 0 3 6 9 12 \
  --date 2022-06-06 \
  --score-threshold 0.5 \
  --overwrite
```

`--self-test` runs without real data; `--w-*` flags adjust component
weights (priors dominate; continuity/SNR are weak auxiliaries).

## Stage 12 — Learned sequence-prior track scoring

The first **learned trajectory-shape** stage: trains three denoising
autoencoders (**mlp_dae**, **gru_ae**, **tcn_ae**, PyTorch) on
origin/heading-normalized trajectory windows from clean stage-5 relocated
truth (stage-4 fallback), then scores stage-8 confirmed tracks by
**reconstruction plausibility** calibrated against holdout-truth errors
(score 1 at/below the validation p50 error, 0 at/above the p99).
Compared four-way against stage 8 / stage 9 (hand physics) / stage 11
(empirical marginal priors). **Not VAE, not diffusion, not the model
zoo** — stage 13 will implement a VAE trajectory prior over the same
windows. Truth labels are evaluation-only.

- **Training inputs:** `data/active/radar_truth_relocated/` (preferred) or
  `data/active/trajectories_10s/`; whole days held out for calibration.
- **Scoring inputs:** stage-8 tracks (`data/active/tracks_kalman/`, both
  schemas) + trained models in `models/sequence_priors/`.
- **Outputs:** model checkpoints (`.pt`, git-ignored) + committed JSON
  metadata in `models/sequence_priors/`; compact tables, plots, and
  `sequence_prior_report.md` in `reports/stage12_sequence_priors/`.

```bash
# Step A: train
python scripts/12_train_sequence_priors.py \
  --truth-dir data/active/radar_truth_relocated \
  --models-dir models/sequence_priors \
  --report-dir reports/stage12_sequence_priors \
  --model mlp_dae gru_ae tcn_ae \
  --window-len 20 --stride 5 --epochs 20 --batch-size 512 \
  --max-train-windows 500000 --max-val-windows 100000 \
  --holdout-date 2022-06-27 --overwrite

# Step B: score
python scripts/12_score_tracks_sequence_prior.py \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/sequence_priors \
  --stage09-dir reports/stage09_physics_scoring \
  --stage11-dir reports/stage11_adsb_prior_scoring \
  --report-dir reports/stage12_sequence_priors \
  --threshold-db -5 0 3 6 9 12 \
  --date 2022-06-06 --score-threshold 0.5 --overwrite
```

Both scripts have `--self-test`. Requires PyTorch (the scripts fail with a
clear message if `torch` is missing). On CPU-only machines, reduce
`--max-train-windows`/`--epochs` (the committed run used 200k windows x 10
epochs on CPU; the manifest records the exact settings).

### Stage 12.5 — Noise-matched calibration

Stage 12's models trained on **clean** truth windows separate true and
false tracks strongly, but their clean-truth score band is miscalibrated
for **noisy** stage-8 Kalman tracks: genuine tracks reconstruct worse than
clean truth and collapse toward score 0, so true-track retention at the 0.5
threshold is far too low. Stage 12.5 **recalibrates the reconstruction-
error→score mapping** using high-purity stage-8 true tracks instead of
clean truth windows — the p50→1 / p99→0 anchors now come from the noisy-
track domain, making the 0.5 threshold meaningful.

- This **does not retrain the models** — the `.pt` weights are untouched;
  only the calibration quantiles change.
- **Truth labels are used only to select the calibration tracks** and to
  evaluate metrics; they never enter the score itself. Still **not VAE**
  (stage 13) and **not diffusion** (stage 14).
- Run this **before comparing stage 12 to stage 9 / stage 11** — the
  four-way comparison should reflect the noise-matched calibration.
- Higher detection thresholds (3/6/9/12 dB) are used for the calibration
  set because they yield cleaner, more reliable high-purity true tracks;
  the full threshold set (including −5/0) is still evaluated.
- **Outputs:** `reports/stage12_sequence_priors/calibration/` —
  `sequence_track_calibration.json`/`.csv` (the band) and
  `sequence_calibration_comparison.csv` (clean-truth vs track-purity), plus
  three calibration plots and a Stage 12.5 report section.

```bash
python scripts/12_score_tracks_sequence_prior.py \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/sequence_priors \
  --stage09-dir reports/stage09_physics_scoring \
  --stage11-dir reports/stage11_adsb_prior_scoring \
  --report-dir reports/stage12_sequence_priors \
  --threshold-db -5 0 3 6 9 12 \
  --date 2022-06-06 \
  --calibration-mode track_purity \
  --calibration-date 2022-06-06 \
  --calibration-threshold-db 3 6 9 12 \
  --calibration-min-target-fraction 0.95 \
  --calibration-min-purity 0.95 \
  --score-threshold 0.5 \
  --compare-calibration \
  --overwrite
```

`scripts/12_calibrate_sequence_prior.py` builds the same calibration JSON
standalone (without scoring), and
`--self-test --calibration-mode track_purity` runs the calibration self-
test.

## Stage 13 — VAE trajectory prior

The first **probabilistic** sequence-prior stage: trains a variational
autoencoder (**SequenceVAE**, PyTorch) over the same origin/heading-
normalized trajectory windows as stage 12, then scores stage-8 confirmed
tracks by anomaly under two variants — **reconstruction** (recon error
only) and **elbo** (recon + `beta_score`·KL). Scores use the **noise-
matched track-purity calibration** from stage 12.5 (score 1 at/below the
calibration p50 anomaly, 0 at/above the p99), and the run is compared
**five-way** against stages 8 / 9 / 11 / 12 (best stage-12 model chosen
automatically). The research question is whether a probabilistic latent
model beats the stage-12.5 deterministic autoencoders or adds useful latent
structure. **Not diffusion (stage 14), not the full model zoo.** Truth
labels are used only for calibration-track selection and evaluation. Where
available, stage 13 **reuses the stage-12 normalizer** (`models/sequence_priors/normalizer.json`)
so the two stages are directly comparable.

- **Training inputs:** `data/active/radar_truth_relocated/` (stage-4
  fallback); whole days held out for validation/calibration.
- **Scoring inputs:** stage-8 tracks (`data/active/tracks_kalman/`, both
  schemas) + the trained VAE in `models/vae_priors/`, plus the stage-9/11/12
  reports for the five-way comparison.
- **Outputs:** `vae_prior.pt` (git-ignored) + committed `vae_config.json`,
  `vae_training_manifest.json`, `vae_calibration.json` in
  `models/vae_priors/`; compact tables, six plots, and `vae_prior_report.md`
  in `reports/stage13_vae_prior/`.

```bash
# Step A: train
python scripts/13_train_vae_prior.py \
  --truth-dir data/active/radar_truth_relocated \
  --models-dir models/vae_priors \
  --sequence-models-dir models/sequence_priors \
  --report-dir reports/stage13_vae_prior \
  --window-len 20 --stride 5 --epochs 20 --batch-size 512 \
  --max-train-windows 200000 --max-val-windows 60000 \
  --holdout-date 2022-06-27 --latent-dim 32 --beta 0.001 --overwrite

# Step B: score
python scripts/13_score_tracks_vae_prior.py \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/vae_priors \
  --sequence-models-dir models/sequence_priors \
  --stage09-dir reports/stage09_physics_scoring \
  --stage11-dir reports/stage11_adsb_prior_scoring \
  --stage12-dir reports/stage12_sequence_priors \
  --report-dir reports/stage13_vae_prior \
  --threshold-db -5 0 3 6 9 12 \
  --date 2022-06-06 \
  --calibration-mode track_purity \
  --calibration-date 2022-06-06 \
  --calibration-threshold-db 3 6 9 12 \
  --calibration-min-target-fraction 0.95 \
  --calibration-min-purity 0.95 \
  --score-threshold 0.5 --overwrite
```

Both scripts have `--self-test`. Requires PyTorch (they fail with a clear
message if `torch` is missing). **Stage 14** consolidates and ranks the
existing methods before any diffusion / model-zoo work.

## Stage 14 — Unified method benchmark and operating-point selection

A **consolidation** stage: it **adds no new model and retrains nothing**.
Stage 14 reads the compact report CSVs from stages 07 / 08 / 09 / 11 / 12.5
/ 13, normalizes every track-level keep/reject method into one operating-
point schema, and answers *which existing method to use before building
anything new*. It compares **Stage 09** (hand physics), **Stage 11**
(ADS-B marginal priors), **Stage 12.5** (deterministic sequence
autoencoders), and **Stage 13** (VAE) — with Stage 08 as the no-filter
baseline and Stage 07 as frame-level context.

It produces a method inventory (missing files flagged), unified metrics,
best method per detection threshold, matched-retention and matched-false-
reduction comparisons, a Pareto frontier, descriptive rankings, a runtime
inventory, and (when the large per-track score CSVs are present locally)
failure-case candidates. **Headline:** the deterministic **Stage 12.5**
sequence autoencoders are the strongest learned filter (at −5 dB, matched
to ~0.97 true retention, they remove ~100% of false tracks vs ~79% for
hand physics and ~30% for ADS-B priors); **Stage 13 VAE does not improve on
them**; **Stage 09 remains the strongest interpretable fallback**. The
report states plainly that the utility scores are descriptive, not absolute.

- **Inputs:** `reports/stage07..stage13` report directories (robust to
  missing files — it warns and continues).
- **Outputs:** `reports/stage14_method_benchmark/` — 11 CSVs
  (`unified_method_metrics.csv`, `best_method_by_threshold.csv`,
  matched/Pareto/rankings/runtime/inventory/failure-case), the report, and
  six plots in `plots/`.

```bash
python scripts/14_benchmark_methods.py \
  --stage07-dir reports/stage07_threshold_only \
  --stage08-dir reports/stage08_kalman_baseline \
  --stage09-dir reports/stage09_physics_scoring \
  --stage11-dir reports/stage11_adsb_prior_scoring \
  --stage12-dir reports/stage12_sequence_priors \
  --stage13-dir reports/stage13_vae_prior \
  --output-dir reports/stage14_method_benchmark \
  --date 2022-06-06 \
  --target-retention 0.97 \
  --target-false-reduction 0.95 \
  --overwrite
```

`--self-test` runs on tiny synthetic reports. **Stage 15** tests diffusion
specifically for denoising / gap filling.

## Stage 15 — Diffusion trajectory denoising and gap filling

A **narrowly scoped** study — **not** a new primary false-track classifier.
Stage 14 showed stage 12.5 already removes nearly all false tracks, so
Stage 15 instead asks whether a **DDPM-style diffusion denoiser** over the
shared normalized trajectory windows adds value for **noisy-track
regularization** and **short-gap filling**, with a denoising-residual
anomaly score kept explicitly secondary. It trains a lightweight temporal
1D-conv noise-prediction network on clean stage-5 relocated truth windows
(reusing the stage-12 normalizer) and evaluates three tasks on stage-8
tracks: (1) synthetic-corruption recovery + smoothness regularization, (2)
gap filling vs linear interpolation, (3) a track-purity-calibrated residual
score compared five-way against stage 12.5 / 13. Evaluation uses fast
single-step denoising (Mode A). **Stage 12.5 remains the primary false-track
suppression method unless diffusion clearly beats it.**

- **Training inputs:** `data/active/radar_truth_relocated/` (stage-4
  fallback); a whole day held out for validation.
- **Evaluation inputs:** stage-8 tracks (`data/active/tracks_kalman/`) +
  the trained denoiser in `models/diffusion_denoisers/`, plus the stage-12/13
  reports for the comparison table.
- **Outputs:** `diffusion_denoiser.pt` (git-ignored) + committed
  `diffusion_config.json` / `diffusion_training_manifest.json` /
  `diffusion_calibration.json`; compact tables, six plots, and
  `diffusion_denoising_report.md` in `reports/stage15_diffusion_denoising/`.

```bash
# Step A: train
python scripts/15_train_diffusion_denoiser.py \
  --truth-dir data/active/radar_truth_relocated \
  --models-dir models/diffusion_denoisers \
  --sequence-models-dir models/sequence_priors \
  --report-dir reports/stage15_diffusion_denoising \
  --window-len 20 --stride 5 --epochs 20 --batch-size 512 \
  --max-train-windows 200000 --max-val-windows 60000 \
  --holdout-date 2022-06-27 --hidden-dim 128 --num-blocks 4 \
  --num-diffusion-steps 100 --overwrite

# Step B: evaluate (thresholds -5/0/3/6 only; 9/12 dB have windowability
# denominator issues documented in stage 14)
python scripts/15_evaluate_diffusion_denoiser.py \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/diffusion_denoisers \
  --sequence-models-dir models/sequence_priors \
  --stage12-dir reports/stage12_sequence_priors \
  --stage13-dir reports/stage13_vae_prior \
  --report-dir reports/stage15_diffusion_denoising \
  --threshold-db -5 0 3 6 \
  --date 2022-06-06 --denoise-t 20 --score-threshold 0.5 --overwrite
```

Both scripts have `--self-test` and require PyTorch (clear failure message
if missing). Depending on results, **Stage 16** consolidates robustness and
ablation evidence for the winner before any final packaging.

## Stage 16 — Robustness and ablation study

A **consolidation** stage: it **adds no new model** and retrains nothing
(only the optional `--run-missing` may call the existing stage-12 scorer for
day/threshold combinations that lack compact scores). It reads the compact
reports from stages 08 / 09 / 12 / 14 / 15 and stress-tests the current best
method — **stage-12.5 deterministic sequence autoencoders with noise-matched
calibration** — answering whether the result holds beyond one configuration:
robustness by day and threshold, an **MLP/GRU/TCN model ablation**, a
**clean-truth vs track-purity calibration ablation**, score-threshold
sensitivity with recommended operating points, a **windowability audit** (the
stage-14 9/12 dB denominator caveat), range-bin robustness, and a failure-mode
summary, all rolled up into a key-findings table.

**Findings:** stage 12.5 holds across all four informative thresholds
(mean ~0.984 false reduction at ~0.978 true retention); **track-purity
calibration is necessary** (clean-truth retention ~0.08 → track-purity ~0.96);
**mlp_dae ranks best** with gru_ae very close and tcn_ae weaker; 9/12 dB have
**zero windowable false tracks** (caveat confirmed); and the evidence is
currently **single-day (2022-06-06)** — multi-day confirmation is the main
remaining gap. Stage 12.5 remains the recommended primary filter and Stage 09
the interpretable fallback.

- **Inputs:** the stage-08/09/12/14/15 report directories (robust to missing
  files); optionally `data/active/tracks_kalman/` for `--run-missing`.
- **Outputs:** `reports/stage16_robustness/` — 11 compact CSVs (inventory,
  by-day, by-threshold, model/calibration ablations, sensitivity,
  windowability, range-bin, failure-mode, key-findings), the report, and
  seven plots.

```bash
python scripts/16_robustness_ablation.py \
  --stage08-dir reports/stage08_kalman_baseline \
  --stage09-dir reports/stage09_physics_scoring \
  --stage12-dir reports/stage12_sequence_priors \
  --stage14-dir reports/stage14_method_benchmark \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/sequence_priors \
  --output-dir reports/stage16_robustness \
  --threshold-db -5 0 3 6 \
  --overwrite
```

Add `--include-high-thresholds` to fold 9/12 dB into the main tables (with the
documented caveat), or `--run-missing` to score any missing all-day
combinations first (may be expensive). `--self-test` runs on tiny synthetic
reports. Run Stage 16 **before final report packaging or a broader model
zoo** — it establishes whether the stage-12.5 result is stable enough to
build on.

## Stage 17 — Four-day validation

Closes the single-day gap flagged by Stage 16: does the selected **Stage 12.5**
method (deterministic sequence autoencoder, track-purity calibration) hold
across **all four ADS-B days** (2022-06-06/13/20/27) at thresholds −5/0/3/6 dB?
It **adds no new model** and retrains nothing; it consolidates per-day
stage-08/09/12.5 results and, only when `--run-missing-*` is passed, calls the
existing stage-08/09/12 scripts to generate any missing day/threshold outputs.
It validates two primary models (`mlp_dae`, `gru_ae`) with Stage 09 hand physics
as the interpretable fallback and Stage 08 Kalman-only as the denominator
baseline; 9/12 dB stay audit-only (windowability caveat). VAE/diffusion are
referenced as prior findings, not rerun.

It writes an input-availability matrix, a **run plan** with the exact commands
for anything missing, four-day stage-08/09 context, four-day stage-12 metrics,
per-day / per-threshold / overall summaries, an MLP-vs-GRU comparison, a
stage-09 fallback comparison, a windowability audit, a failure-case rollup, a
key-findings table, eight plots, and a report that states whether the single-day
limitation is **closed** or **still open**.

- **Inputs:** the stage-08/09/12/14 report dirs; optionally
  `data/active/tracks_kalman/` + stage-12 checkpoints for `--run-missing-*`.
- **Outputs:** `reports/stage17_four_day_validation/` — 14 CSVs, the report, and
  eight plots.

```bash
# consolidation only (no reruns)
python scripts/17_four_day_validation.py \
  --stage08-dir reports/stage08_kalman_baseline \
  --stage09-dir reports/stage09_physics_scoring \
  --stage12-dir reports/stage12_sequence_priors \
  --tracks-dir data/active/tracks_kalman \
  --models-dir models/sequence_priors \
  --output-dir reports/stage17_four_day_validation \
  --date 2022-06-06 2022-06-13 2022-06-20 2022-06-27 \
  --threshold-db -5 0 3 6 \
  --overwrite
```

```bash
# optional: actually generate the missing days (expensive; stage-6 detections
# and stage-12 checkpoints must exist locally)
python scripts/17_four_day_validation.py \
  --date 2022-06-06 2022-06-13 2022-06-20 2022-06-27 \
  --threshold-db -5 0 3 6 \
  --run-missing-stage08 --run-missing-stage09 --run-missing-stage12 \
  --overwrite
```

`--dry-run` writes only the availability matrix and run plan; `--self-test`
runs on tiny synthetic reports. When all four days are present the report
declares the single-day limitation **closed**; otherwise it stays **open** and
the run plan lists exactly what to run.

### Stage 17.5 — Reproducibility hardening

A bug-fix pass, **not a new model** — see
[`reports/stage17_four_day_validation/stage17p5_repro_hardening.md`](reports/stage17_four_day_validation/stage17p5_repro_hardening.md).
It fixes three defects found while running the four-day validation, with
regression checks so they cannot return:

1. **`sys.executable`** for every internal Python subprocess (a bare `python`
   does not exist on all machines).
2. **Calibration-overwrite prevention** — `--calibration-output` now defaults to
   `<--report-dir>/calibration/`, and orchestrators sandbox it inside their own
   output directory, so a per-day rerun can never clobber the canonical stage-12
   calibration artifact.
3. **Undefined false-reduction cells** — a zero false-track denominator is now
   `NaN` + an explicit `undefined_reason`, never `0`/`1` and never misattributed
   to missing stage-09 data. Aggregates average defined cells and report the
   undefined count.

```bash
python scripts/17p5_regression_checks.py   # 11 guards; run before packaging
```

Results are scientifically unchanged (canonical calibration byte-identical; no
diff in stage-14/16 key tables) — only labels, paths, and added diagnostic
columns.

## Stage 18 — Final report / paper package

Packages the completed stage 01–17.5 pipeline into a paper-ready deliverable.
**Stage 18 adds no new model and runs no new experiment**; it retrains nothing
and changes no scientific result. It reads the **hardened Stage 17.5 artifacts
as the source of truth** for headline metrics — four-day values are never
replaced by stale single-day ones.

It produces six final summary CSVs (results, method comparison, ablations,
reproducibility checklist, limitations, contributions), paper-facing tables, a
full `final_report.md`, **25 figures**, and a complete **Overleaf package**
(`main.tex` + `references.bib` + `figures/`, all referenced paths resolving).

> **Scope caveat carried through every artifact.** The radar model is a
> **point-detection simulation** — there is no raw RF/IQ and no gridded
> range-Doppler intensity simulation. The "pseudo range-Doppler" figures are
> **scatter plots of point detections** in (radial velocity, range) space, *not*
> raw radar RD intensity heatmaps. `references.bib` marks three entries as
> explicit placeholders rather than inventing citations.

- **Inputs:** `reports/stage07…stage17_four_day_validation/`; optionally
  `data/active/sim_detections_relocated/` and `tracks_kalman/` for illustrative
  figures (missing large files degrade to *reasoned* placeholders, never a crash).
- **Outputs:** `reports/stage18_final_package/` — report, `tables/`, `figures/`,
  `overleaf/`, and a manifest recording each figure as data-driven, schematic,
  copied, or placeholder.

```bash
python scripts/18_build_final_report.py \
  --reports-root reports \
  --output-dir reports/stage18_final_package \
  --project-title "ADS-B-Guided Weak-Target Radar Tracking Under Low SNR and Clutter" \
  --overwrite
```

`--self-test` builds a tiny package from synthetic mini-reports (and checks that
an absent detection file yields a *reasoned* pseudo-RD placeholder). Set
`--author` / `--institution` for the LaTeX title block.

## Stage 22 — F04 cube-derived detection evaluation

Stage 22 runs the existing F03 stack (stage 08 tracking, then stage 12.5
sequence-prior scoring) over **F04 cube-derived CFAR detections** instead of the
F02 point-detection simulator's output. It answers one question: *does the
learned sequence-prior filter still suppress false tracks when the detections
come from structured radar-cube CFAR output?*

It **retrains nothing**, adds no model, and does not modify F04 or the stage-08
tracker.

**Input.** One F04 Stage-21 export:
`../F04-RADAR-CUBE/data/active/f03_exports/f04_cube_detections_2022-06-06_cfar_ca_scale_6p0_cap_2000.csv`

### Two warnings that matter

**Small scope.** This is a **200-frame, single-day, single-operating-point**
stress test. The F02/F03 headline result is a four-day, multi-threshold
experiment. The F04 run is a structured-clutter stress test, **not** a
replacement for it, and the two differ in detection source, frame count, day
count and threshold semantics all at once — no single difference between them is
attributable to any one cause.

**CFAR scale is not dB.** F04's operating point is `threshold_scale = 6.0`, a
*linear multiplier* on the local CFAR noise estimate — about **7.78 dB** above
it. F03 discovers detection files as `detections_<date>_thr_<token>dB.csv` and
filters them with a numeric `--threshold-db`, so Stage 22 must create a
`thr_6p0dB` alias. That token is a **compatibility filename label only**; it is
*not* a 6 dB SNR threshold. Every Stage 22 artifact carries the warning, and
`data/active/f04_cube_detections/f04_cube_alias_metadata_<date>.json` records the
real operating point.

### Command

```bash
python scripts/22_evaluate_f04_cube_detections.py --self-test

python scripts/22_evaluate_f04_cube_detections.py \
  --f04-export ../F04-RADAR-CUBE/data/active/f03_exports/f04_cube_detections_2022-06-06_cfar_ca_scale_6p0_cap_2000.csv \
  --truth-dir data/active/radar_truth_relocated \
  --detections-dir data/active/f04_cube_detections \
  --tracks-dir data/active/tracks_kalman_f04_cube \
  --report-dir reports/stage22_f04_cube_eval \
  --date 2022-06-06 --cfar-type ca --threshold-scale 6.0 --cap 2000 \
  --max-frames 200 --run-stage08 --run-stage12 --overwrite
```

`--run-stage09` additionally attempts hand-physics scoring on the F04 tracks and
documents the outcome if the schema is incompatible.

### Import adaptations

Stage 22 never edits the F04 export. It writes an adapted copy plus an alias into
`data/active/f04_cube_detections/`, and records every adaptation in the alias
metadata JSON and the report. Currently one adaptation is required: the F04
Stage-21 writer serializes with `float_format="%.6g"`, which rounds epoch seconds
to six significant digits and collapses 200 distinct frame times onto 2 values.
Stage 08 is immune (it takes `dt` from `--frame-period-s`), but stage 12's window
features differentiate position with respect to `timestamp`, so the raw column
would silently corrupt every velocity feature. Stage 22 reconstructs
`timestamp = frame_id * frame_period_s` — F04 defines
`frame_id = floor(timestamp / scan_period_s)` — and refuses to proceed unless the
reconstruction rounds back to the value the export carries. **This is a real bug
in F04's Stage 21 writer and should be fixed there.**

### Calibration provenance

The noise-matched (`track_purity`) calibration is built from the **F04 cube
tracks themselves** by pinning `--calibration-tracks-dir` at the F04 track
directory. That flag defaults to the canonical F02 `data/active/tracks_kalman`,
which also contains a `thr_6p0dB` file — so omitting it would quietly calibrate
on F02 point-detection tracks while scoring F04 cube-CFAR tracks, reintroducing
exactly the noise mismatch stage 12.5 exists to remove. The calibration JSON is
written inside the Stage 22 report directory; the canonical stage-12 calibration
is never touched, and the validation gate hashes it to prove so.

### Outputs

```text
reports/stage22_f04_cube_eval/
  stage22_f04_cube_eval_report.md
  f04_import_summary.csv                  what was imported and what was adapted
  f04_stage08_summary.csv                 tracking on cube detections
  f04_stage12_summary.csv                 sequence-prior filter metrics
  f04_cube_vs_f02_point_comparison.csv    3 clearly-labelled rows, not a single delta
  stage22_key_findings.csv
  plots/  f04_detection_counts.png  f04_track_counts.png
          f04_stage12_filter_effect.png   f04_vs_f02_comparison.png
  stage08_kalman_f04_cube/   stage12_sequence_f04_cube/   generated/ (run logs)
```

Imported detection CSVs and F04 track CSVs are git-ignored (large, regenerable).

### Next stage

If stage 08 and stage 12.5 both completed, **Stage 23** should sweep 2–3 of the
Stage 20.5 CFAR operating points (`A_high_pd` scale 4 / cap 10000, `B_balanced`
scale 6 / cap 2000, `C_conservative` scale 12 / cap 2000) through F03, so
false-track difficulty can be traced against detector aggressiveness rather than
inferred from a single point. If they did not, Stage 23 should fix the adapter
first.

## Audit

`scripts/06_audit_relocated_experiment.py` is the read-only audit of the
relocated wide-area experiment (copied here from F02 because it is
evaluation, not simulation); its report lives at
`reports/relocated_experiment_audit.md`.
