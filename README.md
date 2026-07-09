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
if missing). Depending on results, **Stage 16** will be either a compact
model-zoo benchmark of only the promising families, or a robustness /
ablation study across all four days and clutter/noise levels.

## Audit

`scripts/06_audit_relocated_experiment.py` is the read-only audit of the
relocated wide-area experiment (copied here from F02 because it is
evaluation, not simulation); its report lives at
`reports/relocated_experiment_audit.md`.
