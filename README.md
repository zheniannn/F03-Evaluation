# F03-Evaluation

Evaluation stages of the light-GA radar pipeline. **F03-Evaluation consumes
outputs from F02-RADAR** (which ends at stage 6 synthetic detections):

- **Stage 07** — threshold-only baseline evaluation (no tracking)
- **Stage 08** — constant-velocity Kalman detect-then-track baseline
- **Stage 09+** (future) — physics-guided / advanced tracking evaluation

Upstream: F01-PREPROCESSING owns stages 1–4 (ADS-B → uniform trajectories),
F02-RADAR owns stages 5–6 (radar truth → synthetic detections).

## Structure

```
F03-Evaluation/
├── scripts/
│   ├── 06_audit_relocated_experiment.py   # read-only audit of the F02 experiment
│   ├── 07_evaluate_threshold_only.py      # stage 7: threshold-only baseline
│   └── 08_run_kalman_baseline.py          # stage 8: CV-Kalman detect-then-track
├── utils/
│   ├── threshold_eval.py                  # stage 7 logic
│   ├── kalman_tracker.py                  # stage 8 tracker (CV-KF + greedy gated NN)
│   └── track_eval.py                      # stage 8 evaluation + report
├── data/active/
│   ├── radar_truth_relocated/             # INPUT from F02 (git-ignored)
│   ├── sim_detections_relocated/          # INPUT from F02 (git-ignored)
│   └── tracks_kalman/                     # stage-8 track output (git-ignored)
└── reports/
    ├── relocated_experiment_audit.md
    ├── stage07_threshold_only/            # committed compact tables/plots/report
    └── stage08_kalman_baseline/           # stage-8 summary + report
```

## Input contract from F02-RADAR

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

## How to copy inputs from F02-RADAR

```bash
cp ../F02-RADAR/data/active/radar_truth_relocated/radar_truth_*.csv \
   data/active/radar_truth_relocated/

cp ../F02-RADAR/data/active/sim_detections_relocated/detections_*.csv \
   data/active/sim_detections_relocated/

cp ../F02-RADAR/data/active/sim_detections_relocated/sim_detection_summary.csv \
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

Track CSVs go to `data/active/tracks_kalman/` (git-ignored); the summary
CSV and Markdown report go to `reports/stage08_kalman_baseline/`.
`--self-test` runs without real data. Documented baseline simplifications
(stage 9+ targets): isotropic measurement noise, Euclidean rather than
Mahalanobis gating, radial velocity unused.

## Audit

`scripts/06_audit_relocated_experiment.py` is the read-only audit of the
relocated wide-area experiment (copied here from F02 because it is
evaluation, not simulation); its report lives at
`reports/relocated_experiment_audit.md`.
