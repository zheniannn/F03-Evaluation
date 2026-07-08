# Stage 08 run log

Incremental evaluation runs of the constant-velocity Kalman detect-then-track
baseline. Tracker configuration for every run below: gate 2500 m,
q_accel 1 m/s², confirm after 3 hits, delete after 3 misses, frame period
10 s. Truth labels were used only in evaluation, never by the tracker.

## Smoke test: 2022-06-06, threshold 0 dB, first 1,000 frames

Historical first run (commit 239a339). 129,171 detections; 2,369 confirmed
tracks (2,176 true / 193 false); trajectory coverage 0.997; fragmentation
1.63; target absorption 98.0%; clutter absorption 4.0%. **These results are
from the first 1,000 frames only and are not full-day performance.**

## Full-day baseline: 2022-06-06, threshold 0 dB

Command:

```bash
python scripts/08_run_kalman_baseline.py \
  --detections-dir data/active/sim_detections_relocated \
  --truth-dir data/active/radar_truth_relocated \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage08_kalman_baseline \
  --threshold-db 0 --date 2022-06-06 --overwrite
```

- Completed successfully. Runtime **2m29s**; track CSV **410 MB**
  (git-ignored).
- 8,639 frames; 1,705,595 detections in.
- 32,158 confirmed tracks: **30,578 true / 1,580 false**.
- Trajectory coverage **0.986** (18,171 / 18,426 trackable); fragmentation 1.68.
- Target detection absorption 98.2%; clutter absorption 4.0%.
- Position RMSE over true tracks: mean 219.7 m, median 206.1 m
  (posterior track position vs truth position of the associated detection).

Caveats: single day, single threshold, untuned default tracker parameters.
The smoke test's coverage (0.997) was slightly optimistic vs the full day
(0.986) — always quote the full-day number.

## One-day threshold sweep: 2022-06-06, thresholds -5, 0, 3, 6 dB

Command:

```bash
python scripts/08_run_kalman_baseline.py \
  --detections-dir data/active/sim_detections_relocated \
  --truth-dir data/active/radar_truth_relocated \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage08_kalman_baseline \
  --threshold-db -5 0 3 6 --date 2022-06-06 --overwrite
```

Completed successfully. Runtime **11m04s** for all four thresholds. Track
CSVs (git-ignored): 543 MB (−5 dB), 410 MB (0 dB), 345 MB (3 dB),
284 MB (6 dB).

| threshold (dB) | detections in | confirmed | true | false | coverage | fragmentation | target abs. | clutter abs. | RMSE mean/median (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −5 | 2,221,123 | 31,626 | 25,501 | 6,125 | 0.985 (18,157/18,426) | 1.40 | 0.990 | 0.069 | 228.2 / 199.2 |
| 0 | 1,705,595 | 32,158 | 30,578 | 1,580 | 0.986 (18,171/18,426) | 1.68 | 0.982 | 0.040 | 219.7 / 206.1 |
| 3 | 1,417,466 | 33,826 | 33,174 | 652 | 0.977 (17,968/18,389) | 1.85 | 0.970 | 0.028 | 211.1 / 200.7 |
| 6 | 1,147,885 | 31,927 | 31,620 | 307 | 0.950 (16,833/17,726) | 1.88 | 0.954 | 0.020 | 195.4 / 184.8 |

Caveats:

- **These results are for 2022-06-06 only and should be treated as a
  one-day baseline before the full 4-day sweep.**
- The coverage denominator ("trackable trajectories" = ≥5 target detections
  at that threshold) is itself threshold-dependent: it shrinks from 18,426
  at −5/0 dB to 17,726 at 6 dB, because weak far trajectories stop being
  detected at all. Coverage numbers therefore compare recovery of what was
  detectable at each threshold, not of a fixed trajectory set.
- Thresholds 9 and 12 dB were deliberately not run yet (per the incremental
  plan). Tracker parameters are untuned defaults.

Observations:

- Coverage stays within 0.95–0.99 across the sweep while frame-level Pd
  (stage 7) spans 0.79 → 0.47: temporal integration is doing exactly what
  it should.
- False tracks fall 6,125 → 307 as the threshold rises — the main cost of
  a low threshold is clutter-born tracks, not lost trajectories.
- Fragmentation rises with threshold (1.40 → 1.88): sparser detections
  break trajectories into more track segments.
