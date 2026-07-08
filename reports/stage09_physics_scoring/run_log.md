# Stage 09 run log

## Initial run: 2022-06-06, thresholds -5, 0, 3, 6 dB (commit cc31bbd)

First real run, score threshold 0.5, runtime 3m37s. Superseded by the full
sweep below (same code, wider threshold set).

## Full one-day physics scoring: 2022-06-06, thresholds -5, 0, 3, 6, 9, 12 dB

Command:

```bash
python scripts/09_score_tracks_physics.py \
  --tracks-dir data/active/tracks_kalman \
  --report-dir reports/stage09_physics_scoring \
  --threshold-db -5 0 3 6 9 12 \
  --date 2022-06-06 \
  --score-threshold 0.5 \
  --overwrite
```

Completed successfully. Runtime **5m41s**; score threshold **0.5**;
180,194 confirmed tracks scored. Scored per-track CSVs (git-ignored) total
~100 MB in `data/active/tracks_scored_physics/`.

| threshold (dB) | stage 08 false tracks → stage 09 kept | false-track reduction | true-track retention | precision before → after |
|---:|---:|---:|---:|---:|
| −5 | 6,138 → 1,302 | 0.788 | 0.973 | 0.806 → 0.950 |
| 0 | 1,604 → 583 | 0.637 | 0.985 | 0.950 → 0.981 |
| 3 | 659 → 295 | 0.552 | 0.984 | 0.981 → 0.991 |
| 6 | 274 → 143 | 0.478 | 0.982 | 0.991 → 0.995 |
| 9 | 104 → 60 | 0.423 | 0.976 | 0.996 → 0.998 |
| 12 | 49 → 30 | 0.388 | 0.971 | 0.998 → 0.999 |

Median physics score: true tracks 0.851 vs false tracks 0.407 (separable).

Caveats:

- **These are one-day results for 2022-06-06 only** — treat as a baseline
  before any multi-day sweep.
- True/false track counts use the stage-9 definition (purity and
  target_fraction >= 0.8), stricter than the stage-8 report's 0.5-based
  definition, so stage-8 baseline numbers here differ slightly from the
  stage-8 run log.
- The association (Mahalanobis) penalty channel is unavailable for the
  baseline stage-8 outputs and is excluded from the score weighting.
- Score threshold 0.5 is untuned; see `physics_filter_sweep.csv` for the
  full retention-vs-reduction trade-off.
- **Stage 10 (empirical ADS-B prior learning) is intentionally deferred.**
  Stage 09 remains hand-designed physics scoring — not empirical prior
  learning and not VAE.

Observations:

- The filter's value concentrates exactly where clutter dominates: at
  −5 dB it removes 78.8% of false tracks for a 2.7% true-track cost.
- At high detection thresholds there is little left to clean (49 false
  tracks at 12 dB), and reduction naturally falls toward ~0.4 — the
  surviving false tracks are the kinematically plausible ones.
- The filtered −5 dB stream (precision 0.950, 24,812 true tracks kept)
  now rivals the unfiltered 0 dB stream's precision while feeding from the
  richest detection set — the operating point stage 10+ should exploit.
