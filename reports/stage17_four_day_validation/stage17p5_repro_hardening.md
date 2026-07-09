# Stage 17.5 Reproducibility Hardening

## Purpose

This is a **bug-fix and reproducibility hardening pass, not a new model and not a
new research stage.** No model was retrained, no research logic was changed, and
no score was recomputed. Stage 17's four-day conclusion stands exactly as
published; this pass makes it *reproducible on another machine* and prevents
five concrete defects — four found while running the Stage 17 four-day
validation, one found during this pass — from silently returning.

The scientific result is unchanged:

| model | pooled true retention | pooled false reduction | false tracks kept |
|---:|---:|---:|---:|
| `mlp_dae` | 0.973 | 0.9939 | 5 / 815 |
| `gru_ae` | 0.972 | 0.9791 | 17 / 815 |

## Issues fixed

Four were carried over from Stage 17; a fifth (§5) was discovered during this pass.

### 1. `sys.executable` for internal subprocess calls

Orchestrators built command lists starting with the literal `"python"`, which does
not exist on this machine (only `python3`). Every internal Python invocation now
uses `sys.executable`, so an orchestrated rerun uses the *same interpreter and
environment* as the parent process.

- `scripts/16_robustness_ablation.py` — `maybe_run_missing()` stage-12 command.
- `scripts/17_four_day_validation.py` — stage-08, stage-09 and stage-12 commands
  in `run_missing()`. The earlier string-substitution workaround was removed in
  favour of building the list correctly and **asserting** `cmd[0] == sys.executable`.

User-facing README examples and the `run_plan.csv` command column still show
`python …` for readability — those are documentation, not automation.

### 2. Calibration-overwrite prevention

`scripts/12_score_tracks_sequence_prior.py` defaulted `--calibration-output` to the
**canonical** path
`reports/stage12_sequence_priors/calibration/sequence_track_calibration.json`.
Any per-day rerun that omitted the flag therefore overwrote the committed
2022-06-06 calibration with that day's band — silently desynchronizing the Stage 12
report from its own calibration artifact. This actually happened during Stage 17 and
was caught in `git status` before it was committed.

Fixes, defence in depth:

- **Safer default.** `--calibration-output` now defaults to `None` and is resolved by
  `resolve_calibration_output()` to `<--report-dir>/calibration/…`. A normal Stage 12
  run (`--report-dir reports/stage12_sequence_priors`) still produces the canonical
  path, so behaviour is unchanged; but a run pointed anywhere else can no longer
  reach into the canonical directory.
- **Orchestrators sandbox themselves.** Stage 17 writes per-day calibration to
  `reports/stage17_four_day_validation/generated/stage12/<date>/calibration/`; Stage 16
  writes to its own `generated_calibration/`. Both pass `--calibration-output` explicitly.
- **Help text** on both `12_score_tracks_sequence_prior.py` and
  `12_calibrate_sequence_prior.py` now states who may target the canonical path.

### 3. Undefined false-reduction cells handled explicitly

False-track reduction is `1 - kept_false / total_false`. When a cell has **no false
tracks in its denominator** — at high thresholds, or when the sequence methods have no
*windowable* false tracks — the reduction is **undefined**. It is not `0` (a total
failure), not `1` (a perfect filter), and it is **not** evidence that Stage 09 data is
missing. The Stage 17 fallback table previously labelled such cells
`"stage-09 unavailable"`, which was wrong on both counts (Stage 09 had data), and the
win count reported a pessimistic `14/16`.

Fixes:

- New shared helpers in `utils/common.py`: `safe_reduction()`, `safe_ratio()`,
  `undefined_reason()`, `summarize_defined()`. A zero or non-finite denominator always
  yields `NaN`.
- `utils/method_benchmark.py` and `utils/four_day_validation.py` now compute every
  reduction/ratio through those helpers.
- New columns: `false_reduction_defined`, `false_reduction_denominator`,
  `undefined_reason` (Stage 17), plus `n_false_reduction_defined` /
  `n_false_reduction_undefined` on every Stage 17 summary table.
- Aggregates average **defined cells only** and report the undefined count alongside.
- Key finding F3 now reads **`14/14 (2 undefined)`** instead of `14/16`.

The two undefined cells are 6 dB on 2022-06-13 and 2022-06-27, both with
`false_reduction_denominator = 0` and reason
`"undefined: no windowable false tracks for this cell"`.

### 4. Large generated artifacts remain git-ignored

Verified (not assumed) via `git check-ignore` that the following are ignored:

- `reports/stage17_four_day_validation/generated/**/sequence_track_scores.csv`
- `reports/stage17_four_day_validation/generated/**/physics_track_scores.csv`
- `reports/stage12_sequence_priors/sequence_track_scores.csv`
- `data/active/tracks_kalman/*.csv`
- `models/**/*.pt`

And that compact artifacts are **still tracked**: the Stage 17 summary CSVs and
`reports/stage12_sequence_priors/calibration/sequence_track_calibration.json`.

### 5. Stage 16 mislabelled single-day evidence as four-day

Found *during* this hardening pass, not during Stage 17. `utils/robustness_analysis.py`
`detect_dates()` took the **union** of stage-08 and stage-12 dates. Once Stage 17
generated stage-08 tracks for all four days, Stage 16 began reporting
`"4 days aggregated"` and `F6 -> n_days = 4`, even though the stage-12 metrics it
actually audits are still a single day (2022-06-06). That is a false claim about the
strength of the evidence, and it would have been committed silently.

Fix: `detect_dates()` now derives dates from the **stage-12 evidence** (its subject),
falling back to stage-08 coverage only when no stage-12 per-track scores exist.
Stage 16 again reports `SINGLE-DAY evidence only` and `F6 -> 1`, which is correct —
closing that gap is precisely what Stage 17 is for.

## Regression checks

`scripts/17p5_regression_checks.py` pins all five issues (12 assertions, all passing).
It monkeypatches `subprocess.run` to capture the command lists the orchestrators
actually build, rather than trusting the source to look right.

| id | check |
|---|---|
| R1a | Stage-17 run-missing commands all start with `sys.executable` |
| R1b | No `["python", …]` command construction anywhere in `scripts/` or `utils/` |
| R2a | Stage-17 passes `--calibration-output` inside its own output dir |
| R2b | Stage-16 sandboxes its calibration output |
| R2c | Omitting `--calibration-output` follows `--report-dir`, never the canonical dir |
| R3a | `safe_reduction` returns NaN on a zero denominator (and `1.0` when kept=0 with a real denominator) |
| R3b | `undefined_reason` cites the denominator and never blames Stage 09 |
| R3c | The fallback table marks zero-denominator cells undefined, not "stage-09 unavailable" |
| R3d | Aggregates exclude undefined cells and count them |
| R4a | Large per-track CSVs and checkpoints are git-ignored |
| R4b | Compact reports and the calibration JSON remain tracked |
| R5a | Stage-16 dates itself from stage-12 evidence (1 day), not stage-08 coverage (4 days) |

R1b was verified with a **negative control**: injecting `_BAD = ["python", "scripts/x.py"]`
into `utils/common.py` made it fail, and removing it made it pass again — so the check
detects a real regression rather than passing vacuously.

Also re-run and passing:

```text
python scripts/12_score_tracks_sequence_prior.py --self-test --calibration-mode track_purity
python scripts/12_score_tracks_sequence_prior.py --self-test
python scripts/16_robustness_ablation.py --self-test
python scripts/17_four_day_validation.py --self-test
python scripts/17p5_regression_checks.py
```

## Result

**The reports remain scientifically unchanged; only labels, paths and added
diagnostic columns differ.**

Evidence:

- The canonical Stage 12 calibration JSON is **byte-identical** before and after the
  Stage 17 consolidation rerun (md5 `fcd676b6…`, and `git diff` is empty).
- `reports/stage14_method_benchmark/best_method_by_threshold.csv` and
  `reports/stage16_robustness/robustness_by_threshold.csv` show **no diff** after
  regenerating with the hardened code.
- Stage 17 consolidation still reports all four days complete, `mlp_dae` best,
  pooled false reduction 0.994 at 0.973 retention — identical to the published run.

The only substantive report changes are **corrections**, not different results:

- Two mislabelled Stage 17 cells are now `undefined` (previously
  `"stage-09 unavailable"`), and F3 reads `14/14 (2 undefined)` instead of `14/16`.
  Those cells were never Stage 12 losses.
- Stage 16 again states `SINGLE-DAY evidence only` / `F6 -> 1` after the `detect_dates`
  fix, instead of the false `4 days aggregated` it briefly produced once Stage 17 had
  populated four days of stage-08 tracks.
