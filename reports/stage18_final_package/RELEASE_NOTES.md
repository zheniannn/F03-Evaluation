# Stage 18 Final Package Release Notes

## Release

- **Tag:** `stage18-final-package-v1`
- **Tagged commit:** the release commit that adds these notes
  (resolve with `git rev-list -n 1 stage18-final-package-v1`)
- **Parent (Stage 18 package commit):** `4b89cbf`
- **Date:** 2026-07-09
- **Branch:** main
- **Remote:** git@github.com:zheniannn/F03-Evaluation.git

> A hash cannot be self-referentially embedded in the commit it names, so the tagged
> commit is identified by the tag itself rather than a hardcoded (and necessarily wrong)
> hash. The Stage 18 package content is frozen at parent `4b89cbf`.

## Scope

This release freezes the completed ADS-B-guided weak-target radar tracking evaluation
pipeline through Stage 18.

It includes:

- final Markdown report (`final_report.md`)
- Overleaf-ready LaTeX package (`overleaf/`)
- final result tables (6 summary CSVs + 5 paper-facing tables)
- 25 final figures (25 mirrored into `overleaf/figures/`)
- manifest (`stage18_final_package_manifest.json`) and reproducibility notes

## Final validated claim

Across four days and thresholds -5/0/3/6 dB, the selected Stage 12.5 MLP denoising
autoencoder retained **97.3%** of true Kalman tracks while reducing the strict/windowable
false-track set from **815 to 5**.

Companion result: `gru_ae` retained 97.2% while reducing 815 to 17. Stage 09 hand-designed
physics scoring remains the recommended interpretable fallback.

> **Which false tracks?** Two false-track populations exist in this project and must not be
> conflated. Stage 08 labels every confirmed track by its own criterion (purity >= 0.5),
> giving 34,773 false tracks over four days and four thresholds. All filter metrics — including
> the headline above — are measured on the stricter, **windowable** subset (purity >= 0.8 and
> long enough to form a window): **815** false tracks.

## Important caveat

This is a radar **point-detection simulation** and evaluation framework. It is **not** raw
RF/IQ simulation and **not** a true gridded range-Doppler intensity simulation. Pseudo
range-Doppler figures are **point-detection scatter visualizations**, not raw radar heatmaps.

Additionally, `overleaf/references.bib` contains three deliberate placeholder `@misc` entries
(`tbd`, `adsbpred`, `aeanomaly`); no citation was invented. Replace them with concrete
references before submission — see `overleaf/README_OVERLEAF.md`.

## Reproducibility status

- Stage 17.5 reproducibility hardening completed (12 regression checks, incl. a negative control).
- Canonical calibration overwrite protection added.
- Internal subprocess calls use `sys.executable`.
- Undefined false-reduction cells are handled explicitly (zero denominator => NaN + reason,
  never 0/1 and never misattributed to missing Stage 09 data).
- Large generated CSVs, detection files, track files and model checkpoints are git-ignored.
- Stage 18 package validation passed (8/8 read-only checks).

### Release-hygiene fix applied in this release

Two large per-track score CSVs had been tracked since Stages 09/11, predating the ignore
rules used for their Stage 12/13/15/17 siblings:

- `reports/stage09_physics_scoring/physics_track_scores.csv` (~33 MB)
- `reports/stage11_adsb_prior_scoring/adsb_prior_track_scores.csv` (~48 MB)

They are now **untracked and git-ignored** (local copies retained). They are regenerable stage
outputs, and the only consumer — Stage 14's failure-case analysis — already degrades gracefully
when they are absent. **Git history still contains these blobs**; purging them would require a
history rewrite, which was deliberately **not** performed because it would invalidate the
published commit hashes (including `4b89cbf`) and the pushed branch.

## Package validation summary

| check | result |
|---|---|
| `final_report.md` required strings | 8/8 present |
| `overleaf/main.tex` structure | 10 sections, 10 `\includegraphics` |
| Overleaf figure paths resolve | 10/10 |
| Manifest valid JSON | 25 figures (21 data-driven, 2 schematic, 2 copied) |
| Figure counts | 25 figures, 25 Overleaf figures |
| Required final CSVs | 6/6 |
| Bibliography citations resolve | 7/7, no orphans |
| Tracked package file > 1 MB | none (largest 127 KB) |

## Recommended use

- Use `reports/stage18_final_package/overleaf/` for Overleaf submission/review
  (or upload `overleaf_stage18_final_package.zip`).
- Use `reports/stage18_final_package/final_report.md` for quick technical review.

## Reproducing the package

```bash
python scripts/17p5_regression_checks.py          # reproducibility guards
python scripts/18_build_final_report.py --self-test
python scripts/18_build_final_report.py --overwrite
```

No model training or expensive scoring is required to rebuild the package; it reads the
committed compact stage reports.
