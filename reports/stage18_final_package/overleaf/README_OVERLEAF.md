# Overleaf package

Upload this whole `overleaf/` directory to Overleaf (or compile locally).

```
main.tex          complete LaTeX article
references.bib    bibliography (see the caveat below)
figures/          25 PNG figures, all referenced paths resolve
```

Compile with `pdflatex -> bibtex -> pdflatex -> pdflatex`, or just press Recompile
in Overleaf (it runs bibtex automatically).

## Figure scope caveat

`figures/03_pseudo_range_doppler_frame_low_threshold.png` and its high-threshold
counterpart are **point-detection pseudo range-Doppler visualizations, not raw radar
intensity maps.** The simulation produces per-scan point detections, not gridded
range-Doppler intensity. The caption in `main.tex` states this; please keep it.

## Bibliography caveat

`references.bib` contains **exact** entries for Kalman (1960), the OpenSky network,
Kingma \& Welling (VAE), and Ho et al. (DDPM). Three entries -- `tbd`, `adsbpred`
and `aeanomaly` -- are deliberate **placeholder `@misc` entries** with descriptive
titles and notes, because the specific references were not fixed in this project's
documentation. Replace them with the concrete citations from your literature review
before submission. No citation here was invented to look authoritative.

## Author fields

`main.tex` currently reads:

```
\author{Your Name \\ Your Institution}
```

Set `--author` and `--institution` when running `scripts/18_build_final_report.py`,
or edit `main.tex` directly.
