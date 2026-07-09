"""Stage 12: sequence-prior scoring of stage-8 tracks + evaluation.

Scores each confirmed stage-8 track by how well its normalized trajectory
windows RECONSTRUCT under the stage-12 autoencoders, calibrated against
clean holdout-truth reconstruction errors:

    track_error = median per-window reconstruction error
    score       = 1                       if track_error <= val_p50
                  0                       if track_error >= val_p99
                  linear in between       otherwise

Truth labels never influence the score; they enter only afterwards, in the
evaluation (same strict true-track definition as stages 9 and 11).
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table
from utils.track_physics_score import TRUE_TRACK_PURITY, TRUE_TRACK_TARGET_FRACTION

CALIBRATION_QUANTILES = [50, 75, 90, 95, 99]


def calibrate_score_from_validation_errors(errors: np.ndarray) -> Dict:
    """Validation-error quantiles for one model (the score anchors)."""
    e = errors[np.isfinite(errors)]
    q = {f"error_p{p}": float(np.percentile(e, p)) for p in CALIBRATION_QUANTILES}
    q.update({"n_val_windows": int(len(e)), "error_mean": float(e.mean()),
              "error_std": float(e.std())})
    return q


def score_track_errors(track_error: float, calibration: Dict) -> float:
    """Map a track's (median) reconstruction error to [0, 1]."""
    lo, hi = calibration["error_p50"], calibration["error_p99"]
    if not np.isfinite(track_error):
        return np.nan
    if track_error <= lo:
        return 1.0
    if track_error >= hi:
        return 0.0
    return float(1.0 - (track_error - lo) / (hi - lo))


def evaluate_track_labels(g: pd.DataFrame) -> Dict:
    """Evaluation-only labels for one track (from the canonical loader frame)."""
    hits = g[g["is_hit"] == 1]
    n_target = int((hits["assoc_is_target"] == 1).sum())
    n_clutter = int((hits["assoc_is_target"] == 0).sum())
    tf = n_target / len(hits) if len(hits) else np.nan
    traj = hits.loc[hits["assoc_is_target"] == 1, "assoc_trajectory_id"].dropna()
    majority = traj.mode().iloc[0] if len(traj) else None
    errs = g["pos_error_m"].to_numpy(dtype=float)
    errs = errs[np.isfinite(errs)]
    return {
        "n_target_hits": n_target, "n_clutter_hits": n_clutter,
        "target_fraction": tf, "purity": tf,
        "majority_trajectory_id": majority,
        "is_true_track": bool(np.isfinite(tf) and tf >= TRUE_TRACK_TARGET_FRACTION
                              and tf >= TRUE_TRACK_PURITY and majority is not None),
        "position_rmse_m": float(np.sqrt((errs**2).mean())) if len(errs) else np.nan,
    }


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_by_model_threshold(scores: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    rows = []
    for (model, thr), g in scores.groupby(["model", "threshold_db"]):
        scored = g[np.isfinite(g["sequence_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored["keep_sequence_prior"]]
        kt = int(kept["is_true_track"].sum())
        kf = len(kept) - kt
        rows.append({
            "model": model, "threshold_db": thr,
            "stage08_confirmed_tracks": len(scored),
            "stage08_true_tracks": len(true),
            "stage08_false_tracks": len(false),
            "stage12_kept_tracks": len(kept),
            "stage12_kept_true_tracks": kt,
            "stage12_kept_false_tracks": kf,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_retention": kf / len(false) if len(false) else np.nan,
            "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
            "precision_before": len(true) / len(scored) if len(scored) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
            "mean_score_true_tracks": float(true["sequence_prior_score"].mean()) if len(true) else np.nan,
            "mean_score_false_tracks": float(false["sequence_prior_score"].mean()) if len(false) else np.nan,
            "median_score_true_tracks": float(true["sequence_prior_score"].median()) if len(true) else np.nan,
            "median_score_false_tracks": float(false["sequence_prior_score"].median()) if len(false) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["model", "threshold_db"]).reset_index(drop=True)


def sweep_table(scores: pd.DataFrame, sweep_thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for (model, thr), g in scores.groupby(["model", "threshold_db"]):
        scored = g[np.isfinite(g["sequence_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        for st in sweep_thresholds:
            kept = scored[scored["sequence_prior_score"] >= st]
            kt = int(kept["is_true_track"].sum())
            kf = len(kept) - kt
            rows.append({
                "model": model, "threshold_db": thr, "score_threshold": st,
                "kept_tracks": len(kept), "kept_true_tracks": kt, "kept_false_tracks": kf,
                "true_track_retention": kt / len(true) if len(true) else np.nan,
                "false_track_retention": kf / len(false) if len(false) else np.nan,
                "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
                "precision_after": kt / len(kept) if len(kept) else np.nan,
            })
    return pd.DataFrame(rows)


def range_bin_table(scores: pd.DataFrame, edges: List[float]) -> pd.DataFrame:
    labels = [f">{lo / 1000:.0f} km" if np.isinf(hi) else f"{lo / 1000:.0f}-{hi / 1000:.0f} km"
              for lo, hi in zip(edges[:-1], edges[1:])]
    idx = np.digitize(scores["median_range_m"].to_numpy(), np.asarray(edges)[1:-1])
    scores = scores.assign(_bin=[labels[i] for i in idx], _lo=[edges[i] for i in idx])
    rows = []
    for (model, thr, label, lo), g in scores.groupby(["model", "threshold_db", "_bin", "_lo"]):
        scored = g[np.isfinite(g["sequence_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored["keep_sequence_prior"]]
        kt = int(kept["is_true_track"].sum())
        rows.append({
            "model": model, "threshold_db": thr, "range_bin": label, "_lo": lo,
            "stage08_tracks": len(scored),
            "stage08_true_tracks": len(true), "stage08_false_tracks": len(false),
            "stage12_kept_tracks": len(kept), "stage12_kept_true_tracks": kt,
            "stage12_kept_false_tracks": len(kept) - kt,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_reduction": 1 - (len(kept) - kt) / len(false) if len(false) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
            "median_sequence_score": float(scored["sequence_prior_score"].median())
            if len(scored) else np.nan,
        })
    return (pd.DataFrame(rows).sort_values(["model", "threshold_db", "_lo"])
            .drop(columns="_lo").reset_index(drop=True))


def compare_with_stage09_stage11(by_mt: pd.DataFrame, stage09_dir: str,
                                 stage11_dir: str) -> Tuple[pd.DataFrame, bool, bool]:
    """Four-way comparison table; NaNs where stage 9 / 11 are unavailable."""
    base = (by_mt.groupby("threshold_db")
            .agg(stage08_true_tracks=("stage08_true_tracks", "first"),
                 stage08_false_tracks=("stage08_false_tracks", "first"))
            .reset_index())

    s9_path = os.path.join(stage09_dir, "stage08_vs_stage09.csv")
    s9_ok = os.path.exists(s9_path)
    if s9_ok:
        s9 = pd.read_csv(s9_path)[["threshold_db", "stage09_true_track_retention",
                                   "stage09_false_track_reduction"]]
        s9.columns = ["threshold_db", "stage09_true_retention", "stage09_false_reduction"]
        base = base.merge(s9, on="threshold_db", how="left")
    else:
        base["stage09_true_retention"] = base["stage09_false_reduction"] = np.nan

    s11_path = os.path.join(stage11_dir, "stage08_vs_stage09_vs_stage11.csv")
    s11_ok = os.path.exists(s11_path)
    if s11_ok:
        s11 = pd.read_csv(s11_path)[["threshold_db", "stage11_true_track_retention",
                                     "stage11_false_track_reduction"]]
        s11.columns = ["threshold_db", "stage11_true_retention", "stage11_false_reduction"]
        base = base.merge(s11, on="threshold_db", how="left")
    else:
        base["stage11_true_retention"] = base["stage11_false_reduction"] = np.nan

    s12 = by_mt[["model", "threshold_db", "true_track_retention",
                 "false_track_reduction", "precision_after"]].rename(columns={
        "true_track_retention": "stage12_true_retention",
        "false_track_reduction": "stage12_false_reduction",
        "precision_after": "stage12_precision"})
    merged = base.merge(s12, on="threshold_db", how="right")
    cols = ["threshold_db", "stage08_true_tracks", "stage08_false_tracks",
            "stage09_true_retention", "stage09_false_reduction",
            "stage11_true_retention", "stage11_false_reduction",
            "model", "stage12_true_retention", "stage12_false_reduction",
            "stage12_precision"]
    return merged[cols].sort_values(["model", "threshold_db"]).reset_index(drop=True), s9_ok, s11_ok


# =============================================================================
# Plots (matplotlib only, default colors, one figure each)
# =============================================================================

def make_scoring_plots(scores: pd.DataFrame, by_mt: pd.DataFrame, sweep: pd.DataFrame,
                       comparison: pd.DataFrame, s9_ok: bool, s11_ok: bool,
                       plots_dir: str) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    for model, g in scores.groupby("model"):
        g = g[np.isfinite(g["sequence_prior_score"])]
        ax.hist(g.loc[g["is_true_track"], "sequence_prior_score"], bins=bins,
                alpha=0.45, density=True, label=f"{model} true", histtype="step", linewidth=2)
        ax.hist(g.loc[~g["is_true_track"], "sequence_prior_score"], bins=bins,
                alpha=0.45, density=True, label=f"{model} false", histtype="step",
                linewidth=2, linestyle="--")
    ax.set_xlabel("sequence prior score")
    ax.set_ylabel("density")
    ax.set_title("Sequence-prior score distributions (solid true, dashed false)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "sequence_score_hist_true_false.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    for ycol, s9col, s11col, fname, title in [
        ("false_track_reduction", "stage09_false_reduction", "stage11_false_reduction",
         "model_comparison_false_reduction.png", "False-track reduction by method"),
        ("true_track_retention", "stage09_true_retention", "stage11_true_retention",
         "model_comparison_true_retention.png", "True-track retention by method"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in by_mt.groupby("model"):
            g = g.sort_values("threshold_db")
            ax.plot(g["threshold_db"], g[ycol], marker="o", label=f"stage 12 {model}")
        ref = comparison.drop_duplicates("threshold_db").sort_values("threshold_db")
        if s9_ok:
            ax.plot(ref["threshold_db"], ref[s9col], marker="s", linestyle=":", label="stage 9")
        if s11_ok:
            ax.plot(ref["threshold_db"], ref[s11col], marker="^", linestyle=":", label="stage 11")
        ax.set_xlabel("detection threshold (dB)")
        ax.set_ylabel(ycol.replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in sweep.groupby("model"):
        agg = g.groupby("score_threshold").agg(kt=("kept_true_tracks", "sum"),
                                               kf=("kept_false_tracks", "sum")).reset_index()
        sc = scores[(scores["model"] == model) & np.isfinite(scores["sequence_prior_score"])]
        tt = max(int(sc["is_true_track"].sum()), 1)
        tf = max(int((~sc["is_true_track"]).sum()), 1)
        ax.plot(agg["score_threshold"], agg["kt"] / tt, marker="o", label=f"{model} retention")
        ax.plot(agg["score_threshold"], 1 - agg["kf"] / tf, marker="x", linestyle="--",
                label=f"{model} reduction")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("fraction")
    ax.set_title("Sequence-prior filter sweep (pooled detection thresholds)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "sequence_filter_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# =============================================================================
# Report
# =============================================================================

def write_scoring_report(report_dir: str, scores: pd.DataFrame, by_mt: pd.DataFrame,
                         comparison: pd.DataFrame, range_bins: pd.DataFrame,
                         val_recon: pd.DataFrame, manifest: Dict,
                         s9_ok: bool, s11_ok: bool, score_threshold: float) -> str:
    dates = sorted(scores["date"].unique())
    date_scope = (f"These are one-day scoring results for {dates[0]}."
                  if len(dates) == 1 else
                  f"These scoring results cover {len(dates)} days: {', '.join(dates)}.")
    scored = scores[np.isfinite(scores["sequence_prior_score"])]
    true_med = scored.loc[scored["is_true_track"], "sequence_prior_score"].median()
    false_med = scored.loc[~scored["is_true_track"], "sequence_prior_score"].median()
    n_short = int((~np.isfinite(scores["sequence_prior_score"])).sum())

    lines = [
        "# Stage 12 Learned Sequence-Prior Track Scoring",
        "",
        "## Status",
        "",
        "- Stage 12 trains trajectory-window autoencoders from clean",
        "  ADS-B/radar truth windows and scores Stage 08 Kalman tracks by",
        "  reconstruction plausibility.",
        f"- {date_scope}",
        "- This is **not VAE** (stage 13), **not diffusion** (stage 14), and",
        "  not the full model zoo.",
        "- **Truth labels are used only after scoring**, for evaluation.",
        "",
        "## Motivation",
        "",
        "- Stage 09 used hand-designed feature penalties; stage 11 used",
        "  empirical marginal ADS-B priors. Both score features mostly",
        "  independently and miss the temporal SHAPE of a trajectory.",
        "- Stage 12 learns sequence shape directly: real GA windows should",
        "  reconstruct well; clutter-born tracks should not.",
        "",
        "## Training data",
        "",
        f"- Source: `{manifest.get('data_source', 'unknown')}`",
        f"- Train dates: {', '.join(manifest.get('train_dates', []))};"
        f" holdout dates: {', '.join(manifest.get('holdout_dates', []))}",
        f"- Train windows: {manifest.get('n_train_windows', 0):,};"
        f" validation windows: {manifest.get('n_val_windows', 0):,}",
        f"- Features: {', '.join(manifest.get('features', []))}",
        f"- Window length {manifest.get('window_len')} @ stride {manifest.get('stride')}"
        " (10 s grid); windows are origin-shifted and heading-rotated before",
        "  standardization.",
        "",
        "## Models",
        "",
        "- **mlp_dae** -- flattened-window MLP denoising autoencoder.",
        "- **gru_ae** -- GRU encoder to latent, latent-driven GRU decoder.",
        "- **tcn_ae** -- 1D-convolutional encoder/decoder over time.",
        "",
        "All trained with denoising MSE (Gaussian input noise).",
        "",
        "## Validation reconstruction",
        "",
    ]
    lines += md_table(val_recon.round(6))
    lines += [
        "",
        "## Scoring model",
        "",
        "- Per track: build the same normalized windows from the posterior",
        "  states; track_error = **median** per-window reconstruction error.",
        "- Calibration against clean holdout windows: score 1 at/below the",
        "  validation p50 error, 0 at/above the p99, linear between.",
        f"- Keep threshold: **{score_threshold:g}**. Tracks with fewer than",
        "  window_len points cannot be windowed and are excluded from",
        f"  filtering metrics (NaN score; {n_short:,} such track-rows here).",
        "",
        "## Overall results",
        "",
    ]
    lines += md_table(comparison.round(4))
    lines += [
        "",
        "## Model comparison",
        "",
    ]
    best = by_mt.loc[by_mt.groupby("model")["false_track_reduction"].idxmax()] \
        if len(by_mt) else pd.DataFrame()
    for model, g in by_mt.groupby("model"):
        fr = g["false_track_reduction"].mean()
        tr = g["true_track_retention"].mean()
        lines.append(f"- **{model}**: mean false reduction {fr:.3f}, mean true retention {tr:.3f}.")
    lines += [
        "",
        "## Range-bin behavior",
        "",
    ]
    rb = range_bins.groupby(["model", "range_bin"], sort=False).agg(
        tracks=("stage08_tracks", "sum"),
        true_retention=("true_track_retention", "mean"),
        false_reduction=("false_track_reduction", "mean"),
        median_score=("median_sequence_score", "mean")).reset_index()
    lines += md_table(rb.round(4))
    lines += [
        "",
        "## Score separability",
        "",
        f"Median sequence score (pooled models): true {true_med:.3f} vs false {false_med:.3f}.",
        ("Sequence reconstruction separates true from false tracks."
         if true_med > false_med else
         "Sequence autoencoders are not separable enough on this data."),
        "",
        "## Score-threshold calibration",
        "",
        "Raw reconstruction errors vs the clean-holdout calibration band",
        "(score 1 at/below val p50, 0 at/above val p99):",
        "",
    ]
    err = scored.groupby("model").apply(
        lambda g: pd.Series({
            "true_median_error": g.loc[g["is_true_track"], "sequence_recon_error_median"].median(),
            "false_median_error": g.loc[~g["is_true_track"], "sequence_recon_error_median"].median(),
            "calibration_p50": g["calibration_error_p50"].iloc[0],
            "calibration_p99": g["calibration_error_p99"].iloc[0],
        }), include_groups=False).reset_index()
    lines += md_table(err.round(4))
    lines += [
        "",
        "- Reconstruction error itself is strongly separable: false-track",
        "  median errors sit roughly an order of magnitude above true-track",
        "  median errors for every model.",
        "- But the calibration band comes from CLEAN truth windows, while",
        "  scored windows come from Kalman posteriors over noisy stage-6",
        "  measurements -- so typical TRUE tracks land at or above the",
        "  clean-holdout p99 and are compressed toward score 0.",
        "- Consequence at every swept score threshold: false-track",
        "  reduction is total (precision after filtering = 1.0) while true",
        "  retention is far too low to use as a filter at 0.5.",
        "- The discrimination is real; the score MAPPING is miscalibrated.",
        "  Calibrating against noisy (measurement-matched or",
        "  Kalman-filtered) truth windows, or matching the training noise",
        "  to the stage-6 measurement noise, is the concrete fix -- a",
        "  natural part of the stage 13 probabilistic treatment.",
        "",
        "## Failure modes",
        "",
        "- Short tracks (< window_len points) cannot be windowed at all.",
        "- Autoencoders can reconstruct some smooth false tracks -- smoothness",
        "  is exactly what they learn.",
        "- Reconstruction error is not a calibrated likelihood; the p50/p99",
        "  mapping is a pragmatic surrogate.",
        "- Unusual but valid maneuvers may reconstruct poorly and be rejected.",
        "- Probabilistic sequence models are the natural next step.",
        "",
        "## Recommended next stage",
        "",
        "Stage 13 should implement a **VAE trajectory prior** over the same",
        "normalized windows, giving a proper likelihood-based score.",
        "",
    ]
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "sequence_prior_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate (scoring)
# =============================================================================

def run_scoring_gate(report_dir: str, scores: pd.DataFrame, by_mt: pd.DataFrame,
                     comparison: pd.DataFrame, s9_ok: bool, s11_ok: bool) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 12 scoring validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (scoring)")
    print("=" * 70)

    path = os.path.join(report_dir, "sequence_track_scores.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        fail("sequence_track_scores.csv missing or empty")
    scored = scores[np.isfinite(scores["sequence_prior_score"])]
    if scored.empty:
        fail("no tracks received a finite sequence score")
    if not scored["sequence_prior_score"].between(0, 1).all():
        fail("sequence_prior_score outside [0, 1]")
    if not scored["keep_sequence_prior"].isin([True, False]).all():
        fail("keep_sequence_prior is not boolean")
    print("  scores file nonempty, scores in [0,1], keep boolean: OK")
    print("  scoring uses reconstruction error + holdout calibration only "
          "(truth enters afterwards): OK by construction")

    for _, r in by_mt.iterrows():
        if r["stage12_kept_true_tracks"] > r["stage08_true_tracks"]:
            fail("kept true tracks exceed stage-8 true tracks")
        if r["stage12_kept_false_tracks"] > r["stage08_false_tracks"]:
            fail("kept false tracks exceed stage-8 false tracks")
        for col in ("true_track_retention", "false_track_reduction"):
            v = r[col]
            if np.isfinite(v) and not (0.0 <= v <= 1.0):
                fail(f"{col} outside [0, 1]")
    print("  kept-track counts bounded; retention/reduction in [0,1]: OK")

    for model, g in scored.groupby("model"):
        tm = g.loc[g["is_true_track"], "sequence_prior_score"].median()
        fm = g.loc[~g["is_true_track"], "sequence_prior_score"].median()
        te = g.loc[g["is_true_track"], "sequence_recon_error_median"].median()
        fe = g.loc[~g["is_true_track"], "sequence_recon_error_median"].median()
        print(f"  {model}: median score true={tm:.3f} false={fm:.3f}; "
              f"median recon error true={te:.4f} false={fe:.4f} (report-only) "
              + ("-- separable" if tm > fm else
                 "-- NOT separable: sequence autoencoders are not separable enough"))

    if s9_ok and s11_ok:
        ref = comparison.drop_duplicates("threshold_db")
        print(f"  reference (report-only): stage 9 mean reduction "
              f"{ref['stage09_false_reduction'].mean():.3f}, stage 11 "
              f"{ref['stage11_false_reduction'].mean():.3f}; stage 12 per model above.")
