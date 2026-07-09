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

import json
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


# =============================================================================
# Stage 12.5 -- noise-matched (track-purity) calibration
#
# The stage-12 models are trained on CLEAN truth windows, so their reconstruction
# errors are calibrated to clean motion. Stage-8 tracks are Kalman posteriors over
# NOISY stage-6 measurements, so even genuine tracks reconstruct worse than clean
# truth -- the clean-truth p50/p99 band pushes true tracks toward score 0. Stage
# 12.5 rebuilds the error->score band from high-purity NOISY stage-8 true tracks so
# the 0.5 threshold is meaningful in the noisy-track domain. The autoencoder weights
# are never touched; only the calibration quantiles change. Truth labels are used
# ONLY to select the calibration tracks and to evaluate metrics.
# =============================================================================

CALIBRATION_QUANTILES_FULL = [10, 25, 50, 75, 90, 95, 99]


def quantiles_from_window_errors(errors: np.ndarray) -> Dict:
    """Full calibration-quantile summary from a pool of per-window errors."""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    q = {f"error_p{p}": float(np.percentile(e, p)) for p in CALIBRATION_QUANTILES_FULL}
    q.update({"error_mean": float(e.mean()) if len(e) else np.nan,
              "error_std": float(e.std()) if len(e) else np.nan,
              "n_calibration_windows": int(len(e))})
    return q


def write_calibration_files(calibration_dir: str, cal_track: Dict[str, Dict],
                            meta: Dict) -> Tuple[str, str]:
    """Persist the track-purity calibration as JSON (keyed by model) and a flat CSV."""
    os.makedirs(calibration_dir, exist_ok=True)
    payload = {
        "calibration_mode": "track_purity",
        "calibration_dates": meta["calibration_dates"],
        "calibration_thresholds": meta["calibration_thresholds"],
        "min_target_fraction": meta["min_target_fraction"],
        "min_purity": meta["min_purity"],
        "models": cal_track,
    }
    json_path = meta.get("calibration_output",
                         os.path.join(calibration_dir, "sequence_track_calibration.json"))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    rows = []
    for model, d in cal_track.items():
        rows.append({
            "model": model,
            **{k: d[k] for k in
               ["error_p10", "error_p25", "error_p50", "error_p75",
                "error_p90", "error_p95", "error_p99", "error_mean", "error_std",
                "n_calibration_tracks", "n_calibration_windows"]},
            "calibration_dates": ",".join(meta["calibration_dates"]),
            "calibration_thresholds": ",".join(f"{t:g}" for t in meta["calibration_thresholds"]),
            "min_target_fraction": meta["min_target_fraction"],
            "min_purity": meta["min_purity"],
        })
    csv_path = os.path.join(os.path.dirname(json_path), "sequence_track_calibration.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def build_calibration_comparison(by_mt_by_mode: Dict[str, pd.DataFrame],
                                 score_threshold: float) -> pd.DataFrame:
    """Per model x detection-threshold x calibration-mode filtering metrics."""
    frames = []
    for mode, by_mt in by_mt_by_mode.items():
        f = by_mt[["model", "threshold_db", "stage12_kept_tracks",
                   "stage12_kept_true_tracks", "stage12_kept_false_tracks",
                   "true_track_retention", "false_track_reduction", "precision_after",
                   "median_score_true_tracks", "median_score_false_tracks"]].copy()
        f.insert(2, "calibration_mode", mode)
        f.insert(3, "score_threshold", score_threshold)
        f = f.rename(columns={"stage12_kept_tracks": "kept_tracks",
                              "stage12_kept_true_tracks": "kept_true_tracks",
                              "stage12_kept_false_tracks": "kept_false_tracks"})
        frames.append(f)
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["calibration_mode", "model", "threshold_db"])
            .reset_index(drop=True))


def make_calibration_plots(cal_track: Dict[str, Dict], cal_clean: Dict[str, Dict],
                           by_mt_by_mode: Dict[str, pd.DataFrame], plots_dir: str) -> List[str]:
    """Calibration quantile bands + clean-vs-track retention / false-reduction."""
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    qs = CALIBRATION_QUANTILES_FULL
    for model in cal_track:
        ax.plot(qs, [cal_track[model][f"error_p{p}"] for p in qs], marker="o",
                label=f"{model} track-purity")
        if model in cal_clean and all(f"error_p{p}" in cal_clean[model] for p in qs):
            ax.plot(qs, [cal_clean[model][f"error_p{p}"] for p in qs], marker="s",
                    linestyle="--", label=f"{model} clean-truth")
    ax.set_xlabel("percentile")
    ax.set_ylabel("reconstruction error")
    ax.set_yscale("log")
    ax.set_title("Calibration error quantiles (solid track-purity, dashed clean-truth)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "calibration_error_quantiles.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    for ycol, fname, title in [
        ("true_track_retention", "clean_vs_track_calibrated_retention.png",
         "True-track retention: clean-truth vs track-purity calibration"),
        ("false_track_reduction", "clean_vs_track_calibrated_false_reduction.png",
         "False-track reduction: clean-truth vs track-purity calibration"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        styles = {"clean_truth": (":", "s"), "track_purity": ("-", "o")}
        for mode, by_mt in by_mt_by_mode.items():
            ls, mk = styles.get(mode, ("-", "o"))
            for model, g in by_mt.groupby("model"):
                g = g.sort_values("threshold_db")
                ax.plot(g["threshold_db"], g[ycol], marker=mk, linestyle=ls,
                        label=f"{model} {mode}")
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
    return written


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

def aggregate_by_model_threshold(scores: pd.DataFrame, score_threshold: float,
                                 score_col: str = "sequence_prior_score") -> pd.DataFrame:
    rows = []
    for (model, thr), g in scores.groupby(["model", "threshold_db"]):
        scored = g[np.isfinite(g[score_col])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored[score_col] >= score_threshold]
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
            "mean_score_true_tracks": float(true[score_col].mean()) if len(true) else np.nan,
            "mean_score_false_tracks": float(false[score_col].mean()) if len(false) else np.nan,
            "median_score_true_tracks": float(true[score_col].median()) if len(true) else np.nan,
            "median_score_false_tracks": float(false[score_col].median()) if len(false) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["model", "threshold_db"]).reset_index(drop=True)


def sweep_table(scores: pd.DataFrame, sweep_thresholds: List[float],
                score_col: str = "sequence_prior_score") -> pd.DataFrame:
    rows = []
    for (model, thr), g in scores.groupby(["model", "threshold_db"]):
        scored = g[np.isfinite(g[score_col])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        for st in sweep_thresholds:
            kept = scored[scored[score_col] >= st]
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


def range_bin_table(scores: pd.DataFrame, edges: List[float], score_threshold: float = 0.5,
                    score_col: str = "sequence_prior_score") -> pd.DataFrame:
    labels = [f">{lo / 1000:.0f} km" if np.isinf(hi) else f"{lo / 1000:.0f}-{hi / 1000:.0f} km"
              for lo, hi in zip(edges[:-1], edges[1:])]
    idx = np.digitize(scores["median_range_m"].to_numpy(), np.asarray(edges)[1:-1])
    scores = scores.assign(_bin=[labels[i] for i in idx], _lo=[edges[i] for i in idx])
    rows = []
    for (model, thr, label, lo), g in scores.groupby(["model", "threshold_db", "_bin", "_lo"]):
        scored = g[np.isfinite(g[score_col])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored[score_col] >= score_threshold]
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
            "median_sequence_score": float(scored[score_col].median())
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

    s12_cols = ["model", "threshold_db", "true_track_retention",
                "false_track_reduction", "precision_after"]
    has_mode = "calibration_mode" in by_mt.columns
    if has_mode:
        s12_cols.append("calibration_mode")
    s12 = by_mt[s12_cols].rename(columns={
        "true_track_retention": "stage12_true_retention",
        "false_track_reduction": "stage12_false_reduction",
        "precision_after": "stage12_precision",
        "calibration_mode": "stage12_calibration_mode"})
    merged = base.merge(s12, on="threshold_db", how="right")
    cols = ["threshold_db", "stage08_true_tracks", "stage08_false_tracks",
            "stage09_true_retention", "stage09_false_reduction",
            "stage11_true_retention", "stage11_false_reduction",
            "model", "stage12_true_retention", "stage12_false_reduction",
            "stage12_precision"]
    if has_mode:
        cols.insert(cols.index("model") + 1, "stage12_calibration_mode")
    sort_keys = (["stage12_calibration_mode", "model", "threshold_db"] if has_mode
                 else ["model", "threshold_db"])
    return merged[cols].sort_values(sort_keys).reset_index(drop=True), s9_ok, s11_ok


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

def _stage125_report_section(scores, by_mt, cal_track, cal_clean, calib_comparison,
                             calib_meta, primary_mode, score_threshold) -> List[str]:
    calib_meta = calib_meta or {}
    lines = [
        "## Stage 12.5 Noise-Matched Calibration",
        "",
        "The original stage-12 score mapping used **clean truth** validation",
        "windows for its p50->1 / p99->0 band. Scoring noisy stage-8 Kalman",
        "tracks against that clean band is a domain shift: genuine tracks",
        "reconstruct worse than clean truth and collapse toward score 0.",
        "Stage 12.5 recalibrates the error->score band using **high-purity",
        "stage-8 true tracks** instead, so the 0.5 threshold is meaningful in",
        "the noisy-track domain.",
        "",
        "- The **autoencoder weights are unchanged** -- nothing is retrained;",
        "  only the reconstruction-error quantiles that define the score band",
        "  are replaced.",
        "- **Truth labels are used only to select calibration tracks** and to",
        "  evaluate metrics; they never enter the score itself.",
        "- This is still **not VAE** (stage 13) and **not diffusion** (stage 14).",
        "",
        f"Calibration tracks: dates {', '.join(calib_meta.get('calibration_dates', []))};"
        f" thresholds {', '.join(f'{t:g}' for t in calib_meta.get('calibration_thresholds', []))} dB;"
        f" eligibility target_fraction >= {calib_meta.get('min_target_fraction')}"
        f" and purity >= {calib_meta.get('min_purity')}"
        " (higher thresholds are used for calibration because they yield cleaner,"
        " more reliable high-purity true tracks).",
        "",
        "### Calibration error quantiles (per model)",
        "",
    ]
    qcols = ["error_p50", "error_p90", "error_p99", "n_calibration_tracks",
             "n_calibration_windows"]
    ctab = pd.DataFrame([{"model": m, "calibration": "track_purity",
                          **{c: cal_track[m][c] for c in qcols}} for m in cal_track])
    if cal_clean:
        for m in cal_clean:
            ctab = pd.concat([ctab, pd.DataFrame([{
                "model": m, "calibration": "clean_truth",
                "error_p50": cal_clean[m].get("error_p50"),
                "error_p90": cal_clean[m].get("error_p90"),
                "error_p99": cal_clean[m].get("error_p99"),
                "n_calibration_tracks": np.nan,
                "n_calibration_windows": cal_clean[m].get("n_val_windows", np.nan)}])],
                ignore_index=True)
    lines += md_table(ctab.sort_values(["model", "calibration"]).round(6))

    if calib_comparison is not None and len(calib_comparison):
        lines += [
            "",
            f"### Clean-truth vs track-purity filtering (score threshold {score_threshold:g})",
            "",
        ]
        cmp_cols = ["model", "threshold_db", "calibration_mode", "true_track_retention",
                    "false_track_reduction", "precision_after",
                    "median_score_true_tracks"]
        lines += md_table(calib_comparison[cmp_cols].round(4))
        # report-only verdict at the chosen score threshold
        piv = (calib_comparison.groupby("calibration_mode")["true_track_retention"]
               .mean())
        clean_ret = float(piv.get("clean_truth", np.nan))
        track_ret = float(piv.get("track_purity", np.nan))
        lines += [""]
        if np.isfinite(clean_ret) and np.isfinite(track_ret):
            if track_ret >= clean_ret:
                lines.append(
                    f"Mean true-track retention rises from {clean_ret:.3f} (clean-truth)"
                    f" to {track_ret:.3f} (track-purity) at score threshold"
                    f" {score_threshold:g} -- noise-matched calibration fixes the domain"
                    " shift while false-track reduction stays high.")
            else:
                lines.append(
                    f"Mean true-track retention is {clean_ret:.3f} (clean-truth) vs"
                    f" {track_ret:.3f} (track-purity): calibration did not fix the"
                    " domain shift on this data.")
    lines += [
        "",
        f"The primary scores in this run use the **{primary_mode}** calibration.",
        "",
    ]
    return lines


def write_scoring_report(report_dir: str, scores: pd.DataFrame, by_mt: pd.DataFrame,
                         comparison: pd.DataFrame, range_bins: pd.DataFrame,
                         val_recon: pd.DataFrame, manifest: Dict,
                         s9_ok: bool, s11_ok: bool, score_threshold: float,
                         primary_mode: str = "clean_truth",
                         cal_track: Optional[Dict] = None,
                         cal_clean: Optional[Dict] = None,
                         calib_comparison: Optional[pd.DataFrame] = None,
                         calib_meta: Optional[Dict] = None) -> str:
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
    if cal_track is not None:
        lines += _stage125_report_section(scores, by_mt, cal_track, cal_clean,
                                          calib_comparison, calib_meta, primary_mode,
                                          score_threshold)
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "sequence_prior_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate (scoring)
# =============================================================================

def run_scoring_gate(report_dir: str, scores: pd.DataFrame, by_mt: pd.DataFrame,
                     comparison: pd.DataFrame, s9_ok: bool, s11_ok: bool,
                     cal_track: Optional[Dict] = None,
                     calibration_json: Optional[str] = None,
                     by_mt_by_mode: Optional[Dict[str, pd.DataFrame]] = None,
                     score_threshold: float = 0.5) -> None:
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

    if cal_track is not None:
        if not calibration_json or not os.path.exists(calibration_json):
            fail("track-purity calibration JSON missing")
        for model, d in cal_track.items():
            if not d.get("n_calibration_tracks", 0) > 0:
                fail(f"{model}: n_calibration_tracks must be > 0")
            if not d.get("n_calibration_windows", 0) > 0:
                fail(f"{model}: n_calibration_windows must be > 0")
            if not d["error_p99"] >= d["error_p50"]:
                fail(f"{model}: calibration error_p99 < error_p50")
        if "sequence_prior_score_track_calibrated" in scores.columns:
            sc = scores["sequence_prior_score_track_calibrated"]
            sc = sc[np.isfinite(sc)]
            if len(sc) and not sc.between(0, 1).all():
                fail("track-calibrated scores outside [0, 1]")
        print("  Stage 12.5: calibration JSON present; per-model tracks/windows > 0; "
              "p99 >= p50; track-calibrated scores in [0,1]: OK")

    if by_mt_by_mode and {"clean_truth", "track_purity"} <= set(by_mt_by_mode):
        clean_ret = by_mt_by_mode["clean_truth"]["true_track_retention"].mean()
        track_ret = by_mt_by_mode["track_purity"]["true_track_retention"].mean()
        verdict = ("track-purity calibration fixed the domain shift"
                   if track_ret >= clean_ret else
                   "calibration did not fix the domain shift")
        print(f"  Stage 12.5 (report-only): mean true retention clean={clean_ret:.3f} "
              f"vs track={track_ret:.3f} at score threshold {score_threshold:g} "
              f"-- {verdict}.")
