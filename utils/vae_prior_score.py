"""Stage 13: VAE-prior scoring of stage-8 tracks + evaluation.

Scores each confirmed stage-8 track by how anomalous its normalized
trajectory windows look under the stage-13 sequence VAE, using two variants:

  * reconstruction -- anomaly = per-window reconstruction error;
  * elbo           -- anomaly = reconstruction error + beta_score * KL.

The track score input is the MEDIAN per-window anomaly, mapped to [0, 1]
against a noise-matched calibration band (score 1 at/below the calibration
p50 anomaly, 0 at/above the p99) built from high-purity stage-8 true tracks
(the stage-12.5 lesson). Truth labels are used ONLY to select calibration
tracks and to evaluate; they never enter the score itself.
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
from utils.sequence_prior_score import evaluate_track_labels  # noqa: F401  (re-exported)

VARIANTS = ["reconstruction", "elbo"]


# =============================================================================
# Score mapping + calibration quantiles
# =============================================================================

def score_from_band(value: float, p50: float, p99: float) -> float:
    """Map an anomaly value to [0, 1]: 1 at/below p50, 0 at/above p99."""
    if not np.isfinite(value):
        return np.nan
    if p99 <= p50:
        return float(value <= p50)
    if value <= p50:
        return 1.0
    if value >= p99:
        return 0.0
    return float(1.0 - (value - p50) / (p99 - p50))


def calibration_quantiles(values: np.ndarray) -> Dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"error_p50": float(np.percentile(v, 50)) if len(v) else np.nan,
            "error_p90": float(np.percentile(v, 90)) if len(v) else np.nan,
            "error_p99": float(np.percentile(v, 99)) if len(v) else np.nan,
            "error_mean": float(v.mean()) if len(v) else np.nan,
            "n_calibration_windows": int(len(v))}


def write_vae_calibration(models_dir: str, report_dir: str, variants_cal: Dict[str, Dict],
                          meta: Dict) -> Tuple[str, str]:
    """Persist the two-variant calibration band to JSON (models) and CSV (report)."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    payload = {
        "calibration_mode": meta.get("calibration_mode", "track_purity"),
        "calibration_dates": meta.get("calibration_dates", []),
        "calibration_thresholds": meta.get("calibration_thresholds", []),
        "min_target_fraction": meta.get("min_target_fraction"),
        "min_purity": meta.get("min_purity"),
        "beta_score": meta.get("beta_score"),
        "variants": {v: {k: variants_cal[v][k] for k in
                         ("error_p50", "error_p99", "n_calibration_tracks",
                          "n_calibration_windows")} for v in variants_cal},
    }
    json_path = os.path.join(models_dir, "vae_calibration.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    rows = []
    for v, d in variants_cal.items():
        rows.append({"variant": v, **{k: d.get(k) for k in
                     ("error_p50", "error_p90", "error_p99", "error_mean",
                      "n_calibration_tracks", "n_calibration_windows")},
                     "calibration_mode": payload["calibration_mode"],
                     "calibration_dates": ",".join(meta.get("calibration_dates", [])),
                     "calibration_thresholds": ",".join(
                         f"{t:g}" for t in meta.get("calibration_thresholds", [])),
                     "min_target_fraction": meta.get("min_target_fraction"),
                     "min_purity": meta.get("min_purity")})
    csv_path = os.path.join(report_dir, "vae_calibration.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_by_variant_threshold(scores: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    rows = []
    for (variant, thr), g in scores.groupby(["variant", "threshold_db"]):
        scored = g[np.isfinite(g["vae_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored["vae_prior_score"] >= score_threshold]
        kt = int(kept["is_true_track"].sum())
        kf = len(kept) - kt
        rows.append({
            "variant": variant, "threshold_db": thr,
            "stage08_confirmed_tracks": len(scored),
            "stage08_true_tracks": len(true), "stage08_false_tracks": len(false),
            "stage13_kept_tracks": len(kept), "stage13_kept_true_tracks": kt,
            "stage13_kept_false_tracks": kf,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_retention": kf / len(false) if len(false) else np.nan,
            "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
            "precision_before": len(true) / len(scored) if len(scored) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
            "mean_score_true_tracks": float(true["vae_prior_score"].mean()) if len(true) else np.nan,
            "mean_score_false_tracks": float(false["vae_prior_score"].mean()) if len(false) else np.nan,
            "median_score_true_tracks": float(true["vae_prior_score"].median()) if len(true) else np.nan,
            "median_score_false_tracks": float(false["vae_prior_score"].median()) if len(false) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["variant", "threshold_db"]).reset_index(drop=True)


def variant_sweep_table(scores: pd.DataFrame, sweep_thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for (variant, thr), g in scores.groupby(["variant", "threshold_db"]):
        scored = g[np.isfinite(g["vae_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        for st in sweep_thresholds:
            kept = scored[scored["vae_prior_score"] >= st]
            kt = int(kept["is_true_track"].sum())
            kf = len(kept) - kt
            rows.append({
                "variant": variant, "threshold_db": thr, "score_threshold": st,
                "kept_tracks": len(kept), "kept_true_tracks": kt, "kept_false_tracks": kf,
                "true_track_retention": kt / len(true) if len(true) else np.nan,
                "false_track_retention": kf / len(false) if len(false) else np.nan,
                "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
                "precision_after": kt / len(kept) if len(kept) else np.nan,
            })
    return pd.DataFrame(rows)


def variant_range_bin_table(scores: pd.DataFrame, edges: List[float],
                            score_threshold: float) -> pd.DataFrame:
    labels = [f">{lo / 1000:.0f} km" if np.isinf(hi) else f"{lo / 1000:.0f}-{hi / 1000:.0f} km"
              for lo, hi in zip(edges[:-1], edges[1:])]
    idx = np.digitize(scores["median_range_m"].to_numpy(), np.asarray(edges)[1:-1])
    scores = scores.assign(_bin=[labels[i] for i in idx], _lo=[edges[i] for i in idx])
    rows = []
    for (variant, thr, label, lo), g in scores.groupby(
            ["variant", "threshold_db", "_bin", "_lo"]):
        scored = g[np.isfinite(g["vae_prior_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored["vae_prior_score"] >= score_threshold]
        kt = int(kept["is_true_track"].sum())
        rows.append({
            "variant": variant, "threshold_db": thr, "range_bin": label, "_lo": lo,
            "stage08_tracks": len(scored), "stage08_true_tracks": len(true),
            "stage08_false_tracks": len(false),
            "stage13_kept_tracks": len(kept), "stage13_kept_true_tracks": kt,
            "stage13_kept_false_tracks": len(kept) - kt,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_reduction": 1 - (len(kept) - kt) / len(false) if len(false) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
            "median_vae_score": float(scored["vae_prior_score"].median()) if len(scored) else np.nan,
        })
    return (pd.DataFrame(rows).sort_values(["variant", "threshold_db", "_lo"])
            .drop(columns="_lo").reset_index(drop=True))


def latent_summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, threshold, is_true_track) latent-diagnostic summary."""
    rows = []
    for (variant, thr, is_true), g in scores.groupby(
            ["variant", "threshold_db", "is_true_track"]):
        g = g[np.isfinite(g["latent_mu_mean_norm"])]
        rows.append({
            "variant": variant, "threshold_db": thr, "is_true_track": bool(is_true),
            "n_windows": int(g["n_windows"].sum()),
            "mu_norm_p50": float(g["latent_mu_mean_norm"].median()) if len(g) else np.nan,
            "mu_norm_p90": float(g["latent_mu_mean_norm"].quantile(0.9)) if len(g) else np.nan,
            "mu_dim_std_mean": float(g["latent_mu_std_mean"].mean()) if len(g) else np.nan,
            "kl_p50": float(g["vae_kl_median"].median()) if len(g) else np.nan,
            "kl_p90": float(g["vae_kl_median"].quantile(0.9)) if len(g) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["variant", "threshold_db", "is_true_track"]).reset_index(drop=True)


# =============================================================================
# Five-way comparison (best stage-12 model + VAE variants)
# =============================================================================

def select_best_stage12(stage12_csv: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Best stage-12 model: max mean false-reduction s.t. mean true-retention >= 0.95,
    else best F1-like tradeoff. Returns (model_name, stage-12 rows for that model)."""
    if not stage12_csv or not os.path.exists(stage12_csv):
        return None, None
    s12 = pd.read_csv(stage12_csv)
    if "model" not in s12.columns:
        return None, None
    agg = s12.groupby("model").agg(
        tr=("stage12_true_retention", "mean"),
        fr=("stage12_false_reduction", "mean")).reset_index()
    ok = agg[agg["tr"] >= 0.95]
    if len(ok):
        best = ok.sort_values("fr", ascending=False).iloc[0]["model"]
    else:
        agg["f1"] = 2 * agg["tr"] * agg["fr"] / (agg["tr"] + agg["fr"]).replace(0, np.nan)
        best = agg.sort_values("f1", ascending=False).iloc[0]["model"]
    return best, s12[s12["model"] == best].copy()


def build_five_way_comparison(by_vt: pd.DataFrame, stage12_csv: str,
                              stage09_dir: str, stage11_dir: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Join stage 8/9/11/12(best model)/13(per variant) by detection threshold."""
    best_model, s12 = select_best_stage12(stage12_csv)

    if s12 is not None:
        base = s12[["threshold_db", "stage08_true_tracks", "stage08_false_tracks",
                    "stage09_true_retention", "stage09_false_reduction",
                    "stage11_true_retention", "stage11_false_reduction",
                    "stage12_true_retention", "stage12_false_reduction"]].copy()
        base["stage12_model"] = best_model
        base["stage12_calibration_mode"] = (
            s12["stage12_calibration_mode"].iloc[0]
            if "stage12_calibration_mode" in s12.columns else "unknown")
    else:
        base = (by_vt.groupby("threshold_db")
                .agg(stage08_true_tracks=("stage08_true_tracks", "first"),
                     stage08_false_tracks=("stage08_false_tracks", "first"))
                .reset_index())
        for c in ("stage09_true_retention", "stage09_false_reduction",
                  "stage11_true_retention", "stage11_false_reduction",
                  "stage12_true_retention", "stage12_false_reduction"):
            base[c] = np.nan
        base["stage12_model"] = None
        base["stage12_calibration_mode"] = None

    s13 = by_vt[["variant", "threshold_db", "true_track_retention",
                 "false_track_reduction", "precision_after"]].rename(columns={
        "variant": "stage13_variant",
        "true_track_retention": "stage13_true_retention",
        "false_track_reduction": "stage13_false_reduction",
        "precision_after": "stage13_precision"})
    merged = base.merge(s13, on="threshold_db", how="right")
    cols = ["threshold_db", "stage08_true_tracks", "stage08_false_tracks",
            "stage09_true_retention", "stage09_false_reduction",
            "stage11_true_retention", "stage11_false_reduction",
            "stage12_model", "stage12_calibration_mode",
            "stage12_true_retention", "stage12_false_reduction",
            "stage13_variant", "stage13_true_retention", "stage13_false_reduction",
            "stage13_precision"]
    return (merged[cols].sort_values(["stage13_variant", "threshold_db"])
            .reset_index(drop=True), best_model)


# =============================================================================
# Plots (matplotlib only, default colors)
# =============================================================================

def make_training_plot(history: Dict, plots_dir: str) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    ep = np.arange(1, len(history["train_recon"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ep, history["train_recon"], label="train recon")
    ax1.plot(ep, history["val_recon"], linestyle="--", label="val recon")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("reconstruction MSE")
    ax1.set_yscale("log")
    ax1.set_title("VAE reconstruction loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.5)
    ax2.plot(ep, history["train_kl"], label="train KL")
    ax2.plot(ep, history["val_kl"], linestyle="--", label="val KL")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("KL divergence")
    ax2.set_title("VAE KL divergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "vae_loss_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def make_vae_scoring_plots(scores: pd.DataFrame, by_vt: pd.DataFrame,
                           comparison: pd.DataFrame, sweep: pd.DataFrame,
                           best_stage12_model: Optional[str], plots_dir: str) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    for variant, g in scores.groupby("variant"):
        g = g[np.isfinite(g["vae_prior_score"])]
        ax.hist(g.loc[g["is_true_track"], "vae_prior_score"], bins=bins, density=True,
                histtype="step", linewidth=2, label=f"{variant} true")
        ax.hist(g.loc[~g["is_true_track"], "vae_prior_score"], bins=bins, density=True,
                histtype="step", linewidth=2, linestyle="--", label=f"{variant} false")
    ax.set_xlabel("VAE prior score")
    ax.set_ylabel("density")
    ax.set_title("VAE prior score (solid true, dashed false)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "vae_score_hist_true_false.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    for ycol, s12col, fname, title in [
        ("stage13_false_reduction", "stage12_false_reduction",
         "stage12_vs_stage13_false_reduction.png", "False-track reduction: stage 12 vs 13"),
        ("stage13_true_retention", "stage12_true_retention",
         "stage12_vs_stage13_true_retention.png", "True-track retention: stage 12 vs 13"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for variant, g in comparison.groupby("stage13_variant"):
            g = g.sort_values("threshold_db")
            ax.plot(g["threshold_db"], g[ycol], marker="o", label=f"stage 13 {variant}")
        ref = comparison.drop_duplicates("threshold_db").sort_values("threshold_db")
        if ref[s12col].notna().any():
            ax.plot(ref["threshold_db"], ref[s12col], marker="s", linestyle=":",
                    label=f"stage 12 {best_stage12_model}")
        ax.set_xlabel("detection threshold (dB)")
        ax.set_ylabel(ycol.replace("stage13_", "").replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, g in sweep.groupby("variant"):
        agg = g.groupby("score_threshold").agg(kt=("kept_true_tracks", "sum"),
                                               kf=("kept_false_tracks", "sum")).reset_index()
        sc = scores[(scores["variant"] == variant) & np.isfinite(scores["vae_prior_score"])]
        tt = max(int(sc["is_true_track"].sum()), 1)
        tf = max(int((~sc["is_true_track"]).sum()), 1)
        ax.plot(agg["score_threshold"], agg["kt"] / tt, marker="o", label=f"{variant} retention")
        ax.plot(agg["score_threshold"], 1 - agg["kf"] / tf, marker="x", linestyle="--",
                label=f"{variant} reduction")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("fraction")
    ax.set_title("VAE filter sweep (pooled detection thresholds)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "vae_filter_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


def make_latent_pca_plot(latent_mu: np.ndarray, is_true: np.ndarray, plots_dir: str) -> Optional[str]:
    """2-component PCA scatter of per-track latent means (numpy SVD, no sklearn)."""
    os.makedirs(plots_dir, exist_ok=True)
    mu = np.asarray(latent_mu, dtype=float)
    mask = np.isfinite(mu).all(axis=1)
    mu, is_true = mu[mask], np.asarray(is_true)[mask]
    if len(mu) < 3 or mu.shape[1] < 2:
        return None
    centered = mu - mu.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lbl, name, marker in [(True, "true", "o"), (False, "false", "x")]:
        sel = is_true == lbl
        if sel.any():
            ax.scatter(proj[sel, 0], proj[sel, 1], s=10, alpha=0.5, marker=marker, label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("VAE latent means (PCA), true vs false tracks")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "vae_latent_pca.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# =============================================================================
# Report
# =============================================================================

def write_vae_report(report_dir: str, scores: pd.DataFrame, by_vt: pd.DataFrame,
                     comparison: pd.DataFrame, range_bins: pd.DataFrame,
                     latent: pd.DataFrame, val_recon: pd.DataFrame, manifest: Dict,
                     variants_cal: Dict, calib_meta: Dict, best_stage12_model: Optional[str],
                     score_threshold: float) -> str:
    dates = sorted(scores["date"].unique())
    date_scope = (f"These are one-day results for {dates[0]}." if len(dates) == 1 else
                  f"These results cover {len(dates)} days: {', '.join(dates)}.")

    def variant_means(variant):
        g = by_vt[by_vt["variant"] == variant]
        return g["false_track_reduction"].mean(), g["true_track_retention"].mean()

    lines = [
        "# Stage 13 VAE Trajectory Prior",
        "",
        "## Status",
        "",
        "- Stage 13 trains a VAE over normalized trajectory windows.",
        "- Stage 13 scores Stage 08 Kalman tracks using reconstruction and",
        "  ELBO-like anomaly scores.",
        "- Score calibration uses high-purity noisy Stage 08 tracks by default.",
        f"- {date_scope}",
        "- This is **not diffusion** and not the full model zoo.",
        "- Truth labels are used only for calibration track selection and evaluation.",
        "",
        "## Motivation",
        "",
        "- Stage 12.5 deterministic autoencoders worked very well after",
        "  noise-matched calibration (mean true retention ~0.96 with high",
        "  false-track reduction).",
        "- Stage 13 tests whether a probabilistic latent trajectory model (a",
        "  VAE) adds value: a reconstruction score, KL/latent diagnostics, and",
        "  latent motion structure.",
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
        " (origin-shifted, heading-rotated, standardized).",
        f"- Normalizer: {manifest.get('normalizer_source', 'unknown')}",
        "",
        "## Model",
        "",
        f"- Fully-connected sequence VAE; latent dim {manifest.get('latent_dim')},"
        f" hidden dim {manifest.get('hidden_dim')}.",
        "- Encoder: flatten window -> 2x Linear+ReLU -> (mu, logvar).",
        "- Decoder: latent -> 2x Linear+ReLU -> Linear -> window.",
        "- Loss: MSE reconstruction + beta * KL to N(0, I);"
        f" beta {manifest.get('beta')}, KL annealed over"
        f" {manifest.get('kl_anneal_epochs')} epochs.",
        "",
        "## Validation",
        "",
    ]
    lines += md_table(val_recon.round(6))
    lines += [
        "",
        "## Calibration",
        "",
        "- Noise-matched **track-purity** calibration (the stage-12.5 lesson):",
        "  anomaly quantiles come from high-purity stage-8 true tracks, not",
        "  clean truth. Score 1 at/below p50, 0 at/above p99.",
        f"- Calibration dates {', '.join(calib_meta.get('calibration_dates', []))};"
        f" thresholds {', '.join(f'{t:g}' for t in calib_meta.get('calibration_thresholds', []))} dB;"
        f" eligibility target_fraction >= {calib_meta.get('min_target_fraction')}"
        f" and purity >= {calib_meta.get('min_purity')}.",
        "- Two variants: **reconstruction** (recon error only) and **elbo**"
        f" (recon + beta_score * KL, beta_score {calib_meta.get('beta_score')}).",
        "",
    ]
    cal_tab = pd.DataFrame([{"variant": v, "error_p50": variants_cal[v]["error_p50"],
                             "error_p99": variants_cal[v]["error_p99"],
                             "n_calibration_tracks": variants_cal[v]["n_calibration_tracks"],
                             "n_calibration_windows": variants_cal[v]["n_calibration_windows"]}
                            for v in variants_cal])
    lines += md_table(cal_tab.round(6))
    lines += [
        "",
        "## Results",
        "",
        f"Best stage-12 model for comparison: **{best_stage12_model}**"
        " (max mean false-reduction subject to mean true-retention >= 0.95).",
        "",
    ]
    lines += md_table(comparison.round(4))
    lines += ["", "## VAE vs deterministic autoencoders", ""]
    for v in VARIANTS:
        if (by_vt["variant"] == v).any():
            fr, tr = variant_means(v)
            lines.append(f"- **{v}**: mean false reduction {fr:.3f}, mean true retention {tr:.3f}.")
    s12_tr = comparison["stage12_true_retention"].mean()
    s12_fr = comparison["stage12_false_reduction"].mean()
    if np.isfinite(s12_tr):
        best_v = max(VARIANTS, key=lambda v: variant_means(v)[0] if (by_vt["variant"] == v).any() else -1)
        bfr, btr = variant_means(best_v)
        if bfr >= s12_fr and btr >= 0.95:
            lines.append(f"- The VAE ({best_v}) matches or exceeds stage 12"
                         f" ({best_stage12_model}: mean reduction {s12_fr:.3f},"
                         f" retention {s12_tr:.3f}).")
        else:
            lines.append(f"- The VAE does not beat stage 12"
                         f" ({best_stage12_model}: mean reduction {s12_fr:.3f},"
                         f" retention {s12_tr:.3f}); **deterministic autoencoders"
                         " remain the stronger baseline.**")
    lines += [
        "- ELBO vs reconstruction-only: compare the two variant rows above --"
        " when KL adds no separation, the reconstruction term dominates and"
        " the variants are near-identical.",
        "",
        "## Range-bin behavior",
        "",
    ]
    rb = range_bins.groupby(["variant", "range_bin"], sort=False).agg(
        tracks=("stage08_tracks", "sum"),
        true_retention=("true_track_retention", "mean"),
        false_reduction=("false_track_reduction", "mean"),
        median_vae_score=("median_vae_score", "mean")).reset_index()
    lines += md_table(rb.round(4))
    lines += [
        "",
        "## Latent diagnostics",
        "",
    ]
    lines += md_table(latent.round(4))
    lines += [
        "",
        "## Failure modes",
        "",
        "- Posterior collapse is possible (KL -> 0); then the VAE degenerates",
        "  toward a plain autoencoder and the latent carries no structure.",
        "- KL may not improve anomaly scoring over reconstruction error alone.",
        "- Reconstruction error may still dominate the ELBO-like score.",
        "- The VAE may smooth rare but valid maneuvers and reject them.",
        "- Calibration remains essential: clean-truth quantiles under-retain",
        "  noisy tracks exactly as in stage 12.",
        "",
        "## Recommended next stage",
        "",
        "Stage 14 should test diffusion or a broader model-zoo benchmark,",
        "depending on whether the VAE adds value over Stage 12.5.",
        "",
    ]
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "vae_prior_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate (scoring)
# =============================================================================

def run_vae_scoring_gate(report_dir: str, scores: pd.DataFrame, by_vt: pd.DataFrame,
                         comparison: pd.DataFrame, variants_cal: Dict,
                         calibration_json: str, best_stage12_model: Optional[str]) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 13 scoring validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (scoring)")
    print("=" * 70)

    path = os.path.join(report_dir, "vae_track_scores.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        fail("vae_track_scores.csv missing or empty")
    scored = scores[np.isfinite(scores["vae_prior_score"])]
    if scored.empty:
        fail("no tracks received a finite VAE score")
    if not scored["vae_prior_score"].between(0, 1).all():
        fail("vae_prior_score outside [0, 1]")
    if not scored["keep_vae_prior"].isin([True, False]).all():
        fail("keep_vae_prior is not boolean")
    print("  scores file nonempty, scores in [0,1], keep boolean: OK")

    if not calibration_json or not os.path.exists(calibration_json):
        fail("calibration JSON missing")
    for v, d in variants_cal.items():
        if not d.get("n_calibration_tracks", 0) > 0:
            fail(f"{v}: n_calibration_tracks must be > 0")
        if not d.get("n_calibration_windows", 0) > 0:
            fail(f"{v}: n_calibration_windows must be > 0")
    print("  calibration JSON present; per-variant tracks/windows > 0: OK")

    for _, r in by_vt.iterrows():
        if r["stage13_kept_true_tracks"] > r["stage08_true_tracks"]:
            fail("kept true tracks exceed stage-8 true tracks")
        if r["stage13_kept_false_tracks"] > r["stage08_false_tracks"]:
            fail("kept false tracks exceed stage-8 false tracks")
        for col in ("true_track_retention", "false_track_reduction"):
            v = r[col]
            if np.isfinite(v) and not (0.0 <= v <= 1.0):
                fail(f"{col} outside [0, 1]")
    print("  kept-track counts bounded; retention/reduction in [0,1]: OK")

    for variant, g in scored.groupby("variant"):
        tm = g.loc[g["is_true_track"], "vae_prior_score"].median()
        fm = g.loc[~g["is_true_track"], "vae_prior_score"].median()
        print(f"  {variant}: median score true={tm:.3f} false={fm:.3f} (report-only) "
              + ("-- separable" if tm > fm else "-- NOT separable"))

    # report-only: does the VAE beat the best stage-12 model?
    s12_fr = comparison["stage12_false_reduction"].mean()
    s12_tr = comparison["stage12_true_retention"].mean()
    if np.isfinite(s12_fr):
        best_v = by_vt.groupby("variant")["false_track_reduction"].mean().idxmax()
        vfr = by_vt[by_vt["variant"] == best_v]["false_track_reduction"].mean()
        vtr = by_vt[by_vt["variant"] == best_v]["true_track_retention"].mean()
        if vfr >= s12_fr and vtr >= 0.95:
            print(f"  Stage 13 (report-only): VAE {best_v} (fr {vfr:.3f}/tr {vtr:.3f}) "
                  f">= stage 12 {best_stage12_model} (fr {s12_fr:.3f}/tr {s12_tr:.3f}).")
        else:
            print(f"  Stage 13 (report-only): VAE {best_v} (fr {vfr:.3f}/tr {vtr:.3f}) does "
                  f"NOT beat stage 12 {best_stage12_model} (fr {s12_fr:.3f}/tr {s12_tr:.3f}) "
                  "-- deterministic autoencoders remain the stronger baseline.")
