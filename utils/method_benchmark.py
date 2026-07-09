"""Stage 14: unify existing track-scoring methods into one benchmark.

Stage 14 introduces NO new model and retrains nothing. It reads the compact
report CSVs already produced by stages 07-13, normalizes every track-level
keep/reject method into a common operating-point schema, and derives the
tables the benchmark needs: a method inventory, unified metrics, best method
per threshold, matched-retention / matched-false-reduction comparisons, a
Pareto frontier, descriptive rankings, a runtime inventory, and optional
failure-case candidates.

Everything is defensive: a missing input warns and is skipped; the run only
fails if no usable track-level method exists.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table, safe_ratio, safe_reduction, summarize_defined

# tie-break preference: simpler / more interpretable families win ties
FAMILY_RANK = {"hand_physics": 0, "sequence_autoencoder": 1, "adsb_marginal_prior": 2,
               "vae_prior": 3, "kalman_only": 4, "threshold_only": 5}

UNIFIED_COLUMNS = [
    "date", "threshold_db", "method_id", "method_family", "stage", "model", "variant",
    "calibration_mode", "score_threshold", "stage08_true_tracks", "stage08_false_tracks",
    "kept_true_tracks", "kept_false_tracks", "true_track_retention", "false_track_reduction",
    "false_track_retention", "false_reduction_defined", "false_reduction_denominator",
    "precision_before", "precision_after", "delta_precision",
    "track_reduction_total", "notes",
]


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - corrupt file
        print(f"  WARNING: could not read {path}: {exc}")
        return None


def _metric_row(date, threshold_db, meta: Dict, st, sf, kt, kf, score_threshold,
                notes: str = "") -> Dict:
    st, sf, kt, kf = float(st), float(sf), float(kt), float(kf)
    tot, kept = st + sf, kt + kf
    pb = safe_ratio(st, tot)
    pa = safe_ratio(kt, kept)
    # zero false-track denominator => reduction is UNDEFINED (NaN), never 0 or 1
    reduction = safe_reduction(sf, kf)
    return {
        "date": date, "threshold_db": float(threshold_db),
        "method_id": meta["method_id"], "method_family": meta["method_family"],
        "stage": meta["stage"], "model": meta.get("model"), "variant": meta.get("variant"),
        "calibration_mode": meta.get("calibration_mode"),
        "score_threshold": (float(score_threshold) if score_threshold is not None
                            and np.isfinite(score_threshold) else np.nan),
        "stage08_true_tracks": st, "stage08_false_tracks": sf,
        "kept_true_tracks": kt, "kept_false_tracks": kf,
        "true_track_retention": safe_ratio(kt, st),
        "false_track_reduction": reduction,
        "false_track_retention": safe_ratio(kf, sf),
        "false_reduction_defined": bool(np.isfinite(reduction)),
        "false_reduction_denominator": sf,
        "precision_before": pb, "precision_after": pa,
        "delta_precision": (pa - pb) if np.isfinite(pa) and np.isfinite(pb) else np.nan,
        "track_reduction_total": safe_reduction(tot, kept),
        "notes": notes,
    }


def _inv(stage, family, method_id, source_file, df, note=""):
    return {"stage": stage, "method_family": family, "method_id": method_id,
            "source_file": source_file, "available": df is not None,
            "rows_loaded": (len(df) if df is not None else 0), "notes": note}


# =============================================================================
# Per-stage loaders
# =============================================================================

def _load_kalman(stage08_dir, date, rows, inv):
    path = os.path.join(stage08_dir, "kalman_metrics_by_day.csv")
    df = read_csv_safe(path)
    inv.append(_inv("08", "kalman_only", "kalman_only", path, df,
                    "no-filter baseline" if df is not None else "missing"))
    if df is None:
        return
    d = df[df["date"].astype(str) == str(date)] if "date" in df.columns else df
    meta = {"method_id": "kalman_only", "method_family": "kalman_only", "stage": "08"}
    for _, r in d.iterrows():
        st, sf = r["true_tracks"], r["false_tracks"]
        rows.append(_metric_row(date, r["threshold_db"], meta, st, sf, st, sf, None,
                                "kept = all confirmed tracks (no scoring filter)"))


def _load_sweep_family(counts, sweep, key_cols, meta_fn, date, rows, count_cols):
    """Merge a score-threshold sweep with its stage-8 track counts into unified rows."""
    if counts is None or sweep is None:
        return
    m = sweep.merge(counts[key_cols + count_cols], on=key_cols, how="left")
    for _, r in m.iterrows():
        rows.append(_metric_row(date, r["threshold_db"], meta_fn(r),
                                r[count_cols[0]], r[count_cols[1]],
                                r["kept_true_tracks"], r["kept_false_tracks"],
                                r.get("score_threshold")))


def _load_stage09(stage09_dir, date, rows, inv):
    counts = read_csv_safe(os.path.join(stage09_dir, "physics_metrics_by_threshold.csv"))
    sweep = read_csv_safe(os.path.join(stage09_dir, "physics_filter_sweep.csv"))
    inv.append(_inv("09", "hand_physics", "stage09_hand_physics",
                    os.path.join(stage09_dir, "physics_filter_sweep.csv"), sweep))
    meta = lambda r: {"method_id": "stage09_hand_physics", "method_family": "hand_physics",
                      "stage": "09", "model": None, "variant": None,
                      "calibration_mode": None}
    _load_sweep_family(counts, sweep, ["threshold_db"], meta, date, rows,
                       ["stage08_true_tracks", "stage08_false_tracks"])


def _load_stage11(stage11_dir, date, rows, inv):
    counts = read_csv_safe(os.path.join(stage11_dir, "adsb_prior_metrics_by_threshold.csv"))
    sweep = read_csv_safe(os.path.join(stage11_dir, "adsb_prior_filter_sweep.csv"))
    inv.append(_inv("11", "adsb_marginal_prior", "stage11_adsb_marginal",
                    os.path.join(stage11_dir, "adsb_prior_filter_sweep.csv"), sweep))
    meta = lambda r: {"method_id": "stage11_adsb_marginal",
                      "method_family": "adsb_marginal_prior", "stage": "11",
                      "model": None, "variant": None, "calibration_mode": None}
    _load_sweep_family(counts, sweep, ["threshold_db"], meta, date, rows,
                       ["stage08_true_tracks", "stage08_false_tracks"])


def _load_stage12(stage12_dir, date, rows, inv):
    counts = read_csv_safe(os.path.join(stage12_dir, "sequence_metrics_by_model_threshold.csv"))
    sweep = read_csv_safe(os.path.join(stage12_dir, "sequence_filter_sweep.csv"))
    inv.append(_inv("12.5", "sequence_autoencoder", "stage12_sequence_autoencoders",
                    os.path.join(stage12_dir, "sequence_filter_sweep.csv"), sweep))
    meta = lambda r: {"method_id": f"stage12_{r['model']}_track_calibrated",
                      "method_family": "sequence_autoencoder", "stage": "12.5",
                      "model": r["model"], "variant": None, "calibration_mode": "track_purity"}
    _load_sweep_family(counts, sweep, ["model", "threshold_db"], meta, date, rows,
                       ["stage08_true_tracks", "stage08_false_tracks"])

    # optional clean-truth operating points (single score threshold), marked as such
    cc = read_csv_safe(os.path.join(stage12_dir, "calibration",
                                    "sequence_calibration_comparison.csv"))
    inv.append(_inv("12.5", "sequence_autoencoder", "stage12_clean_truth",
                    os.path.join(stage12_dir, "calibration",
                                 "sequence_calibration_comparison.csv"), cc,
                    "clean-truth calibration (for contrast)"))
    if cc is not None and counts is not None:
        clean = cc[cc["calibration_mode"] == "clean_truth"]
        m = clean.merge(counts[["model", "threshold_db", "stage08_true_tracks",
                                "stage08_false_tracks"]], on=["model", "threshold_db"], how="left")
        for _, r in m.iterrows():
            meta_ct = {"method_id": f"stage12_{r['model']}_clean_truth",
                       "method_family": "sequence_autoencoder", "stage": "12.5",
                       "model": r["model"], "variant": None, "calibration_mode": "clean_truth"}
            rows.append(_metric_row(date, r["threshold_db"], meta_ct,
                                    r["stage08_true_tracks"], r["stage08_false_tracks"],
                                    r["kept_true_tracks"], r["kept_false_tracks"],
                                    r.get("score_threshold"), "clean-truth calibration"))


def _load_stage13(stage13_dir, date, rows, inv):
    counts = read_csv_safe(os.path.join(stage13_dir, "vae_metrics_by_threshold.csv"))
    sweep = read_csv_safe(os.path.join(stage13_dir, "vae_filter_sweep.csv"))
    inv.append(_inv("13", "vae_prior", "stage13_vae_prior",
                    os.path.join(stage13_dir, "vae_filter_sweep.csv"), sweep))
    meta = lambda r: {"method_id": f"stage13_vae_{r['variant']}", "method_family": "vae_prior",
                      "stage": "13", "model": None, "variant": r["variant"],
                      "calibration_mode": "track_purity"}
    _load_sweep_family(counts, sweep, ["variant", "threshold_db"], meta, date, rows,
                       ["stage08_true_tracks", "stage08_false_tracks"])


def load_threshold_context(stage07_dir, output_dir, inv) -> Optional[pd.DataFrame]:
    overall = read_csv_safe(os.path.join(stage07_dir, "threshold_overall.csv"))
    inv.append(_inv("07", "threshold_only", "threshold_only",
                    os.path.join(stage07_dir, "threshold_overall.csv"), overall,
                    "frame-level context only; not a track filter"))
    if overall is not None:
        overall.to_csv(os.path.join(output_dir, "threshold_context.csv"), index=False)
    return overall


# =============================================================================
# Build unified metrics + inventory
# =============================================================================

def build_unified(dirs: Dict[str, str], date: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    inv: List[Dict] = []
    load_threshold_context(dirs["stage07"], output_dir, inv)
    _load_kalman(dirs["stage08"], date, rows, inv)
    _load_stage09(dirs["stage09"], date, rows, inv)
    _load_stage11(dirs["stage11"], date, rows, inv)
    _load_stage12(dirs["stage12"], date, rows, inv)
    _load_stage13(dirs["stage13"], date, rows, inv)

    unified = pd.DataFrame(rows, columns=UNIFIED_COLUMNS) if rows else pd.DataFrame(
        columns=UNIFIED_COLUMNS)
    inventory = pd.DataFrame(inv)
    return unified, inventory


# =============================================================================
# Best method per threshold
# =============================================================================

def _fam_rank(family: str) -> int:
    return FAMILY_RANK.get(family, 99)


def best_method_by_threshold(unified: pd.DataFrame, min_true_retention: float) -> pd.DataFrame:
    rows = []
    track = unified[unified["method_family"] != "threshold_only"]
    for thr, g in track.groupby("threshold_db"):
        g = g[np.isfinite(g["true_track_retention"])]
        if g.empty:
            continue
        qual = g[g["true_track_retention"] >= min_true_retention]
        rule = f"retention>={min_true_retention:g}, max false_reduction"
        if qual.empty:
            qual = g[g["true_track_retention"] == g["true_track_retention"].max()]
            rule = "no method met retention target; highest retention"
        qual = qual.assign(
            _red=qual["false_track_reduction"].fillna(-1.0),
            _prec=qual["precision_after"].fillna(-1.0),
            _fam=qual["method_family"].map(_fam_rank))
        best = qual.sort_values(["_red", "_prec", "_fam"],
                                ascending=[False, False, True]).iloc[0]
        rows.append({
            "threshold_db": thr, "best_method_id": best["method_id"],
            "best_method_family": best["method_family"],
            "score_threshold": best["score_threshold"],
            "true_track_retention": best["true_track_retention"],
            "false_track_reduction": best["false_track_reduction"],
            "precision_after": best["precision_after"],
            "kept_true_tracks": best["kept_true_tracks"],
            "kept_false_tracks": best["kept_false_tracks"],
            "selection_rule": rule,
        })
    return pd.DataFrame(rows).sort_values("threshold_db").reset_index(drop=True)


# =============================================================================
# Matched-target comparisons
# =============================================================================

def _matched(unified: pd.DataFrame, target: float, col: str, gap_name: str,
             extra_cols: List[str]) -> pd.DataFrame:
    rows = []
    track = unified[unified["method_family"] != "threshold_only"]
    for (mid, thr), g in track.groupby(["method_id", "threshold_db"]):
        g = g[np.isfinite(g[col])]
        if g.empty:
            continue
        at_or_above = g[g[col] >= target]
        pick = (at_or_above.iloc[(at_or_above[col] - target).abs().to_numpy().argmin()]
                if len(at_or_above) else g.iloc[(g[col] - target).abs().to_numpy().argmin()])
        row = {"threshold_db": thr, "method_id": mid,
               "method_family": pick["method_family"],
               "score_threshold": pick["score_threshold"]}
        for c in extra_cols:
            row[c] = pick[c]
        row[gap_name] = pick[col] - target
        rows.append(row)
    return pd.DataFrame(rows)


def matched_retention_comparison(unified, target_retention: float) -> pd.DataFrame:
    df = _matched(unified, target_retention, "true_track_retention", "retention_gap",
                  ["true_track_retention", "false_track_reduction", "precision_after"])
    df["target_retention"] = target_retention
    cols = ["threshold_db", "method_id", "method_family", "score_threshold",
            "target_retention", "true_track_retention", "false_track_reduction",
            "precision_after", "retention_gap"]
    return df[cols].sort_values(["threshold_db", "method_family", "method_id"]).reset_index(drop=True)


def matched_false_reduction_comparison(unified, target_false_reduction: float) -> pd.DataFrame:
    df = _matched(unified, target_false_reduction, "false_track_reduction",
                  "false_reduction_gap",
                  ["true_track_retention", "false_track_reduction", "precision_after"])
    df["target_false_reduction"] = target_false_reduction
    cols = ["threshold_db", "method_id", "method_family", "score_threshold",
            "target_false_reduction", "true_track_retention", "false_track_reduction",
            "precision_after", "false_reduction_gap"]
    return df[cols].sort_values(["threshold_db", "method_family", "method_id"]).reset_index(drop=True)


# =============================================================================
# Pareto frontier
# =============================================================================

def pareto_frontier(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    track = unified[unified["method_family"] != "threshold_only"]
    for thr, g in track.groupby("threshold_db"):
        g = g[np.isfinite(g["true_track_retention"]) & np.isfinite(g["false_track_reduction"])]
        if g.empty:
            continue
        pts = g[["true_track_retention", "false_track_reduction"]].to_numpy()
        is_par = np.ones(len(g), dtype=bool)
        for i in range(len(g)):
            for j in range(len(g)):
                if i == j:
                    continue
                if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]
                        and (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                    is_par[i] = False
                    break
        for k, (_, r) in enumerate(g.iterrows()):
            rows.append({"threshold_db": thr, "method_id": r["method_id"],
                         "method_family": r["method_family"],
                         "score_threshold": r["score_threshold"],
                         "true_track_retention": r["true_track_retention"],
                         "false_track_reduction": r["false_track_reduction"],
                         "precision_after": r["precision_after"], "is_pareto": bool(is_par[k])})
    return pd.DataFrame(rows).sort_values(
        ["threshold_db", "is_pareto", "false_track_reduction"],
        ascending=[True, False, False]).reset_index(drop=True)


# =============================================================================
# Rankings
# =============================================================================

def rankings(unified: pd.DataFrame) -> pd.DataFrame:
    g = unified[unified["method_family"] != "threshold_only"].copy()
    g = g[np.isfinite(g["true_track_retention"]) & np.isfinite(g["false_track_reduction"])]
    tr, fr = g["true_track_retention"], g["false_track_reduction"]
    g["utility"] = 0.5 * tr + 0.5 * fr
    g["utility_recall_weighted"] = 0.7 * tr + 0.3 * fr
    g["utility_precision_weighted"] = 0.3 * tr + 0.7 * fr
    g["rank_utility"] = g.groupby("threshold_db")["utility"].rank(ascending=False, method="min")
    g["rank_recall_weighted"] = g.groupby("threshold_db")["utility_recall_weighted"].rank(
        ascending=False, method="min")
    g["rank_precision_weighted"] = g.groupby("threshold_db")["utility_precision_weighted"].rank(
        ascending=False, method="min")
    cols = ["threshold_db", "method_id", "method_family", "score_threshold",
            "true_track_retention", "false_track_reduction", "precision_after",
            "utility", "utility_recall_weighted", "utility_precision_weighted",
            "rank_utility", "rank_recall_weighted", "rank_precision_weighted"]
    return g[cols].sort_values(["threshold_db", "rank_utility"]).reset_index(drop=True)


# =============================================================================
# Runtime inventory
# =============================================================================

def runtime_inventory(dirs: Dict[str, str]) -> pd.DataFrame:
    sources = [
        ("08", "kalman_only", os.path.join(dirs["stage08"], "run_log.md")),
        ("09", "hand_physics", os.path.join(dirs["stage09"], "run_log.md")),
        ("11", "adsb_marginal_prior", os.path.join(dirs["stage11"], "run_log.md")),
        ("12.5", "sequence_autoencoder", os.path.join(dirs["stage12"], "run_log.md")),
        ("13", "vae_prior", os.path.join(dirs["stage13"], "vae_prior_report.md")),
    ]
    pat = re.compile(r"(\d+)\s*m\s*(\d+)\s*s")
    rows = []
    for stage, family, path in sources:
        runtime, note = "unknown", "no run log parsed"
        if os.path.exists(path):
            try:
                text = open(path).read()
                matches = pat.findall(text)
                if matches:
                    runtime = "; ".join(f"{a}m{b}s" for a, b in matches[:3])
                    note = f"parsed from {os.path.basename(path)}"
                else:
                    note = f"{os.path.basename(path)} present, no runtime pattern"
            except Exception:
                note = "read error"
        rows.append({"stage": stage, "method_family": family,
                     "runtime_reported": runtime, "notes": note})
    return pd.DataFrame(rows)


# =============================================================================
# Global recommendations
# =============================================================================

def global_recommendations(unified: pd.DataFrame, best_by_thr: pd.DataFrame) -> Dict:
    g = unified[np.isfinite(unified["true_track_retention"])
                & np.isfinite(unified["false_track_reduction"])].copy()
    g["utility"] = 0.5 * g["true_track_retention"] + 0.5 * g["false_track_reduction"]

    def best_of(sub):
        sub = sub[sub["true_track_retention"] >= 0.95]
        if sub.empty:
            return None
        by_method = sub.groupby("method_id")["utility"].mean()
        return by_method.idxmax()

    learned = g[g["method_family"].isin(["sequence_autoencoder", "vae_prior"])]
    interp = g[g["method_family"].isin(["hand_physics", "adsb_marginal_prior"])]
    low_thr = None
    if len(best_by_thr):
        lo = best_by_thr.sort_values("threshold_db").iloc[0]
        low_thr = lo["best_method_id"]
    return {
        "best_overall_method_id": best_of(g),
        "best_low_threshold_method_id": low_thr,
        "best_interpretable_method_id": best_of(interp),
        "best_learned_method_id": best_of(learned),
    }


# =============================================================================
# Failure-case candidates (optional; needs per-track compact score CSVs)
# =============================================================================

def failure_case_candidates(dirs: Dict[str, str], date: str, output_dir: str,
                            per_type: int = 20) -> Tuple[pd.DataFrame, str]:
    """Best-effort: join stage-9/12/13 per-track scores to surface disagreements.

    Requires the compact per-track score CSVs (large / usually git-ignored). If
    they are missing, returns an empty frame and a note string instead."""
    s9 = os.path.join(dirs["stage09"], "physics_track_scores.csv")
    s12 = os.path.join(dirs["stage12"], "sequence_track_scores.csv")
    s13 = os.path.join(dirs["stage13"], "vae_track_scores.csv")
    if not (os.path.exists(s9) and os.path.exists(s12) and os.path.exists(s13)):
        return (pd.DataFrame(), "Failure-case analysis requires the per-track score CSVs "
                "(physics_track_scores.csv, sequence_track_scores.csv, vae_track_scores.csv); "
                "one or more are unavailable (they are large and usually git-ignored).")
    try:
        d9 = pd.read_csv(s9, usecols=["date", "threshold_db", "track_id", "physics_score",
                                      "keep_physics", "is_true_track", "target_fraction",
                                      "purity", "position_rmse_m", "median_range_m"])
        d12 = pd.read_csv(s12, usecols=["date", "threshold_db", "model", "track_id",
                                        "sequence_prior_score", "keep_sequence_prior"])
        d12 = d12[d12["model"] == "mlp_dae"].drop(columns="model")
        d13 = pd.read_csv(s13, usecols=["date", "threshold_db", "variant", "track_id",
                                        "vae_prior_score", "keep_vae_prior",
                                        "latent_mu_mean_norm"])
        d13 = d13[d13["variant"] == "elbo"].drop(columns="variant")
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame(), f"Failure-case analysis skipped (read error): {exc}"

    j = d9.merge(d12, on=["date", "threshold_db", "track_id"], how="inner", suffixes=("", "_s12"))
    j = j.merge(d13, on=["date", "threshold_db", "track_id"], how="inner")
    j = j.rename(columns={"physics_score": "score_stage09",
                          "sequence_prior_score": "score_stage12",
                          "vae_prior_score": "score_stage13"})

    out = []

    def take(sel, case_type, ctx, reason):
        sub = j[sel].head(per_type)
        for _, r in sub.iterrows():
            out.append({"case_type": case_type, "date": r["date"],
                        "threshold_db": r["threshold_db"], "track_id": r["track_id"],
                        "method_context": ctx, "reason": reason,
                        "score_stage09": r["score_stage09"], "score_stage12": r["score_stage12"],
                        "score_stage13": r["score_stage13"],
                        "target_fraction": r["target_fraction"], "purity": r["purity"],
                        "position_rmse_m": r["position_rmse_m"],
                        "median_range_m": r["median_range_m"]})

    false_ = ~j["is_true_track"]
    true_ = j["is_true_track"]
    take(false_ & (j["keep_physics"] == True) & (j["keep_sequence_prior"] == False),
         "false_survives_s09_rejected_s12", "stage09 vs stage12.5",
         "false track kept by hand physics but rejected by sequence AE")
    take(false_ & (j["keep_sequence_prior"] == True),
         "false_survives_s12", "stage12.5", "false track survives sequence AE filter")
    take(true_ & (j["keep_sequence_prior"] == False),
         "true_rejected_s12", "stage12.5", "true track rejected by sequence AE")
    take(true_ & (j["keep_sequence_prior"] == True) & (j["keep_vae_prior"] == False),
         "true_kept_s12_rejected_vae", "stage12.5 vs stage13",
         "true track kept by sequence AE but rejected by VAE")
    if true_.any():
        thr_hi = j.loc[true_, "latent_mu_mean_norm"].quantile(0.99)
        take(true_ & (j["latent_mu_mean_norm"] >= thr_hi),
             "vae_latent_outlier_true", "stage13", "true track with outlier VAE latent norm")

    return pd.DataFrame(out), ""


# =============================================================================
# Plots (matplotlib only, default colors)
# =============================================================================

def make_plots(unified, best_by_thr, pareto, ranks, plots_dir) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []
    track = unified[unified["method_family"] != "threshold_only"]
    fin = track[np.isfinite(track["true_track_retention"])
                & np.isfinite(track["false_track_reduction"])]

    # 1. false reduction vs true retention scatter (per family)
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, g in fin.groupby("method_family"):
        ax.scatter(g["true_track_retention"], g["false_track_reduction"], s=14, alpha=0.6,
                   label=fam)
    ax.set_xlabel("true-track retention")
    ax.set_ylabel("false-track reduction")
    ax.set_title("Operating points: false reduction vs true retention")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "false_reduction_vs_true_retention.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. precision by threshold at score_threshold 0.5 (per family, one method each)
    half = fin[np.isclose(fin["score_threshold"].fillna(-1), 0.5)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, g in half.groupby("method_id"):
        g = g.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["precision_after"], marker="o", label=mid)
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("precision after filtering")
    ax.set_title("Precision after filtering (score threshold 0.5)")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "precision_by_threshold.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. kept false tracks by method (score threshold 0.5); include kalman baseline
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = track[track["method_family"] == "kalman_only"].sort_values("threshold_db")
    if len(base):
        ax.plot(base["threshold_db"], base["kept_false_tracks"], marker="s", linestyle=":",
                label="kalman (unfiltered)")
    for mid, g in half.groupby("method_id"):
        g = g.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["kept_false_tracks"], marker="o", label=mid)
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("kept false tracks")
    ax.set_title("Kept false tracks by method (score threshold 0.5)")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "false_tracks_by_method.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 4. true retention by method (score threshold 0.5)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid, g in half.groupby("method_id"):
        g = g.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["true_track_retention"], marker="o", label=mid)
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("true-track retention")
    ax.set_title("True-track retention by method (score threshold 0.5)")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "true_retention_by_method.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 5. pareto frontier (one representative threshold: the lowest, most false tracks)
    if len(pareto):
        thr0 = sorted(pareto["threshold_db"].unique())[0]
        g = pareto[pareto["threshold_db"] == thr0]
        fig, ax = plt.subplots(figsize=(7, 5))
        dom = g[~g["is_pareto"]]
        par = g[g["is_pareto"]].sort_values("true_track_retention")
        ax.scatter(dom["true_track_retention"], dom["false_track_reduction"], s=14, alpha=0.4,
                   label="dominated")
        ax.plot(par["true_track_retention"], par["false_track_reduction"], marker="o",
                label="Pareto frontier")
        ax.set_xlabel("true-track retention")
        ax.set_ylabel("false-track reduction")
        ax.set_title(f"Pareto frontier at {thr0:g} dB")
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "pareto_frontier.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    # 6. method rank heatmap (utility rank; methods x thresholds)
    if len(ranks):
        piv = ranks.pivot_table(index="method_id", columns="threshold_db",
                                values="rank_utility", aggfunc="min")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(piv) + 1)))
        im = ax.imshow(piv.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(list(piv.index), fontsize=7)
        ax.set_xlabel("detection threshold (dB)")
        ax.set_title("Utility rank by method and threshold (1 = best)")
        fig.colorbar(im, ax=ax, label="rank")
        fig.tight_layout()
        p = os.path.join(plots_dir, "method_rank_heatmap.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(output_dir, date, inventory, unified, best_by_thr, matched_ret,
                 matched_fr, pareto, ranks, runtime, recs, failure_note) -> str:
    n_methods = unified["method_id"].nunique()
    lines = [
        "# Stage 14 Unified Method Benchmark",
        "",
        "## Status",
        "",
        "- **Stage 14 does not introduce a new model** and retrains nothing;",
        "  it consolidates Stages 07, 08, 09, 11, 12.5, and 13.",
        f"- Results are for {date} unless more dates are processed.",
        "- The goal is **operating-point selection** before diffusion or",
        "  broader model-zoo work (Stage 15).",
        f"- {n_methods} track-level methods (operating points across score thresholds)"
        " were unified.",
        "",
        "## Methods compared",
        "",
        "- threshold-only (Stage 07) -- frame-level context only, not a track filter",
        "- Kalman only (Stage 08) -- no-filter track baseline",
        "- hand physics (Stage 09)",
        "- ADS-B marginal priors (Stage 11)",
        "- deterministic sequence autoencoders (Stage 12.5)",
        "- VAE prior (Stage 13)",
        "",
        "## Input inventory",
        "",
    ]
    lines += md_table(inventory)
    lines += ["", "## Unified operating curves", "",
              "Per-method true-track retention, false-track reduction, and precision",
              "after filtering are in `unified_method_metrics.csv`"
              f" ({len(unified)} operating-point rows).", ""]

    lines += ["## Best method by threshold", ""]
    lines += md_table(best_by_thr.round(4))
    lines += [
        "",
        "> **Denominator caveat.** Each stage reports its own stage-8 true/false-track",
        "> counts, and the learned stages (12.5, 13) only score tracks long enough to",
        "> window (>= window_len points, >= 5 hits), so at high detection thresholds",
        "> (9/12 dB) their windowable false-track count is ~0 while the physics/ADS-B",
        "> stages still count short false tracks. Cross-stage false-reduction is",
        "> therefore only apples-to-apples where every method sees a comparable false",
        "> population -- i.e. the informative low/mid thresholds (-5, 0, 3, 6 dB). At",
        "> 9/12 dB the 'best method' collapses to whichever stage still has false tracks",
        "> to remove; read those rows with care.",
        "",
        f"- **Best overall method:** {recs['best_overall_method_id']}",
              f"- **Best low-threshold method:** {recs['best_low_threshold_method_id']}",
              f"- **Best interpretable method:** {recs['best_interpretable_method_id']}",
              f"- **Best learned method:** {recs['best_learned_method_id']}", ""]

    lines += ["## Matched-retention comparison", "",
              "At similar true-track retention (target 0.97), which method removes the",
              "most false tracks?", ""]
    lines += md_table(matched_ret.round(4))

    lines += ["", "## Matched-false-reduction comparison", "",
              "At similar false-track reduction (target 0.95), which method retains the",
              "most true tracks?", ""]
    lines += md_table(matched_fr.round(4))

    lines += ["", "## Pareto frontier", "",
              "Non-dominated operating points (a point is dominated if another has >=",
              "retention AND >= false reduction, with at least one strictly greater):", ""]
    lines += md_table(pareto[pareto["is_pareto"]].round(4))

    lines += ["", "## Rankings", "",
              "Descriptive utility = 0.5*retention + 0.5*false_reduction (plus recall- and",
              "precision-weighted variants). **These utility scores are descriptive, not",
              "absolute truth -- operational priorities determine the final method choice.**",
              "Top operating point per threshold by balanced utility:", ""]
    top = (ranks.sort_values(["threshold_db", "rank_utility"])
           .groupby("threshold_db").head(1))
    lines += md_table(top[["threshold_db", "method_id", "true_track_retention",
                           "false_track_reduction", "utility"]].round(4))

    lines += ["", "## Runtime inventory", ""]
    lines += md_table(runtime)

    lines += [
        "", "## Interpretation", "",
        "- **Stage 12.5 deterministic sequence autoencoders are currently the",
        "  strongest learned method** for keep/reject scoring.",
        "- **Stage 13 VAE does not improve over the deterministic autoencoders**",
        "  for keep/reject scoring, though its latent diagnostics separate true and",
        "  false tracks.",
        "- **Stage 09 hand physics remains the strongest simple/interpretable",
        "  baseline** and is a close, transparent fallback.",
        "- **Stage 11 empirical marginal priors** are useful evidence but not enough",
        "  alone -- they do not clearly beat Stage 09.",
        "- Future diffusion or model-zoo work should target remaining gaps, not",
        "  repeat solved comparisons.",
        "",
        "## Recommended operating point", "",
        f"- **Best conservative / overall method:** {recs['best_overall_method_id']}"
        " (highest balanced utility at retention >= 0.95).",
        f"- **Best low-threshold method:** {recs['best_low_threshold_method_id']}"
        " (worst clutter regime, -5 dB).",
        f"- **Best interpretable method:** {recs['best_interpretable_method_id']}"
        " (transparent, no training).",
        f"- **Best learned method:** {recs['best_learned_method_id']}.",
        "",
        "## Failure cases to inspect", "",
    ]
    if failure_note:
        lines.append(failure_note)
    else:
        lines.append("Candidate disagreement cases are in `failure_case_candidates.csv`"
                     " (false tracks surviving each filter, true tracks wrongly rejected,"
                     " and VAE latent outliers).")
    lines += [
        "",
        "## Next stage", "",
        "Stage 15 should either:",
        "1. test **diffusion** specifically for denoising / gap filling, or",
        "2. build a broader **model-zoo** benchmark only if there is a clearly defined",
        "   remaining gap.",
        "",
    ]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stage14_method_benchmark_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_gate(output_dir, inventory, unified, best_by_thr, pareto) -> None:
    def fail(msg):
        raise ValueError(f"Stage 14 benchmark validation failed: {msg}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (benchmark)")
    print("=" * 70)

    inv_path = os.path.join(output_dir, "method_inventory.csv")
    uni_path = os.path.join(output_dir, "unified_method_metrics.csv")
    if not os.path.exists(inv_path) or inventory.empty:
        fail("method_inventory.csv missing or empty")
    if not os.path.exists(uni_path) or unified.empty:
        fail("unified_method_metrics.csv missing or empty")

    fams = set(unified["method_family"].unique())
    if "kalman_only" not in fams:
        fail("Stage 08 (kalman_only) baseline not available")
    if not ({"hand_physics", "adsb_marginal_prior", "sequence_autoencoder", "vae_prior"} & fams):
        fail("no scoring method available")
    print(f"  Stage 08 + scoring methods available: {sorted(fams)}")

    for col in ("true_track_retention", "false_track_reduction", "precision_after"):
        v = unified[col].dropna()
        if len(v) and not v.between(0, 1).all():
            fail(f"{col} outside [0, 1]")
    print("  retention / false-reduction / precision in [0, 1]: OK")

    if best_by_thr.empty:
        fail("best_method_by_threshold.csv is empty")
    if pareto.empty or not pareto["is_pareto"].any():
        fail("pareto_frontier.csv has no Pareto point")
    print("  best-method table nonempty; Pareto frontier has >= 1 point: OK")

