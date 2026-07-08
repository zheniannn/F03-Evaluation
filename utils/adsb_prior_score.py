"""Stage 11: empirical ADS-B-prior track scoring.

Applies the stage-10 empirical motion priors to stage-8 confirmed Kalman
tracks and evaluates whether data-derived likelihoods suppress false tracks
better than stage-9's hand-designed physics knees. Stage 8 tracker logic,
stage 9 scoring, and stage 10 priors are all left untouched; truth labels
enter only AFTER scoring, for evaluation (same strict true-track definition
as stage 9: confirmed, target_fraction >= 0.8, purity >= 0.8, non-null
majority trajectory).

Scoring model (documented choice):
  * Per feature (speed, |accel|, vector accel, |turn rate|, |vertical
    speed|), the penalty is a QUANTILE-EXCEEDANCE mix taken from the
    stage-10 prior JSON quantiles -- nothing hand-coded:
        raw = frac(> p99) + 2 * frac(> p995) + 4 * frac(> p999)
    (speed additionally penalizes the low side, frac(< p01) + 4 *
    frac(< p001), because implausibly SLOW is as suspicious as fast).
    raw is normalized by its maximum and inverted: score = 1 - raw/max.
  * Mean histogram log-density per feature (via stage 10's
    evaluate_empirical_logpdf) is carried as a DIAGNOSTIC column, not mixed
    into the score, so the score stays interpretable in quantile terms.
  * The joint prior score compares the track's MEDIAN feature vector
    (log1p) against the stage-10 joint prior median with its log1p
    covariance: d2 = (x-m)^T inv(C + eps I) (x-m), score =
    exp(-0.5 * min(d2, 50)). The track median -- not the spec'd p95 -- is
    compared because the prior's median/covariance describe per-SAMPLE
    values; a track's typical sample should look like the prior's typical
    sample. This is a deliberate, documented deviation.
  * Continuity (hit rate) and SNR are weak auxiliary terms; unavailable
    components (e.g. SNR without a detections join) are excluded from the
    weight normalization and disclosed.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table, threshold_to_token
from utils.motion_priors import (
    PRIOR_FILENAMES,
    evaluate_empirical_logpdf,
    load_prior_json,
)
from utils.track_physics_score import (
    TRUE_TRACK_PURITY,
    TRUE_TRACK_TARGET_FRACTION,
    discover_stage08_runs,
    load_run,
)

MIN_SPEED_FOR_TURN_MPS = 10.0

# feature -> (prior filename key, one_sided)
SCORED_FEATURES = {
    "speed_mps": ("speed_mps", False),
    "accel_abs_mps2": ("accel_abs_mps2", True),
    "accel_vector_mps2": ("accel_vector_mps2", True),
    "turn_rate_abs_deg_s": ("turn_rate_abs_deg_s", True),
    "vertical_speed_abs_mps": ("vertical_speed_abs_mps", True),
}
FEATURE_SHORT = {
    "speed_mps": "speed", "accel_abs_mps2": "accel",
    "accel_vector_mps2": "vector_accel", "turn_rate_abs_deg_s": "turn",
    "vertical_speed_abs_mps": "vertical",
}
PENALTY_MAX_ONE_SIDED = 7.0    # frac>p99 + 2*frac>p995 + 4*frac>p999, all = 1
PENALTY_MAX_TWO_SIDED = 12.0   # + frac<p01 + 4*frac<p001, all = 1
JOINT_D2_CLIP = 50.0
SNR_SCORE_RANGE_DB = (-10.0, 3.0)   # weak auxiliary mapping only


@dataclass
class AdsbPriorScoreConfig:
    """All stage-11 tunables in one place (populated from the CLI)."""
    tracks_dir: str = "data/active/tracks_kalman"
    priors_dir: str = "models/motion_priors"
    stage09_dir: str = "reports/stage09_physics_scoring"
    report_dir: str = "reports/stage11_adsb_prior_scoring"
    detections_dir: str = "data/active/sim_detections_relocated"   # SNR recovery only
    thresholds_db: Optional[List[float]] = None
    dates: Optional[List[str]] = None
    range_bins_m: List[float] = field(default_factory=lambda: [0, 50_000.0, 100_000.0, 200_000.0, float("inf")])
    score_threshold: float = 0.5
    sweep_thresholds: List[float] = field(default_factory=lambda: [round(0.1 * k, 1) for k in range(1, 10)])
    min_hits: int = 3
    chunksize: int = 1_000_000
    overwrite: bool = False
    no_plots: bool = False
    w_speed: float = 1.0
    w_accel: float = 1.0
    w_vector_accel: float = 1.0
    w_turn: float = 1.0
    w_vertical: float = 1.0
    w_joint: float = 0.5
    w_continuity: float = 0.5
    w_snr: float = 0.25


# =============================================================================
# Priors
# =============================================================================

def load_priors(priors_dir: str) -> Tuple[Dict[str, Dict], Optional[Dict]]:
    """Load the stage-10 marginal priors (whichever exist) and the joint prior."""
    priors = {}
    for feature, fname in PRIOR_FILENAMES.items():
        path = os.path.join(priors_dir, fname)
        if os.path.exists(path):
            priors[feature] = load_prior_json(path)
    joint_path = os.path.join(priors_dir, "joint_motion_prior.json")
    joint = load_prior_json(joint_path) if os.path.exists(joint_path) else None
    if joint is not None and joint.get("n_samples", 0) <= 0:
        joint = None
    return priors, joint


def prepare_joint(joint: Optional[Dict]) -> Optional[Dict]:
    """Precompute the inverse covariance for the joint score; None if the
    joint prior is unavailable or singular."""
    if joint is None or "log1p_covariance" not in joint:
        return None
    features = joint["log1p_covariance"]["features"]
    cov = np.asarray(joint["log1p_covariance"]["matrix"], dtype=float)
    median = np.array([np.log1p(max(joint["median"][f], 0.0)) for f in features])
    try:
        inv = np.linalg.inv(cov + 1e-9 * np.eye(len(features)))
    except np.linalg.LinAlgError:
        return None
    return {"features": features, "median_log1p": median, "inv_cov": inv}


# =============================================================================
# Per-track features + score
# =============================================================================

def compute_track_series(g: pd.DataFrame) -> Dict:
    """Per-sample motion series + basic/eval features for one track (rows
    sorted by timestamp; posterior states)."""
    g = g.sort_values("timestamp")
    t = g["timestamp"].to_numpy(dtype=float)
    v = g[["vx", "vy", "vz"]].to_numpy(dtype=float)
    pos = g[["x", "y", "z"]].to_numpy(dtype=float)

    speed = np.linalg.norm(v, axis=1)
    dt = np.diff(t)
    ok = dt > 0
    accel_abs = np.abs(np.diff(speed))[ok] / dt[ok]
    accel_vec = np.linalg.norm(np.diff(v, axis=0), axis=1)[ok] / dt[ok]

    heading = np.degrees(np.arctan2(v[:, 0], v[:, 1]))      # x=east, y=north
    dh = (np.diff(heading) + 180.0) % 360.0 - 180.0
    hspeed = np.linalg.norm(v[:, :2], axis=1)
    turn_ok = ok & (hspeed[:-1] >= MIN_SPEED_FOR_TURN_MPS) & (hspeed[1:] >= MIN_SPEED_FOR_TURN_MPS)
    turn_abs = np.abs(dh[turn_ok] / dt[turn_ok]) if turn_ok.any() else np.array([])

    rng = np.linalg.norm(pos, axis=1)
    n_points = len(g)
    n_hits = int(g["is_hit"].sum())

    hits = g[g["is_hit"] == 1]
    n_target = int((hits["assoc_is_target"] == 1).sum())
    n_clutter = int((hits["assoc_is_target"] == 0).sum())
    target_fraction = n_target / len(hits) if len(hits) else np.nan
    traj = hits.loc[hits["assoc_is_target"] == 1, "assoc_trajectory_id"].dropna()
    majority = traj.mode().iloc[0] if len(traj) else None
    errs = g["pos_error_m"].to_numpy(dtype=float)
    errs = errs[np.isfinite(errs)]
    snr = g["snr_db"].to_numpy(dtype=float)
    snr = snr[np.isfinite(snr)]

    return {
        "series": {
            "speed_mps": speed,
            "accel_abs_mps2": accel_abs,
            "accel_vector_mps2": accel_vec,
            "turn_rate_abs_deg_s": turn_abs,
            "vertical_speed_abs_mps": np.abs(v[:, 2]),
        },
        "n_hits": n_hits,
        "n_misses": n_points - n_hits,
        "duration_s": float(t[-1] - t[0]) if n_points > 1 else 0.0,
        "hit_rate": n_hits / n_points if n_points else np.nan,
        "median_range_m": float(np.median(rng)) if len(rng) else np.nan,
        "max_range_m": float(rng.max()) if len(rng) else np.nan,
        "snr_p50": float(np.median(snr)) if len(snr) else np.nan,
        "n_target_hits": n_target,
        "n_clutter_hits": n_clutter,
        "target_fraction": target_fraction,
        "purity": target_fraction,
        "majority_trajectory_id": majority,
        "position_rmse_m": float(np.sqrt((errs**2).mean())) if len(errs) else np.nan,
    }


def quantile_exceedance_penalty(values: np.ndarray, prior: Dict, one_sided: bool) -> float:
    """Empirical anomaly penalty in [0, 1] from stage-10 prior quantiles."""
    v = values[np.isfinite(values)]
    if not len(v):
        return np.nan
    q = prior["quantiles"]
    raw = (float((v > q["p99"]).mean())
           + 2.0 * float((v > q["p995"]).mean())
           + 4.0 * float((v > q["p999"]).mean()))
    if not one_sided:
        raw += float((v < q["p01"]).mean()) + 4.0 * float((v < q["p001"]).mean())
        return min(raw / PENALTY_MAX_TWO_SIDED, 1.0)
    return min(raw / PENALTY_MAX_ONE_SIDED, 1.0)


def score_track(feat: Dict, priors: Dict[str, Dict], joint_prep: Optional[Dict],
                cfg: AdsbPriorScoreConfig, snr_available: bool) -> Dict:
    """Empirical-prior score for one track. Uses ONLY physics/measurement
    features and the stage-10 priors -- never truth labels."""
    weights = {"speed": cfg.w_speed, "accel": cfg.w_accel,
               "vector_accel": cfg.w_vector_accel, "turn": cfg.w_turn,
               "vertical": cfg.w_vertical, "joint": cfg.w_joint,
               "continuity": cfg.w_continuity, "snr": cfg.w_snr}
    out: Dict = {}
    components: Dict[str, float] = {}

    for feature, (prior_key, one_sided) in SCORED_FEATURES.items():
        short = FEATURE_SHORT[feature]
        series = feat["series"][feature]
        prior = priors.get(prior_key)
        if prior is None or not len(series[np.isfinite(series)]):
            out[f"{short}_prior_penalty"] = np.nan
            out[f"{short}_prior_score"] = np.nan
            out[f"{short}_mean_logpdf"] = np.nan
            continue
        penalty = quantile_exceedance_penalty(series, prior, one_sided)
        out[f"{short}_prior_penalty"] = penalty
        out[f"{short}_prior_score"] = 1.0 - penalty
        lp = evaluate_empirical_logpdf(series[np.isfinite(series)], prior)
        out[f"{short}_mean_logpdf"] = float(lp.mean())
        components[short] = 1.0 - penalty

    # Joint score on the track's MEDIAN feature vector (see module docstring).
    if joint_prep is not None:
        x = []
        valid = True
        for f in joint_prep["features"]:
            series = feat["series"].get(f, np.array([]))
            series = series[np.isfinite(series)]
            if not len(series):
                valid = False
                break
            x.append(np.log1p(max(float(np.median(series)), 0.0)))
        if valid:
            d = np.asarray(x) - joint_prep["median_log1p"]
            d2 = float(d @ joint_prep["inv_cov"] @ d)
            score = float(np.exp(-0.5 * min(d2, JOINT_D2_CLIP)))
            out["joint_prior_score"] = score
            out["joint_prior_penalty"] = 1.0 - score
            components["joint"] = score
        else:
            out["joint_prior_score"] = out["joint_prior_penalty"] = np.nan
    else:
        out["joint_prior_score"] = out["joint_prior_penalty"] = np.nan

    # Weak auxiliary terms.
    out["continuity_score"] = float(np.clip(feat["hit_rate"], 0.0, 1.0)) \
        if np.isfinite(feat["hit_rate"]) else np.nan
    if np.isfinite(out["continuity_score"]):
        components["continuity"] = out["continuity_score"]
    if snr_available and np.isfinite(feat["snr_p50"]):
        lo, hi = SNR_SCORE_RANGE_DB
        out["snr_score"] = float(np.clip((feat["snr_p50"] - lo) / (hi - lo), 0.0, 1.0))
        components["snr"] = out["snr_score"]
    else:
        out["snr_score"] = np.nan

    total_w = sum(weights[k] for k in components)
    out["adsb_prior_score"] = float(np.clip(
        sum(weights[k] * v for k, v in components.items()) / total_w, 0.0, 1.0)) \
        if total_w > 0 else 0.0
    return out


# =============================================================================
# Per-run orchestration
# =============================================================================

def score_run(date: str, threshold_db: float, points_path: str,
              cfg: AdsbPriorScoreConfig, priors: Dict[str, Dict],
              joint_prep: Optional[Dict]) -> pd.DataFrame:
    df = load_run(points_path, date, threshold_db, cfg.detections_dir)
    snr_available = bool(np.isfinite(df["snr_db"]).any())

    confirmed_ids = df.loc[df["confirmed"] == 1, "track_id"].unique()
    df = df[df["track_id"].isin(confirmed_ids)]

    rows = []
    for track_id, g in df.groupby("track_id", sort=False):
        feat = compute_track_series(g)
        if feat["n_hits"] < cfg.min_hits:
            continue
        row = {
            "date": date, "threshold_db": threshold_db, "track_id": track_id,
            "n_hits": feat["n_hits"], "n_misses": feat["n_misses"],
            "duration_s": feat["duration_s"], "hit_rate": feat["hit_rate"],
            "median_range_m": feat["median_range_m"], "max_range_m": feat["max_range_m"],
        }
        for feature in SCORED_FEATURES:
            series = feat["series"][feature]
            series = series[np.isfinite(series)]
            for q in (50, 95, 99):
                row[f"{feature}_p{q}"] = float(np.percentile(series, q)) if len(series) else np.nan
        row.update(score_track(feat, priors, joint_prep, cfg, snr_available))
        row["keep_adsb_prior"] = bool(row["adsb_prior_score"] >= cfg.score_threshold)
        tf, pu = feat["target_fraction"], feat["purity"]
        row.update({
            "n_target_hits": feat["n_target_hits"], "n_clutter_hits": feat["n_clutter_hits"],
            "target_fraction": tf, "purity": pu,
            "majority_trajectory_id": feat["majority_trajectory_id"],
            "is_true_track": bool(np.isfinite(tf) and tf >= TRUE_TRACK_TARGET_FRACTION
                                  and np.isfinite(pu) and pu >= TRUE_TRACK_PURITY
                                  and feat["majority_trajectory_id"] is not None),
            "position_rmse_m": feat["position_rmse_m"],
        })
        rows.append(row)

    scores = pd.DataFrame(rows)
    scores.attrs["snr_available"] = snr_available
    return scores


# =============================================================================
# Aggregation / comparison
# =============================================================================

def aggregate_by_threshold(scores: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    rows = []
    for thr, g in scores.groupby("threshold_db"):
        true = g[g["is_true_track"]]
        false = g[~g["is_true_track"]]
        kept = g[g["keep_adsb_prior"]]
        kt = int(kept["is_true_track"].sum())
        kf = len(kept) - kt
        rows.append({
            "threshold_db": thr,
            "stage08_confirmed_tracks": len(g),
            "stage08_true_tracks": len(true),
            "stage08_false_tracks": len(false),
            "stage11_kept_tracks": len(kept),
            "stage11_kept_true_tracks": kt,
            "stage11_kept_false_tracks": kf,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_retention": kf / len(false) if len(false) else np.nan,
            "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
            "precision_before": len(true) / len(g) if len(g) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
            "mean_score_true_tracks": float(true["adsb_prior_score"].mean()) if len(true) else np.nan,
            "mean_score_false_tracks": float(false["adsb_prior_score"].mean()) if len(false) else np.nan,
            "median_score_true_tracks": float(true["adsb_prior_score"].median()) if len(true) else np.nan,
            "median_score_false_tracks": float(false["adsb_prior_score"].median()) if len(false) else np.nan,
            "score_threshold": score_threshold,
        })
    return pd.DataFrame(rows).sort_values("threshold_db").reset_index(drop=True)


def sweep_table(scores: pd.DataFrame, sweep_thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for thr, g in scores.groupby("threshold_db"):
        true = g[g["is_true_track"]]
        false = g[~g["is_true_track"]]
        for st in sweep_thresholds:
            kept = g[g["adsb_prior_score"] >= st]
            kt = int(kept["is_true_track"].sum())
            kf = len(kept) - kt
            rows.append({
                "threshold_db": thr, "score_threshold": st,
                "kept_tracks": len(kept), "kept_true_tracks": kt, "kept_false_tracks": kf,
                "true_track_retention": kt / len(true) if len(true) else np.nan,
                "false_track_retention": kf / len(false) if len(false) else np.nan,
                "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
                "precision_after": kt / len(kept) if len(kept) else np.nan,
            })
    return pd.DataFrame(rows)


def range_bin_table(scores: pd.DataFrame, edges: List[float]) -> pd.DataFrame:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f">{lo / 1000:.0f} km" if np.isinf(hi) else f"{lo / 1000:.0f}-{hi / 1000:.0f} km")
    idx = np.digitize(scores["median_range_m"].to_numpy(), np.asarray(edges)[1:-1])
    scores = scores.assign(_bin=[labels[i] for i in idx], _bin_lo=[edges[i] for i in idx])
    rows = []
    for (thr, label, lo), g in scores.groupby(["threshold_db", "_bin", "_bin_lo"]):
        true = g[g["is_true_track"]]
        false = g[~g["is_true_track"]]
        kept = g[g["keep_adsb_prior"]]
        kt = int(kept["is_true_track"].sum())
        rows.append({
            "threshold_db": thr, "range_bin": label, "_bin_lo": lo,
            "stage08_tracks": len(g),
            "stage08_true_tracks": len(true), "stage08_false_tracks": len(false),
            "stage11_kept_tracks": len(kept), "stage11_kept_true_tracks": kt,
            "stage11_kept_false_tracks": len(kept) - kt,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_reduction": 1 - (len(kept) - kt) / len(false) if len(false) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
        })
    return (pd.DataFrame(rows).sort_values(["threshold_db", "_bin_lo"])
            .drop(columns="_bin_lo").reset_index(drop=True))


def build_comparison(by_thr: pd.DataFrame, stage09_dir: str) -> Tuple[pd.DataFrame, bool]:
    """Merge stage-9's committed comparison when available."""
    s11 = by_thr.rename(columns={
        "score_threshold": "stage11_score_threshold",
        "stage11_kept_true_tracks": "stage11_kept_true_tracks",
        "true_track_retention": "stage11_true_track_retention",
        "false_track_reduction": "stage11_false_track_reduction",
        "precision_after": "stage11_precision",
    })[["threshold_db", "stage08_confirmed_tracks", "stage08_true_tracks",
        "stage08_false_tracks", "stage11_score_threshold", "stage11_kept_true_tracks",
        "stage11_kept_false_tracks", "stage11_true_track_retention",
        "stage11_false_track_reduction", "stage11_precision"]]

    s9_path = os.path.join(stage09_dir, "stage08_vs_stage09.csv")
    if not os.path.exists(s9_path):
        return s11, False
    s9 = pd.read_csv(s9_path)[[
        "threshold_db", "stage09_kept_true_tracks", "stage09_kept_false_tracks",
        "stage09_true_track_retention", "stage09_false_track_reduction", "stage09_precision"]]
    merged = s11.merge(s9, on="threshold_db", how="left")
    cols = ["threshold_db", "stage08_confirmed_tracks", "stage08_true_tracks",
            "stage08_false_tracks",
            "stage09_kept_true_tracks", "stage09_kept_false_tracks",
            "stage09_true_track_retention", "stage09_false_track_reduction",
            "stage09_precision",
            "stage11_score_threshold", "stage11_kept_true_tracks",
            "stage11_kept_false_tracks", "stage11_true_track_retention",
            "stage11_false_track_reduction", "stage11_precision"]
    return merged[cols], True


# =============================================================================
# Plots (matplotlib only, default colors, one figure each)
# =============================================================================

def _bin_labels(edges: List[float]) -> List[str]:
    return [f">{lo / 1000:.0f} km" if np.isinf(hi) else f"{lo / 1000:.0f}-{hi / 1000:.0f} km"
            for lo, hi in zip(edges[:-1], edges[1:])]


def make_plots(scores: pd.DataFrame, sweep: pd.DataFrame, comparison: pd.DataFrame,
               stage09_available: bool, edges: List[float], plots_dir: str) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores.loc[scores["is_true_track"], "adsb_prior_score"], bins=bins,
            alpha=0.6, label="true tracks", density=True)
    ax.hist(scores.loc[~scores["is_true_track"], "adsb_prior_score"], bins=bins,
            alpha=0.6, label="false tracks", density=True)
    ax.set_xlabel("ADS-B prior score")
    ax.set_ylabel("density")
    ax.set_title("ADS-B-prior score distributions, true vs false tracks")
    ax.legend()
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "adsb_score_hist_true_false.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    if stage09_available:
        o = comparison.sort_values("threshold_db")
        for s9col, s11col, ylabel, fname, title in [
            ("stage09_false_track_reduction", "stage11_false_track_reduction",
             "false-track reduction", "stage09_vs_stage11_false_reduction.png",
             "False-track reduction: stage 9 vs stage 11"),
            ("stage09_true_track_retention", "stage11_true_track_retention",
             "true-track retention", "stage09_vs_stage11_true_retention.png",
             "True-track retention: stage 9 vs stage 11"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(o["threshold_db"], o[s9col], marker="o", label="stage 9 (hand knees)")
            ax.plot(o["threshold_db"], o[s11col], marker="o", label="stage 11 (ADS-B priors)")
            ax.set_xlabel("detection threshold (dB)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linewidth=0.5)
            fig.tight_layout()
            p = os.path.join(plots_dir, fname)
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    total_true = max(int(scores["is_true_track"].sum()), 1)
    total_false = max(int((~scores["is_true_track"]).sum()), 1)
    agg = sweep.groupby("score_threshold").agg(
        kept_true=("kept_true_tracks", "sum"), kept_false=("kept_false_tracks", "sum")).reset_index()
    ax.plot(agg["score_threshold"], agg["kept_true"] / total_true, marker="o",
            label="true-track retention")
    ax.plot(agg["score_threshold"], 1 - agg["kept_false"] / total_false, marker="o",
            label="false-track reduction")
    ax.set_xlabel("score threshold")
    ax.set_ylabel("fraction")
    ax.set_title("ADS-B-prior filter sweep (all detection thresholds pooled)")
    ax.legend()
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "adsb_filter_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # median score by track median-range bin, true vs false
    labels = _bin_labels(edges)
    idx = np.digitize(scores["median_range_m"].to_numpy(), np.asarray(edges)[1:-1])
    binned = scores.assign(_bin=idx)
    xs = np.arange(len(labels))
    med_true = [binned.loc[(binned["_bin"] == i) & binned["is_true_track"],
                           "adsb_prior_score"].median() for i in xs]
    med_false = [binned.loc[(binned["_bin"] == i) & ~binned["is_true_track"],
                            "adsb_prior_score"].median() for i in xs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    ax.bar(xs - width / 2, med_true, width, label="true tracks")
    ax.bar(xs + width / 2, med_false, width, label="false tracks")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("track median range bin")
    ax.set_ylabel("median ADS-B prior score")
    ax.set_title("Median ADS-B-prior score by range bin")
    ax.legend()
    ax.grid(True, linewidth=0.5, axis="y")
    fig.tight_layout()
    p = os.path.join(plots_dir, "adsb_score_by_range_bin.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(report_dir: str, scores: pd.DataFrame, by_thr: pd.DataFrame,
                 comparison: pd.DataFrame, range_bins: pd.DataFrame,
                 priors: Dict[str, Dict], joint_available: bool,
                 stage09_available: bool, snr_available: bool,
                 cfg: AdsbPriorScoreConfig) -> str:
    dates = sorted(scores["date"].unique())
    date_scope = (f"These are one-day results for {dates[0]}." if len(dates) == 1
                  else f"These results cover {len(dates)} days: {', '.join(dates)}.")
    true_med = scores.loc[scores["is_true_track"], "adsb_prior_score"].median()
    false_med = scores.loc[~scores["is_true_track"], "adsb_prior_score"].median()

    prior_rows = []
    for feature, prior in priors.items():
        q = prior["quantiles"]
        prior_rows.append({"prior": PRIOR_FILENAMES[feature], "feature": feature,
                           "units": prior["units"], "p95": round(q["p95"], 3),
                           "p99": round(q["p99"], 3), "p999": round(q["p999"], 3)})

    lines = [
        "# Stage 11 Empirical ADS-B-Prior Track Scoring",
        "",
        "## Status",
        "",
        "- Stage 11 applies the **Stage 10 empirical ADS-B priors** to the",
        "  Stage 08 Kalman tracks.",
        f"- {date_scope}",
        "- This is **not VAE**, diffusion, or neural ML -- the priors are",
        "  transparent empirical histograms and quantiles.",
        "- **Truth labels are used only after scoring**, for evaluation.",
        "",
        "## Purpose",
        "",
        "- Stage 09 used hand-designed physics thresholds (soft knees).",
        "- Stage 10 learned empirical GA motion priors from 9.68M ADS-B",
        "  samples.",
        "- Stage 11 tests whether those data-derived priors suppress false",
        "  tracks better than the hand knees while retaining true tracks.",
        "",
        "## Priors used",
        "",
    ]
    lines += md_table(pd.DataFrame(prior_rows))
    lines += [
        "",
        f"- Joint prior: {'loaded and used (weight ' + format(cfg.w_joint, 'g') + ')' if joint_available else 'unavailable/singular -- excluded from weighting'}.",
        f"- SNR auxiliary term: {'available' if snr_available else 'unavailable -- excluded from weighting'}.",
        "",
        "## Scoring model",
        "",
        "- **Empirical quantile-exceedance penalties** per feature, from the",
        "  prior JSON quantiles (nothing hand-coded): raw = frac(>p99) +",
        "  2*frac(>p995) + 4*frac(>p999), normalized and inverted to a [0,1]",
        "  score. Speed also penalizes the LOW side (frac(<p01) + 4*frac(<p001))",
        "  because implausibly slow tracks are as suspicious as fast ones.",
        "- **Mean histogram log-densities** per feature are carried as",
        "  diagnostic columns (speed/accel/vector/turn/vertical _mean_logpdf),",
        "  not mixed into the score.",
        "- **Joint prior score**: robust Mahalanobis-like distance of the",
        "  track's median log1p feature vector to the stage-10 joint median",
        "  under the log1p covariance, score = exp(-0.5 * min(d2, 50)). The",
        "  track median (not p95) is compared because the joint prior",
        "  describes per-sample values.",
        "- **Weak auxiliary terms**: continuity = hit rate"
        f" (w={cfg.w_continuity:g}), SNR mapped over {SNR_SCORE_RANGE_DB} dB"
        f" (w={cfg.w_snr:g}).",
        "- Final score = weighted mean of available components, clamped to",
        f"  [0,1]; keep threshold **{cfg.score_threshold:g}**.",
        "",
        "## Overall results",
        "",
    ]
    lines += md_table(comparison.round(4))
    lines += [
        "",
        "## Comparison with Stage 09",
        "",
    ]
    if stage09_available:
        s9fr = comparison["stage09_false_track_reduction"].mean()
        s11fr = comparison["stage11_false_track_reduction"].mean()
        s9tr = comparison["stage09_true_track_retention"].mean()
        s11tr = comparison["stage11_true_track_retention"].mean()
        lines += [
            f"- Mean false-track reduction: stage 9 {s9fr:.3f} vs stage 11 {s11fr:.3f}"
            + (" -- **ADS-B priors remove more false tracks**." if s11fr > s9fr else
               " -- hand knees removed more; empirical marginal priors alone are"
               " not sufficient here."),
            f"- Mean true-track retention: stage 9 {s9tr:.3f} vs stage 11 {s11tr:.3f}.",
            "- Where stage 11 is stricter, it is because the real GA priors are",
            "  much tighter than the stage-9 knees (empirical p95 speed 92.7 vs",
            "  knee 110 m/s; |accel| 0.43 vs 3.0 m/s^2; turn 2.5 vs 5 deg/s;",
            "  vertical 4.1 vs 10 m/s) -- clutter chains that slipped under the",
            "  generous knees now sit in the far tail of the data distribution.",
        ]
    else:
        lines += ["- Stage 09 comparison files were unavailable; only stage 8 vs",
                  "  stage 11 is reported."]

    # Score-threshold calibration: the exceedance penalties are normalized by
    # their maximum, so scores cluster high and the default 0.5 threshold can
    # be inert. The pooled sweep shows the score's actual discriminative power.
    total_true = max(int(scores["is_true_track"].sum()), 1)
    total_false = max(int((~scores["is_true_track"]).sum()), 1)
    cal_rows = []
    for st in cfg.sweep_thresholds:
        kept = scores[scores["adsb_prior_score"] >= st]
        kt = int(kept["is_true_track"].sum())
        cal_rows.append({"score_threshold": st,
                         "true_retention": round(kt / total_true, 3),
                         "false_reduction": round(1 - (len(kept) - kt) / total_false, 3)})
    lines += [
        "",
        "### Score-threshold calibration (pooled over detection thresholds)",
        "",
        "The exceedance penalties are normalized by their theoretical maximum,",
        "so scores cluster near 1 and the nominal 0.5 keep-threshold is nearly",
        "inert. The pooled sweep below shows the score's actual operating",
        "curve -- pick the keep-threshold from here, not from the 0.5 default:",
        "",
    ]
    lines += md_table(pd.DataFrame(cal_rows))
    lines += [
        "",
        "## Range-bin behavior",
        "",
    ]
    rb = range_bins.groupby("range_bin", sort=False).agg(
        tracks=("stage08_tracks", "sum"),
        true_retention=("true_track_retention", "mean"),
        false_reduction=("false_track_reduction", "mean")).reset_index()
    lines += md_table(rb.round(4))
    lines += [
        "",
        "## Score separability",
        "",
        f"Median ADS-B prior score: true tracks {true_med:.3f}, false tracks {false_med:.3f}.",
        ("True and false tracks separate under the empirical priors."
         if true_med > false_med else
         "True tracks do NOT out-score false tracks -- empirical marginal priors"
         " alone are insufficient on this data."),
        "",
        "## Failure modes",
        "",
        "- Real but unusual maneuvers (aerobatics, tight pattern work) sit in",
        "  the empirical tails and get penalized like clutter.",
        "- Smooth clutter chains whose kinematics land inside the priors can",
        "  still survive.",
        "- Histogram priors treat samples independently -- they ignore the",
        "  temporal SHAPE of the trajectory, which is exactly the information",
        "  a sequence model can use.",
        "- These motivate Stage 12 learned sequence priors.",
        "",
        "## Recommended next stage",
        "",
        "Stage 12 should test learned trajectory-window models:",
        "",
        "- VAE trajectory prior",
        "- denoising autoencoder",
        "- TCN/GRU autoencoder",
        "- diffusion denoiser later",
        "",
    ]
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "adsb_prior_scoring_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_validation_gate(report_dir: str, scores: pd.DataFrame, by_thr: pd.DataFrame,
                        priors: Dict[str, Dict], comparison: pd.DataFrame,
                        stage09_available: bool) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 11 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    path = os.path.join(report_dir, "adsb_prior_track_scores.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        fail("adsb_prior_track_scores.csv missing or empty")
    if scores.empty:
        fail("no confirmed tracks were scored")
    if not scores["adsb_prior_score"].between(0, 1).all():
        fail("adsb_prior_score outside [0, 1]")
    if not scores["keep_adsb_prior"].isin([True, False]).all():
        fail("keep_adsb_prior is not boolean")
    print("  scores file nonempty, adsb_prior_score in [0,1], keep boolean: OK")
    print("  score computation uses stage-10 priors + physics features only "
          "(truth enters afterwards, in evaluation): OK by construction")

    for name in ["speed_mps", "accel_abs_mps2", "turn_rate_abs_deg_s", "vertical_speed_abs_mps"]:
        if name not in priors:
            fail(f"required stage-10 prior not loaded: {name}")
    print("  required stage-10 priors loaded: OK")

    for _, r in by_thr.iterrows():
        if r["stage11_kept_true_tracks"] > r["stage08_true_tracks"]:
            fail("kept true tracks exceed stage-8 true tracks")
        if r["stage11_kept_false_tracks"] > r["stage08_false_tracks"]:
            fail("kept false tracks exceed stage-8 false tracks")
        for col in ("true_track_retention", "false_track_reduction"):
            v = r[col]
            if np.isfinite(v) and not (0.0 <= v <= 1.0):
                fail(f"{col} outside [0, 1]")
    print("  kept-track counts bounded; retention/reduction in [0,1]: OK")

    true_med = scores.loc[scores["is_true_track"], "adsb_prior_score"].median()
    false_med = scores.loc[~scores["is_true_track"], "adsb_prior_score"].median()
    print(f"  median score true={true_med:.3f} vs false={false_med:.3f} (report-only): "
          + ("separable" if true_med > false_med else
             "NOT separable -- empirical marginal priors alone are insufficient"))

    if stage09_available:
        s9 = comparison["stage09_false_track_reduction"].mean()
        s11 = comparison["stage11_false_track_reduction"].mean()
        s9t = comparison["stage09_true_track_retention"].mean()
        s11t = comparison["stage11_true_track_retention"].mean()
        print(f"  stage 9 vs 11 (report-only): false reduction {s9:.3f} -> {s11:.3f}, "
              f"true retention {s9t:.3f} -> {s11t:.3f}"
              + ("" if s11 >= s9 else
                 "  (stage 11 underperforms stage 9 on reduction: empirical "
                 "marginal priors alone are insufficient)"))
