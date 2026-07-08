"""Stage 9: physics-guided track scoring and false-track suppression.

Post-tracking stage: consumes stage-8 confirmed Kalman tracks and assigns
each a transparent, rule-based physics plausibility score in [0, 1]
(1 = very plausible fixed-wing GA track, 0 = likely clutter-born). Nothing
is trained; the tracker is not changed; and the score is computed ONLY from
physics/measurement features -- truth labels (is_target, trajectory_id,
purity, target_fraction, truth positions) are used strictly AFTER scoring,
for evaluation.

Input schemas: the loader accepts either the richer
track_points_*/track_summary_* schema (state_*, event_type,
innovation_norm_m, mahalanobis_d2, snr_db, ...) or the actual stage-8
baseline output (tracks_*_thr_*dB.csv), for which the per-track summary is
derived from the points and snr_db is recovered by joining the stage-6
detections on detection_id. Penalty channels whose inputs are absent
(e.g. Mahalanobis d2 for baseline outputs) are dropped from the weight
normalization and disclosed in the report.

True-track definition here (per the stage-9 spec) is STRICTER than the
stage-8 report's: confirmed AND target_fraction >= 0.8 AND purity >= 0.8
AND a non-null majority trajectory. Stage-8 baseline counts in this
stage's tables are recomputed under this stricter definition, so they will
not numerically match the stage-8 report (which used 0.5 thresholds).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table, token_to_threshold

TRACKS_PATTERN = re.compile(r"^tracks_(\d{4}-\d{2}-\d{2})_thr_(.+)dB\.csv$")
POINTS_PATTERN = re.compile(r"^track_points_(\d{4}-\d{2}-\d{2})_thr_(.+)dB\.csv$")

MIN_SPEED_FOR_TURN_MPS = 10.0     # skip turn-rate samples below this speed

# soft-penalty (good, bad) knees for fixed-wing GA physics
SPEED_P95_GOOD_BAD = (110.0, 160.0)
ACCEL_P95_GOOD_BAD = (3.0, 8.0)
TURN_P95_GOOD_BAD = (5.0, 15.0)
VERTICAL_P95_GOOD_BAD = (10.0, 25.0)
HITRATE_GOOD_BAD = (0.8, 0.4)     # reversed: lower is worse
SNR_P50_GOOD_BAD = (3.0, -10.0)   # reversed: lower is worse
MAHAL_P95_GOOD_BAD = (10.0, 25.0)

TRUE_TRACK_TARGET_FRACTION = 0.8
TRUE_TRACK_PURITY = 0.8

SCORE_COLUMNS = [
    "date", "threshold_db", "track_id",
    "n_hits", "n_misses", "duration_s", "hit_rate",
    "physics_score", "keep_physics",
    "speed_penalty", "accel_penalty", "turn_penalty", "vertical_penalty",
    "continuity_penalty", "snr_penalty", "association_penalty",
    "speed_mps_p50", "speed_mps_p95", "speed_mps_max",
    "accel_mps2_p50", "accel_mps2_p95", "accel_mps2_max",
    "vertical_speed_mps_p95_abs", "turn_rate_deg_s_p95_abs",
    "innovation_norm_m_p95", "mahalanobis_d2_p95", "snr_db_p50",
    "median_range_m", "max_range_m",
    "n_target_hits", "n_clutter_hits", "target_fraction", "purity",
    "majority_trajectory_id", "is_true_track", "position_rmse_m",
]


@dataclass
class PhysicsScoreConfig:
    """All stage-9 tunables in one place (populated from the CLI)."""
    tracks_dir: str = "data/active/tracks_kalman"
    detections_dir: str = "data/active/sim_detections_relocated"
    scored_tracks_dir: str = "data/active/tracks_scored_physics"
    report_dir: str = "reports/stage09_physics_scoring"
    thresholds_db: Optional[List[float]] = None
    dates: Optional[List[str]] = None
    range_bins_m: List[float] = field(default_factory=lambda: [0, 50_000.0, 100_000.0, 200_000.0, float("inf")])
    score_threshold: float = 0.5
    sweep_thresholds: List[float] = field(default_factory=lambda: [round(0.1 * k, 1) for k in range(1, 10)])
    min_hits: int = 3
    chunksize: int = 1_000_000
    overwrite: bool = False
    no_plots: bool = False
    # penalty weights
    w_speed: float = 1.0
    w_accel: float = 1.0
    w_turn: float = 1.0
    w_vertical: float = 1.0
    w_continuity: float = 1.0
    w_snr: float = 0.5
    w_clutter: float = 1.0            # weight of the association-consistency penalty


def soft_penalty(value: float, good: float, bad: float) -> float:
    """0 at/inside `good`, 1 at/beyond `bad`, linear in between. Works for
    reversed knees (good > bad, where LOWER values are worse). NaN -> 0.5."""
    if value is None or not np.isfinite(value):
        return 0.5
    if good <= bad:   # normal: big is bad
        if value <= good:
            return 0.0
        if value >= bad:
            return 1.0
        return (value - good) / (bad - good)
    # reversed: small is bad
    if value >= good:
        return 0.0
    if value <= bad:
        return 1.0
    return (good - value) / (good - bad)


# =============================================================================
# Discovery / loading (both stage-8 schemas)
# =============================================================================

def discover_stage08_runs(tracks_dir: str) -> List[Tuple[str, float, str]]:
    """Sorted (date, threshold_db, points_path). Prefers the richer
    track_points_* schema; falls back to the baseline tracks_* schema."""
    runs = {}
    for name in sorted(os.listdir(tracks_dir)):
        for pattern in (POINTS_PATTERN, TRACKS_PATTERN):
            m = pattern.match(name)
            if m:
                thr = token_to_threshold(m.group(2))
                key = (m.group(1), thr)
                # points schema wins if both exist
                if key not in runs or pattern is POINTS_PATTERN:
                    runs[key] = os.path.join(tracks_dir, name)
    return sorted((d, t, p) for (d, t), p in runs.items())


def _snr_lookup(detections_dir: str, date: str, threshold_db: float) -> Optional[np.ndarray]:
    """snr_db indexed by detection_id (stage 6 assigns sequential ids)."""
    token = f"{threshold_db:.1f}".replace("-", "m").replace(".", "p")
    path = os.path.join(detections_dir, f"detections_{date}_thr_{token}dB.csv")
    if not os.path.exists(path):
        return None
    snr = pd.read_csv(path, usecols=["detection_id", "snr_db"])
    arr = np.full(int(snr["detection_id"].max()) + 1, np.nan)
    arr[snr["detection_id"].to_numpy()] = snr["snr_db"].to_numpy()
    return arr


def load_run(points_path: str, date: str, threshold_db: float,
             detections_dir: str) -> pd.DataFrame:
    """Load one stage-8 run into the canonical internal schema:
    track_id, timestamp, x, y, z, vx, vy, vz, confirmed, is_hit,
    assoc_is_target, assoc_trajectory_id, pos_error_m,
    snr_db / innovation_norm_m / mahalanobis_d2 (NaN where unavailable)."""
    name = os.path.basename(points_path)
    if POINTS_PATTERN.match(name):
        df = pd.read_csv(points_path, dtype={"trajectory_id": str}, low_memory=False)
        out = pd.DataFrame({
            "track_id": df["track_id"],
            "timestamp": df["timestamp"],
            "x": df["state_x_m"], "y": df["state_y_m"], "z": df["state_z_m"],
            "vx": df["state_vx_mps"], "vy": df["state_vy_mps"], "vz": df["state_vz_mps"],
            "confirmed": df["is_confirmed"].astype(int),
            "is_hit": (df["assigned_detection_id"] >= 0).astype(int)
            if "assigned_detection_id" in df.columns else (df["event_type"] == "hit").astype(int),
            "assoc_is_target": df["is_target"].fillna(-1).astype(int),
            "assoc_trajectory_id": df["trajectory_id"],
            "snr_db": df["snr_db"] if "snr_db" in df.columns else np.nan,
            "innovation_norm_m": df["innovation_norm_m"] if "innovation_norm_m" in df.columns else np.nan,
            "mahalanobis_d2": df["mahalanobis_d2"] if "mahalanobis_d2" in df.columns else np.nan,
        })
        if "truth_x_m" in df.columns:
            err = np.sqrt((df["state_x_m"] - df["truth_x_m"])**2
                          + (df["state_y_m"] - df["truth_y_m"])**2
                          + (df["state_z_m"] - df["truth_z_m"])**2)
            out["pos_error_m"] = err.where(df["is_target"] == 1)
        else:
            out["pos_error_m"] = np.nan
        return out

    # baseline stage-8 schema (tracks_*.csv)
    df = pd.read_csv(points_path, dtype={"assoc_trajectory_id": str}, low_memory=False)
    out = pd.DataFrame({
        "track_id": df["track_id"],
        "timestamp": df["timestamp"],
        "x": df["x_m"], "y": df["y_m"], "z": df["z_m"],
        "vx": df["vx_mps"], "vy": df["vy_mps"], "vz": df["vz_mps"],
        "confirmed": df["confirmed"],
        "is_hit": (df["detection_id"] >= 0).astype(int),
        "assoc_is_target": df["assoc_is_target"],
        "assoc_trajectory_id": df["assoc_trajectory_id"],
        "pos_error_m": df["pos_error_m"],
        "innovation_norm_m": np.nan,
        "mahalanobis_d2": np.nan,
    })
    snr_arr = _snr_lookup(detections_dir, date, threshold_db)
    if snr_arr is not None:
        det_ids = df["detection_id"].to_numpy()
        snr = np.full(len(df), np.nan)
        hit = det_ids >= 0
        valid = hit & (det_ids < len(snr_arr))
        snr[valid] = snr_arr[det_ids[valid]]
        out["snr_db"] = snr
    else:
        out["snr_db"] = np.nan
    return out


# =============================================================================
# Per-track features + score
# =============================================================================

def compute_track_features(g: pd.DataFrame) -> Dict:
    """Physics + evaluation features for one track (rows sorted by time)."""
    g = g.sort_values("timestamp")
    t = g["timestamp"].to_numpy(dtype=float)
    v = g[["vx", "vy", "vz"]].to_numpy(dtype=float)
    pos = g[["x", "y", "z"]].to_numpy(dtype=float)

    n_points = len(g)
    n_hits = int(g["is_hit"].sum())
    n_misses = n_points - n_hits
    duration = float(t[-1] - t[0]) if n_points > 1 else 0.0
    hit_rate = n_hits / n_points if n_points else np.nan

    dt = np.diff(t)
    ok = dt > 0
    speed = np.linalg.norm(v, axis=1)
    dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
    accel = dv[ok] / dt[ok]

    heading = np.degrees(np.arctan2(v[:, 0], v[:, 1]))      # x=east, y=north
    dh = np.diff(heading)
    dh = (dh + 180.0) % 360.0 - 180.0
    hspeed = np.linalg.norm(v[:, :2], axis=1)
    turn_ok = ok & (hspeed[:-1] >= MIN_SPEED_FOR_TURN_MPS) & (hspeed[1:] >= MIN_SPEED_FOR_TURN_MPS)
    turn = np.abs(dh[turn_ok] / dt[turn_ok]) if turn_ok.any() else np.array([])

    rng = np.linalg.norm(pos, axis=1)
    vz_abs = np.abs(v[:, 2])
    med_dt = float(np.median(dt[ok])) if ok.any() else np.nan

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else np.nan

    snr = g["snr_db"].to_numpy(dtype=float)
    snr = snr[np.isfinite(snr)]
    innov = g["innovation_norm_m"].to_numpy(dtype=float)
    innov = innov[np.isfinite(innov)]
    mahal = g["mahalanobis_d2"].to_numpy(dtype=float)
    mahal = mahal[np.isfinite(mahal)]

    # ---- evaluation-only labels (never used for the score) ----------------
    hits = g[g["is_hit"] == 1]
    n_target_hits = int((hits["assoc_is_target"] == 1).sum())
    n_clutter_hits = int((hits["assoc_is_target"] == 0).sum())
    target_fraction = n_target_hits / len(hits) if len(hits) else np.nan
    traj = hits.loc[hits["assoc_is_target"] == 1, "assoc_trajectory_id"].dropna()
    majority = traj.mode().iloc[0] if len(traj) else None
    purity = target_fraction   # target hits / associated hits (same convention)
    errs = g["pos_error_m"].to_numpy(dtype=float)
    errs = errs[np.isfinite(errs)]

    return {
        "n_points": n_points, "n_hits": n_hits, "n_misses": n_misses,
        "duration_s": duration, "hit_rate": hit_rate,
        "median_dt_s": med_dt,
        "max_gap_s": float(dt[ok].max()) if ok.any() else np.nan,
        "gap_fraction": float((dt[ok] > 2 * med_dt).mean()) if ok.any() and np.isfinite(med_dt) else np.nan,
        "speed_mps_p50": pct(speed, 50), "speed_mps_p95": pct(speed, 95),
        "speed_mps_max": float(speed.max()) if len(speed) else np.nan,
        "accel_mps2_p50": pct(accel, 50), "accel_mps2_p95": pct(accel, 95),
        "accel_mps2_max": float(accel.max()) if len(accel) else np.nan,
        "vertical_speed_mps_p50_abs": pct(vz_abs, 50),
        "vertical_speed_mps_p95_abs": pct(vz_abs, 95),
        "vertical_speed_mps_max_abs": float(vz_abs.max()) if len(vz_abs) else np.nan,
        "turn_rate_deg_s_p50_abs": pct(turn, 50),
        "turn_rate_deg_s_p95_abs": pct(turn, 95),
        "turn_rate_deg_s_max_abs": float(turn.max()) if len(turn) else np.nan,
        "innovation_norm_m_p50": pct(innov, 50), "innovation_norm_m_p95": pct(innov, 95),
        "mahalanobis_d2_p50": pct(mahal, 50), "mahalanobis_d2_p95": pct(mahal, 95),
        "snr_db_p50": pct(snr, 50), "snr_db_p10": pct(snr, 10), "snr_db_p90": pct(snr, 90),
        "first_range_m": float(rng[0]) if len(rng) else np.nan,
        "median_range_m": pct(rng, 50),
        "max_range_m": float(rng.max()) if len(rng) else np.nan,
        # evaluation-only
        "n_target_hits": n_target_hits, "n_clutter_hits": n_clutter_hits,
        "target_fraction": target_fraction, "purity": purity,
        "majority_trajectory_id": majority,
        "position_rmse_m": float(np.sqrt((errs**2).mean())) if len(errs) else np.nan,
    }


def score_features(f: Dict, cfg: PhysicsScoreConfig,
                   snr_available: bool, association_available: bool) -> Dict:
    """Rule-based physics plausibility score from PHYSICS features only.
    Channels whose inputs don't exist in this run are excluded from the
    weight normalization (not silently scored 0.5)."""
    penalties = {
        "speed_penalty": (soft_penalty(f["speed_mps_p95"], *SPEED_P95_GOOD_BAD), cfg.w_speed, True),
        "accel_penalty": (soft_penalty(f["accel_mps2_p95"], *ACCEL_P95_GOOD_BAD), cfg.w_accel, True),
        "turn_penalty": (soft_penalty(f["turn_rate_deg_s_p95_abs"], *TURN_P95_GOOD_BAD), cfg.w_turn, True),
        "vertical_penalty": (soft_penalty(f["vertical_speed_mps_p95_abs"], *VERTICAL_P95_GOOD_BAD), cfg.w_vertical, True),
        "continuity_penalty": (soft_penalty(f["hit_rate"], *HITRATE_GOOD_BAD), cfg.w_continuity, True),
        "snr_penalty": (soft_penalty(f["snr_db_p50"], *SNR_P50_GOOD_BAD), cfg.w_snr, snr_available),
        "association_penalty": (soft_penalty(f["mahalanobis_d2_p95"], *MAHAL_P95_GOOD_BAD),
                                cfg.w_clutter, association_available),
    }
    total_w = sum(w for _, w, avail in penalties.values() if avail)
    total_p = sum(p * w for p, w, avail in penalties.values() if avail)
    score = 1.0 - (total_p / total_w if total_w > 0 else 0.5)
    score = float(np.clip(score, 0.0, 1.0))

    out = {name: p for name, (p, _, _) in penalties.items()}
    out["physics_score"] = score
    return out


# =============================================================================
# Orchestration per run
# =============================================================================

def score_run(date: str, threshold_db: float, points_path: str,
              cfg: PhysicsScoreConfig) -> pd.DataFrame:
    """Score every confirmed track of one (date, threshold) stage-8 run."""
    df = load_run(points_path, date, threshold_db, cfg.detections_dir)
    snr_available = bool(np.isfinite(df["snr_db"]).any())
    association_available = bool(np.isfinite(df["mahalanobis_d2"]).any())

    confirmed_ids = df.loc[df["confirmed"] == 1, "track_id"].unique()
    df = df[df["track_id"].isin(confirmed_ids)]

    rows = []
    for track_id, g in df.groupby("track_id", sort=False):
        f = compute_track_features(g)
        if f["n_hits"] < cfg.min_hits:
            continue
        s = score_features(f, cfg, snr_available, association_available)
        f.update(s)
        f["keep_physics"] = bool(f["physics_score"] >= cfg.score_threshold)
        tf, pu = f["target_fraction"], f["purity"]
        f["is_true_track"] = bool(
            np.isfinite(tf) and tf >= TRUE_TRACK_TARGET_FRACTION
            and np.isfinite(pu) and pu >= TRUE_TRACK_PURITY
            and f["majority_trajectory_id"] is not None)
        f.update({"date": date, "threshold_db": threshold_db, "track_id": track_id})
        rows.append(f)

    scores = pd.DataFrame(rows)
    meta = {"snr_available": snr_available, "association_available": association_available}
    scores.attrs.update(meta)
    return scores


# =============================================================================
# Aggregates
# =============================================================================

def aggregate_scores(scores: pd.DataFrame, by: List[str], score_threshold: float) -> pd.DataFrame:
    rows = []
    for key, g in scores.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        true = g[g["is_true_track"]]
        false = g[~g["is_true_track"]]
        kept = g[g["keep_physics"]]
        kept_true = kept[kept["is_true_track"]]
        kept_false = kept[~kept["is_true_track"]]
        row = dict(zip(by, key))
        row.update({
            "stage08_confirmed_tracks": len(g),
            "stage08_true_tracks": len(true),
            "stage08_false_tracks": len(false),
            "stage09_kept_tracks": len(kept),
            "stage09_kept_true_tracks": len(kept_true),
            "stage09_kept_false_tracks": len(kept_false),
            "true_track_retention": len(kept_true) / len(true) if len(true) else np.nan,
            "false_track_retention": len(kept_false) / len(false) if len(false) else np.nan,
            "false_track_reduction": 1 - len(kept_false) / len(false) if len(false) else np.nan,
            "precision_before": len(true) / len(g) if len(g) else np.nan,
            "precision_after": len(kept_true) / len(kept) if len(kept) else np.nan,
            "coverage_proxy_before": np.nan,   # track-level stage; see report note
            "coverage_proxy_after": np.nan,
            "mean_score_true_tracks": float(true["physics_score"].mean()) if len(true) else np.nan,
            "mean_score_false_tracks": float(false["physics_score"].mean()) if len(false) else np.nan,
            "median_score_true_tracks": float(true["physics_score"].median()) if len(true) else np.nan,
            "median_score_false_tracks": float(false["physics_score"].median()) if len(false) else np.nan,
            "score_threshold": score_threshold,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def sweep_table(scores: pd.DataFrame, sweep_thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for thr_db, g in scores.groupby("threshold_db"):
        true = g[g["is_true_track"]]
        false = g[~g["is_true_track"]]
        for st in sweep_thresholds:
            kept = g[g["physics_score"] >= st]
            kt = int(kept["is_true_track"].sum())
            kf = len(kept) - kt
            rows.append({
                "threshold_db": thr_db, "score_threshold": st,
                "kept_tracks": len(kept), "kept_true_tracks": kt, "kept_false_tracks": kf,
                "true_track_retention": kt / len(true) if len(true) else np.nan,
                "false_track_retention": kf / len(false) if len(false) else np.nan,
                "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
                "precision_after": kt / len(kept) if len(kept) else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Plots (matplotlib only, default colors, one figure each)
# =============================================================================

def make_plots(scores: pd.DataFrame, by_thr: pd.DataFrame, sweep: pd.DataFrame,
               plots_dir: str) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(scores.loc[scores["is_true_track"], "physics_score"], bins=bins,
            alpha=0.6, label="true tracks", density=True)
    ax.hist(scores.loc[~scores["is_true_track"], "physics_score"], bins=bins,
            alpha=0.6, label="false tracks", density=True)
    ax.set_xlabel("physics score")
    ax.set_ylabel("density")
    ax.set_title("Physics-score distributions, true vs false tracks")
    ax.legend()
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "score_hist_true_false.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for thr_db, g in sweep.groupby("threshold_db"):
        g = g.sort_values("score_threshold")
        ax.plot(g["false_track_reduction"], g["true_track_retention"], marker="o",
                label=f"{thr_db:g} dB")
    ax.set_xlabel("false-track reduction")
    ax.set_ylabel("true-track retention")
    ax.set_title("Retention vs reduction across score thresholds")
    ax.legend(title="detection threshold")
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "false_track_reduction_vs_true_retention.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    o = by_thr.sort_values("threshold_db")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(o["threshold_db"], o["stage08_false_tracks"], marker="o", label="stage 08 false tracks")
    ax.plot(o["threshold_db"], o["stage09_kept_false_tracks"], marker="o", label="stage 09 kept false tracks")
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("false confirmed tracks")
    ax.set_title("False tracks before vs after physics filtering")
    ax.legend()
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "stage08_vs_stage09_false_tracks.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(o["threshold_db"], o["stage08_true_tracks"], marker="o", label="stage 08 true tracks")
    ax.plot(o["threshold_db"], o["stage09_kept_true_tracks"], marker="o", label="stage 09 kept true tracks")
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("true confirmed tracks")
    ax.set_title("True tracks before vs after physics filtering")
    ax.legend()
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "stage08_vs_stage09_track_detection.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(report_dir: str, scores: pd.DataFrame, by_thr: pd.DataFrame,
                 comparison: pd.DataFrame, sweep: pd.DataFrame,
                 cfg: PhysicsScoreConfig, channels_note: str) -> str:
    true_med = scores.loc[scores["is_true_track"], "physics_score"].median()
    false_med = scores.loc[~scores["is_true_track"], "physics_score"].median()
    separable = np.isfinite(true_med) and np.isfinite(false_med) and true_med > false_med
    dates = sorted(scores["date"].unique())
    date_scope = (f"These are one-day results for {dates[0]}." if len(dates) == 1
                  else f"These results cover {len(dates)} days: {', '.join(dates)}.")

    lines = [
        "# Stage 09 Physics-Guided Track Scoring",
        "",
        "## Status",
        "",
        f"- {date_scope}",
        "- **Stage 10 is intentionally deferred.**",
        "- Stage 09 remains **hand-designed physics scoring**, not empirical",
        "  ADS-B prior learning and not VAE.",
        "",
        "## Experiment definition",
        "",
        "- Input is **Stage 08 confirmed Kalman tracks**; Stage 09 does not",
        "  change the tracker in any way.",
        "- Stage 09 computes a **rule-based physics plausibility score** in",
        "  [0, 1] per confirmed track, from posterior-state kinematics and",
        "  measurement features only.",
        "- **Truth labels are used only after scoring**, to evaluate retention",
        "  and reduction. The score itself never sees is_target,",
        "  trajectory_id, purity, target_fraction, or truth positions.",
        "- This is **not ML** and not a VAE — nothing is trained; every",
        "  penalty is a transparent soft threshold.",
        "- True-track definition in this stage (target_fraction >= 0.8 and",
        "  purity >= 0.8) is stricter than the stage-8 report's 0.5-based",
        "  definition, so the stage-8 baseline counts here are recomputed",
        "  under the stricter rule and will not match the stage-8 report",
        "  numbers exactly.",
        f"- {channels_note}",
        "",
        "## Scoring model",
        "",
        "Soft penalties (0 inside the good knee, 1 beyond the bad knee,",
        "linear between, NaN -> 0.5), combined as a weighted mean and",
        "inverted: score = 1 - normalized penalty.",
        "",
        f"- speed: p95 speed good <= {SPEED_P95_GOOD_BAD[0]:.0f} m/s, bad >= {SPEED_P95_GOOD_BAD[1]:.0f} m/s (w={cfg.w_speed:g})",
        f"- acceleration: p95 good <= {ACCEL_P95_GOOD_BAD[0]:.1f} m/s^2, bad >= {ACCEL_P95_GOOD_BAD[1]:.1f} m/s^2 (w={cfg.w_accel:g})",
        f"- turn rate: p95 |turn| good <= {TURN_P95_GOOD_BAD[0]:.0f} deg/s, bad >= {TURN_P95_GOOD_BAD[1]:.0f} deg/s (w={cfg.w_turn:g})",
        f"- vertical speed: p95 |vz| good <= {VERTICAL_P95_GOOD_BAD[0]:.0f} m/s, bad >= {VERTICAL_P95_GOOD_BAD[1]:.0f} m/s (w={cfg.w_vertical:g})",
        f"- continuity: hit rate good >= {HITRATE_GOOD_BAD[0]:.1f}, bad <= {HITRATE_GOOD_BAD[1]:.1f} (reversed; w={cfg.w_continuity:g})",
        f"- SNR: median good >= {SNR_P50_GOOD_BAD[0]:.0f} dB, bad <= {SNR_P50_GOOD_BAD[1]:.0f} dB (reversed; lower weight w={cfg.w_snr:g} because weak real targets are valid)",
        f"- association consistency: p95 Mahalanobis d2 good <= {MAHAL_P95_GOOD_BAD[0]:.0f}, bad >= {MAHAL_P95_GOOD_BAD[1]:.0f} (w={cfg.w_clutter:g})",
        "",
        f"Keep/reject threshold used: **physics_score >= {cfg.score_threshold:g}**.",
        "",
        "## Overall results",
        "",
    ]
    lines += md_table(comparison)
    lines += [
        "",
        "coverage_proxy columns are NaN: this stage evaluates **track-level",
        "retention**, not full trajectory coverage — kept-track trajectory",
        "coverage belongs to a stage-8-style re-evaluation over the filtered",
        "track set.",
        "",
        "## Threshold sweep",
        "",
        "How false-track reduction trades against true-track retention as the",
        "score threshold moves (per detection threshold):",
        "",
    ]
    pivot = sweep.pivot_table(index="score_threshold", columns="threshold_db",
                              values="false_track_reduction").round(3).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    lines += ["False-track reduction:", ""] + md_table(pivot)
    pivot2 = sweep.pivot_table(index="score_threshold", columns="threshold_db",
                               values="true_track_retention").round(3).reset_index()
    pivot2.columns = [str(c) for c in pivot2.columns]
    lines += ["", "True-track retention:", ""] + md_table(pivot2)
    lines += [
        "",
        "## Score separability",
        "",
        f"Median physics score: true tracks {true_med:.3f}, false tracks {false_med:.3f}.",
        ("True and false tracks separate in score — the physics plausibility"
         " features carry real signal.") if separable else
        ("True tracks do NOT score higher than false tracks — the current"
         " hand-designed features are not separable enough on this data."),
        "",
        "## Failure modes",
        "",
        "- False tracks that happen to look kinematically plausible (clutter",
        "  chains that mimic straight flight) survive the filter.",
        "- Real weak/noisy tracks can be rejected when the continuity/SNR",
        "  penalties are too harsh for genuinely low-SNR targets.",
        "- Hand-tuned knees are brittle across scenarios — this motivates",
        "  empirical ADS-B priors and later VAE/learned motion priors.",
        "",
        "## Recommended next stage",
        "",
        "Stage 10 should learn empirical motion-prior distributions (speed,",
        "acceleration, turn rate, vertical speed) from ADS-B/Stage 4 data or",
        "from true tracks, and replace these hand-tuned penalties with",
        "data-derived likelihoods.",
        "",
    ]
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "physics_scoring_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_validation_gate(report_dir: str, scores: pd.DataFrame, by_thr: pd.DataFrame) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 09 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    path = os.path.join(report_dir, "physics_track_scores.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        fail("physics_track_scores.csv missing or empty")
    if scores.empty:
        fail("no confirmed tracks were scored")
    if not scores["physics_score"].between(0, 1).all():
        fail("physics_score outside [0, 1]")
    if not scores["keep_physics"].isin([True, False]).all():
        fail("keep_physics is not boolean")
    print("  scores file nonempty, physics_score in [0,1], keep_physics boolean: OK")
    print("  score computation uses physics/measurement features only "
          "(truth columns enter afterwards, in evaluation): OK by construction")

    for _, r in by_thr.iterrows():
        if r["stage09_kept_true_tracks"] > r["stage08_true_tracks"]:
            fail("kept true tracks exceed stage-8 true tracks")
        if r["stage09_kept_false_tracks"] > r["stage08_false_tracks"]:
            fail("kept false tracks exceed stage-8 false tracks")
        for col in ("true_track_retention", "false_track_reduction"):
            v = r[col]
            if np.isfinite(v) and not (0.0 <= v <= 1.0):
                fail(f"{col} outside [0, 1]")
    print("  kept-track counts bounded by stage-8 counts; retention/reduction in [0,1]: OK")

    true_med = scores.loc[scores["is_true_track"], "physics_score"].median()
    false_med = scores.loc[~scores["is_true_track"], "physics_score"].median()
    print(f"  median score true={true_med:.3f} vs false={false_med:.3f} (report-only): "
          + ("separable" if true_med > false_med else
         "NOT separable -- current hand-designed features are not separable enough"))
