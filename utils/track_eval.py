"""Stage 8: evaluation of Kalman baseline tracks against truth labels.

Truth enters ONLY here: stage-6 detections carry is_target and
trajectory_id, so every track state that consumed a detection is labeled.
A confirmed track is scored by the composition of the detections it
consumed:

  * purity            = target detections / associated detections
  * dominant fraction = share of associated detections belonging to the
                        track's single most-frequent truth trajectory
  * TRUE track        = confirmed, purity >= 0.5, dominant fraction >= 0.5
  * FALSE track       = confirmed but not true (mostly clutter-born)

Coverage counts a truth trajectory as covered when at least one TRUE track
is dominated by it; the denominator is trajectories with at least
`min_trajectory_dets` target detections in the evaluated frames (a
trajectory that was never detected cannot be tracked by any method).
Fragmentation is true tracks per covered trajectory.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.common import md_table


@dataclass
class TrackEvalConfig:
    purity_threshold: float = 0.5
    dominant_threshold: float = 0.5
    min_trajectory_dets: int = 5     # detections needed for a trajectory to count as trackable


def evaluate_tracks(tracks: pd.DataFrame, detections: pd.DataFrame,
                    cfg: Optional[TrackEvalConfig] = None) -> Dict:
    """Score one tracker run. `tracks` is run_tracker() output; `detections`
    is the same stream the tracker consumed (for the trajectory denominator).
    Returns {'metrics': dict, 'per_track': DataFrame}."""
    cfg = cfg or TrackEvalConfig()

    assoc = tracks[tracks["detection_id"] >= 0]
    confirmed_by_track = tracks.groupby("track_id")["confirmed"].max()
    per_track_rows = []
    for track_id, g in assoc.groupby("track_id"):
        n = len(g)
        n_target = int((g["assoc_is_target"] == 1).sum())
        traj = g.loc[g["assoc_is_target"] == 1, "assoc_trajectory_id"].dropna()
        if len(traj):
            dominant_traj = traj.mode().iloc[0]
            dominant_n = int((g["assoc_trajectory_id"] == dominant_traj).sum())
        else:
            dominant_traj, dominant_n = "", 0
        confirmed = bool(confirmed_by_track.loc[track_id])
        purity = n_target / n
        dominant_fraction = dominant_n / n
        errs = g["pos_error_m"].to_numpy(dtype=float) if "pos_error_m" in g.columns else np.array([])
        errs = errs[np.isfinite(errs)]
        per_track_rows.append({
            "track_id": track_id, "confirmed": confirmed,
            "n_associated": n, "n_target_dets": n_target,
            "purity": purity, "dominant_trajectory_id": dominant_traj,
            "dominant_fraction": dominant_fraction,
            "position_rmse_m": float(np.sqrt((errs**2).mean())) if errs.size else np.nan,
            "is_true_track": bool(confirmed and purity >= cfg.purity_threshold
                                  and dominant_fraction >= cfg.dominant_threshold),
        })
    per_track = pd.DataFrame(per_track_rows)

    n_frames = int(tracks["frame_id"].nunique()) if len(tracks) else 0
    confirmed_tracks = per_track[per_track["confirmed"]] if len(per_track) else per_track
    true_tracks = confirmed_tracks[confirmed_tracks["is_true_track"]] if len(per_track) else per_track

    # Trackable-trajectory denominator, from the same detection stream.
    tgt = detections[detections["is_target"] == 1]
    traj_counts = tgt.groupby("trajectory_id").size()
    trackable = set(traj_counts[traj_counts >= cfg.min_trajectory_dets].index)
    covered = set(true_tracks["dominant_trajectory_id"]) & trackable if len(per_track) else set()

    # Detection absorption into confirmed tracks.
    confirmed_ids = set(confirmed_tracks["track_id"]) if len(per_track) else set()
    assoc_conf = assoc[assoc["track_id"].isin(confirmed_ids)]
    n_target_dets = int((detections["is_target"] == 1).sum())
    n_clutter_dets = int((detections["is_target"] == 0).sum())

    metrics = {
        "frames": n_frames,
        "detections_in": len(detections),
        "tracks_total": int(len(per_track)),
        "tracks_confirmed": int(len(confirmed_tracks)),
        "true_tracks": int(len(true_tracks)),
        "false_tracks": int(len(confirmed_tracks) - len(true_tracks)),
        "false_tracks_per_frame": (len(confirmed_tracks) - len(true_tracks)) / n_frames if n_frames else np.nan,
        "trackable_trajectories": len(trackable),
        "covered_trajectories": len(covered),
        "trajectory_coverage": len(covered) / len(trackable) if trackable else np.nan,
        "fragmentation": len(true_tracks) / len(covered) if covered else np.nan,
        "mean_true_track_purity": float(true_tracks["purity"].mean()) if len(true_tracks) else np.nan,
        "target_det_absorption": (int((assoc_conf["assoc_is_target"] == 1).sum()) / n_target_dets
                                  if n_target_dets else np.nan),
        "clutter_det_absorption": (int((assoc_conf["assoc_is_target"] == 0).sum()) / n_clutter_dets
                                   if n_clutter_dets else np.nan),
        # Purity / state-error aggregates over TRUE tracks (posterior track
        # position vs the truth position of each associated target detection).
        "median_track_purity": float(true_tracks["purity"].median()) if len(true_tracks) else np.nan,
        "mean_position_rmse_m": (float(true_tracks["position_rmse_m"].mean())
                                 if len(true_tracks) else np.nan),
        "median_position_rmse_m": (float(true_tracks["position_rmse_m"].median())
                                   if len(true_tracks) else np.nan),
    }
    # Spec-named aliases used by the stage07-vs-stage08 comparison.
    metrics["track_detection_rate"] = metrics["trajectory_coverage"]
    metrics["target_assignment_rate"] = metrics["target_det_absorption"]
    metrics["clutter_assignment_rate"] = metrics["clutter_det_absorption"]
    metrics["mean_track_purity"] = metrics["mean_true_track_purity"]
    return {"metrics": metrics, "per_track": per_track}


def write_report(report_dir: str, summary_rows: list, tracker_cfg, eval_cfg: TrackEvalConfig) -> str:
    """Markdown report over all evaluated (date, threshold) runs."""
    df = pd.DataFrame(summary_rows)
    lines = [
        "# Stage 08 Kalman Detect-Then-Track Baseline",
        "",
        "## Method",
        "",
        "- Constant-velocity Kalman filter per track (6-state position +",
        "  velocity, discrete white-noise acceleration process model).",
        "- Greedy gated nearest-neighbor association on predicted position",
        f"  (Euclidean gate {tracker_cfg.gate_m:.0f} m).",
        f"- Tracks confirm after {tracker_cfg.confirm_hits} hits and delete after",
        f"  {tracker_cfg.max_misses} consecutive misses.",
        "- The tracker never sees truth labels; is_target / trajectory_id are",
        "  used only in this evaluation.",
        f"- A TRUE track is confirmed with purity >= {eval_cfg.purity_threshold}",
        f"  and dominant-trajectory fraction >= {eval_cfg.dominant_threshold};",
        "  coverage denominators are trajectories with >=",
        f"  {eval_cfg.min_trajectory_dets} target detections in the evaluated frames.",
        "- Baseline simplifications (stage 9+ targets): isotropic measurement",
        "  noise, Euclidean (not Mahalanobis) gating, radial velocity unused.",
        "",
        "## Results",
        "",
    ]
    if len(df):
        cols = ["date", "threshold_db", "frames", "detections_in", "tracks_confirmed",
                "true_tracks", "false_tracks", "trajectory_coverage", "fragmentation",
                "mean_true_track_purity", "target_det_absorption", "clutter_det_absorption"]
        lines += md_table(df[[c for c in cols if c in df.columns]])
    lines += [
        "",
        "## Interpretation",
        "",
        "The Kalman baseline turns point detections into persistent objects:",
        "true tracks follow real aircraft across frames, while clutter-born",
        "tracks rarely survive confirmation because false alarms are spatially",
        "independent between scans. Compare trajectory coverage here against",
        "the stage-7 single-frame Pd at the same threshold to quantify what",
        "temporal integration buys before any ML model is introduced.",
        "",
    ]
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "kalman_baseline_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
