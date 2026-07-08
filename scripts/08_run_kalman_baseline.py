"""Entry point: stage 8 constant-velocity Kalman detect-then-track baseline.

Runs the classical tracker over stage-6 detection streams (one (date,
threshold) file at a time), writes per-track state CSVs and an evaluation
report. The tracker never sees truth labels; the is_target /
trajectory_id columns carried by stage-6 detections are used only by the
post-hoc evaluation. No ML, no smoothing beyond the Kalman filter itself.

Usage:
    python scripts/08_run_kalman_baseline.py --threshold-db 0 --date 2022-06-06 --max-frames 1000
    python scripts/08_run_kalman_baseline.py --self-test
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kalman_tracker import KalmanTrackerConfig, run_tracker, spherical_to_cartesian
from utils.track_eval import TrackEvalConfig, evaluate_tracks, write_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION_PATTERN = re.compile(r"detections_(\d{4}-\d{2}-\d{2})_thr_(.+)dB\.csv$")

DETECTION_USECOLS = ["frame_id", "timestamp", "detection_id", "is_target", "trajectory_id",
                     "meas_range_m", "meas_azimuth_rad", "meas_elevation_rad"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Constant-velocity Kalman detect-then-track baseline over stage-6 detections.")
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    parser.add_argument("--truth-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"),
                        help="Stage-5 truth (reserved for future state-error metrics; the "
                             "current evaluation uses the labels carried by the detections).")
    parser.add_argument("--tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline"))
    parser.add_argument("--threshold-db", type=float, default=0.0,
                        help="Which stage-6 threshold stream to track (default: 0).")
    parser.add_argument("--date", type=str, default=None,
                        help="Restrict to one day (default: every day found).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Track only the first N frames of each day (default: all).")
    parser.add_argument("--gate-m", type=float, default=2500.0)
    parser.add_argument("--q-accel-mps2", type=float, default=1.0)
    parser.add_argument("--confirm-hits", type=int, default=3)
    parser.add_argument("--max-misses", type=int, default=3)
    parser.add_argument("--frame-period-s", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate track CSVs that already exist (default: skip them).")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def discover(detections_dir: str, threshold_db: float, date: str = None):
    out = []
    for name in sorted(os.listdir(detections_dir)):
        m = DETECTION_PATTERN.search(name)
        if not m:
            continue
        thr = float(m.group(2).replace("m", "-").replace("p", "."))
        if abs(thr - threshold_db) > 1e-9:
            continue
        if date is not None and m.group(1) != date:
            continue
        out.append((m.group(1), thr, os.path.join(detections_dir, name)))
    return out


def run_one(date: str, threshold_db: float, path: str, tracks_dir: str,
            cfg: KalmanTrackerConfig, eval_cfg: TrackEvalConfig,
            max_frames, overwrite: bool):
    token = f"{threshold_db:.1f}".replace("-", "m").replace(".", "p")
    out_path = os.path.join(tracks_dir, f"tracks_{date}_thr_{token}dB.csv")
    if os.path.exists(out_path) and not overwrite:
        print(f"[{date} thr={threshold_db:g}dB] tracks exist, skipping "
              f"(pass --overwrite to regenerate): {out_path}")
        return None

    det = pd.read_csv(path, usecols=DETECTION_USECOLS,
                      dtype={"trajectory_id": str}, low_memory=False)
    if max_frames is not None:
        keep_frames = np.sort(det["frame_id"].unique())[:max_frames]
        det = det[det["frame_id"].isin(keep_frames)]

    tracks = run_tracker(det, cfg)
    os.makedirs(tracks_dir, exist_ok=True)
    tracks.to_csv(out_path, index=False)

    result = evaluate_tracks(tracks, det, eval_cfg)
    m = result["metrics"]
    m.update({"date": date, "threshold_db": threshold_db,
              "tracks_file": os.path.abspath(out_path)})

    print(f"\n--- {date}  threshold {threshold_db:g} dB ---")
    print(f"frames tracked:        {m['frames']}")
    print(f"detections in:         {m['detections_in']}")
    print(f"confirmed tracks:      {m['tracks_confirmed']}")
    print(f"true tracks:           {m['true_tracks']}")
    print(f"false tracks:          {m['false_tracks']}")
    print(f"trajectory coverage:   {m['trajectory_coverage']:.3f} "
          f"({m['covered_trajectories']}/{m['trackable_trajectories']})")
    print(f"fragmentation:         {m['fragmentation']:.2f}" if np.isfinite(m["fragmentation"])
          else "fragmentation:         n/a")
    print(f"target det absorption: {m['target_det_absorption']:.3f}")
    print(f"clutter det absorption:{m['clutter_det_absorption']:.3f}")
    print(f"tracks written:        {out_path}")
    return m


def self_test() -> None:
    """Two crossing constant-velocity targets + clutter over 30 frames."""
    rng = np.random.default_rng(42)
    frames = 30
    dt = 10.0
    rows = []
    det_id = 0
    targets = [  # (traj_id, p0, v)
        ("traj_A", np.array([-15_000.0, 5_000.0, 1_000.0]), np.array([50.0, 0.0, 0.0])),
        ("traj_B", np.array([10_000.0, -12_000.0, 1_500.0]), np.array([-20.0, 40.0, 0.0])),
    ]
    for k in range(frames):
        t = 1_000.0 + dt * k
        for traj_id, p0, v in targets:
            p = p0 + v * (dt * k) + rng.normal(0, 30.0, 3)
            r = np.linalg.norm(p)
            az = np.arctan2(p[0], p[1]) % (2 * np.pi)
            el = np.arcsin(np.clip(p[2] / r, -1, 1))
            rows.append(dict(frame_id=k, timestamp=t, detection_id=det_id, is_target=1,
                             trajectory_id=traj_id, meas_range_m=r, meas_azimuth_rad=az,
                             meas_elevation_rad=el))
            det_id += 1
        for _ in range(3):   # spatially independent clutter
            r = rng.uniform(5_000, 60_000)
            az = rng.uniform(0, 2 * np.pi)
            el = rng.uniform(0.0, 0.3)
            rows.append(dict(frame_id=k, timestamp=t, detection_id=det_id, is_target=0,
                             trajectory_id=np.nan, meas_range_m=r, meas_azimuth_rad=az,
                             meas_elevation_rad=el))
            det_id += 1
    det = pd.DataFrame(rows)

    cfg = KalmanTrackerConfig()
    eval_cfg = TrackEvalConfig()
    tracks = run_tracker(det, cfg)
    result = evaluate_tracks(tracks, det, eval_cfg)
    m = result["metrics"]

    # sanity: the coordinate conversion round-trips
    x, y, z = spherical_to_cartesian(det["meas_range_m"], det["meas_azimuth_rad"],
                                     det["meas_elevation_rad"])
    assert np.allclose(np.sqrt(x**2 + y**2 + z**2), det["meas_range_m"], rtol=1e-9)

    assert m["true_tracks"] >= 2, f"expected both targets tracked, got {m['true_tracks']} true tracks"
    assert m["trajectory_coverage"] == 1.0, f"coverage {m['trajectory_coverage']} != 1.0"
    assert m["false_tracks"] <= 2, f"too many clutter-born confirmed tracks: {m['false_tracks']}"
    per_track = result["per_track"]
    true_purity = per_track[per_track["is_true_track"]]["purity"]
    assert (true_purity >= 0.9).all(), f"true-track purity too low: {true_purity.tolist()}"

    with tempfile.TemporaryDirectory() as tmp:
        m.update({"date": "2022-01-01", "threshold_db": 0.0})
        path = write_report(tmp, [m], cfg, eval_cfg)
        text = open(path).read()
        for needle in ["Kalman", "nearest-neighbor", "never sees truth", "TRUE track"]:
            assert needle in text, f"report missing {needle!r}"

    print(f"true tracks: {m['true_tracks']}, false tracks: {m['false_tracks']}, "
          f"coverage: {m['trajectory_coverage']:.2f}, "
          f"clutter absorption: {m['clutter_det_absorption']:.3f}")
    print("\nStage 08 self-test passed.")


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = KalmanTrackerConfig(
        frame_period_s=args.frame_period_s, gate_m=args.gate_m,
        q_accel_mps2=args.q_accel_mps2,
        confirm_hits=args.confirm_hits, max_misses=args.max_misses,
    )
    eval_cfg = TrackEvalConfig()

    found = discover(args.detections_dir, args.threshold_db, args.date)
    if not found:
        print(f"No detection files for threshold {args.threshold_db:g} dB"
              f"{f' and date {args.date}' if args.date else ''} in {args.detections_dir}")
        return

    summary_rows = []
    for date, thr, path in found:
        m = run_one(date, thr, path, args.tracks_dir, cfg, eval_cfg,
                    args.max_frames, args.overwrite)
        if m is not None:
            summary_rows.append(m)

    if summary_rows:
        os.makedirs(args.report_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.report_dir, "kalman_baseline_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        report_path = write_report(args.report_dir, summary_rows, cfg, eval_cfg)
        print(f"\nSummary written to: {os.path.abspath(summary_path)}")
        print(f"Report written to:  {os.path.abspath(report_path)}")

    print("\n08_run_kalman_baseline completed successfully.")


if __name__ == "__main__":
    main()
