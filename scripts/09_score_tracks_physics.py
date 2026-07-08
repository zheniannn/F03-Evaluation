"""Entry point: stage 9 physics-guided track scoring and false-track
suppression.

Scores every stage-8 confirmed Kalman track with a transparent, rule-based
physics plausibility score (nothing trained, tracker untouched), then
evaluates -- using truth labels only AFTER scoring -- how much of the
false-track population a score threshold removes while retaining true
tracks.

Usage:
    python scripts/09_score_tracks_physics.py --threshold-db -5 0 3 6 \
        --date 2022-06-06 --score-threshold 0.5 --overwrite
    python scripts/09_score_tracks_physics.py --self-test
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.track_physics_score import (
    SCORE_COLUMNS,
    PhysicsScoreConfig,
    _aggregate,
    discover_stage08_runs,
    make_plots,
    run_validation_gate,
    score_run,
    sweep_table,
    write_report,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physics-guided scoring of stage-8 Kalman tracks (post-tracking, rule-based).")
    parser.add_argument("--tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"),
                        help="Stage-6 detections, used only to recover per-detection snr_db "
                             "for baseline stage-8 outputs that lack it.")
    parser.add_argument("--scored-tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_scored_physics"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    parser.add_argument("--threshold-db", type=float, nargs="+", default=None,
                        help="Detection thresholds to process (default: all available).")
    parser.add_argument("--date", type=str, nargs="*", default=None,
                        help="Dates to process (default: all available).")
    parser.add_argument("--range-bins-m", type=str, default="0,50000,100000,200000,inf")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", type=str,
                        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--w-accel", type=float, default=1.0)
    parser.add_argument("--w-turn", type=float, default=1.0)
    parser.add_argument("--w-vertical", type=float, default=1.0)
    parser.add_argument("--w-continuity", type=float, default=1.0)
    parser.add_argument("--w-snr", type=float, default=0.5)
    parser.add_argument("--w-clutter", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def run_all(cfg: PhysicsScoreConfig) -> dict:
    key_output = os.path.join(cfg.report_dir, "physics_track_scores.csv")
    if os.path.exists(key_output) and not cfg.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")

    runs = discover_stage08_runs(cfg.tracks_dir)
    if cfg.thresholds_db is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in cfg.thresholds_db)]
    if cfg.dates:
        runs = [r for r in runs if r[0] in cfg.dates]
    if not runs:
        raise SystemExit(f"No stage-8 track files found in {cfg.tracks_dir} for the requested "
                         f"dates/thresholds")

    os.makedirs(cfg.report_dir, exist_ok=True)
    os.makedirs(cfg.scored_tracks_dir, exist_ok=True)

    all_scores = []
    channels = {"snr": True, "association": True}
    for date, thr, path in runs:
        print(f"[{date} thr={thr:g}dB] scoring {os.path.basename(path)} ...", flush=True)
        scores = score_run(date, thr, path, cfg)
        channels["snr"] &= scores.attrs.get("snr_available", False)
        channels["association"] &= scores.attrs.get("association_available", False)
        token = f"{thr:.1f}".replace("-", "m").replace(".", "p")
        scores.to_csv(os.path.join(cfg.scored_tracks_dir,
                                   f"scored_tracks_{date}_thr_{token}dB.csv"), index=False)
        n_kept = int(scores["keep_physics"].sum())
        print(f"[{date} thr={thr:g}dB] confirmed tracks scored: {len(scores)}, "
              f"kept at {cfg.score_threshold:g}: {n_kept}")
        all_scores.append(scores)

    scores = pd.concat(all_scores, ignore_index=True)
    scores_out = scores[[c for c in SCORE_COLUMNS if c in scores.columns]]
    scores_out.to_csv(key_output, index=False, float_format="%.5g")

    by_thr = _aggregate(scores, ["threshold_db"], cfg.score_threshold)
    by_day = _aggregate(scores, ["date", "threshold_db"], cfg.score_threshold)
    sweep = sweep_table(scores, cfg.sweep_thresholds)

    comparison = by_thr[[
        "threshold_db", "stage08_confirmed_tracks", "stage08_true_tracks",
        "stage08_false_tracks", "score_threshold", "stage09_kept_tracks",
        "stage09_kept_true_tracks", "stage09_kept_false_tracks",
        "true_track_retention", "false_track_reduction",
        "precision_before", "precision_after",
    ]].rename(columns={
        "score_threshold": "stage09_score_threshold",
        "true_track_retention": "stage09_true_track_retention",
        "false_track_reduction": "stage09_false_track_reduction",
        "precision_before": "stage08_precision",
        "precision_after": "stage09_precision",
    })

    by_thr.drop(columns=["score_threshold"]).to_csv(
        os.path.join(cfg.report_dir, "physics_metrics_by_threshold.csv"), index=False)
    by_day.drop(columns=["score_threshold"]).to_csv(
        os.path.join(cfg.report_dir, "physics_metrics_by_day.csv"), index=False)
    sweep.to_csv(os.path.join(cfg.report_dir, "physics_filter_sweep.csv"), index=False)
    comparison.to_csv(os.path.join(cfg.report_dir, "stage08_vs_stage09.csv"), index=False)

    if not cfg.no_plots:
        make_plots(scores, by_thr, sweep, os.path.join(cfg.report_dir, "plots"))

    channels_note = ("All penalty channels were available." if all(channels.values()) else
                     "Channels unavailable in these stage-8 outputs and excluded from the "
                     "weighting: "
                     + ", ".join(n for n, ok in [("association (Mahalanobis d2)", channels["association"]),
                                                 ("SNR", channels["snr"])] if not ok) + ".")
    report_path = write_report(cfg.report_dir, scores, by_thr, comparison, sweep, cfg, channels_note)
    run_validation_gate(cfg.report_dir, scores, by_thr)

    print(f"\nreport: {os.path.abspath(report_path)}")
    print("\nstage 8 vs stage 9 (per detection threshold):")
    print(comparison.to_string(index=False))
    return {"scores": scores, "by_thr": by_thr, "sweep": sweep, "comparison": comparison}


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """Four synthetic stage-8-like tracks: two smooth true, one wildly
    implausible false, one smooth-but-weak (low SNR, poor association,
    gappy) false."""
    date, dt, n = "2022-01-01", 10.0, 40
    t0 = 1_000.0

    def make_track(track_id, p0, vfun, is_target, traj, snr, mahal, miss_every=None):
        rows = []
        p = np.array(p0, dtype=float)
        for k in range(n):
            v = np.array(vfun(k), dtype=float)
            p = p + v * dt
            miss = miss_every is not None and k % miss_every == 1
            rows.append(dict(
                date=date, threshold_db=0.0, frame_id=k, timestamp=t0 + dt * k,
                track_id=track_id, track_status="confirmed", is_confirmed=1,
                event_type="miss" if miss else "hit",
                assigned_detection_id=-1 if miss else k,
                state_x_m=p[0], state_y_m=p[1], state_z_m=p[2],
                state_vx_mps=v[0], state_vy_mps=v[1], state_vz_mps=v[2],
                meas_x_m=p[0], meas_y_m=p[1], meas_z_m=p[2],
                meas_range_m=np.linalg.norm(p), meas_azimuth_rad=0.5,
                meas_elevation_rad=0.05, meas_radial_velocity_mps=0.0,
                innovation_norm_m=50.0, mahalanobis_d2=np.nan if miss else mahal,
                is_target=np.nan if miss else is_target,
                trajectory_id=None if (miss or not is_target) else traj,
                truth_x_m=p[0], truth_y_m=p[1], truth_z_m=p[2],
                truth_range_m=np.linalg.norm(p), snr_db=np.nan if miss else snr, pd=0.9,
            ))
        return rows

    rows = []
    # T1: straight and level at 60 m/s
    rows += make_track(0, (10_000, 20_000, 1_200), lambda k: (60.0, 0.0, 0.0), 1, "tA", 12.0, 2.0)
    # T2: gentle 1 deg/s turn at ~55 m/s with mild climb
    rows += make_track(1, (-15_000, 8_000, 900),
                       lambda k: (55 * np.sin(np.radians(10 + k)), 55 * np.cos(np.radians(10 + k)), 2.0),
                       1, "tB", 10.0, 2.5)
    # F1: wildly implausible -- velocity flips every frame
    rows += make_track(2, (5_000, -5_000, 800),
                       lambda k: (200.0, 0.0, 30.0) if k % 2 == 0 else (-200.0, 50.0, -30.0),
                       0, None, 6.0, 4.0)
    # F2: smooth like T1 but weak: low SNR, bad association, gappy (miss every 2nd)
    rows += make_track(3, (-8_000, -12_000, 1_000), lambda k: (58.0, 5.0, 0.0),
                       0, None, -15.0, 40.0, miss_every=2)

    points = pd.DataFrame(rows)
    summary = points.groupby("track_id").agg(
        n_hits=("event_type", lambda s: int((s == "hit").sum())),
        n_misses=("event_type", lambda s: int((s == "miss").sum()))).reset_index()
    summary["date"], summary["threshold_db"], summary["confirmed"] = date, 0.0, 1

    with tempfile.TemporaryDirectory() as tmp:
        tracks_dir = os.path.join(tmp, "tracks")
        report_dir = os.path.join(tmp, "reports")
        scored_dir = os.path.join(tmp, "scored")
        os.makedirs(tracks_dir)
        points.to_csv(os.path.join(tracks_dir, f"track_points_{date}_thr_0p0dB.csv"), index=False)
        summary.to_csv(os.path.join(tracks_dir, f"track_summary_{date}_thr_0p0dB.csv"), index=False)

        cfg = PhysicsScoreConfig(tracks_dir=tracks_dir, detections_dir=tmp,
                                 scored_tracks_dir=scored_dir, report_dir=report_dir,
                                 score_threshold=0.5, overwrite=True)
        out = run_all(cfg)
        scores = out["scores"].set_index("track_id")

        assert os.path.exists(os.path.join(report_dir, "physics_track_scores.csv"))
        assert len(scores) == 4, f"expected 4 scored tracks, got {len(scores)}"
        assert scores["physics_score"].between(0, 1).all()
        true_mean = scores.loc[[0, 1], "physics_score"].mean()
        assert true_mean > scores.loc[2, "physics_score"], \
            f"smooth true tracks ({true_mean:.3f}) should outscore the wild false track " \
            f"({scores.loc[2, 'physics_score']:.3f})"
        kept = scores["keep_physics"]
        assert not kept.loc[2], "the kinematically wild false track must be rejected at 0.5"
        assert kept.loc[0] and kept.loc[1], "true tracks must be retained at 0.5"
        assert scores.loc[[0, 1], "is_true_track"].all() and not scores.loc[[2, 3], "is_true_track"].any()

        text = open(os.path.join(report_dir, "physics_scoring_report.md")).read()
        for needle in ["physics plausibility", "Truth labels are used only after scoring",
                       "not ML", "Stage 10"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print(f"\nscores: T1={scores.loc[0, 'physics_score']:.3f} "
              f"T2={scores.loc[1, 'physics_score']:.3f} "
              f"F1(wild)={scores.loc[2, 'physics_score']:.3f} "
              f"F2(weak)={scores.loc[3, 'physics_score']:.3f}")

    print("\nStage 09 self-test passed.")


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = PhysicsScoreConfig(
        tracks_dir=args.tracks_dir,
        detections_dir=args.detections_dir,
        scored_tracks_dir=args.scored_tracks_dir,
        report_dir=args.report_dir,
        thresholds_db=args.threshold_db,
        dates=args.date if args.date else None,
        range_bins_m=[float(p) for p in args.range_bins_m.split(",")],
        score_threshold=args.score_threshold,
        sweep_thresholds=[float(p) for p in args.sweep_thresholds.split(",")],
        min_hits=args.min_hits,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        no_plots=args.no_plots,
        w_speed=args.w_speed, w_accel=args.w_accel, w_turn=args.w_turn,
        w_vertical=args.w_vertical, w_continuity=args.w_continuity,
        w_snr=args.w_snr, w_clutter=args.w_clutter,
    )
    run_all(cfg)
    print("\n09_score_tracks_physics completed successfully.")


if __name__ == "__main__":
    main()
