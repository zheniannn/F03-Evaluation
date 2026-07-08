"""Entry point: stage 11 -- empirical ADS-B-prior track scoring.

Scores stage-8 confirmed Kalman tracks with the stage-10 empirical motion
priors (quantile-exceedance penalties + joint prior; truth labels used only
afterwards, for evaluation) and compares against stage-9 hand-designed
physics scoring. No VAE/diffusion/ML; stages 8-10 are not modified.

Usage:
    python scripts/11_score_tracks_adsb_prior.py --threshold-db -5 0 3 6 9 12 \
        --date 2022-06-06 --score-threshold 0.5 --overwrite
    python scripts/11_score_tracks_adsb_prior.py --self-test
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.adsb_prior_score import (
    AdsbPriorScoreConfig,
    aggregate_by_threshold,
    build_comparison,
    discover_stage08_runs,
    load_priors,
    make_plots,
    prepare_joint,
    range_bin_table,
    run_validation_gate,
    score_run,
    sweep_table,
    write_report,
)

SCORE_CSV_COLUMNS = [
    "date", "threshold_db", "track_id", "n_hits", "n_misses", "duration_s", "hit_rate",
    "adsb_prior_score", "keep_adsb_prior",
    "speed_prior_score", "accel_prior_score", "vector_accel_prior_score",
    "turn_prior_score", "vertical_prior_score", "joint_prior_score",
    "continuity_score", "snr_score",
    "speed_prior_penalty", "accel_prior_penalty", "vector_accel_prior_penalty",
    "turn_prior_penalty", "vertical_prior_penalty", "joint_prior_penalty",
    "speed_mps_p95", "accel_abs_mps2_p95", "accel_vector_mps2_p95",
    "turn_rate_abs_deg_s_p95", "vertical_speed_abs_mps_p95",
    "speed_mean_logpdf", "accel_mean_logpdf", "vector_accel_mean_logpdf",
    "turn_mean_logpdf", "vertical_mean_logpdf",
    "median_range_m", "max_range_m",
    "n_target_hits", "n_clutter_hits", "target_fraction", "purity",
    "majority_trajectory_id", "is_true_track", "position_rmse_m",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score stage-8 tracks with stage-10 empirical ADS-B priors.")
    parser.add_argument("--tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    parser.add_argument("--priors-dir", type=str,
                        default=os.path.join(REPO_ROOT, "models", "motion_priors"))
    parser.add_argument("--stage09-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage11_adsb_prior_scoring"))
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"),
                        help="Stage-6 detections, used only to recover per-detection snr_db.")
    parser.add_argument("--threshold-db", type=float, nargs="+", default=None)
    parser.add_argument("--date", type=str, nargs="*", default=None)
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
    parser.add_argument("--w-vector-accel", type=float, default=1.0)
    parser.add_argument("--w-turn", type=float, default=1.0)
    parser.add_argument("--w-vertical", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.5)
    parser.add_argument("--w-continuity", type=float, default=0.5)
    parser.add_argument("--w-snr", type=float, default=0.25)
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def run_all(cfg: AdsbPriorScoreConfig) -> dict:
    key_output = os.path.join(cfg.report_dir, "adsb_prior_track_scores.csv")
    if os.path.exists(key_output) and not cfg.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")

    priors, joint = load_priors(cfg.priors_dir)
    if not priors:
        raise SystemExit(f"No stage-10 prior JSONs found in {cfg.priors_dir} -- run stage 10 first")
    joint_prep = prepare_joint(joint)

    runs = discover_stage08_runs(cfg.tracks_dir)
    if cfg.thresholds_db is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in cfg.thresholds_db)]
    if cfg.dates:
        runs = [r for r in runs if r[0] in cfg.dates]
    if not runs:
        raise SystemExit(f"No stage-8 track files found in {cfg.tracks_dir}")

    os.makedirs(cfg.report_dir, exist_ok=True)
    all_scores = []
    snr_available = True
    for date, thr, path in runs:
        print(f"[{date} thr={thr:g}dB] scoring {os.path.basename(path)} ...", flush=True)
        scores = score_run(date, thr, path, cfg, priors, joint_prep)
        snr_available &= scores.attrs.get("snr_available", False)
        print(f"[{date} thr={thr:g}dB] tracks scored: {len(scores)}, "
              f"kept at {cfg.score_threshold:g}: {int(scores['keep_adsb_prior'].sum())}")
        all_scores.append(scores)

    scores = pd.concat(all_scores, ignore_index=True)
    scores[[c for c in SCORE_CSV_COLUMNS if c in scores.columns]].to_csv(
        key_output, index=False, float_format="%.5g")

    by_thr = aggregate_by_threshold(scores, cfg.score_threshold)
    sweep = sweep_table(scores, cfg.sweep_thresholds)
    range_bins = range_bin_table(scores, cfg.range_bins_m)
    comparison, stage09_available = build_comparison(by_thr, cfg.stage09_dir)

    by_thr.drop(columns=["score_threshold"]).to_csv(
        os.path.join(cfg.report_dir, "adsb_prior_metrics_by_threshold.csv"), index=False)
    sweep.to_csv(os.path.join(cfg.report_dir, "adsb_prior_filter_sweep.csv"), index=False)
    range_bins.to_csv(os.path.join(cfg.report_dir, "adsb_prior_range_bin_metrics.csv"), index=False)
    comparison.to_csv(os.path.join(cfg.report_dir, "stage08_vs_stage09_vs_stage11.csv"), index=False)

    if not cfg.no_plots:
        make_plots(scores, sweep, comparison, stage09_available, cfg.range_bins_m,
                   os.path.join(cfg.report_dir, "plots"))
    report_path = write_report(cfg.report_dir, scores, by_thr, comparison, range_bins,
                               priors, joint_prep is not None, stage09_available,
                               snr_available, cfg)
    run_validation_gate(cfg.report_dir, scores, by_thr, priors, comparison, stage09_available)

    print(f"\nreport: {os.path.abspath(report_path)}")
    show = comparison[[c for c in ["threshold_db", "stage08_false_tracks",
                                   "stage09_kept_false_tracks", "stage11_kept_false_tracks",
                                   "stage09_true_track_retention", "stage11_true_track_retention",
                                   "stage09_precision", "stage11_precision"]
                       if c in comparison.columns]]
    print("\nstage 8 vs 9 vs 11 (per detection threshold):")
    print(show.to_string(index=False))
    return {"scores": scores, "by_thr": by_thr, "comparison": comparison,
            "stage09_available": stage09_available}


def self_test() -> None:
    """Synthetic priors + four synthetic tracks: two near the priors (true),
    one wildly implausible (false), one smooth but off-distribution (false)."""
    from utils.motion_priors import fit_empirical_prior, write_prior_json, build_joint_motion_prior

    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        priors_dir = os.path.join(tmp, "models")
        tracks_dir = os.path.join(tmp, "tracks")
        report_dir = os.path.join(tmp, "reports")
        os.makedirs(priors_dir)
        os.makedirs(tracks_dir)

        # --- synthetic stage-10 priors ---------------------------------
        prior_specs = {
            "speed_mps": ("speed_prior.json", "m/s", np.abs(rng.normal(50, 8, 200_000)), (5, 160)),
            "accel_abs_mps2": ("acceleration_prior.json", "m/s^2",
                               np.abs(rng.normal(0.2, 0.1, 200_000)), (0, 15)),
            "accel_vector_mps2": ("vector_acceleration_prior.json", "m/s^2",
                                  np.abs(rng.normal(0.4, 0.2, 200_000)), (0, 15)),
            "turn_rate_abs_deg_s": ("turn_rate_prior.json", "deg/s",
                                    np.abs(rng.normal(1.0, 0.5, 200_000)), (0, 30)),
            "vertical_speed_abs_mps": ("vertical_speed_prior.json", "m/s",
                                       np.abs(rng.normal(1.0, 0.5, 200_000)), (0, 40)),
        }
        samples = {}
        for feature, (fname, units, vals, (lo, hi)) in prior_specs.items():
            vals = vals[(vals >= lo) & (vals <= hi)]
            counts, edges = np.histogram(vals, bins=100, range=(lo, hi))
            prior = fit_empirical_prior(vals, counts, edges, feature, units,
                                        len(vals), len(vals), {"lo": lo, "hi": hi})
            write_prior_json(os.path.join(priors_dir, fname), prior)
            samples[feature] = vals[:100_000]
        n = min(len(v) for v in samples.values())
        joint_df = pd.DataFrame({k: v[:n] for k, v in samples.items()})
        cfg_dummy = None
        write_prior_json(os.path.join(priors_dir, "joint_motion_prior.json"),
                         build_joint_motion_prior(joint_df, cfg_dummy))
        write_prior_json(os.path.join(priors_dir, "motion_prior_manifest.json"),
                         {"files": sorted(os.listdir(priors_dir)), "created_by": "Stage 10"})

        # --- synthetic stage-8 tracks (spec schema) ---------------------
        date, dt, n = "2022-01-01", 10.0, 40

        def make_track(track_id, vfun, is_target, traj):
            rows = []
            p = np.array([10_000.0, 10_000.0, 1_000.0])
            for k in range(n):
                v = np.array(vfun(k), dtype=float)
                p = p + v * dt
                rows.append(dict(
                    date=date, threshold_db=0.0, frame_id=k, timestamp=1_000.0 + dt * k,
                    track_id=track_id, is_confirmed=1, event_type="hit",
                    assigned_detection_id=k,
                    state_x_m=p[0], state_y_m=p[1], state_z_m=p[2],
                    state_vx_mps=v[0], state_vy_mps=v[1], state_vz_mps=v[2],
                    is_target=is_target, trajectory_id=traj if is_target else None,
                    snr_db=8.0, mahalanobis_d2=2.0, innovation_norm_m=50.0,
                ))
            return rows

        rows = []
        # T0/T1: near the priors (speed ~50, gentle turn, small vz)
        rows += make_track(0, lambda k: (50.0 + 0.2 * np.sin(k / 5), 2.0, 1.0), 1, "tA")
        rows += make_track(1, lambda k: (48 * np.sin(np.radians(20 + k)),
                                         48 * np.cos(np.radians(20 + k)), -1.0), 1, "tB")
        # F2: wildly implausible (speed ~200, huge accel/turn)
        rows += make_track(2, lambda k: (200.0, 0.0, 20.0) if k % 2 == 0 else (-150.0, 80.0, -20.0),
                           0, None)
        # F3: smooth but off-distribution (speed 120 -- far tail of the 50+-8 prior)
        rows += make_track(3, lambda k: (120.0, 5.0, 0.5), 0, None)

        pd.DataFrame(rows).to_csv(
            os.path.join(tracks_dir, f"track_points_{date}_thr_0p0dB.csv"), index=False)
        pd.DataFrame({"track_id": [0, 1, 2, 3]}).to_csv(
            os.path.join(tracks_dir, f"track_summary_{date}_thr_0p0dB.csv"), index=False)

        cfg = AdsbPriorScoreConfig(tracks_dir=tracks_dir, priors_dir=priors_dir,
                                   stage09_dir=os.path.join(tmp, "nonexistent"),
                                   report_dir=report_dir, detections_dir=tmp,
                                   score_threshold=0.5, overwrite=True)
        out = run_all(cfg)
        scores = out["scores"].set_index("track_id")

        assert os.path.exists(os.path.join(report_dir, "adsb_prior_track_scores.csv"))
        assert len(scores) == 4, f"expected 4 scored tracks, got {len(scores)}"
        assert scores["adsb_prior_score"].between(0, 1).all()
        true_mean = scores.loc[[0, 1], "adsb_prior_score"].mean()
        assert true_mean > scores.loc[2, "adsb_prior_score"], \
            "true tracks should out-score the wild false track"
        kept = scores["keep_adsb_prior"]
        assert not kept.loc[2] or not kept.loc[3], "at least one false track must be rejected at 0.5"
        assert kept.loc[0] and kept.loc[1], "true tracks must be retained at 0.5"

        text = open(os.path.join(report_dir, "adsb_prior_scoring_report.md")).read()
        for needle in ["Stage 11 Empirical ADS-B-Prior Track Scoring",
                       "Truth labels are used only after scoring", "not VAE", "Stage 12"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print(f"\nscores: T0={scores.loc[0, 'adsb_prior_score']:.3f} "
              f"T1={scores.loc[1, 'adsb_prior_score']:.3f} "
              f"F2(wild)={scores.loc[2, 'adsb_prior_score']:.3f} "
              f"F3(off-dist)={scores.loc[3, 'adsb_prior_score']:.3f}")

    print("\nStage 11 self-test passed.")


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = AdsbPriorScoreConfig(
        tracks_dir=args.tracks_dir, priors_dir=args.priors_dir,
        stage09_dir=args.stage09_dir, report_dir=args.report_dir,
        detections_dir=args.detections_dir,
        thresholds_db=args.threshold_db, dates=args.date if args.date else None,
        range_bins_m=[float(p) for p in args.range_bins_m.split(",")],
        score_threshold=args.score_threshold,
        sweep_thresholds=[float(p) for p in args.sweep_thresholds.split(",")],
        min_hits=args.min_hits, chunksize=args.chunksize,
        overwrite=args.overwrite, no_plots=args.no_plots,
        w_speed=args.w_speed, w_accel=args.w_accel, w_vector_accel=args.w_vector_accel,
        w_turn=args.w_turn, w_vertical=args.w_vertical, w_joint=args.w_joint,
        w_continuity=args.w_continuity, w_snr=args.w_snr,
    )
    run_all(cfg)
    print("\n11_score_tracks_adsb_prior completed successfully.")


if __name__ == "__main__":
    main()
