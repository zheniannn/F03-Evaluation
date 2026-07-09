"""Entry point: stage 15 step B -- evaluate the diffusion denoiser.

Three tasks over stage-8 track windows: (1) denoise true tracks (synthetic-
corruption recovery + smoothness regularization), (2) gap filling vs linear
interpolation, (3) a SECONDARY residual/anomaly score calibrated in the noisy-
track domain and compared against stage 12.5 / 13. Truth labels are used only
for evaluation and track-purity calibration, never for denoising itself.

Usage:
    python scripts/15_evaluate_diffusion_denoiser.py --threshold-db -5 0 3 6 \
        --date 2022-06-06 --denoise-t 20 --overwrite
    python scripts/15_evaluate_diffusion_denoiser.py --self-test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.sequence_windows import (
    WindowConfig, apply_normalizer, build_windows_by_group, discover_track_files,
    load_normalizer, load_tracks_for_windows,
)
from utils.sequence_prior_score import evaluate_track_labels
from utils import diffusion_denoise as dd


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the stage-15 diffusion denoiser.")
    p.add_argument("--tracks-dir", default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "diffusion_denoisers"))
    p.add_argument("--sequence-models-dir", default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--stage12-dir", default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    p.add_argument("--stage13-dir", default=os.path.join(REPO_ROOT, "reports", "stage13_vae_prior"))
    p.add_argument("--report-dir", default=os.path.join(REPO_ROOT, "reports", "stage15_diffusion_denoising"))
    p.add_argument("--detections-dir",
                   default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"),
                   help="Only used by the shared stage-8 loader (SNR column; unused here).")
    p.add_argument("--threshold-db", type=float, nargs="+", default=[-5, 0, 3, 6])
    p.add_argument("--date", nargs="*", default=["2022-06-06"])
    p.add_argument("--range-bins-m", default="0,50000,100000,200000,inf")
    p.add_argument("--denoise-t", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--sweep-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--min-track-hits", type=int, default=5)
    p.add_argument("--task-window-cap", type=int, default=20000,
                   help="Cap windows used for the denoising/gap-filling tasks (per run).")
    # track-purity calibration (stage 12.5 pattern)
    p.add_argument("--calibration-tracks-dir",
                   default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--calibration-date", nargs="*", default=None)
    p.add_argument("--calibration-threshold-db", type=float, nargs="+", default=[3, 6, 9, 12])
    p.add_argument("--calibration-min-target-fraction", type=float, default=0.95)
    p.add_argument("--calibration-min-purity", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return p.parse_args()


def _run_windows(date, thr, path, normalizer, wcfg, args, want_labels=True):
    """Return (normed_windows, gids, basics) for one stage-8 run."""
    df = load_tracks_for_windows(path, date, thr, args.detections_dir)
    confirmed = df.loc[df["confirmed"] == 1, "group_id"].unique()
    df = df[df["group_id"].isin(confirmed)]
    t_all = df["timestamp"].to_numpy(dtype=float)
    basics = {}
    for gid, pos in df.groupby("group_id", sort=False).indices.items():
        g = df.iloc[pos]
        n_hits = int(g["is_hit"].sum())
        if n_hits < args.min_track_hits:
            continue
        rng_m = np.sqrt(g["x"]**2 + g["y"]**2 + g["z"]**2)
        basics[gid] = {"n_hits": n_hits, "n_misses": len(g) - n_hits,
                       "duration_s": float(t_all[pos].max() - t_all[pos].min()),
                       "median_range_m": float(np.median(rng_m)), "max_range_m": float(rng_m.max()),
                       **(evaluate_track_labels(g) if want_labels else {})}
    df = df[df["group_id"].isin(basics)]
    windows, gids = build_windows_by_group(df, "group_id", wcfg, return_group_index=True)
    normed = apply_normalizer(windows, normalizer) if len(windows) else windows
    return normed, np.asarray(gids, dtype=object), basics


def build_residual_calibration(model, sched, normalizer, wcfg, args, device):
    cal_dates = args.calibration_date if args.calibration_date else args.date
    runs = discover_track_files(args.calibration_tracks_dir)
    runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in args.calibration_threshold_db)]
    if cal_dates:
        runs = [r for r in runs if r[0] in cal_dates]
    if not runs:
        raise SystemExit("No calibration track files found for the requested dates/thresholds")
    pooled, n_tracks = [], 0
    for date, thr, path in runs:
        print(f"[calib {date} thr={thr:g}dB] {os.path.basename(path)} ...", flush=True)
        normed, gids, basics = _run_windows(date, thr, path, normalizer, wcfg, args)
        if not len(normed):
            continue
        per = dd.residual_per_track(model, sched, normed, gids, args.denoise_t,
                                    args.batch_size, device)
        for gid, b in basics.items():
            if (b.get("is_true_track") and np.isfinite(b["target_fraction"])
                    and b["target_fraction"] >= args.calibration_min_target_fraction
                    and b["purity"] >= args.calibration_min_purity and gid in per):
                pooled.append(per[gid]["all"])
                n_tracks += 1
    if not pooled:
        raise SystemExit("No calibration windows collected")
    q = dd.calibration_quantiles(np.concatenate(pooled))
    q["n_calibration_tracks"] = n_tracks
    meta = {"calibration_dates": sorted({r[0] for r in runs}),
            "calibration_thresholds": sorted({r[1] for r in runs}),
            "min_target_fraction": args.calibration_min_target_fraction,
            "min_purity": args.calibration_min_purity}
    return q, meta


def run_evaluation(args) -> dict:
    from utils.diffusion_models import NoiseSchedule, load_diffusion, resolve_device

    key = os.path.join(args.report_dir, "diffusion_residual_scores.csv")
    if os.path.exists(key) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key}")

    device = resolve_device(args.device)
    with open(os.path.join(args.models_dir, "diffusion_config.json")) as f:
        cfg = json.load(f)
    wcfg = WindowConfig(window_len=cfg["window_len"], stride=cfg["stride"], features=cfg["features"])
    normalizer = load_normalizer(os.path.join(args.models_dir, "normalizer.json"))
    with open(os.path.join(args.models_dir, "diffusion_training_manifest.json")) as f:
        manifest = json.load(f)
    sched = NoiseSchedule(cfg["num_diffusion_steps"], cfg["beta_start"], cfg["beta_end"])
    ckpt = os.path.join(args.models_dir, "diffusion_denoiser.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Diffusion checkpoint missing: {ckpt} -- run training first")
    model = load_diffusion(ckpt, cfg, device)
    os.makedirs(args.report_dir, exist_ok=True)

    # calibration first (needs the model)
    calib, calib_meta = build_residual_calibration(model, sched, normalizer, wcfg, args, device)
    with open(os.path.join(args.models_dir, "diffusion_calibration.json"), "w") as f:
        json.dump({"calibration_mode": "track_purity",
                   "error_p50": calib["error_p50"], "error_p99": calib["error_p99"],
                   "n_calibration_tracks": calib["n_calibration_tracks"],
                   "n_calibration_windows": calib["n_calibration_windows"], **calib_meta}, f, indent=2)

    runs = discover_track_files(args.tracks_dir)
    runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in args.threshold_db)]
    if args.date:
        runs = [r for r in runs if r[0] in args.date]
    if not runs:
        raise SystemExit(f"No stage-8 track files found in {args.tracks_dir}")

    task1_rows, task2_rows, score_rows = [], [], []
    example = None
    for date, thr, path in runs:
        print(f"[{date} thr={thr:g}dB] evaluating {os.path.basename(path)} ...", flush=True)
        normed, gids, basics = _run_windows(date, thr, path, normalizer, wcfg, args)
        if not len(normed):
            continue

        # Task 3: residual score for ALL tracks
        per = dd.residual_per_track(model, sched, normed, gids, args.denoise_t,
                                    args.batch_size, device)
        for gid, b in basics.items():
            r = per.get(gid)
            if r is None:
                continue
            score = dd.score_from_band(r["median"], calib["error_p50"], calib["error_p99"])
            score_rows.append({"date": date, "threshold_db": thr, "track_id": gid,
                               "n_windows": r["n"], "diffusion_residual_median": r["median"],
                               "diffusion_residual_p90": r["p90"], "diffusion_score": score,
                               "keep_diffusion_score": bool(np.isfinite(score)
                                                            and score >= args.score_threshold),
                               "is_true_track": b["is_true_track"],
                               "target_fraction": b["target_fraction"], "purity": b["purity"],
                               "position_rmse_m": b["position_rmse_m"],
                               "median_range_m": b["median_range_m"]})

        # Tasks 1 & 2: TRUE tracks only
        true_gids = [g for g, b in basics.items() if b["is_true_track"]]
        true_mask = np.isin(gids, np.array(true_gids, dtype=object))
        Xtrue = normed[true_mask]
        Xtrue = dd.sample_rows(Xtrue, args.task_window_cap, seed=int(abs(thr)) + 7)
        if len(Xtrue):
            t1 = dd.denoising_metrics_for_run(model, sched, Xtrue, args.denoise_t,
                                              args.batch_size, device, seed=13)
            task1_rows.append({"date": date, "threshold_db": thr, "n_tracks": len(true_gids), **t1})
            for gr in dd.gap_filling_for_run(model, sched, Xtrue, args.denoise_t,
                                             args.batch_size, device, seed=17):
                task2_rows.append({"date": date, "threshold_db": thr, **gr})
            if example is None:
                from utils.diffusion_models import denoise_windows
                x0 = denoise_windows(model, Xtrue[:1], args.denoise_t, sched, args.batch_size, device)
                example = {"input": Xtrue[0], "denoised": x0[0]}

    scores = pd.DataFrame(score_rows)
    task1 = pd.DataFrame(task1_rows)
    task2 = pd.DataFrame(task2_rows)
    scores.to_csv(key, index=False, float_format="%.6g")
    task1.to_csv(os.path.join(args.report_dir, "diffusion_track_denoising_metrics.csv"), index=False)
    task2.to_csv(os.path.join(args.report_dir, "diffusion_gap_filling_metrics.csv"), index=False)

    by_thr = dd.aggregate_residual_by_threshold(scores, args.score_threshold)
    sweep = dd.residual_sweep(scores, [float(x) for x in args.sweep_thresholds.split(",")])
    by_thr.to_csv(os.path.join(args.report_dir, "diffusion_residual_by_threshold.csv"), index=False)
    sweep.to_csv(os.path.join(args.report_dir, "diffusion_filter_sweep.csv"), index=False)

    comparison = dd.build_comparison(by_thr, task1, task2, args.stage12_dir, args.stage13_dir)
    comparison.to_csv(os.path.join(args.report_dir, "stage12_vs_stage13_vs_stage15.csv"), index=False)

    val_denoise = pd.read_csv(manifest["validation_denoising"]) \
        if os.path.exists(manifest.get("validation_denoising", "")) else pd.DataFrame()
    training_summary = pd.read_csv(os.path.join(args.report_dir, "diffusion_training_summary.csv")) \
        if os.path.exists(os.path.join(args.report_dir, "diffusion_training_summary.csv")) \
        else pd.DataFrame()
    runtime = pd.DataFrame({"component": ["training", "evaluation"],
                            "seconds": [np.nan, np.nan],
                            "notes": ["see manifest / run log", "single-step Mode A inference"]})

    if not args.no_plots:
        dd.make_plots(training_summary, val_denoise, task1, task2, scores, comparison,
                      example, runtime, os.path.join(args.report_dir, "plots"))

    report = dd.write_report(args.report_dir, ",".join(args.date), manifest, val_denoise, task1,
                             task2, scores, by_thr, comparison, calib, calib_meta, runtime,
                             args.denoise_t, args.score_threshold)
    dd.run_eval_gate(args.report_dir, task1, task2, scores, by_thr, calib)

    print(f"\nreport: {os.path.abspath(report)}")
    print(f"\nmean denoise RMSE improvement: {task1['rmse_improvement_ratio'].mean():.4f}")
    print(f"mean gap-fill improvement:     {task2['improvement_ratio'].mean():.4f}")
    print(f"mean smoothness improvement:   {task1['smoothness_improvement_ratio'].mean():.4f}")
    print("residual filter (score 0.5) by threshold:")
    print(by_thr[["threshold_db", "true_track_retention", "false_track_reduction",
                  "precision_after"]].to_string(index=False))
    return {"scores": scores, "task1": task1, "task2": task2, "by_thr": by_thr,
            "comparison": comparison, "calib": calib}


# =============================================================================
# Self-test
# =============================================================================

def _load_train15():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train15", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "15_train_diffusion_denoiser.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _track_rows(track_id, date, thr, vfun, is_target, traj, n=40, dt=10.0,
                p0=(15_000.0, 12_000.0, 1_000.0)):
    rows, p = [], np.array(p0, dtype=float)
    for k in range(n):
        v = np.array(vfun(k), dtype=float)
        p = p + v * dt
        rows.append(dict(date=date, threshold_db=thr, frame_id=k, timestamp=1_000.0 + dt * k,
                         track_id=track_id, is_confirmed=1, event_type="hit",
                         assigned_detection_id=k, state_x_m=p[0], state_y_m=p[1], state_z_m=p[2],
                         state_vx_mps=v[0], state_vy_mps=v[1], state_vz_mps=v[2],
                         is_target=is_target, trajectory_id=traj if is_target else None, snr_db=8.0))
    return rows


def _write_tracks(prefix, rows, ids):
    pd.DataFrame(rows).to_csv(f"{prefix}.csv", index=False)
    pd.DataFrame({"track_id": ids}).to_csv(
        f"{prefix.replace('track_points_', 'track_summary_')}.csv", index=False)


def self_test() -> None:
    train15 = _load_train15()
    rng = np.random.default_rng(15)
    with tempfile.TemporaryDirectory() as tmp:
        truth_dir = os.path.join(tmp, "truth")
        os.makedirs(truth_dir)
        for i, date in enumerate(["2022-01-01", "2022-01-02"]):
            train15.make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"),
                                         date, seed=i)
        saved = sys.argv
        sys.argv = [saved[0]]
        try:
            targs = train15.parse_args()
        finally:
            sys.argv = saved
        targs.truth_dir = truth_dir
        targs.sequence_models_dir = os.path.join(tmp, "seq_missing")
        targs.models_dir = os.path.join(tmp, "models")
        targs.report_dir = os.path.join(tmp, "reports")
        targs.holdout_date = ["2022-01-02"]
        targs.window_len, targs.stride = 10, 2
        targs.epochs, targs.batch_size = 20, 64
        targs.hidden_dim, targs.num_blocks = 64, 3
        targs.num_diffusion_steps = 50
        targs.overwrite = True
        train15.run_training(targs)

        tracks_dir = os.path.join(tmp, "tracks")
        os.makedirs(tracks_dir)

        def noisy_true(tid, date, thr, base, traj, sigma=5.0):
            def vfun(k):
                ang = np.radians(20 + 0.4 * k)
                j = rng.normal(0, sigma, size=3)
                return (base * np.sin(ang) + j[0], base * np.cos(ang) + j[1], 0.4 + 0.3 * j[2])
            return _track_rows(tid, date, thr, vfun, 1, traj)

        cal_rows = []
        for tid in range(10, 16):
            cal_rows += noisy_true(tid, "2022-01-04", 6.0, 50.0 + tid, f"c{tid}")
        _write_tracks(os.path.join(tracks_dir, "track_points_2022-01-04_thr_6p0dB"),
                      cal_rows, list(range(10, 16)))

        eval_rows = []
        eval_rows += noisy_true(0, "2022-01-03", 0.0, 52.0, "tA")
        eval_rows += noisy_true(1, "2022-01-03", 0.0, 54.0, "tB")
        eval_rows += _track_rows(2, "2022-01-03", 0.0, lambda k: (150.0, -60.0, 15.0) if k % 2 == 0
                                 else (-120.0, 90.0, -15.0), 0, None)
        _write_tracks(os.path.join(tracks_dir, "track_points_2022-01-03_thr_0p0dB"),
                      eval_rows, [0, 1, 2])

        eargs = parse_args()
        eargs.tracks_dir = tracks_dir
        eargs.calibration_tracks_dir = tracks_dir
        eargs.models_dir = targs.models_dir
        eargs.stage12_dir = os.path.join(tmp, "no12")
        eargs.stage13_dir = os.path.join(tmp, "no13")
        eargs.report_dir = os.path.join(tmp, "eval_reports")
        eargs.detections_dir = tmp
        eargs.threshold_db = [0.0]
        eargs.date = ["2022-01-03"]
        eargs.calibration_date = ["2022-01-04"]
        eargs.calibration_threshold_db = [6.0]
        eargs.denoise_t = 20
        eargs.overwrite = True
        out = run_evaluation(eargs)

        o = eargs.report_dir
        for f_ in ["diffusion_track_denoising_metrics.csv", "diffusion_gap_filling_metrics.csv",
                   "diffusion_residual_scores.csv", "diffusion_denoising_report.md"]:
            assert os.path.exists(os.path.join(o, f_)), f"missing {f_}"
        assert np.isfinite(out["task1"][["kalman_position_rmse_norm",
                                         "diffusion_position_rmse_norm"]].to_numpy()).all()
        assert np.isfinite(out["task2"][["interp_mse", "diffusion_mse"]].to_numpy()).all()
        sc = out["scores"]
        fin = sc[np.isfinite(sc["diffusion_score"])]
        assert fin["diffusion_score"].between(0, 1).all(), "scores must be in [0, 1]"

        text = open(os.path.join(o, "diffusion_denoising_report.md")).read()
        for needle in ["Stage 15 Diffusion Trajectory Denoising Study",
                       "not a new\n  primary false-track classifier", "Stage 16"]:
            key = needle.replace("\n  ", " ")
            assert key in text.replace("\n  ", " "), f"report missing: {key!r}"

        print("\nresidual median score true vs false:")
        print(fin.groupby("is_true_track")["diffusion_score"].median().to_string())

    print("\nStage 15 diffusion evaluation self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_evaluation(args)
    print("\n15_evaluate_diffusion_denoiser completed successfully.")


if __name__ == "__main__":
    main()
