"""Entry point: stage 12 step B -- score stage-8 Kalman tracks by
autoencoder reconstruction plausibility.

Builds the same normalized trajectory windows from each confirmed track's
posterior states, computes per-window reconstruction errors under the
trained stage-12 models, calibrates against clean holdout-truth errors,
and compares against stage 9 (hand physics) and stage 11 (empirical
marginal priors). Truth labels are used only after scoring, for
evaluation. Not VAE (stage 13), not diffusion (stage 14).

Usage:
    python scripts/12_score_tracks_sequence_prior.py --threshold-db -5 0 3 6 9 12 \
        --date 2022-06-06 --score-threshold 0.5 --overwrite
    python scripts/12_score_tracks_sequence_prior.py --self-test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.sequence_windows import (
    WindowConfig,
    apply_normalizer,
    build_windows_by_group,
    discover_track_files,
    load_normalizer,
    load_tracks_for_windows,
)
from utils.sequence_prior_score import (
    aggregate_by_model_threshold,
    build_calibration_comparison,
    compare_with_stage09_stage11,
    evaluate_track_labels,
    make_calibration_plots,
    make_scoring_plots,
    range_bin_table,
    run_scoring_gate,
    score_track_errors,
    sweep_table,
    write_calibration_files,
    write_scoring_report,
)

SCORE_CSV_COLUMNS = [
    "date", "threshold_db", "model", "track_id",
    "n_hits", "n_misses", "duration_s", "median_range_m", "max_range_m", "n_windows",
    "sequence_prior_score", "keep_sequence_prior", "calibration_mode",
    "sequence_prior_score_clean_truth", "sequence_prior_score_track_calibrated",
    "sequence_recon_error_mean", "sequence_recon_error_median",
    "sequence_recon_error_p90", "sequence_recon_error_max",
    "calibration_error_p50", "calibration_error_p99",
    "n_target_hits", "n_clutter_hits", "target_fraction", "purity",
    "majority_trajectory_id", "is_true_track", "position_rmse_m",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score stage-8 tracks with trained stage-12 sequence priors.")
    parser.add_argument("--tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    parser.add_argument("--models-dir", type=str,
                        default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    parser.add_argument("--stage09-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    parser.add_argument("--stage11-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage11_adsb_prior_scoring"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    parser.add_argument("--detections-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"),
                        help="Only used by the shared stage-8 loader (SNR column; unused here).")
    parser.add_argument("--model", type=str, nargs="+", default=["mlp_dae", "gru_ae", "tcn_ae"])
    parser.add_argument("--threshold-db", type=float, nargs="+", default=None)
    parser.add_argument("--date", type=str, nargs="*", default=None)
    parser.add_argument("--range-bins-m", type=str, default="0,50000,100000,200000,inf")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", type=str,
                        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--min-track-hits", type=int, default=5)
    parser.add_argument("--window-len", type=int, default=None,
                        help="Default: read from window_config.json.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Default: read from window_config.json.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")

    # --- Stage 12.5: noise-matched (track-purity) calibration -------------
    parser.add_argument("--calibration-mode", choices=["clean_truth", "track_purity"],
                        default="clean_truth",
                        help="clean_truth (stage-12 default) or track_purity (stage 12.5).")
    parser.add_argument("--calibration-tracks-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"),
                        help="Stage-8 tracks used to build the track-purity calibration set.")
    parser.add_argument("--calibration-date", type=str, nargs="*", default=None,
                        help="Dates for calibration tracks (default: --date).")
    parser.add_argument("--calibration-threshold-db", type=float, nargs="+", default=None,
                        help="Thresholds for calibration tracks (default: --threshold-db).")
    parser.add_argument("--calibration-min-target-fraction", type=float, default=0.95)
    parser.add_argument("--calibration-min-purity", type=float, default=0.95)
    parser.add_argument("--calibration-max-false-tracks", type=int, default=0,
                        help="Sanity guard: calibration tracks should be true tracks only.")
    parser.add_argument("--calibration-output", type=str, default=None,
                        help="Where the track-purity calibration JSON is written. Defaults to "
                             "<--report-dir>/calibration/sequence_track_calibration.json. "
                             "ORCHESTRATORS MUST SET THIS to a path inside their own report "
                             "directory, so a per-day rerun can never overwrite the canonical "
                             "stage-12 calibration artifact.")
    parser.add_argument("--compare-calibration", action="store_true",
                        help="Score under both clean-truth and track-purity calibrations.")
    return parser.parse_args()


def _compute_run_errors(date, threshold_db, path, models, normalizer, wcfg, args, device):
    """Per-window reconstruction errors for one stage-8 run.

    Returns (basics, errors_by_model_gid) where basics maps track id -> per-track
    basic/evaluation fields and errors_by_model_gid maps model -> {track id -> per-
    window error array}. Truth labels sit in basics for later use; they never touch
    the reconstruction errors themselves.
    """
    from utils.sequence_models import reconstruction_errors

    df = load_tracks_for_windows(path, date, threshold_db, args.detections_dir)
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
        basics[gid] = {
            "n_hits": n_hits, "n_misses": len(g) - n_hits,
            "duration_s": float(t_all[pos].max() - t_all[pos].min()),
            "median_range_m": float(np.median(rng_m)), "max_range_m": float(rng_m.max()),
            **evaluate_track_labels(g),
        }
    df = df[df["group_id"].isin(basics)]

    windows, gids = build_windows_by_group(df, "group_id", wcfg, return_group_index=True)
    errors_by_model_gid = {m: {} for m in args.model}
    if len(windows):
        normed = apply_normalizer(windows, normalizer)
        gid_arr = np.asarray(gids, dtype=object)
        for model_name in args.model:
            errors = reconstruction_errors(models[model_name], normed, args.batch_size, device)
            for gid, errs in pd.Series(errors).groupby(gid_arr):
                errors_by_model_gid[model_name][gid] = errs.to_numpy()
    return basics, errors_by_model_gid


def score_one_run(date, threshold_db, path, models, cal_by_mode, primary_mode,
                  compare, normalizer, wcfg, args, device):
    """Score every confirmed track of one stage-8 run under every model + calibration."""
    basics, errors_by_model_gid = _compute_run_errors(
        date, threshold_db, path, models, normalizer, wcfg, args, device)

    rows = []
    for model_name in args.model:
        errors_by_gid = errors_by_model_gid[model_name]
        for gid, b in basics.items():
            errs = errors_by_gid.get(gid)
            if errs is None or not len(errs):
                stats = dict(mean=np.nan, median=np.nan, p90=np.nan, mx=np.nan, n=0)
                score_by_mode = {m: np.nan for m in cal_by_mode}
            else:
                stats = dict(mean=float(errs.mean()), median=float(np.median(errs)),
                             p90=float(np.percentile(errs, 90)), mx=float(errs.max()),
                             n=len(errs))
                score_by_mode = {m: score_track_errors(stats["median"], cal[model_name])
                                 for m, cal in cal_by_mode.items()}
            primary = score_by_mode[primary_mode]
            cal_primary = cal_by_mode[primary_mode][model_name]
            row = {
                "date": date, "threshold_db": threshold_db, "model": model_name,
                "track_id": gid, **{k: b[k] for k in
                                    ("n_hits", "n_misses", "duration_s",
                                     "median_range_m", "max_range_m")},
                "n_windows": stats["n"],
                "sequence_prior_score": primary,
                "keep_sequence_prior": bool(np.isfinite(primary)
                                            and primary >= args.score_threshold),
                "calibration_mode": primary_mode,
                "sequence_recon_error_mean": stats["mean"],
                "sequence_recon_error_median": stats["median"],
                "sequence_recon_error_p90": stats["p90"],
                "sequence_recon_error_max": stats["mx"],
                "calibration_error_p50": cal_primary["error_p50"],
                "calibration_error_p99": cal_primary["error_p99"],
                **{k: b[k] for k in ("n_target_hits", "n_clutter_hits", "target_fraction",
                                     "purity", "majority_trajectory_id", "is_true_track",
                                     "position_rmse_m")},
            }
            if compare:
                row["sequence_prior_score_clean_truth"] = score_by_mode.get(
                    "clean_truth", np.nan)
                row["sequence_prior_score_track_calibrated"] = score_by_mode.get(
                    "track_purity", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def build_track_purity_calibration(models, normalizer, wcfg, args, device):
    """Build noise-matched calibration quantiles from high-purity stage-8 true tracks."""
    cal_dates = args.calibration_date if args.calibration_date else args.date
    cal_thr = (args.calibration_threshold_db if args.calibration_threshold_db is not None
               else args.threshold_db)
    runs = discover_track_files(args.calibration_tracks_dir)
    if cal_thr is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in cal_thr)]
    if cal_dates:
        runs = [r for r in runs if r[0] in cal_dates]
    if not runs:
        raise SystemExit(
            f"No calibration track files found in {args.calibration_tracks_dir} "
            f"for dates={cal_dates} thresholds={cal_thr}")

    from utils.sequence_prior_score import quantiles_from_window_errors

    pooled = {m: [] for m in args.model}
    n_tracks = 0
    n_false = 0
    for date, thr, path in runs:
        print(f"[calib {date} thr={thr:g}dB] {os.path.basename(path)} ...", flush=True)
        basics, errors_by_model_gid = _compute_run_errors(
            date, thr, path, models, normalizer, wcfg, args, device)
        eligible = [gid for gid, b in basics.items()
                    if np.isfinite(b["target_fraction"])
                    and b["target_fraction"] >= args.calibration_min_target_fraction
                    and np.isfinite(b["purity"])
                    and b["purity"] >= args.calibration_min_purity
                    and b["is_true_track"]]
        n_false += sum(1 for gid in eligible if not basics[gid]["is_true_track"])
        for gid in eligible:
            if any(len(errors_by_model_gid[m].get(gid, [])) for m in args.model):
                n_tracks += 1
        for m in args.model:
            for gid in eligible:
                errs = errors_by_model_gid[m].get(gid)
                if errs is not None and len(errs) and np.all(np.isfinite(errs)):
                    pooled[m].append(errs)

    if n_false > args.calibration_max_false_tracks:
        raise SystemExit(
            f"Calibration selected {n_false} false tracks (> "
            f"--calibration-max-false-tracks {args.calibration_max_false_tracks})")

    cal_track = {}
    per_model_tracks = n_tracks // max(len(args.model), 1)
    for m in args.model:
        if not pooled[m]:
            raise SystemExit(f"No calibration windows collected for model {m}")
        errs = np.concatenate(pooled[m])
        q = quantiles_from_window_errors(errs)
        q["n_calibration_tracks"] = per_model_tracks
        cal_track[m] = q

    meta = {
        "calibration_dates": sorted({r[0] for r in runs}),
        "calibration_thresholds": sorted({r[1] for r in runs}),
        "min_target_fraction": args.calibration_min_target_fraction,
        "min_purity": args.calibration_min_purity,
        "calibration_output": args.calibration_output,
    }
    return cal_track, meta


def resolve_calibration_output(args) -> str:
    """Calibration JSON lives under the run's OWN report dir unless overridden.

    This keeps a per-day/orchestrated rerun from silently overwriting the canonical
    stage-12 calibration artifact: the canonical path is only produced when the run
    is genuinely targeting the canonical --report-dir (stage 17.5 hardening)."""
    if getattr(args, "calibration_output", None):
        return args.calibration_output
    return os.path.join(args.report_dir, "calibration", "sequence_track_calibration.json")


def run_scoring(args) -> dict:
    from utils.sequence_models import load_model, resolve_device

    args.calibration_output = resolve_calibration_output(args)
    key_output = os.path.join(args.report_dir, "sequence_track_scores.csv")
    if os.path.exists(key_output) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")

    device = resolve_device(args.device)
    normalizer = load_normalizer(os.path.join(args.models_dir, "normalizer.json"))
    with open(os.path.join(args.models_dir, "window_config.json")) as f:
        window_config = json.load(f)
    wcfg = WindowConfig(
        window_len=args.window_len or window_config["window_len"],
        stride=args.stride or window_config["stride"],
        features=window_config["features"])
    with open(os.path.join(args.models_dir, "training_manifest.json")) as f:
        manifest = json.load(f)
    val_recon = pd.read_csv(manifest["validation_reconstruction"]) \
        if os.path.exists(manifest["validation_reconstruction"]) else \
        pd.read_csv(os.path.join(args.report_dir, "validation_reconstruction.csv"))

    models = {}
    cal_clean = {}
    input_dim = len(wcfg.features)
    for name in args.model:
        ckpt = os.path.join(args.models_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Model checkpoint missing: {ckpt} -- run training first")
        models[name] = load_model(ckpt, name, input_dim, wcfg.window_len, device)
        row = val_recon[val_recon["model"] == name].iloc[0]
        cal_clean[name] = {c: float(row[c]) for c in row.index
                           if str(c).startswith("error_p")}
        cal_clean[name]["n_val_windows"] = int(row.get("n_val_windows", 0))

    # --- calibrations: which modes do we need? ---------------------------
    primary_mode = args.calibration_mode
    compare = bool(args.compare_calibration)
    need_track = (primary_mode == "track_purity") or compare
    cal_by_mode = {"clean_truth": cal_clean}
    cal_track, calib_meta = None, None
    if need_track:
        cal_track, calib_meta = build_track_purity_calibration(
            models, normalizer, wcfg, args, device)
        cal_by_mode["track_purity"] = cal_track
        calib_dir = os.path.dirname(args.calibration_output)
        json_path, csv_path = write_calibration_files(calib_dir, cal_track, calib_meta)
        print(f"track-purity calibration written: {json_path}")

    runs = discover_track_files(args.tracks_dir)
    if args.threshold_db is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in args.threshold_db)]
    if args.date:
        runs = [r for r in runs if r[0] in args.date]
    if not runs:
        raise SystemExit(f"No stage-8 track files found in {args.tracks_dir}")

    os.makedirs(args.report_dir, exist_ok=True)
    all_scores = []
    for date, thr, path in runs:
        print(f"[{date} thr={thr:g}dB] scoring {os.path.basename(path)} ...", flush=True)
        scores = score_one_run(date, thr, path, models, cal_by_mode, primary_mode,
                               compare, normalizer, wcfg, args, device)
        kept = scores[scores["keep_sequence_prior"]]
        print(f"[{date} thr={thr:g}dB] track-model rows: {len(scores)}, kept: {len(kept)}")
        all_scores.append(scores)

    scores = pd.concat(all_scores, ignore_index=True)
    scores[[c for c in SCORE_CSV_COLUMNS if c in scores.columns]].to_csv(
        key_output, index=False, float_format="%.6g")

    sweep_cols = [float(p) for p in args.sweep_thresholds.split(",")]
    edges = [float(p) for p in args.range_bins_m.split(",")]

    # primary aggregation drives the main report + four-way comparison
    by_mt = aggregate_by_model_threshold(scores, args.score_threshold)
    by_mt["calibration_mode"] = primary_mode
    sweep = sweep_table(scores, sweep_cols)
    range_bins = range_bin_table(scores, edges, args.score_threshold)

    # per-mode aggregations for the calibration comparison (both when comparing)
    by_mt_by_mode = {primary_mode: by_mt}
    calib_comparison = None
    if compare:
        by_mt_by_mode = {}
        frames_for_compare = []
        for mode in ("clean_truth", "track_purity"):
            col = ("sequence_prior_score_clean_truth" if mode == "clean_truth"
                   else "sequence_prior_score_track_calibrated")
            m_by_mt = aggregate_by_model_threshold(scores, args.score_threshold, col)
            m_by_mt["calibration_mode"] = mode
            by_mt_by_mode[mode] = m_by_mt
            frames_for_compare.append(m_by_mt)
        calib_comparison = build_calibration_comparison(by_mt_by_mode, args.score_threshold)

    comparison, s9_ok, s11_ok = compare_with_stage09_stage11(
        by_mt, args.stage09_dir, args.stage11_dir)

    by_mt.to_csv(os.path.join(args.report_dir, "sequence_metrics_by_model_threshold.csv"),
                 index=False)
    sweep.to_csv(os.path.join(args.report_dir, "sequence_filter_sweep.csv"), index=False)
    range_bins.to_csv(os.path.join(args.report_dir, "sequence_range_bin_metrics.csv"),
                      index=False)
    comparison.to_csv(os.path.join(args.report_dir,
                                   "stage08_vs_stage09_vs_stage11_vs_stage12.csv"), index=False)
    if calib_comparison is not None:
        calib_dir = os.path.dirname(args.calibration_output)
        os.makedirs(calib_dir, exist_ok=True)
        calib_comparison.to_csv(
            os.path.join(calib_dir, "sequence_calibration_comparison.csv"), index=False)

    if not args.no_plots:
        make_scoring_plots(scores, by_mt, sweep, comparison, s9_ok, s11_ok,
                           os.path.join(args.report_dir, "plots"))
        if need_track:
            make_calibration_plots(cal_track, cal_clean, by_mt_by_mode,
                                   os.path.join(args.report_dir, "plots"))
    report_path = write_scoring_report(
        args.report_dir, scores, by_mt, comparison, range_bins, val_recon, manifest,
        s9_ok, s11_ok, args.score_threshold, primary_mode=primary_mode,
        cal_track=cal_track, cal_clean=cal_clean, calib_comparison=calib_comparison,
        calib_meta=calib_meta)
    run_scoring_gate(args.report_dir, scores, by_mt, comparison, s9_ok, s11_ok,
                     cal_track=cal_track,
                     calibration_json=(args.calibration_output if need_track else None),
                     by_mt_by_mode=(by_mt_by_mode if compare else None),
                     score_threshold=args.score_threshold)

    print(f"\nreport: {os.path.abspath(report_path)}")
    print(f"\nprimary calibration mode: {primary_mode}")
    print("mean per model (false reduction / true retention):")
    for model, g in by_mt.groupby("model"):
        print(f"  {model:<10} {g['false_track_reduction'].mean():.3f} / "
              f"{g['true_track_retention'].mean():.3f}")
    if compare:
        print("clean-truth vs track-purity mean true retention:")
        for mode in ("clean_truth", "track_purity"):
            print(f"  {mode:<13} {by_mt_by_mode[mode]['true_track_retention'].mean():.3f}")
    return {"scores": scores, "by_mt": by_mt, "comparison": comparison,
            "cal_track": cal_track, "by_mt_by_mode": by_mt_by_mode}


def _load_train12():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train12", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "12_train_sequence_priors.py"))
    train12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train12)
    return train12


def _tiny_train(tmp, train12):
    """Train tiny models on smooth synthetic truth; return the models dir."""
    truth_dir = os.path.join(tmp, "truth")
    os.makedirs(truth_dir)
    for i, date in enumerate(["2022-01-01", "2022-01-02"]):
        train12.make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"),
                                     date, n_traj=8, n=80, seed=i)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]  # the training parser must not see the score CLI flags
    try:
        targs = train12.parse_args()
    finally:
        sys.argv = saved_argv
    targs.truth_dir = truth_dir
    targs.models_dir = os.path.join(tmp, "models")
    targs.report_dir = os.path.join(tmp, "reports")
    targs.holdout_date = ["2022-01-02"]
    targs.window_len, targs.stride = 10, 2
    targs.epochs, targs.batch_size = 6, 64
    targs.hidden_dim, targs.latent_dim, targs.num_layers = 32, 8, 1
    targs.overwrite = True
    targs.model = ["mlp_dae", "gru_ae", "tcn_ae"]
    targs.learning_rate, targs.weight_decay, targs.noise_std = 1e-3, 1e-5, 0.05
    targs.seed, targs.device = 42, "auto"
    train12.run_training(targs)
    return targs.models_dir


def _write_spec_tracks(path_prefix, rows, track_ids):
    pd.DataFrame(rows).to_csv(f"{path_prefix}.csv", index=False)
    summary = path_prefix.replace("track_points_", "track_summary_")
    pd.DataFrame({"track_id": track_ids}).to_csv(f"{summary}.csv", index=False)


def _track_rows(track_id, date, thr, vfun, is_target, traj, n=40, dt=10.0,
                p0=(15_000.0, 12_000.0, 1_000.0)):
    rows, p = [], np.array(p0, dtype=float)
    for k in range(n):
        v = np.array(vfun(k), dtype=float)
        p = p + v * dt
        rows.append(dict(
            date=date, threshold_db=thr, frame_id=k, timestamp=1_000.0 + dt * k,
            track_id=track_id, is_confirmed=1, event_type="hit", assigned_detection_id=k,
            state_x_m=p[0], state_y_m=p[1], state_z_m=p[2],
            state_vx_mps=v[0], state_vy_mps=v[1], state_vz_mps=v[2],
            is_target=is_target, trajectory_id=traj if is_target else None, snr_db=8.0))
    return rows


def self_test() -> None:
    """Train tiny models on synthetic truth, then score four synthetic tracks."""
    train12 = _load_train12()
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = _tiny_train(tmp, train12)

        tracks_dir = os.path.join(tmp, "tracks")
        os.makedirs(tracks_dir)
        date = "2022-01-03"
        rows = []
        rows += _track_rows(0, date, 0.0, lambda k: (55 * np.sin(np.radians(30 + 0.5 * k)),
                                                     55 * np.cos(np.radians(30 + 0.5 * k)),
                                                     0.5), 1, "tA")
        rows += _track_rows(1, date, 0.0, lambda k: (52.0, 3.0, -0.5), 1, "tB")
        rows += _track_rows(2, date, 0.0, lambda k: (150.0, -60.0, 15.0) if k % 2 == 0
                            else (-120.0, 90.0, -15.0), 0, None)
        rows += _track_rows(3, date, 0.0, lambda k: (15.0, 0.0, 12.0 * np.sin(k / 2)), 0, None)
        _write_spec_tracks(os.path.join(tracks_dir, f"track_points_{date}_thr_0p0dB"),
                           rows, [0, 1, 2, 3])

        sargs = parse_args()
        sargs.tracks_dir = tracks_dir
        sargs.models_dir = models_dir
        sargs.stage09_dir = os.path.join(tmp, "no9")
        sargs.stage11_dir = os.path.join(tmp, "no11")
        sargs.report_dir = os.path.join(tmp, "score_reports")
        sargs.detections_dir = tmp
        sargs.model = ["mlp_dae", "gru_ae", "tcn_ae"]
        sargs.score_threshold = 0.5
        sargs.overwrite = True
        out = run_scoring(sargs)
        scores = out["scores"]

        assert os.path.exists(os.path.join(sargs.report_dir, "sequence_track_scores.csv"))
        finite = scores[np.isfinite(scores["sequence_prior_score"])]
        assert finite["sequence_prior_score"].between(0, 1).all()
        piv = scores.pivot_table(index="track_id", columns="model",
                                 values="sequence_prior_score")
        true_mean = piv.loc[[0, 1]].mean().mean()
        jagged = piv.loc[2].mean()
        assert true_mean > jagged, \
            f"true tracks ({true_mean:.3f}) should out-score the jagged false track ({jagged:.3f})"
        kept = scores.groupby("track_id")["keep_sequence_prior"].any()
        assert not scores[(scores["track_id"] == 2)]["keep_sequence_prior"].all(), \
            "jagged false track should be rejected by at least one model at 0.5"
        assert kept.loc[0] or kept.loc[1], "at least one true track must be retained at 0.5"

        text = open(os.path.join(sargs.report_dir, "sequence_prior_report.md")).read()
        for needle in ["Stage 12 Learned Sequence-Prior Track Scoring", "not VAE",
                       "not diffusion", "Stage 13"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print("\nper-track mean scores across models:")
        print(piv.mean(axis=1).round(3).to_string())

    print("\nStage 12 scoring self-test passed.")


def self_test_track_purity() -> None:
    """Stage 12.5: show clean-truth calibration under-retains noisy true tracks and
    track-purity calibration recovers them, while still rejecting a false track."""
    train12 = _load_train12()
    rng = np.random.default_rng(125)
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = _tiny_train(tmp, train12)  # clean band (low error) from smooth truth

        tracks_dir = os.path.join(tmp, "tracks")
        os.makedirs(tracks_dir)

        def noisy_true(track_id, date, thr, base, traj, sigma=6.0):
            # smooth heading with velocity jitter -> elevated (but genuine) recon error
            def vfun(k):
                ang = np.radians(20 + 0.4 * k)
                bx, by, bz = base * np.sin(ang), base * np.cos(ang), 0.4
                j = rng.normal(0, sigma, size=3)
                return (bx + j[0], by + j[1], bz + 0.3 * j[2])
            return _track_rows(track_id, date, thr, vfun, 1, traj)

        # calibration run: high-purity NOISY true tracks (higher thresholds -> cleaner)
        cal_date = "2022-01-04"
        cal_rows = []
        for tid in range(10, 17):
            cal_rows += noisy_true(tid, cal_date, 6.0, 50.0 + tid, f"c{tid}")
        _write_spec_tracks(os.path.join(tracks_dir, f"track_points_{cal_date}_thr_6p0dB"),
                           cal_rows, list(range(10, 17)))

        # eval run: noisy true tracks (0,1) + jagged false (2)
        eval_date = "2022-01-03"
        eval_rows = []
        eval_rows += noisy_true(0, eval_date, 0.0, 52.0, "tA")
        eval_rows += noisy_true(1, eval_date, 0.0, 54.0, "tB")
        eval_rows += _track_rows(2, eval_date, 0.0, lambda k: (150.0, -60.0, 15.0) if k % 2 == 0
                                 else (-120.0, 90.0, -15.0), 0, None)
        _write_spec_tracks(os.path.join(tracks_dir, f"track_points_{eval_date}_thr_0p0dB"),
                           eval_rows, [0, 1, 2])

        sargs = parse_args()
        sargs.tracks_dir = tracks_dir
        sargs.calibration_tracks_dir = tracks_dir
        sargs.models_dir = models_dir
        sargs.stage09_dir = os.path.join(tmp, "no9")
        sargs.stage11_dir = os.path.join(tmp, "no11")
        sargs.report_dir = os.path.join(tmp, "score_reports")
        sargs.calibration_output = os.path.join(
            tmp, "score_reports", "calibration", "sequence_track_calibration.json")
        sargs.detections_dir = tmp
        sargs.model = ["mlp_dae", "gru_ae", "tcn_ae"]
        sargs.threshold_db = [0.0]
        sargs.date = [eval_date]
        sargs.calibration_mode = "track_purity"
        sargs.calibration_date = [cal_date]
        sargs.calibration_threshold_db = [6.0]
        sargs.calibration_min_target_fraction = 0.95
        sargs.calibration_min_purity = 0.95
        sargs.compare_calibration = True
        sargs.score_threshold = 0.5
        sargs.overwrite = True
        out = run_scoring(sargs)
        scores = out["scores"]

        calib_dir = os.path.dirname(sargs.calibration_output)
        assert os.path.exists(sargs.calibration_output), "calibration JSON missing"
        assert os.path.exists(os.path.join(calib_dir, "sequence_track_calibration.csv")), \
            "calibration CSV missing"
        cal_track = out["cal_track"]
        for m, d in cal_track.items():
            assert d["n_calibration_tracks"] > 0, f"{m}: n_calibration_tracks must be > 0"
            assert d["n_calibration_windows"] > 0, f"{m}: n_calibration_windows must be > 0"

        tc = scores["sequence_prior_score_track_calibrated"]
        tc = tc[np.isfinite(tc)]
        assert tc.between(0, 1).all(), "track-calibrated scores must be in [0, 1]"

        def retention(col):
            true = scores[scores["is_true_track"]]
            true = true[np.isfinite(true[col])]
            kept = true[true[col] >= 0.5]
            return len(kept) / len(true) if len(true) else 0.0

        clean_ret = retention("sequence_prior_score_clean_truth")
        track_ret = retention("sequence_prior_score_track_calibrated")
        assert track_ret >= clean_ret, \
            f"track calibration retention ({track_ret:.3f}) must be >= clean ({clean_ret:.3f})"

        false2 = scores[(scores["track_id"] == 2)]["keep_sequence_prior"]
        assert not false2.all(), "jagged false track must be rejected by >=1 model at 0.5"

        text = open(os.path.join(sargs.report_dir, "sequence_prior_report.md")).read()
        for needle in ["Stage 12.5 Noise-Matched Calibration",
                       "autoencoder weights are unchanged",
                       "Truth labels are used only to select calibration tracks",
                       "not VAE"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print(f"\nclean-truth true-track retention @0.5: {clean_ret:.3f}")
        print(f"track-purity true-track retention @0.5: {track_ret:.3f}")

    print("\nStage 12 scoring self-test passed.")
    print("Stage 12.5 calibration self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        if args.calibration_mode == "track_purity":
            self_test_track_purity()
        else:
            self_test()
        return
    run_scoring(args)
    print("\n12_score_tracks_sequence_prior completed successfully.")


if __name__ == "__main__":
    main()
