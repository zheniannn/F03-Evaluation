"""Entry point: stage 13 step B -- score stage-8 Kalman tracks with the VAE prior.

Builds the same origin/heading-normalized windows from each confirmed track's
posterior states, computes per-window VAE reconstruction error, KL, and latent
means, and scores tracks with two anomaly variants (reconstruction-only and an
ELBO-like recon + beta_score*KL). Scores are calibrated in the noisy-track
domain from high-purity stage-8 true tracks (the stage-12.5 lesson) and compared
five-way against stages 8/9/11/12. Truth labels are used only for calibration
selection and evaluation. Not diffusion (stage 14).

Usage:
    python scripts/13_score_tracks_vae_prior.py --threshold-db -5 0 3 6 9 12 \
        --date 2022-06-06 --calibration-mode track_purity --overwrite
    python scripts/13_score_tracks_vae_prior.py --self-test
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
    WindowConfig,
    apply_normalizer,
    build_windows_by_group,
    discover_track_files,
    load_normalizer,
    load_tracks_for_windows,
)
from utils.sequence_prior_score import evaluate_track_labels
from utils.vae_prior_score import (
    VARIANTS,
    aggregate_by_variant_threshold,
    build_five_way_comparison,
    calibration_quantiles,
    latent_summary_table,
    make_latent_pca_plot,
    make_vae_scoring_plots,
    run_vae_scoring_gate,
    score_from_band,
    variant_range_bin_table,
    variant_sweep_table,
    write_vae_calibration,
    write_vae_report,
)

SCORE_CSV_COLUMNS = [
    "date", "threshold_db", "variant", "track_id",
    "n_hits", "n_misses", "duration_s", "median_range_m", "max_range_m", "n_windows",
    "vae_prior_score", "keep_vae_prior",
    "vae_recon_error_mean", "vae_recon_error_median", "vae_recon_error_p90", "vae_recon_error_max",
    "vae_kl_mean", "vae_kl_median", "vae_kl_p90",
    "vae_elbo_mean", "vae_elbo_median", "vae_elbo_p90",
    "calibration_mode", "calibration_error_p50", "calibration_error_p99",
    "latent_mu_mean_norm", "latent_mu_std_mean",
    "n_target_hits", "n_clutter_hits", "target_fraction", "purity",
    "majority_trajectory_id", "is_true_track", "position_rmse_m",
]


def parse_args():
    p = argparse.ArgumentParser(description="Score stage-8 tracks with the stage-13 VAE prior.")
    p.add_argument("--tracks-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "vae_priors"))
    p.add_argument("--sequence-models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--stage09-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    p.add_argument("--stage11-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage11_adsb_prior_scoring"))
    p.add_argument("--stage12-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    p.add_argument("--report-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage13_vae_prior"))
    p.add_argument("--detections-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"),
                   help="Only used by the shared stage-8 loader (SNR column; unused here).")
    p.add_argument("--threshold-db", type=float, nargs="+", default=None)
    p.add_argument("--date", type=str, nargs="*", default=None)
    p.add_argument("--range-bins-m", type=str, default="0,50000,100000,200000,inf")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--sweep-thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--min-track-hits", type=int, default=5)
    p.add_argument("--beta-score", type=float, default=0.001,
                   help="KL weight in the ELBO-like anomaly (recon + beta_score * KL).")
    p.add_argument("--calibration-mode", choices=["clean_truth", "track_purity"],
                   default="track_purity")
    p.add_argument("--calibration-tracks-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--calibration-date", type=str, nargs="*", default=None)
    p.add_argument("--calibration-threshold-db", type=float, nargs="+", default=[3, 6, 9, 12])
    p.add_argument("--calibration-min-target-fraction", type=float, default=0.95)
    p.add_argument("--calibration-min-purity", type=float, default=0.95)
    p.add_argument("--calibration-max-false-tracks", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return p.parse_args()


def _compute_run_vae(date, threshold_db, path, model, normalizer, wcfg, args, device):
    """Per-track window metrics for one stage-8 run.

    Returns (basics, per_track) where per_track maps track id -> dict with
    recon/kl/elbo per-window arrays and the per-window latent mean matrix.
    """
    from utils.vae_models import vae_window_metrics

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
    per_track = {}
    if len(windows):
        normed = apply_normalizer(windows, normalizer)
        m = vae_window_metrics(model, normed, args.batch_size, device)
        gid_arr = np.asarray(gids, dtype=object)
        elbo = m["recon_error"] + args.beta_score * m["kl"]
        order = pd.Series(np.arange(len(gid_arr))).groupby(gid_arr)
        for gid, idx in order:
            ii = idx.to_numpy()
            per_track[gid] = {"recon": m["recon_error"][ii], "kl": m["kl"][ii],
                              "elbo": elbo[ii], "mu": m["mu"][ii]}
    return basics, per_track


def _track_stat_row(gid, b, wm, variant, cal, beta_score, score_threshold, date, threshold_db):
    r, k, e, mu = wm["recon"], wm["kl"], wm["elbo"], wm["mu"]
    anomaly = e if variant == "elbo" else r
    median_anom = float(np.median(anomaly))
    score = score_from_band(median_anom, cal["error_p50"], cal["error_p99"])
    mu_mean = mu.mean(axis=0)
    return {
        "date": date, "threshold_db": threshold_db, "variant": variant, "track_id": gid,
        **{c: b[c] for c in ("n_hits", "n_misses", "duration_s", "median_range_m", "max_range_m")},
        "n_windows": len(r),
        "vae_prior_score": score,
        "keep_vae_prior": bool(np.isfinite(score) and score >= score_threshold),
        "vae_recon_error_mean": float(r.mean()), "vae_recon_error_median": float(np.median(r)),
        "vae_recon_error_p90": float(np.percentile(r, 90)), "vae_recon_error_max": float(r.max()),
        "vae_kl_mean": float(k.mean()), "vae_kl_median": float(np.median(k)),
        "vae_kl_p90": float(np.percentile(k, 90)),
        "vae_elbo_mean": float(e.mean()), "vae_elbo_median": float(np.median(e)),
        "vae_elbo_p90": float(np.percentile(e, 90)),
        "calibration_mode": cal.get("calibration_mode", "track_purity"),
        "calibration_error_p50": cal["error_p50"], "calibration_error_p99": cal["error_p99"],
        "latent_mu_mean_norm": float(np.linalg.norm(mu_mean)),
        "latent_mu_std_mean": float(mu.std(axis=0).mean()),
        **{c: b[c] for c in ("n_target_hits", "n_clutter_hits", "target_fraction", "purity",
                             "majority_trajectory_id", "is_true_track", "position_rmse_m")},
    }


def score_one_run(date, threshold_db, path, model, cal_by_variant, normalizer, wcfg,
                  args, device):
    basics, per_track = _compute_run_vae(date, threshold_db, path, model, normalizer,
                                         wcfg, args, device)
    rows, latent_rows = [], []
    for gid, b in basics.items():
        wm = per_track.get(gid)
        if wm is None or not len(wm["recon"]):
            continue
        for variant in VARIANTS:
            rows.append(_track_stat_row(gid, b, wm, variant, cal_by_variant[variant],
                                        args.beta_score, args.score_threshold, date, threshold_db))
        latent_rows.append({"mu_mean": wm["mu"].mean(axis=0), "is_true": b["is_true_track"]})
    return pd.DataFrame(rows), latent_rows


def build_calibration(mode, model, normalizer, wcfg, args, device):
    """Return {variant: band-dict} and calibration meta.

    track_purity: per-window recon/elbo anomalies pooled from high-purity stage-8
    true tracks. clean_truth: reconstruction band from the validation reconstruction
    CSV (ELBO falls back to the same band; no per-window val KL is stored)."""
    if mode == "clean_truth":
        vr = pd.read_csv(os.path.join(args.report_dir, "vae_validation_reconstruction.csv"))
        row = vr.iloc[0]
        band = {"error_p50": float(row["error_p50"]), "error_p99": float(row["error_p99"]),
                "error_p90": float(row["error_p90"]), "error_mean": float(row["error_mean"]),
                "n_calibration_tracks": 0, "n_calibration_windows": int(row["n_val_windows"]),
                "calibration_mode": "clean_truth"}
        cal = {v: dict(band) for v in VARIANTS}
        meta = {"calibration_mode": "clean_truth", "calibration_dates": [],
                "calibration_thresholds": [], "min_target_fraction": None,
                "min_purity": None, "beta_score": args.beta_score}
        return cal, meta

    cal_dates = args.calibration_date if args.calibration_date else args.date
    cal_thr = args.calibration_threshold_db
    runs = discover_track_files(args.calibration_tracks_dir)
    if cal_thr is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in cal_thr)]
    if cal_dates:
        runs = [r for r in runs if r[0] in cal_dates]
    if not runs:
        raise SystemExit(f"No calibration track files found in {args.calibration_tracks_dir} "
                         f"for dates={cal_dates} thresholds={cal_thr}")

    pooled = {"reconstruction": [], "elbo": []}
    n_tracks = n_false = 0
    for date, thr, path in runs:
        print(f"[calib {date} thr={thr:g}dB] {os.path.basename(path)} ...", flush=True)
        basics, per_track = _compute_run_vae(date, thr, path, model, normalizer, wcfg,
                                             args, device)
        for gid, b in basics.items():
            wm = per_track.get(gid)
            if wm is None or not len(wm["recon"]):
                continue
            eligible = (np.isfinite(b["target_fraction"])
                        and b["target_fraction"] >= args.calibration_min_target_fraction
                        and np.isfinite(b["purity"]) and b["purity"] >= args.calibration_min_purity
                        and b["is_true_track"])
            if not eligible:
                continue
            n_tracks += 1
            n_false += 0 if b["is_true_track"] else 1
            pooled["reconstruction"].append(wm["recon"])
            pooled["elbo"].append(wm["elbo"])

    if n_false > args.calibration_max_false_tracks:
        raise SystemExit(f"Calibration selected {n_false} false tracks "
                         f"(> --calibration-max-false-tracks {args.calibration_max_false_tracks})")

    cal = {}
    for v in VARIANTS:
        if not pooled[v]:
            raise SystemExit(f"No calibration windows collected for variant {v}")
        q = calibration_quantiles(np.concatenate(pooled[v]))
        q["n_calibration_tracks"] = n_tracks
        q["calibration_mode"] = "track_purity"
        cal[v] = q
    meta = {"calibration_mode": "track_purity",
            "calibration_dates": sorted({r[0] for r in runs}),
            "calibration_thresholds": sorted({r[1] for r in runs}),
            "min_target_fraction": args.calibration_min_target_fraction,
            "min_purity": args.calibration_min_purity, "beta_score": args.beta_score}
    return cal, meta


def run_scoring(args) -> dict:
    from utils.vae_models import load_vae, resolve_device

    key_output = os.path.join(args.report_dir, "vae_track_scores.csv")
    if os.path.exists(key_output) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")

    device = resolve_device(args.device)
    with open(os.path.join(args.models_dir, "vae_config.json")) as f:
        vae_config = json.load(f)
    wcfg = WindowConfig(window_len=vae_config["window_len"], stride=vae_config["stride"],
                        features=vae_config["features"])
    normalizer = load_normalizer(os.path.join(args.models_dir, "normalizer.json"))
    with open(os.path.join(args.models_dir, "vae_training_manifest.json")) as f:
        manifest = json.load(f)
    val_recon = pd.read_csv(manifest["validation_reconstruction"]) \
        if os.path.exists(manifest.get("validation_reconstruction", "")) else \
        pd.read_csv(os.path.join(args.report_dir, "vae_validation_reconstruction.csv"))

    ckpt = os.path.join(args.models_dir, "vae_prior.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"VAE checkpoint missing: {ckpt} -- run training first")
    model = load_vae(ckpt, vae_config, device)

    cal_by_variant, calib_meta = build_calibration(args.calibration_mode, model, normalizer,
                                                   wcfg, args, device)
    os.makedirs(args.report_dir, exist_ok=True)
    json_path, csv_path = write_vae_calibration(args.models_dir, args.report_dir,
                                                cal_by_variant, calib_meta)
    print(f"calibration written: {json_path}")

    runs = discover_track_files(args.tracks_dir)
    if args.threshold_db is not None:
        runs = [r for r in runs if any(abs(r[1] - t) < 1e-9 for t in args.threshold_db)]
    if args.date:
        runs = [r for r in runs if r[0] in args.date]
    if not runs:
        raise SystemExit(f"No stage-8 track files found in {args.tracks_dir}")

    all_scores, all_latent = [], []
    for date, thr, path in runs:
        print(f"[{date} thr={thr:g}dB] scoring {os.path.basename(path)} ...", flush=True)
        s, lat = score_one_run(date, thr, path, model, cal_by_variant, normalizer, wcfg,
                               args, device)
        kept = s[s["keep_vae_prior"]]
        print(f"[{date} thr={thr:g}dB] track-variant rows: {len(s)}, kept: {len(kept)}")
        all_scores.append(s)
        all_latent.extend(lat)

    scores = pd.concat(all_scores, ignore_index=True)
    scores[[c for c in SCORE_CSV_COLUMNS if c in scores.columns]].to_csv(
        key_output, index=False, float_format="%.6g")

    sweep_cols = [float(x) for x in args.sweep_thresholds.split(",")]
    edges = [float(x) for x in args.range_bins_m.split(",")]
    by_vt = aggregate_by_variant_threshold(scores, args.score_threshold)
    sweep = variant_sweep_table(scores, sweep_cols)
    range_bins = variant_range_bin_table(scores, edges, args.score_threshold)
    latent = latent_summary_table(scores)
    stage12_csv = os.path.join(args.stage12_dir,
                               "stage08_vs_stage09_vs_stage11_vs_stage12.csv")
    comparison, best_s12 = build_five_way_comparison(by_vt, stage12_csv,
                                                     args.stage09_dir, args.stage11_dir)

    by_vt.to_csv(os.path.join(args.report_dir, "vae_metrics_by_threshold.csv"), index=False)
    sweep.to_csv(os.path.join(args.report_dir, "vae_filter_sweep.csv"), index=False)
    range_bins.to_csv(os.path.join(args.report_dir, "vae_range_bin_metrics.csv"), index=False)
    latent.to_csv(os.path.join(args.report_dir, "vae_latent_summary.csv"), index=False)
    comparison.to_csv(os.path.join(
        args.report_dir, "stage08_vs_stage09_vs_stage11_vs_stage12_vs_stage13.csv"), index=False)

    if not args.no_plots:
        plots_dir = os.path.join(args.report_dir, "plots")
        make_vae_scoring_plots(scores, by_vt, comparison, sweep, best_s12, plots_dir)
        mu_stack = np.vstack([r["mu_mean"] for r in all_latent]) if all_latent else np.empty((0, 1))
        is_true = np.array([r["is_true"] for r in all_latent], dtype=bool)
        make_latent_pca_plot(mu_stack, is_true, plots_dir)

    report_path = write_vae_report(args.report_dir, scores, by_vt, comparison, range_bins,
                                   latent, val_recon, manifest, cal_by_variant, calib_meta,
                                   best_s12, args.score_threshold)
    run_vae_scoring_gate(args.report_dir, scores, by_vt, comparison, cal_by_variant,
                         json_path, best_s12)

    print(f"\nreport: {os.path.abspath(report_path)}")
    print(f"best stage-12 model for comparison: {best_s12}")
    print("mean per variant (false reduction / true retention):")
    for variant, g in by_vt.groupby("variant"):
        print(f"  {variant:<14} {g['false_track_reduction'].mean():.3f} / "
              f"{g['true_track_retention'].mean():.3f}")
    return {"scores": scores, "by_vt": by_vt, "comparison": comparison,
            "cal": cal_by_variant, "best_stage12": best_s12}


# =============================================================================
# Self-test
# =============================================================================

def _load_train13():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train13", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "13_train_vae_prior.py"))
    train13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train13)
    return train13


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


def _write_spec_tracks(prefix, rows, track_ids):
    pd.DataFrame(rows).to_csv(f"{prefix}.csv", index=False)
    pd.DataFrame({"track_id": track_ids}).to_csv(
        f"{prefix.replace('track_points_', 'track_summary_')}.csv", index=False)


def self_test() -> None:
    train13 = _load_train13()
    rng = np.random.default_rng(13)
    with tempfile.TemporaryDirectory() as tmp:
        # tiny VAE training on smooth truth
        truth_dir = os.path.join(tmp, "truth")
        os.makedirs(truth_dir)
        for i, date in enumerate(["2022-01-01", "2022-01-02"]):
            train13.make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"),
                                         date, seed=i)
        saved_argv = sys.argv
        sys.argv = [saved_argv[0]]
        try:
            targs = train13.parse_args()
        finally:
            sys.argv = saved_argv
        targs.truth_dir = truth_dir
        targs.sequence_models_dir = os.path.join(tmp, "seq_missing")
        targs.models_dir = os.path.join(tmp, "models")
        targs.report_dir = os.path.join(tmp, "reports")
        targs.holdout_date = ["2022-01-02"]
        targs.window_len, targs.stride = 10, 2
        targs.epochs, targs.batch_size = 4, 64
        targs.hidden_dim, targs.latent_dim = 32, 4
        targs.overwrite = True
        train13.run_training(targs)

        # synthetic stage-8 tracks (spec schema): calibration + eval runs
        tracks_dir = os.path.join(tmp, "tracks")
        os.makedirs(tracks_dir)

        def noisy_true(track_id, date, thr, base, traj, sigma=6.0):
            def vfun(k):
                ang = np.radians(20 + 0.4 * k)
                j = rng.normal(0, sigma, size=3)
                return (base * np.sin(ang) + j[0], base * np.cos(ang) + j[1], 0.4 + 0.3 * j[2])
            return _track_rows(track_id, date, thr, vfun, 1, traj)

        cal_date = "2022-01-04"
        cal_rows = []
        for tid in range(10, 17):
            cal_rows += noisy_true(tid, cal_date, 6.0, 50.0 + tid, f"c{tid}")
        _write_spec_tracks(os.path.join(tracks_dir, f"track_points_{cal_date}_thr_6p0dB"),
                           cal_rows, list(range(10, 17)))

        eval_date = "2022-01-03"
        eval_rows = []
        eval_rows += noisy_true(0, eval_date, 0.0, 52.0, "tA")
        eval_rows += noisy_true(1, eval_date, 0.0, 54.0, "tB")
        eval_rows += _track_rows(2, eval_date, 0.0, lambda k: (150.0, -60.0, 15.0) if k % 2 == 0
                                 else (-120.0, 90.0, -15.0), 0, None)          # jagged false
        eval_rows += _track_rows(3, eval_date, 0.0, lambda k: (15.0, 0.0, 12.0 * np.sin(k / 2)),
                                 0, None)                                       # smooth off-pattern
        _write_spec_tracks(os.path.join(tracks_dir, f"track_points_{eval_date}_thr_0p0dB"),
                           eval_rows, [0, 1, 2, 3])

        sargs = parse_args()
        sargs.tracks_dir = tracks_dir
        sargs.calibration_tracks_dir = tracks_dir
        sargs.models_dir = targs.models_dir
        sargs.stage09_dir = os.path.join(tmp, "no9")
        sargs.stage11_dir = os.path.join(tmp, "no11")
        sargs.stage12_dir = os.path.join(tmp, "no12")
        sargs.report_dir = os.path.join(tmp, "score_reports")
        sargs.detections_dir = tmp
        sargs.threshold_db = [0.0]
        sargs.date = [eval_date]
        sargs.calibration_mode = "track_purity"
        sargs.calibration_date = [cal_date]
        sargs.calibration_threshold_db = [6.0]
        sargs.score_threshold = 0.5
        sargs.overwrite = True
        out = run_scoring(sargs)
        scores = out["scores"]

        assert os.path.exists(os.path.join(sargs.report_dir, "vae_track_scores.csv"))
        assert os.path.exists(os.path.join(sargs.models_dir, "vae_calibration.json"))
        finite = scores[np.isfinite(scores["vae_prior_score"])]
        assert finite["vae_prior_score"].between(0, 1).all(), "scores must be in [0, 1]"
        for v, d in out["cal"].items():
            assert d["n_calibration_tracks"] > 0 and d["n_calibration_windows"] > 0

        kept = scores.groupby("track_id")["keep_vae_prior"].any()
        assert not scores[scores["track_id"] == 2]["keep_vae_prior"].all(), \
            "jagged false track must be rejected by at least one variant at 0.5"
        assert kept.loc[0] or kept.loc[1], "at least one true track must be retained at 0.5"

        text = open(os.path.join(sargs.report_dir, "vae_prior_report.md")).read()
        for needle in ["Stage 13 VAE Trajectory Prior", "not diffusion",
                       "Truth labels are used only for calibration track selection and evaluation",
                       "Stage 14"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print("\nper-track mean VAE score across variants:")
        piv = scores.pivot_table(index="track_id", columns="variant", values="vae_prior_score")
        print(piv.round(3).to_string())

    print("\nStage 13 VAE scoring self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_scoring(args)
    print("\n13_score_tracks_vae_prior completed successfully.")


if __name__ == "__main__":
    main()
