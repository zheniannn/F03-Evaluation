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
    compare_with_stage09_stage11,
    evaluate_track_labels,
    make_scoring_plots,
    range_bin_table,
    run_scoring_gate,
    score_track_errors,
    sweep_table,
    write_scoring_report,
)

SCORE_CSV_COLUMNS = [
    "date", "threshold_db", "model", "track_id",
    "n_hits", "n_misses", "duration_s", "median_range_m", "max_range_m", "n_windows",
    "sequence_prior_score", "keep_sequence_prior",
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
    return parser.parse_args()


def score_one_run(date, threshold_db, path, models, calibrations, normalizer,
                  wcfg, args, device):
    """Score every confirmed track of one stage-8 run under every model."""
    from utils.sequence_models import reconstruction_errors

    df = load_tracks_for_windows(path, date, threshold_db, args.detections_dir)
    confirmed = df.loc[df["confirmed"] == 1, "group_id"].unique()
    df = df[df["group_id"].isin(confirmed)]

    # per-track basic + evaluation-only fields
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
    rows = []
    if len(windows):
        normed = apply_normalizer(windows, normalizer)
        gid_arr = np.asarray(gids, dtype=object)
    for model_name in args.model:
        errors_by_gid = {}
        if len(windows):
            errors = reconstruction_errors(models[model_name], normed, args.batch_size, device)
            order = pd.Series(errors).groupby(gid_arr)
            for gid, errs in order:
                errors_by_gid[gid] = errs.to_numpy()
        cal = calibrations[model_name]
        for gid, b in basics.items():
            errs = errors_by_gid.get(gid)
            if errs is None or not len(errs):
                score = np.nan
                stats = dict(mean=np.nan, median=np.nan, p90=np.nan, mx=np.nan, n=0)
            else:
                stats = dict(mean=float(errs.mean()), median=float(np.median(errs)),
                             p90=float(np.percentile(errs, 90)), mx=float(errs.max()),
                             n=len(errs))
                score = score_track_errors(stats["median"], cal)
            rows.append({
                "date": date, "threshold_db": threshold_db, "model": model_name,
                "track_id": gid, **{k: b[k] for k in
                                    ("n_hits", "n_misses", "duration_s",
                                     "median_range_m", "max_range_m")},
                "n_windows": stats["n"],
                "sequence_prior_score": score,
                "keep_sequence_prior": bool(np.isfinite(score) and score >= args.score_threshold),
                "sequence_recon_error_mean": stats["mean"],
                "sequence_recon_error_median": stats["median"],
                "sequence_recon_error_p90": stats["p90"],
                "sequence_recon_error_max": stats["mx"],
                "calibration_error_p50": cal["error_p50"],
                "calibration_error_p99": cal["error_p99"],
                **{k: b[k] for k in ("n_target_hits", "n_clutter_hits", "target_fraction",
                                     "purity", "majority_trajectory_id", "is_true_track",
                                     "position_rmse_m")},
            })
    return pd.DataFrame(rows)


def run_scoring(args) -> dict:
    from utils.sequence_models import load_model, resolve_device

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

    models, calibrations = {}, {}
    input_dim = len(wcfg.features)
    for name in args.model:
        ckpt = os.path.join(args.models_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Model checkpoint missing: {ckpt} -- run training first")
        models[name] = load_model(ckpt, name, input_dim, wcfg.window_len, device)
        row = val_recon[val_recon["model"] == name].iloc[0]
        calibrations[name] = {"error_p50": float(row["error_p50"]),
                              "error_p99": float(row["error_p99"])}

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
        scores = score_one_run(date, thr, path, models, calibrations, normalizer,
                               wcfg, args, device)
        kept = scores[scores["keep_sequence_prior"]]
        print(f"[{date} thr={thr:g}dB] track-model rows: {len(scores)}, kept: {len(kept)}")
        all_scores.append(scores)

    scores = pd.concat(all_scores, ignore_index=True)
    scores[[c for c in SCORE_CSV_COLUMNS if c in scores.columns]].to_csv(
        key_output, index=False, float_format="%.6g")

    by_mt = aggregate_by_model_threshold(scores, args.score_threshold)
    sweep = sweep_table(scores, [float(p) for p in args.sweep_thresholds.split(",")])
    edges = [float(p) for p in args.range_bins_m.split(",")]
    range_bins = range_bin_table(scores, edges)
    comparison, s9_ok, s11_ok = compare_with_stage09_stage11(by_mt, args.stage09_dir,
                                                             args.stage11_dir)

    by_mt.to_csv(os.path.join(args.report_dir, "sequence_metrics_by_model_threshold.csv"),
                 index=False)
    sweep.to_csv(os.path.join(args.report_dir, "sequence_filter_sweep.csv"), index=False)
    range_bins.to_csv(os.path.join(args.report_dir, "sequence_range_bin_metrics.csv"),
                      index=False)
    comparison.to_csv(os.path.join(args.report_dir,
                                   "stage08_vs_stage09_vs_stage11_vs_stage12.csv"), index=False)

    if not args.no_plots:
        make_scoring_plots(scores, by_mt, sweep, comparison, s9_ok, s11_ok,
                           os.path.join(args.report_dir, "plots"))
    report_path = write_scoring_report(args.report_dir, scores, by_mt, comparison,
                                       range_bins, val_recon, manifest, s9_ok, s11_ok,
                                       args.score_threshold)
    run_scoring_gate(args.report_dir, scores, by_mt, comparison, s9_ok, s11_ok)

    print(f"\nreport: {os.path.abspath(report_path)}")
    print("\nmean per model (false reduction / true retention):")
    for model, g in by_mt.groupby("model"):
        print(f"  {model:<10} {g['false_track_reduction'].mean():.3f} / "
              f"{g['true_track_retention'].mean():.3f}")
    return {"scores": scores, "by_mt": by_mt, "comparison": comparison}


def self_test() -> None:
    """Train tiny models on synthetic truth, then score four synthetic tracks."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train12", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "12_train_sequence_priors.py"))
    train12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train12)

    rng = np.random.default_rng(12)
    with tempfile.TemporaryDirectory() as tmp:
        # --- tiny training on smooth synthetic truth ---------------------
        truth_dir = os.path.join(tmp, "truth")
        os.makedirs(truth_dir)
        for i, date in enumerate(["2022-01-01", "2022-01-02"]):
            train12.make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"),
                                         date, n_traj=8, n=80, seed=i)
        targs = train12.parse_args()
        targs.truth_dir = truth_dir
        targs.models_dir = os.path.join(tmp, "models")
        targs.report_dir = os.path.join(tmp, "reports")
        targs.holdout_date = ["2022-01-02"]
        targs.window_len, targs.stride = 10, 2
        targs.epochs, targs.batch_size = 6, 64
        targs.hidden_dim, targs.latent_dim, targs.num_layers = 32, 8, 1
        targs.overwrite = True
        # reuse the training runner (it also validates its own outputs)
        targs.model = ["mlp_dae", "gru_ae", "tcn_ae"]
        targs.learning_rate, targs.weight_decay, targs.noise_std = 1e-3, 1e-5, 0.05
        targs.seed, targs.device = 42, "auto"
        train12.run_training(targs)

        # --- synthetic stage-8 tracks (spec schema) ----------------------
        tracks_dir = os.path.join(tmp, "tracks")
        os.makedirs(tracks_dir)
        date, dt, n = "2022-01-03", 10.0, 40

        def make_track(track_id, vfun, is_target, traj):
            rows = []
            p = np.array([15_000.0, 12_000.0, 1_000.0])
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
                    snr_db=8.0))
            return rows

        rows = []
        rows += make_track(0, lambda k: (55 * np.sin(np.radians(30 + 0.5 * k)),
                                         55 * np.cos(np.radians(30 + 0.5 * k)), 0.5), 1, "tA")
        rows += make_track(1, lambda k: (52.0, 3.0, -0.5), 1, "tB")
        # jagged false: velocity flips wildly
        rows += make_track(2, lambda k: (150.0, -60.0, 15.0) if k % 2 == 0
                           else (-120.0, 90.0, -15.0), 0, None)
        # smooth but off-pattern: extreme vertical oscillation at odd speed
        rows += make_track(3, lambda k: (15.0, 0.0, 12.0 * np.sin(k / 2)), 0, None)

        pd.DataFrame(rows).to_csv(
            os.path.join(tracks_dir, f"track_points_{date}_thr_0p0dB.csv"), index=False)
        pd.DataFrame({"track_id": [0, 1, 2, 3]}).to_csv(
            os.path.join(tracks_dir, f"track_summary_{date}_thr_0p0dB.csv"), index=False)

        sargs = parse_args()
        sargs.tracks_dir = tracks_dir
        sargs.models_dir = targs.models_dir
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


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_scoring(args)
    print("\n12_score_tracks_sequence_prior completed successfully.")


if __name__ == "__main__":
    main()
