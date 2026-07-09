"""Entry point: stage 16 -- robustness + ablation study for the stage-12.5 winner.

Consolidates the compact reports from stages 08/09/12/14/15 into robustness and
ablation tables around the current best method (stage-12.5 deterministic
sequence autoencoder with noise-matched calibration). Adds NO new model and
retrains nothing; only `--run-missing` may call the existing stage-12 scoring
script for day/threshold combinations that have no compact scores yet.

Usage:
    python scripts/16_robustness_ablation.py --threshold-db -5 0 3 6 --overwrite
    python scripts/16_robustness_ablation.py --self-test
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils import robustness_analysis as ra


def parse_args():
    p = argparse.ArgumentParser(description="Stage 16 robustness and ablation study.")
    p.add_argument("--stage08-dir", default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline"))
    p.add_argument("--stage09-dir", default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    p.add_argument("--stage12-dir", default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    p.add_argument("--stage14-dir", default=os.path.join(REPO_ROOT, "reports", "stage14_method_benchmark"))
    p.add_argument("--stage15-dir", default=os.path.join(REPO_ROOT, "reports", "stage15_diffusion_denoising"))
    p.add_argument("--tracks-dir", default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "reports", "stage16_robustness"))
    p.add_argument("--date", nargs="*", default=None)
    p.add_argument("--threshold-db", type=float, nargs="+", default=[-5, 0, 3, 6])
    p.add_argument("--include-high-thresholds", action="store_true")
    p.add_argument("--best-model", default="mlp_dae")
    p.add_argument("--compare-models", default="mlp_dae,gru_ae,tcn_ae")
    p.add_argument("--target-retention", type=float, default=0.97)
    p.add_argument("--score-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--run-missing", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real reports needed) and exit.")
    return p.parse_args()


def _paths(args):
    return {
        "stage08_metrics": os.path.join(args.stage08_dir, "kalman_metrics_by_day.csv"),
        "stage09_metrics": os.path.join(args.stage09_dir, "physics_metrics_by_threshold.csv"),
        "stage12_metrics": os.path.join(args.stage12_dir, "sequence_metrics_by_model_threshold.csv"),
        "stage12_calib": os.path.join(args.stage12_dir, "calibration",
                                      "sequence_calibration_comparison.csv"),
        "stage12_rangebin": os.path.join(args.stage12_dir, "sequence_range_bin_metrics.csv"),
        "stage12_sweep": os.path.join(args.stage12_dir, "sequence_filter_sweep.csv"),
        "stage12_scores": os.path.join(args.stage12_dir, "sequence_track_scores.csv"),
        "stage14_best": os.path.join(args.stage14_dir, "best_method_by_threshold.csv"),
        "stage14_failures": os.path.join(args.stage14_dir, "failure_case_candidates.csv"),
    }


def _tracks_dates(tracks_dir):
    import re
    dates = set()
    if os.path.isdir(tracks_dir):
        for f in os.listdir(tracks_dir):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", f)
            if m:
                dates.add(m.group(1))
    return sorted(dates)


def maybe_run_missing(args, paths, stage12_dates):
    """Optionally score missing day/threshold combos via the existing stage-12 script."""
    track_dates = _tracks_dates(args.tracks_dir)
    missing = [d for d in track_dates if d not in stage12_dates]
    if not missing:
        print("--run-missing: no missing day/threshold combinations (stage-12 scores already "
              "cover all track-file dates).")
        return
    thr = " ".join(f"{t:g}" for t in args.threshold_db)
    cmd = ["python", "scripts/12_score_tracks_sequence_prior.py",
           "--tracks-dir", args.tracks_dir, "--models-dir", args.models_dir,
           "--report-dir", args.stage12_dir, "--threshold-db", *thr.split(),
           "--date", *missing, "--calibration-mode", "track_purity",
           "--calibration-threshold-db", "3", "6", "9", "12", "--score-threshold", "0.5",
           "--overwrite"]
    print(f"--run-missing: would score missing dates {missing}. Command:")
    print("  " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except Exception as exc:
        print(f"--run-missing: scoring failed or track files unavailable ({exc}); "
              "continuing with existing compact data.")


def run(args) -> dict:
    out = args.output_dir
    inv_path = os.path.join(out, "input_coverage_inventory.csv")
    if os.path.exists(inv_path) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {inv_path}")
    os.makedirs(out, exist_ok=True)

    paths = _paths(args)
    thresholds = list(args.threshold_db)
    if args.include_high_thresholds:
        thresholds = sorted(set(thresholds) | {9.0, 12.0})
    compare_models = [m.strip() for m in args.compare_models.split(",") if m.strip()]

    dates = args.date if args.date else ra.detect_dates(paths)
    if args.run_missing:
        maybe_run_missing(args, paths, dates)
        dates = args.date if args.date else ra.detect_dates(paths)

    s12_metrics = ra.read_csv_safe(paths["stage12_metrics"])
    calib_cmp = ra.read_csv_safe(paths["stage12_calib"])
    sweep = ra.read_csv_safe(paths["stage12_sweep"])
    rangebin = ra.read_csv_safe(paths["stage12_rangebin"])
    kalman = ra.read_csv_safe(paths["stage08_metrics"])
    failures = ra.read_csv_safe(paths["stage14_failures"])
    stage15_present = os.path.exists(os.path.join(args.stage15_dir, "diffusion_denoising_report.md"))

    inventory = ra.build_inventory(paths, args.tracks_dir, thresholds)
    inventory.to_csv(inv_path, index=False)
    print("input coverage inventory:")
    print(inventory[["source", "available", "date", "threshold_db", "rows"]].to_string(index=False))

    by_thr = ra.robustness_by_threshold(s12_metrics, args.best_model, thresholds,
                                        args.include_high_thresholds)
    by_day = ra.robustness_by_day(s12_metrics, args.best_model, thresholds, dates)
    model_abl = ra.model_ablation(s12_metrics, compare_models, thresholds)
    calib_abl = ra.calibration_ablation(calib_cmp, thresholds)
    sens = ra.score_threshold_sensitivity(sweep, args.best_model, thresholds, args.target_retention)
    windowability = ra.windowability_audit(paths["stage12_scores"], kalman, args.best_model,
                                           thresholds, args.include_high_thresholds)
    rangebin_rob = ra.range_bin_robustness(rangebin, args.best_model, thresholds)
    failure_summary = ra.failure_mode_summary(failures)
    findings = ra.key_findings(by_thr, model_abl, calib_abl, windowability, dates, stage15_present)

    by_thr.to_csv(os.path.join(out, "robustness_by_threshold.csv"), index=False)
    by_day.to_csv(os.path.join(out, "robustness_by_day.csv"), index=False)
    model_abl.to_csv(os.path.join(out, "model_ablation_mlp_gru_tcn.csv"), index=False)
    calib_abl.to_csv(os.path.join(out, "calibration_ablation.csv"), index=False)
    sens.to_csv(os.path.join(out, "score_threshold_sensitivity.csv"), index=False)
    windowability.to_csv(os.path.join(out, "windowability_audit.csv"), index=False)
    rangebin_rob.to_csv(os.path.join(out, "range_bin_robustness.csv"), index=False)
    failure_summary.to_csv(os.path.join(out, "failure_mode_summary.csv"), index=False)
    findings.to_csv(os.path.join(out, "stage16_key_findings.csv"), index=False)

    if not args.no_plots:
        ra.make_plots(by_thr, by_day, model_abl, calib_abl, sens, windowability, rangebin_rob,
                      os.path.join(out, "plots"))

    report = ra.write_report(out, dates, thresholds, args.include_high_thresholds, args.best_model,
                             inventory, by_day, by_thr, model_abl, calib_abl, sens, windowability,
                             rangebin_rob, failure_summary, findings, stage15_present)
    ra.run_gate(out, inventory, by_thr, model_abl, calib_abl, args.include_high_thresholds)

    print(f"\nreport: {os.path.abspath(report)}")
    print("\nkey findings:")
    for _, r in findings.iterrows():
        print(f"  [{r['finding_id']}] {r['finding']} -> {r['value']}")
    return {"inventory": inventory, "by_thr": by_thr, "by_day": by_day, "model_abl": model_abl,
            "calib_abl": calib_abl, "findings": findings}


# =============================================================================
# Self-test
# =============================================================================

def _w(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _make_mini(root):
    thr = [-5.0, 0.0, 9.0]
    # Stage 08 metrics (with a high threshold for the windowability caveat case)
    _w(os.path.join(root, "s08", "kalman_metrics_by_day.csv"), pd.DataFrame({
        "date": "2022-06-06", "threshold_db": thr, "tracks_confirmed": [3000, 3100, 2500],
        "true_tracks": [2000, 2100, 2400], "false_tracks": [200, 60, 0]}))
    _w(os.path.join(root, "s09", "physics_metrics_by_threshold.csv"), pd.DataFrame({
        "threshold_db": thr, "stage08_true_tracks": [2000, 2100, 2400],
        "stage08_false_tracks": [200, 60, 0]}))

    # Stage 12 metrics: two models, track_purity, mlp best
    rows = []
    for model, kt, kf in [("mlp_dae", [1970, 2080, 2350], [10, 4, 0]),
                          ("gru_ae", [1960, 2070, 2340], [12, 5, 0])]:
        for i, t in enumerate(thr):
            st, sf = [2000, 2100, 2400][i], [200, 60, 0][i]
            rows.append({"model": model, "threshold_db": t, "stage08_confirmed_tracks": st + sf,
                         "stage08_true_tracks": st, "stage08_false_tracks": sf,
                         "stage12_kept_tracks": kt[i] + kf[i], "stage12_kept_true_tracks": kt[i],
                         "stage12_kept_false_tracks": kf[i],
                         "true_track_retention": kt[i] / st,
                         "false_track_retention": (kf[i] / sf) if sf else np.nan,
                         "false_track_reduction": (1 - kf[i] / sf) if sf else np.nan,
                         "precision_before": st / (st + sf),
                         "precision_after": kt[i] / (kt[i] + kf[i]),
                         "median_score_true_tracks": 0.95, "median_score_false_tracks": 0.0,
                         "calibration_mode": "track_purity"})
    _w(os.path.join(root, "s12", "sequence_metrics_by_model_threshold.csv"), pd.DataFrame(rows))

    # Stage 12 filter sweep (mlp)
    sw = []
    for t in thr:
        st, sf = {-5.0: (2000, 200), 0.0: (2100, 60), 9.0: (2400, 0)}[t]
        for stp in [0.3, 0.5, 0.7]:
            kt = int(st * (1.0 - 0.05 * stp))
            kf = int(sf * (1.0 - stp))
            sw.append({"model": "mlp_dae", "threshold_db": t, "score_threshold": stp,
                       "kept_tracks": kt + kf, "kept_true_tracks": kt, "kept_false_tracks": kf,
                       "true_track_retention": kt / st,
                       "false_track_retention": (kf / sf) if sf else np.nan,
                       "false_track_reduction": (1 - kf / sf) if sf else np.nan,
                       "precision_after": kt / (kt + kf)})
    _w(os.path.join(root, "s12", "sequence_filter_sweep.csv"), pd.DataFrame(sw))

    # calibration comparison: clean_truth under-retains vs track_purity
    cc = []
    for model in ["mlp_dae", "gru_ae"]:
        for t in [-5.0, 0.0]:
            st = {-5.0: 2000, 0.0: 2100}[t]
            for mode, ret in [("clean_truth", 0.11), ("track_purity", 0.98)]:
                cc.append({"model": model, "threshold_db": t, "calibration_mode": mode,
                           "score_threshold": 0.5, "kept_tracks": int(st * ret),
                           "kept_true_tracks": int(st * ret), "kept_false_tracks": 0,
                           "true_track_retention": ret, "false_track_reduction": 1.0,
                           "precision_after": 1.0,
                           "median_score_true_tracks": 0.95 if mode == "track_purity" else 0.1,
                           "median_score_false_tracks": 0.0})
    _w(os.path.join(root, "s12", "calibration", "sequence_calibration_comparison.csv"),
       pd.DataFrame(cc))

    # range-bin metrics
    _w(os.path.join(root, "s12", "sequence_range_bin_metrics.csv"), pd.DataFrame([
        {"model": "mlp_dae", "threshold_db": -5.0, "range_bin": "0-50 km", "stage08_tracks": 1500,
         "stage08_true_tracks": 1400, "stage08_false_tracks": 100, "stage12_kept_tracks": 1390,
         "stage12_kept_true_tracks": 1385, "stage12_kept_false_tracks": 5,
         "true_track_retention": 0.989, "false_track_reduction": 0.95, "precision_after": 0.996,
         "median_sequence_score": 1.0},
        {"model": "mlp_dae", "threshold_db": -5.0, "range_bin": "50-100 km", "stage08_tracks": 700,
         "stage08_true_tracks": 600, "stage08_false_tracks": 100, "stage12_kept_tracks": 560,
         "stage12_kept_true_tracks": 555, "stage12_kept_false_tracks": 5,
         "true_track_retention": 0.925, "false_track_reduction": 0.95, "precision_after": 0.991,
         "median_sequence_score": 0.88}]))

    # Stage 14 best + failures
    _w(os.path.join(root, "s14", "best_method_by_threshold.csv"), pd.DataFrame({
        "threshold_db": [-5.0, 0.0], "best_method_id": "stage12_mlp_dae_track_calibrated"}))
    _w(os.path.join(root, "s14", "failure_case_candidates.csv"), pd.DataFrame({
        "case_type": ["false_survives_s12", "true_rejected_s12"], "date": "2022-06-06",
        "threshold_db": [-5.0, -5.0], "track_id": [1, 2], "method_context": "s12", "reason": "x",
        "score_stage09": [0.6, 0.2], "score_stage12": [0.7, 0.1], "score_stage13": [0.5, 0.1],
        "target_fraction": [0.0, 0.9], "purity": [0.0, 0.9], "position_rmse_m": [np.nan, 50.0],
        "median_range_m": [1600.0, 4000.0]}))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_mini(tmp)
        args = parse_args()
        args.stage08_dir = os.path.join(tmp, "s08")
        args.stage09_dir = os.path.join(tmp, "s09")
        args.stage12_dir = os.path.join(tmp, "s12")
        args.stage14_dir = os.path.join(tmp, "s14")
        args.stage15_dir = os.path.join(tmp, "s15_missing")
        args.tracks_dir = os.path.join(tmp, "tracks_missing")
        args.output_dir = os.path.join(tmp, "out")
        args.threshold_db = [-5, 0]
        args.include_high_thresholds = True   # exercise the windowability caveat path
        args.overwrite = True
        out = run(args)

        o = args.output_dir
        for f_ in ["input_coverage_inventory.csv", "robustness_by_threshold.csv",
                   "calibration_ablation.csv", "stage16_key_findings.csv",
                   "stage16_robustness_report.md"]:
            assert os.path.exists(os.path.join(o, f_)), f"missing {f_}"
        assert len(out["calib_abl"]) and \
            (out["calib_abl"]["calibration_mode"] == "clean_truth").any(), "calib ablation missing"
        agg = out["model_abl"][out["model_abl"]["row_type"] == "aggregate"]
        assert agg.sort_values("overall_rank").iloc[0]["model"] == "mlp_dae", \
            "expected mlp_dae to rank best in the toy data"

        text = open(os.path.join(o, "stage16_robustness_report.md")).read()
        for needle in ["Stage 16 adds no new model", "Stage 12.5", "track-purity calibration",
                       "interpretable fallback", "Recommended next stage", "windowability"]:
            assert needle in text, f"report missing: {needle!r}"

        print("\nmodel ablation (aggregate):")
        print(agg[["model", "mean_true_retention", "mean_false_reduction",
                   "overall_rank"]].to_string(index=False))
    print("\nStage 16 robustness self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)
    print("\n16_robustness_ablation completed successfully.")


if __name__ == "__main__":
    main()
