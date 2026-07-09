"""Entry point: stage 17 -- four-day validation of the selected stage-12.5 method.

Consolidates per-day stage-08/09/12.5 results across the four ADS-B days and
reports whether the stage-12.5 sequence-prior conclusion (high false-track
reduction at high true-track retention) holds beyond a single day. Adds NO new
model and retrains nothing; only the `--run-missing-*` flags may call the
existing stage-08/09/12 scripts to generate missing day/threshold outputs.

Usage:
    python scripts/17_four_day_validation.py --date 2022-06-06 2022-06-13 \
        2022-06-20 2022-06-27 --threshold-db -5 0 3 6 --overwrite
    python scripts/17_four_day_validation.py --self-test
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
from utils import four_day_validation as fdv

DEFAULT_DAYS = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 17 four-day validation.")
    p.add_argument("--stage08-dir", default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline"))
    p.add_argument("--stage09-dir", default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    p.add_argument("--stage12-dir", default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    p.add_argument("--stage14-dir", default=os.path.join(REPO_ROOT, "reports", "stage14_method_benchmark"))
    p.add_argument("--tracks-dir", default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--detections-dir", default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    p.add_argument("--truth-dir", default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    p.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "reports", "stage17_four_day_validation"))
    p.add_argument("--date", nargs="+", default=DEFAULT_DAYS)
    p.add_argument("--threshold-db", type=float, nargs="+", default=[-5, 0, 3, 6])
    p.add_argument("--include-high-thresholds", action="store_true")
    p.add_argument("--primary-models", default="mlp_dae,gru_ae")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--calibration-mode", default="track_purity")
    p.add_argument("--run-missing-stage08", action="store_true")
    p.add_argument("--run-missing-stage09", action="store_true")
    p.add_argument("--run-missing-stage12", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real reports needed) and exit.")
    return p.parse_args()


def _read(path):
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def detect_base_date(stage12_dir):
    df = _read(os.path.join(stage12_dir, "sequence_track_scores.csv"))
    if df is not None and "date" in df.columns and len(df):
        return sorted(df["date"].astype(str).unique())[0]
    return "2022-06-06"


def load_stage08_ctx(stage08_dir, dates, thresholds):
    df = _read(os.path.join(stage08_dir, "kalman_metrics_by_day.csv"))
    if df is None:
        return pd.DataFrame()
    df = df[df["date"].astype(str).isin([str(d) for d in dates])
            & df["threshold_db"].isin(thresholds)]
    return pd.DataFrame({
        "date": df["date"].astype(str), "threshold_db": df["threshold_db"],
        "stage08_true_tracks": df["true_tracks"], "stage08_false_tracks": df["false_tracks"],
        "stage08_confirmed_tracks": df.get("tracks_confirmed", np.nan),
        "track_detection_rate": df.get("track_detection_rate", np.nan),
        "target_assignment_rate": df.get("target_assignment_rate", np.nan),
        "clutter_assignment_rate": df.get("clutter_assignment_rate", np.nan),
        "mean_position_rmse_m": df.get("mean_position_rmse_m", np.nan),
        "median_position_rmse_m": df.get("median_position_rmse_m", np.nan)}).reset_index(drop=True)


def _date_keyed_or_base(main_df, base_date, generated_lookup, dates):
    """Yield (date, per-day df) using a date column if present, a base-date fallback
    for the main file, and per-day generated files otherwise."""
    frames = []
    covered = set()
    if main_df is not None and len(main_df):
        if "date" in main_df.columns:
            for d in dates:
                sub = main_df[main_df["date"].astype(str) == str(d)]
                if len(sub):
                    frames.append(sub.assign(date=str(d)))
                    covered.add(str(d))
        else:
            frames.append(main_df.assign(date=str(base_date)))
            covered.add(str(base_date))
    for d in dates:
        if str(d) in covered:
            continue
        gen = generated_lookup(d)
        if gen is not None and len(gen):
            frames.append(gen.assign(date=str(d)))
            covered.add(str(d))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_stage09_ctx(stage09_dir, output_dir, dates, base_date, thresholds):
    main = _read(os.path.join(stage09_dir, "physics_metrics_by_threshold.csv"))
    gen = lambda d: _read(os.path.join(output_dir, "generated", "stage09", str(d),
                                       "physics_metrics_by_threshold.csv"))
    df = _date_keyed_or_base(main, base_date, gen, dates)
    if df.empty:
        return pd.DataFrame()
    df = df[df["threshold_db"].isin(thresholds)]
    return pd.DataFrame({
        "date": df["date"].astype(str), "threshold_db": df["threshold_db"],
        "stage09_true_retention": df["true_track_retention"],
        "stage09_false_reduction": df["false_track_reduction"],
        "stage09_precision_after": df["precision_after"],
        "stage09_kept_true_tracks": df["stage09_kept_true_tracks"],
        "stage09_kept_false_tracks": df["stage09_kept_false_tracks"]}).reset_index(drop=True)


def load_windowability(stage12_dir, output_dir, stage08_ctx, dates, base_date, thresholds,
                       primary_model):
    def scores_for(date):
        if str(date) == str(base_date):
            return _read(os.path.join(stage12_dir, "sequence_track_scores.csv"))
        return _read(os.path.join(output_dir, "generated", "stage12", str(date),
                                  "sequence_track_scores.csv"))
    rows = []
    for date in dates:
        sc = scores_for(date)
        for thr in thresholds:
            confirmed = np.nan
            if len(stage08_ctx):
                m = stage08_ctx[(stage08_ctx["date"] == str(date))
                                & np.isclose(stage08_ctx["threshold_db"], thr)]
                if len(m):
                    confirmed = float(m["stage08_confirmed_tracks"].iloc[0])
            if sc is not None and "n_windows" in sc.columns:
                s = sc[np.isclose(sc["threshold_db"], thr)]
                if "model" in s.columns:
                    s = s[s["model"] == primary_model]
                if len(s):
                    win = s[s["n_windows"] > 0]
                    st = int((s["is_true_track"] == True).sum())
                    sf = int((s["is_true_track"] == False).sum())
                    wt = int((win["is_true_track"] == True).sum())
                    wf = int((win["is_true_track"] == False).sum())
                    rows.append({"date": str(date), "threshold_db": thr,
                                 "stage08_confirmed_tracks": confirmed,
                                 "stage08_true_tracks": st, "stage08_false_tracks": sf,
                                 "windowable_tracks": len(win), "windowable_true_tracks": wt,
                                 "windowable_false_tracks": wf,
                                 "windowable_fraction_all": len(win) / len(s) if len(s) else np.nan,
                                 "windowable_fraction_true": wt / st if st else np.nan,
                                 "windowable_fraction_false": wf / sf if sf else np.nan,
                                 "notes": ""})
    return pd.DataFrame(rows)


def load_stage12_metrics(stage12_dir, output_dir, dates, base_date, models, thresholds,
                         calibration_mode, score_threshold, windowability):
    main = _read(os.path.join(stage12_dir, "sequence_metrics_by_model_threshold.csv"))
    gen = lambda d: _read(os.path.join(output_dir, "generated", "stage12", str(d),
                                       "sequence_metrics_by_model_threshold.csv"))
    df = _date_keyed_or_base(main, base_date, gen, dates)
    if df.empty:
        return pd.DataFrame()
    df = df[df["threshold_db"].isin(thresholds) & df["model"].isin(models)]
    if "calibration_mode" in df.columns:
        df = df[df["calibration_mode"] == calibration_mode]
    win_lookup = {}
    if windowability is not None and len(windowability):
        for _, w in windowability.iterrows():
            win_lookup[(w["date"], round(float(w["threshold_db"]), 6))] = w
    rows = []
    for _, r in df.iterrows():
        w = win_lookup.get((str(r["date"]), round(float(r["threshold_db"]), 6)))
        rows.append({
            "date": str(r["date"]), "threshold_db": r["threshold_db"], "model": r["model"],
            "calibration_mode": r.get("calibration_mode", calibration_mode),
            "score_threshold": score_threshold,
            "stage08_true_tracks": r["stage08_true_tracks"],
            "stage08_false_tracks": r["stage08_false_tracks"],
            "stage12_kept_true_tracks": r["stage12_kept_true_tracks"],
            "stage12_kept_false_tracks": r["stage12_kept_false_tracks"],
            "true_track_retention": r["true_track_retention"],
            "false_track_reduction": r["false_track_reduction"],
            "precision_before": r.get("precision_before", np.nan),
            "precision_after": r["precision_after"],
            "median_score_true_tracks": r.get("median_score_true_tracks", np.nan),
            "median_score_false_tracks": r.get("median_score_false_tracks", np.nan),
            "windowable_tracks": (w["windowable_tracks"] if w is not None else np.nan),
            "windowable_true_tracks": (w["windowable_true_tracks"] if w is not None else np.nan),
            "windowable_false_tracks": (w["windowable_false_tracks"] if w is not None else np.nan),
            "notes": ""})
    return pd.DataFrame(rows)


def run_missing(args, availability):
    """Run requested missing stages via the existing scripts. Returns run-status rows."""
    status = []
    thr = sorted(availability["threshold_db"].unique())
    thr_str = [f"{t:g}" for t in thr]

    def missing_dates(col):
        g = availability.groupby("date")[col].all()
        return [d for d, ok in g.items() if not ok]

    def run(cmd, stage, date):
        print(f"[run-missing:{stage} {date}] " + " ".join(cmd), flush=True)
        try:
            r = subprocess.run(cmd, cwd=REPO_ROOT)
            status.append({"stage": stage, "date": date, "exit_code": r.returncode,
                           "ok": r.returncode == 0})
        except Exception as exc:
            print(f"  failed: {exc}")
            status.append({"stage": stage, "date": date, "exit_code": -1, "ok": False})

    if args.run_missing_stage08:
        for d in missing_dates("stage08_metrics_available"):
            run(["python", "scripts/08_run_kalman_baseline.py", "--detections-dir",
                 args.detections_dir, "--truth-dir", args.truth_dir, "--tracks-dir",
                 args.tracks_dir, "--report-dir", args.stage08_dir, "--threshold-db", *thr_str,
                 "--date", d, "--overwrite"], "stage08", d)
    if args.run_missing_stage09:
        for d in missing_dates("stage09_metrics_available"):
            out = os.path.join(args.output_dir, "generated", "stage09", d)
            run(["python", "scripts/09_score_tracks_physics.py", "--tracks-dir", args.tracks_dir,
                 "--detections-dir", args.detections_dir, "--report-dir", out,
                 "--threshold-db", *thr_str, "--date", d, "--overwrite"], "stage09", d)
    if args.run_missing_stage12:
        for d in [x for x in availability.groupby("date")
                  .agg(a=("stage12_mlp_available", "all"),
                       b=("stage12_gru_available", "all")).query("not (a and b)").index]:
            out = os.path.join(args.output_dir, "generated", "stage12", d)
            run(["python", "scripts/12_score_tracks_sequence_prior.py", "--tracks-dir",
                 args.tracks_dir, "--models-dir", args.models_dir, "--report-dir", out,
                 "--threshold-db", *thr_str, "--date", d, "--calibration-mode",
                 args.calibration_mode, "--calibration-threshold-db", "3", "6", "9", "12",
                 "--score-threshold", str(args.score_threshold), "--overwrite"], "stage12", d)
    return pd.DataFrame(status)


def run(args) -> dict:
    out = args.output_dir
    inv_path = os.path.join(out, "input_availability.csv")
    if os.path.exists(inv_path) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {inv_path}")
    os.makedirs(out, exist_ok=True)

    dates = [str(d) for d in args.date]
    thresholds = list(args.threshold_db)
    audit_thresholds = sorted(set(thresholds) | (fdv.HIGH_THRESHOLDS
                                                 if args.include_high_thresholds else set()))
    models = [m.strip() for m in args.primary_models.split(",") if m.strip()]
    base_date = detect_base_date(args.stage12_dir)
    checkpoints_ok = all(os.path.exists(os.path.join(args.models_dir, f"{m}.pt")) for m in models)
    track_files = {d: any(d in f for f in os.listdir(args.tracks_dir))
                   for d in dates} if os.path.isdir(args.tracks_dir) else {d: False for d in dates}

    def assemble():
        s08 = load_stage08_ctx(args.stage08_dir, dates, audit_thresholds)
        s09 = load_stage09_ctx(args.stage09_dir, out, dates, base_date, audit_thresholds)
        win = load_windowability(args.stage12_dir, out, s08, dates, base_date, audit_thresholds,
                                 models[0])
        s12 = load_stage12_metrics(args.stage12_dir, out, dates, base_date, models,
                                   audit_thresholds, args.calibration_mode, args.score_threshold,
                                   win)
        return s08, s09, s12, win

    s08_ctx, s09_ctx, s12_metrics, windowability = assemble()

    availability = fdv.build_availability(dates, thresholds, s08_ctx, s09_ctx, s12_metrics,
                                          track_files, checkpoints_ok, models)
    availability.to_csv(inv_path, index=False)
    run_plan = fdv.build_run_plan(availability, thresholds, args.tracks_dir, args.models_dir,
                                  args.stage08_dir, args.stage09_dir, args.stage12_dir,
                                  args.detections_dir, args.truth_dir, args.calibration_mode)

    print("input availability by date:")
    print(availability.groupby("date")["status"].value_counts().to_string())

    if args.dry_run:
        run_plan.to_csv(os.path.join(out, "run_plan.csv"), index=False)
        print("\n--dry-run: wrote input_availability.csv and run_plan.csv only.")
        return {"availability": availability, "run_plan": run_plan}

    if args.run_missing_stage08 or args.run_missing_stage09 or args.run_missing_stage12:
        status = run_missing(args, availability)
        if len(status):
            print("\nrun-missing status:")
            print(status.to_string(index=False))
        s08_ctx, s09_ctx, s12_metrics, windowability = assemble()  # reload after generation
        availability = fdv.build_availability(dates, thresholds, s08_ctx, s09_ctx, s12_metrics,
                                              track_files, checkpoints_ok, models)
        availability.to_csv(inv_path, index=False)
        run_plan = fdv.build_run_plan(availability, thresholds, args.tracks_dir, args.models_dir,
                                      args.stage08_dir, args.stage09_dir, args.stage12_dir,
                                      args.detections_dir, args.truth_dir, args.calibration_mode)
        if len(status):
            rp = run_plan.copy()
            run_plan = pd.concat([rp, status.rename(columns={"stage": "missing_stage"})
                                  .assign(date=status["date"], threshold_db="-", required=True,
                                          command="(executed)", will_run=True,
                                          notes="exit=" + status["exit_code"].astype(str))],
                                 ignore_index=True)
    run_plan.to_csv(os.path.join(out, "run_plan.csv"), index=False)

    dates_present = sorted(availability.groupby("date")["status"]
                           .apply(lambda s: (s == "complete").all())
                           .loc[lambda x: x].index.tolist())
    all_days_present = set(DEFAULT_DAYS).issubset(set(dates_present))

    s08_ctx.to_csv(os.path.join(out, "four_day_stage08_context.csv"), index=False)
    s09_ctx.to_csv(os.path.join(out, "four_day_stage09_context.csv"), index=False)
    s12_metrics.to_csv(os.path.join(out, "four_day_stage12_metrics.csv"), index=False)

    by_day = fdv.summary_by_day(s12_metrics, thresholds)
    by_thr = fdv.summary_by_threshold(s12_metrics, thresholds)
    overall, best_model = fdv.summary_overall(s12_metrics, thresholds, all_days_present)
    model_cmp = fdv.model_comparison(s12_metrics, thresholds, models)
    fallback = fdv.interpretable_fallback(s09_ctx, s12_metrics, thresholds, best_model)
    win_audit = fdv.windowability_audit(windowability, thresholds, args.include_high_thresholds)
    failures = fdv.failure_rollup(_read(os.path.join(args.stage14_dir,
                                                     "failure_case_candidates.csv")), best_model)
    findings = fdv.key_findings(overall, by_thr, model_cmp, fallback, win_audit, dates_present,
                                all_days_present, best_model, args.include_high_thresholds)

    by_day.to_csv(os.path.join(out, "four_day_summary_by_day.csv"), index=False)
    by_thr.to_csv(os.path.join(out, "four_day_summary_by_threshold.csv"), index=False)
    overall.to_csv(os.path.join(out, "four_day_summary_overall.csv"), index=False)
    model_cmp.to_csv(os.path.join(out, "model_comparison_mlp_vs_gru.csv"), index=False)
    fallback.to_csv(os.path.join(out, "interpretable_fallback_comparison.csv"), index=False)
    win_audit.to_csv(os.path.join(out, "windowability_four_day_audit.csv"), index=False)
    failures.to_csv(os.path.join(out, "failure_case_rollup.csv"), index=False)
    findings.to_csv(os.path.join(out, "stage17_key_findings.csv"), index=False)

    if not args.no_plots:
        fdv.make_plots(by_day, by_thr, s08_ctx, model_cmp, fallback, win_audit, best_model,
                       models, os.path.join(out, "plots"))

    report = fdv.write_report(out, dates_present, all_days_present, thresholds,
                              args.include_high_thresholds, best_model, availability, run_plan,
                              s08_ctx, by_day, by_thr, overall, model_cmp, fallback, win_audit,
                              failures, findings)
    fdv.run_gate(out, availability, run_plan, s12_metrics, all_days_present)

    print(f"\nreport: {os.path.abspath(report)}")
    print(f"days with complete results: {dates_present} (all four = {all_days_present})")
    print(f"best model: {best_model}")
    print("\nkey findings:")
    for _, r in findings.iterrows():
        print(f"  [{r['finding_id']}] {r['finding']} -> {r['value']}")
    return {"availability": availability, "run_plan": run_plan, "by_day": by_day,
            "overall": overall, "findings": findings, "all_days_present": all_days_present}


# =============================================================================
# Self-test
# =============================================================================

def _w(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _make_mini(root):
    dates = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
    thr = [-5.0, 0.0]
    # Stage 08 metrics (date-keyed) -- all four days
    s08 = []
    for d in dates:
        for t in thr:
            st, sf = (2000, 200) if t == -5 else (2100, 60)
            s08.append({"date": d, "threshold_db": t, "tracks_confirmed": st + sf + 800,
                        "true_tracks": st, "false_tracks": sf, "track_detection_rate": 0.98,
                        "target_assignment_rate": 0.95, "clutter_assignment_rate": 0.3,
                        "mean_position_rmse_m": 120.0, "median_position_rmse_m": 90.0})
    _w(os.path.join(root, "s08", "kalman_metrics_by_day.csv"), pd.DataFrame(s08))

    # Stage 09 metrics WITH a date column (four days); stage12 beats it
    s09 = []
    for d in dates:
        for t in thr:
            st, sf = (2000, 200) if t == -5 else (2100, 60)
            kt, kf = int(st * 0.97), int(sf * 0.4)   # stage09: 60% false reduction
            s09.append({"date": d, "threshold_db": t, "stage08_confirmed_tracks": st + sf,
                        "stage08_true_tracks": st, "stage08_false_tracks": sf,
                        "stage09_kept_tracks": kt + kf, "stage09_kept_true_tracks": kt,
                        "stage09_kept_false_tracks": kf, "true_track_retention": kt / st,
                        "false_track_retention": kf / sf, "false_track_reduction": 1 - kf / sf,
                        "precision_before": st / (st + sf), "precision_after": kt / (kt + kf)})
    _w(os.path.join(root, "s09", "physics_metrics_by_threshold.csv"), pd.DataFrame(s09))

    # Stage 12 metrics WITH date column; omit 2022-06-27 to exercise a missing case
    s12 = []
    for d in dates[:-1]:                              # 3 of 4 days present
        for model, red in [("mlp_dae", 0.02), ("gru_ae", 0.03)]:
            for t in thr:
                st, sf = (2000, 200) if t == -5 else (2100, 60)
                kt, kf = int(st * 0.98), int(sf * red)   # stage12: ~97-98% false reduction
                s12.append({"date": d, "model": model, "threshold_db": t,
                            "calibration_mode": "track_purity",
                            "stage08_confirmed_tracks": st + sf, "stage08_true_tracks": st,
                            "stage08_false_tracks": sf, "stage12_kept_tracks": kt + kf,
                            "stage12_kept_true_tracks": kt, "stage12_kept_false_tracks": kf,
                            "true_track_retention": kt / st, "false_track_retention": kf / sf,
                            "false_track_reduction": 1 - kf / sf, "precision_before": st / (st + sf),
                            "precision_after": kt / (kt + kf), "median_score_true_tracks": 0.95,
                            "median_score_false_tracks": 0.0})
    _w(os.path.join(root, "s12", "sequence_metrics_by_model_threshold.csv"), pd.DataFrame(s12))
    _w(os.path.join(root, "s12", "sequence_track_scores.csv"),
       pd.DataFrame({"date": ["2022-06-06"], "threshold_db": [-5.0], "model": ["mlp_dae"],
                     "track_id": [1], "n_windows": [4], "is_true_track": [True]}))
    _w(os.path.join(root, "s14", "failure_case_candidates.csv"),
       pd.DataFrame({"case_type": ["false_survives_s12", "true_rejected_s12"],
                     "date": "2022-06-06", "threshold_db": [-5.0, -5.0], "track_id": [1, 2],
                     "score_stage12": [0.7, 0.1], "median_range_m": [1600.0, 4000.0]}))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_mini(tmp)
        args = parse_args()
        args.stage08_dir = os.path.join(tmp, "s08")
        args.stage09_dir = os.path.join(tmp, "s09")
        args.stage12_dir = os.path.join(tmp, "s12")
        args.stage14_dir = os.path.join(tmp, "s14")
        args.tracks_dir = os.path.join(tmp, "tracks_missing")
        args.models_dir = os.path.join(tmp, "models_missing")
        args.output_dir = os.path.join(tmp, "out")
        args.date = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
        args.threshold_db = [-5, 0]
        args.overwrite = True
        out = run(args)

        o = args.output_dir
        for f_ in ["input_availability.csv", "run_plan.csv", "four_day_summary_by_day.csv",
                   "four_day_summary_overall.csv", "stage17_key_findings.csv",
                   "stage17_four_day_validation_report.md"]:
            assert os.path.exists(os.path.join(o, f_)), f"missing {f_}"
        # the omitted 2022-06-27 stage-12 must appear in the run plan
        rp = pd.read_csv(os.path.join(o, "run_plan.csv"))
        assert ((rp["missing_stage"] == "stage12") & (rp["date"] == "2022-06-27")).any(), \
            "run plan should capture the missing 2022-06-27 stage-12 output"
        assert not out["all_days_present"], "not all four days should be complete in the toy data"

        text = open(os.path.join(o, "stage17_four_day_validation_report.md")).read()
        for needle in ["Stage 17 adds no new model", "four-day", "Stage 12.5",
                       "Stage 09 interpretable fallback", "windowability",
                       "Recommended next stage", "still open"]:
            assert needle in text, f"report missing: {needle!r}"
        print("\noverall summary:")
        print(out["overall"][["model", "mean_true_retention", "mean_false_reduction",
                              "recommended"]].to_string(index=False))
    print("\nStage 17 four-day validation self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)
    print("\n17_four_day_validation completed successfully.")


if __name__ == "__main__":
    main()
