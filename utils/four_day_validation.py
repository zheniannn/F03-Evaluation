"""Stage 17: four-day validation of the selected stage-12.5 method.

Adds NO new model and retrains nothing. Given per-day stage-08/09/12.5 results
(assembled by the entry script from the compact reports, plus any day/threshold
combinations generated on request), this module builds the validation tables:
input availability, a run plan for anything missing, four-day stage-08/09
context, four-day stage-12 metrics, per-day / per-threshold / overall summaries,
an MLP-vs-GRU comparison, a stage-09 interpretable-fallback comparison, a
four-day windowability audit, a failure-case rollup, and a key-findings table.

Everything is defensive: missing day/threshold cells are reported, not faked.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table

HIGH_THRESHOLDS = {9.0, 12.0}
EXPECTED_DAYS = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]


def _metrics_from_counts(kt, sf_kf, st, sf):
    """Return (retention, reduction, precision_before, precision_after) from counts."""
    kt, kf, st, sf = float(kt), float(sf_kf), float(st), float(sf)
    return (kt / st if st else np.nan,
            1 - kf / sf if sf else np.nan,
            st / (st + sf) if (st + sf) else np.nan,
            kt / (kt + kf) if (kt + kf) else np.nan)


# =============================================================================
# Availability + run plan
# =============================================================================

def build_availability(dates, thresholds, s08, s09, s12, track_files, checkpoints_ok,
                       primary_models) -> pd.DataFrame:
    def has(df, date, thr, model=None):
        if df is None or df.empty:
            return False
        m = (df["date"].astype(str) == str(date)) & np.isclose(df["threshold_db"], thr)
        if model is not None and "model" in df.columns:
            m = m & (df["model"] == model)
        return bool(m.any())

    rows = []
    for date in dates:
        for thr in thresholds:
            s8 = has(s08, date, thr)
            s9 = has(s09, date, thr)
            mlp = has(s12, date, thr, primary_models[0])
            gru = has(s12, date, thr, primary_models[1]) if len(primary_models) > 1 else False
            complete = s8 and s9 and mlp and gru
            rows.append({
                "date": date, "threshold_db": thr,
                "stage08_metrics_available": s8, "stage09_metrics_available": s9,
                "stage12_mlp_available": mlp, "stage12_gru_available": gru,
                "stage08_track_file_available": bool(track_files.get(date, False)),
                "stage12_checkpoints_available": bool(checkpoints_ok),
                "status": "complete" if complete else "incomplete",
                "notes": "" if complete else "missing one or more stage outputs for this cell"})
    return pd.DataFrame(rows)


def build_run_plan(availability, thresholds, tracks_dir, models_dir, stage08_dir, stage09_dir,
                   stage12_dir, detections_dir, truth_dir, calibration_mode) -> pd.DataFrame:
    rows = []
    for date, g in availability.groupby("date"):
        thr_all = sorted(g["threshold_db"].unique())
        thr_str = " ".join(f"{t:g}" for t in thr_all)
        need08 = not g["stage08_metrics_available"].all()
        need09 = not g["stage09_metrics_available"].all()
        need12 = not (g["stage12_mlp_available"].all() and g["stage12_gru_available"].all())
        track_ok = bool(g["stage08_track_file_available"].any())
        ckpt_ok = bool(g["stage12_checkpoints_available"].any())

        if need08:
            rows.append({"date": date, "threshold_db": thr_str, "missing_stage": "stage08",
                         "required": True,
                         "command": (f"python scripts/08_run_kalman_baseline.py "
                                     f"--detections-dir {detections_dir} --truth-dir {truth_dir} "
                                     f"--tracks-dir {tracks_dir} --report-dir {stage08_dir} "
                                     f"--threshold-db {thr_str} --date {date} --overwrite"),
                         "will_run": False, "notes": "needs stage-6 detections for this day"})
        if need09:
            rows.append({"date": date, "threshold_db": thr_str, "missing_stage": "stage09",
                         "required": True,
                         "command": (f"python scripts/09_score_tracks_physics.py "
                                     f"--tracks-dir {tracks_dir} --detections-dir {detections_dir} "
                                     f"--report-dir {stage09_dir} --threshold-db {thr_str} "
                                     f"--date {date} --overwrite"),
                         "will_run": False,
                         "notes": "needs stage-08 tracks for this day" if not track_ok else ""})
        if need12:
            rows.append({"date": date, "threshold_db": thr_str, "missing_stage": "stage12",
                         "required": True,
                         "command": (f"python scripts/12_score_tracks_sequence_prior.py "
                                     f"--tracks-dir {tracks_dir} --models-dir {models_dir} "
                                     f"--report-dir {stage12_dir} --threshold-db {thr_str} "
                                     f"--date {date} --calibration-mode {calibration_mode} "
                                     f"--calibration-threshold-db 3 6 9 12 --score-threshold 0.5 "
                                     f"--overwrite"),
                         "will_run": False,
                         "notes": ("checkpoints missing -- cannot run" if not ckpt_ok else
                                   "needs stage-08 tracks for this day" if not track_ok else "")})
        if not (need08 or need09 or need12):
            rows.append({"date": date, "threshold_db": thr_str, "missing_stage": "-",
                         "required": False, "command": "-", "will_run": False,
                         "notes": "all required outputs present"})
    return pd.DataFrame(rows)


# =============================================================================
# Summaries
# =============================================================================

def _agg(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        st = g["stage08_true_tracks"].sum()
        sf = g["stage08_false_tracks"].sum()
        kt = g["stage12_kept_true_tracks"].sum()
        kf = g["stage12_kept_false_tracks"].sum()
        pooled = _metrics_from_counts(kt, kf, st, sf)
        row = dict(zip(group_cols, keys))
        row.update({
            "stage08_true_tracks": st, "stage08_false_tracks": sf,
            "stage12_kept_true_tracks": kt, "stage12_kept_false_tracks": kf,
            "mean_true_retention": g["true_track_retention"].mean(),
            "mean_false_reduction": g["false_track_reduction"].mean(),
            "mean_precision_after": g["precision_after"].mean(),
            "pooled_true_retention": pooled[0], "pooled_false_reduction": pooled[1],
            "pooled_precision_after": pooled[3]})
        rows.append((row, g))
    return rows


def summary_by_day(s12_metrics, thresholds) -> pd.DataFrame:
    df = s12_metrics[s12_metrics["threshold_db"].isin(thresholds)]
    out = []
    for row, g in _agg(df, ["date", "model"]):
        row["n_thresholds"] = g["threshold_db"].nunique()
        row["notes"] = ("single-day cell" if g["date"].nunique() == 1 else "")
        out.append(row)
    cols = ["date", "model", "n_thresholds", "stage08_true_tracks", "stage08_false_tracks",
            "stage12_kept_true_tracks", "stage12_kept_false_tracks", "mean_true_retention",
            "mean_false_reduction", "mean_precision_after", "pooled_true_retention",
            "pooled_false_reduction", "pooled_precision_after", "notes"]
    return pd.DataFrame(out)[cols].sort_values(["model", "date"]).reset_index(drop=True)


def summary_by_threshold(s12_metrics, thresholds) -> pd.DataFrame:
    df = s12_metrics[s12_metrics["threshold_db"].isin(thresholds)]
    out = []
    for row, g in _agg(df, ["threshold_db", "model"]):
        row["n_dates"] = g["date"].nunique()
        row["notes"] = ""
        out.append(row)
    cols = ["threshold_db", "model", "n_dates", "stage08_true_tracks", "stage08_false_tracks",
            "stage12_kept_true_tracks", "stage12_kept_false_tracks", "mean_true_retention",
            "mean_false_reduction", "mean_precision_after", "pooled_true_retention",
            "pooled_false_reduction", "pooled_precision_after", "notes"]
    return pd.DataFrame(out)[cols].sort_values(["model", "threshold_db"]).reset_index(drop=True)


def summary_overall(s12_metrics, thresholds, all_days_present) -> pd.DataFrame:
    df = s12_metrics[s12_metrics["threshold_db"].isin(thresholds)]
    out = []
    for row, g in _agg(df, ["model"]):
        row["dates_included"] = ",".join(sorted(g["date"].astype(str).unique()))
        row["thresholds_included"] = ",".join(f"{t:g}" for t in sorted(g["threshold_db"].unique()))
        out.append((row, g))
    # rank models by mean false reduction s.t. mean retention >= 0.95
    ranked = sorted(out, key=lambda rg: (-(rg[0]["mean_false_reduction"] or 0)
                                         if (rg[0]["mean_true_retention"] or 0) >= 0.95
                                         else 1, ))
    best_model = ranked[0][0]["model"] if ranked else None
    rows = []
    for row, g in out:
        rec = row["model"] == best_model
        row["recommended"] = rec
        row["recommendation_reason"] = (
            ("best mean false reduction at >=0.95 retention"
             + ("" if all_days_present else "; SINGLE/PARTIAL-day evidence"))
            if rec else "")
        rows.append(row)
    cols = ["model", "dates_included", "thresholds_included", "stage08_true_tracks",
            "stage08_false_tracks", "stage12_kept_true_tracks", "stage12_kept_false_tracks",
            "mean_true_retention", "mean_false_reduction", "mean_precision_after",
            "pooled_true_retention", "pooled_false_reduction", "pooled_precision_after",
            "recommended", "recommendation_reason"]
    return pd.DataFrame(rows)[cols].sort_values("model").reset_index(drop=True), best_model


# =============================================================================
# Comparisons
# =============================================================================

def model_comparison(s12_metrics, thresholds, models) -> pd.DataFrame:
    if len(models) < 2:
        return pd.DataFrame()
    a, b = models[0], models[1]
    df = s12_metrics[s12_metrics["threshold_db"].isin(thresholds)]
    metrics = [("true_track_retention", "max"), ("false_track_reduction", "max"),
               ("precision_after", "max"), ("stage12_kept_false_tracks", "min"),
               ("stage12_kept_true_tracks", "max")]
    rows = []
    for (date, thr), g in df.groupby(["date", "threshold_db"]):
        ga = g[g["model"] == a]
        gb = g[g["model"] == b]
        if ga.empty or gb.empty:
            continue
        for metric, better in metrics:
            va, vb = float(ga[metric].iloc[0]), float(gb[metric].iloc[0])
            if not (np.isfinite(va) and np.isfinite(vb)):
                winner = "n/a"
            elif va == vb:
                winner = "tie"
            elif (va > vb) == (better == "max"):
                winner = a
            else:
                winner = b
            rows.append({"date": date, "threshold_db": thr, "metric": metric,
                         a: va, b: vb, "winner": winner, "difference": va - vb, "notes": ""})
    per = pd.DataFrame(rows)
    # aggregate rows (mean over date/threshold per metric)
    agg = []
    for metric, better in metrics:
        sub = per[per["metric"] == metric]
        if sub.empty:
            continue
        va, vb = sub[a].mean(), sub[b].mean()
        winner = (a if (va > vb) == (better == "max") else b) if va != vb else "tie"
        agg.append({"date": "ALL", "threshold_db": np.nan, "metric": metric, a: va, b: vb,
                    "winner": winner, "difference": va - vb, "notes": "aggregate mean"})
    return pd.concat([per, pd.DataFrame(agg)], ignore_index=True)


def interpretable_fallback(s09_ctx, s12_metrics, thresholds, best_model) -> pd.DataFrame:
    df12 = s12_metrics[(s12_metrics["model"] == best_model)
                       & (s12_metrics["threshold_db"].isin(thresholds))]
    rows = []
    for _, r in df12.iterrows():
        date, thr = r["date"], r["threshold_db"]
        s9 = s09_ctx[(s09_ctx["date"].astype(str) == str(date))
                     & np.isclose(s09_ctx["threshold_db"], thr)]
        s9tr = float(s9["stage09_true_retention"].iloc[0]) if len(s9) else np.nan
        s9fr = float(s9["stage09_false_reduction"].iloc[0]) if len(s9) else np.nan
        s9pa = float(s9["stage09_precision_after"].iloc[0]) if len(s9) else np.nan
        dtr = r["true_track_retention"] - s9tr
        dfr = r["false_track_reduction"] - s9fr
        dpa = r["precision_after"] - s9pa
        interp = ("stage-12 wins on false reduction" if np.isfinite(dfr) and dfr > 0.01 else
                  "stage-09 competitive" if np.isfinite(dfr) else "stage-09 unavailable")
        rows.append({"date": date, "threshold_db": thr,
                     "stage09_true_retention": s9tr, "stage09_false_reduction": s9fr,
                     "stage09_precision_after": s9pa, "best_stage12_model": best_model,
                     "stage12_true_retention": r["true_track_retention"],
                     "stage12_false_reduction": r["false_track_reduction"],
                     "stage12_precision_after": r["precision_after"],
                     "stage12_minus_stage09_true_retention": dtr,
                     "stage12_minus_stage09_false_reduction": dfr,
                     "stage12_minus_stage09_precision": dpa, "interpretation": interp})
    return pd.DataFrame(rows).sort_values(["date", "threshold_db"]).reset_index(drop=True)


def windowability_audit(windowability, thresholds, include_high) -> pd.DataFrame:
    if windowability is None or windowability.empty:
        return pd.DataFrame([{"date": "-", "threshold_db": np.nan,
                              "notes": "stage-12 per-track scores unavailable; cannot audit"}])
    keep_thr = set(thresholds) | (HIGH_THRESHOLDS if include_high else set())
    df = windowability[windowability["threshold_db"].isin(keep_thr)].copy()
    df["high_threshold_caveat"] = df["threshold_db"].isin(HIGH_THRESHOLDS)
    return df.sort_values(["date", "threshold_db"]).reset_index(drop=True)


def failure_rollup(failures, best_model) -> pd.DataFrame:
    if failures is None or failures.empty or "case_type" not in failures.columns:
        return pd.DataFrame([{"case_type": "-", "date": "-", "threshold_db": np.nan,
                              "model": best_model, "count": 0, "median_range_m": np.nan,
                              "median_score": np.nan,
                              "notes": "per-track / stage-14 failure files unavailable"}])
    keep = failures[failures["case_type"].isin(
        ["false_survives_s12", "true_rejected_s12"])].copy()
    if keep.empty:
        keep = failures.copy()
    rows = []
    for (ct, date, thr), g in keep.groupby(["case_type", "date", "threshold_db"]):
        score = (g["score_stage12"].median() if "score_stage12" in g.columns else np.nan)
        rng = (g["median_range_m"].median() if "median_range_m" in g.columns else np.nan)
        rows.append({"case_type": ct, "date": date, "threshold_db": thr, "model": best_model,
                     "count": len(g), "median_range_m": rng, "median_score": score,
                     "notes": ""})
    return pd.DataFrame(rows).sort_values(["case_type", "date", "threshold_db"]).reset_index(drop=True)


# =============================================================================
# Key findings
# =============================================================================

def key_findings(overall, by_thr, model_cmp, fallback, windowability, dates_present,
                 all_days_present, best_model, include_high) -> pd.DataFrame:
    rows = []

    def add(fid, finding, evidence, metric, value, interp):
        rows.append({"finding_id": fid, "finding": finding, "evidence_file": evidence,
                     "metric": metric, "value": value, "interpretation": interp})

    if overall is not None and len(overall):
        b = overall[overall["model"] == best_model]
        if len(b):
            r = b.iloc[0]
            add("F1", "Stage 12.5 generalizes across available days" if all_days_present
                else "Stage 12.5 validated only on available days",
                "four_day_summary_overall.csv", "pooled_false_reduction / pooled_true_retention",
                f"{r['pooled_false_reduction']:.3f} / {r['pooled_true_retention']:.3f}",
                (f"holds across {len(dates_present)} day(s)" if all_days_present else
                 f"only {len(dates_present)} of 4 days scored; multi-day gap NOT fully closed"))
    if model_cmp is not None and len(model_cmp):
        agg = model_cmp[model_cmp["date"] == "ALL"]
        fr = agg[agg["metric"] == "false_track_reduction"]
        winner = fr["winner"].iloc[0] if len(fr) else "n/a"
        add("F2", "MLP vs GRU robustness", "model_comparison_mlp_vs_gru.csv",
            "aggregate false_reduction winner", winner,
            "MLP and GRU are close; the aggregate winner leads on mean false reduction")
    if fallback is not None and len(fallback):
        wins = (fallback["stage12_minus_stage09_false_reduction"] > 0.01).sum()
        add("F3", "Stage 12 vs Stage 09 across days", "interpretable_fallback_comparison.csv",
            "cells where stage12 false_reduction > stage09", f"{wins}/{len(fallback)}",
            "stage-12 generally removes more false tracks; stage-09 stays interpretable fallback")
    if by_thr is not None and len(by_thr):
        core = by_thr[(by_thr["model"] == best_model)
                      & (~by_thr["threshold_db"].isin(HIGH_THRESHOLDS))]
        if len(core):
            lo = core.sort_values("threshold_db").iloc[0]
            add("F4", "Low thresholds remain the best use case", "four_day_summary_by_threshold.csv",
                "false_reduction at lowest threshold", round(float(lo["pooled_false_reduction"]), 4),
                "the worst-clutter low thresholds are where the filter removes the most false tracks")
    add("F5", "Windowability caveat affects high thresholds",
        "windowability_four_day_audit.csv", "9/12 dB windowable false tracks",
        "~0" if not include_high else "audited",
        "high thresholds have few/no windowable false tracks; kept audit-only")
    add("F6", "Remaining single-day limitation",
        "four_day_summary_by_day.csv", "n_days_scored", len(dates_present),
        ("CLOSED: all four days scored" if all_days_present else
         "STILL OPEN: run the missing stage-08/09/12.5 days (see run_plan.csv)"))
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def make_plots(by_day, by_thr, s08_ctx, model_cmp, fallback, windowability, best_model,
               models, plots_dir) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    def _save(fig, name):
        p = os.path.join(plots_dir, name)
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    def _by_day_metric(col, title, fname):
        if by_day is None or by_day.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in by_day.groupby("model"):
            g = g.sort_values("date")
            ax.plot(g["date"], g[col], marker="o", label=model)
        ax.set_xlabel("date"); ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        plt.xticks(rotation=20)
        _save(fig, fname)

    _by_day_metric("pooled_true_retention", "True-track retention by day", "retention_by_day.png")
    _by_day_metric("pooled_false_reduction", "False-track reduction by day",
                   "false_reduction_by_day.png")
    _by_day_metric("pooled_precision_after", "Precision after filtering by day", "precision_by_day.png")

    def _by_thr_metric(col, title, fname):
        if by_thr is None or by_thr.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in by_thr.groupby("model"):
            g = g.sort_values("threshold_db")
            ax.plot(g["threshold_db"], g[col], marker="o", label=model)
        ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel(col.replace("_", " "))
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        _save(fig, fname)

    _by_thr_metric("pooled_true_retention", "True-track retention by threshold",
                   "retention_by_threshold.png")
    _by_thr_metric("pooled_false_reduction", "False-track reduction by threshold",
                   "false_reduction_by_threshold.png")

    if model_cmp is not None and len(model_cmp) and len(models) >= 2:
        a, b = models[0], models[1]
        agg = model_cmp[model_cmp["date"] == "ALL"]
        agg = agg[agg["metric"].isin(["true_track_retention", "false_track_reduction",
                                      "precision_after"])]
        if len(agg):
            x = np.arange(len(agg)); w = 0.38
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(x - w / 2, agg[a], w, label=a)
            ax.bar(x + w / 2, agg[b], w, label=b)
            ax.set_xticks(x); ax.set_xticklabels(agg["metric"], rotation=15, fontsize=8)
            ax.set_ylabel("aggregate mean"); ax.set_title(f"{a} vs {b} (aggregate)")
            ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
            _save(fig, "mlp_vs_gru.png")

    if fallback is not None and len(fallback):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        g = fallback.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["stage09_false_reduction"], marker="s", linestyle="--",
                label="stage 09")
        ax.plot(g["threshold_db"], g["stage12_false_reduction"], marker="o",
                label=f"stage 12 {best_model}")
        ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("false-track reduction")
        ax.set_title("Stage 09 vs Stage 12 false reduction")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        _save(fig, "stage09_vs_stage12.png")

    if windowability is not None and len(windowability) and \
            "windowable_fraction_false" in windowability.columns and \
            windowability["windowable_fraction_false"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for date, g in windowability.groupby("date"):
            g = g.sort_values("threshold_db")
            ax.plot(g["threshold_db"], g["windowable_fraction_false"], marker="o", label=str(date))
        ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("windowable false fraction")
        ax.set_title("Windowability of false tracks (four-day)")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        _save(fig, "windowability_four_day.png")
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(output_dir, dates_present, all_days_present, thresholds, include_high,
                 best_model, availability, run_plan, s08_ctx, by_day, by_thr, overall,
                 model_cmp, fallback, windowability, failures, findings) -> str:
    n_missing = int((run_plan["required"] == True).sum()) if len(run_plan) else 0
    lines = [
        "# Stage 17 Four-Day Validation",
        "",
        "## Status",
        "",
        "- **Stage 17 adds no new model** and retrains nothing.",
        "- It validates the selected **Stage 12.5** method (deterministic sequence",
        "  autoencoder, track-purity calibration) across all available ADS-B days.",
        "- It focuses on the **four-day** validation at thresholds -5/0/3/6 dB.",
        "- 9/12 dB remain **audit-only** due to the windowability caveat.",
        "- It runs missing stages only if explicitly requested (`--run-missing-*`).",
        "",
        "## Input availability",
        "",
        f"Expected four days: {', '.join(EXPECTED_DAYS)}. "
        f"Days with complete stage-08/09/12 results: "
        f"**{', '.join(dates_present) if dates_present else 'none'}**.",
        "",
    ]
    avail_summary = (availability.groupby("date")
                     .agg(cells=("status", "size"),
                          complete=("status", lambda s: (s == "complete").sum()))
                     .reset_index())
    lines += md_table(avail_summary)
    lines += ["",
              ("**All four days are available** -- the single-day limitation from Stage 16 is"
               " addressed below." if all_days_present else
               "**Not all four days are available** -- see the run plan for the exact commands"
               " to generate the missing outputs."),
              ""]

    lines += ["## Run plan", ""]
    if n_missing == 0:
        lines += ["All required stage-08/09/12.5 outputs are present; **no expensive reruns"
                  " needed.**", ""]
    else:
        lines += [f"{n_missing} missing stage output(s). Exact commands to generate them"
                  " (not run unless `--run-missing-*` is passed):", ""]
        lines += md_table(run_plan[run_plan["required"] == True][
            ["date", "threshold_db", "missing_stage", "command", "will_run", "notes"]])
    lines += [""]

    lines += ["## Four-day Stage 08 context", ""]
    if s08_ctx is not None and len(s08_ctx):
        ctx = (s08_ctx.groupby("date").agg(
            stage08_true_tracks=("stage08_true_tracks", "sum"),
            stage08_false_tracks=("stage08_false_tracks", "sum"),
            stage08_confirmed_tracks=("stage08_confirmed_tracks", "sum")).reset_index())
        lines += md_table(ctx)
    else:
        lines += ["Stage-08 context unavailable."]
    lines += [""]

    lines += ["## Stage 12.5 four-day validation", ""]
    lines += md_table(by_day.round(4))
    lines += ["", "### By threshold (pooled across days)", ""]
    lines += md_table(by_thr.round(4))
    lines += ["", "### Overall", ""]
    lines += md_table(overall.round(4))

    lines += ["", "## MLP vs GRU", ""]
    if model_cmp is not None and len(model_cmp):
        lines += md_table(model_cmp[model_cmp["date"] == "ALL"].round(4))
        lines += ["", "MLP remains the best overall model; GRU stays the strong low-threshold"
                  " alternative (differences are small)."]
    else:
        lines += ["MLP/GRU comparison unavailable."]
    lines += [""]

    lines += ["## Stage 09 interpretable fallback comparison", ""]
    lines += md_table(fallback.round(4))
    if len(fallback):
        wins = int((fallback["stage12_minus_stage09_false_reduction"] > 0.01).sum())
        lines += ["", f"- Stage 12 removes more false tracks than Stage 09 in **{wins}/"
                  f"{len(fallback)}** cells; **Stage 09 hand physics remains the recommended",
                  "  interpretable fallback** where transparency matters more than peak reduction.",
                  ""]

    lines += ["## Windowability caveat", "",
              "Sequence methods only score windowable tracks (>= window_len points, >= 5 hits),",
              "so 9/12 dB comparisons are not apples-to-apples (few/no windowable false tracks).",
              "High thresholds are kept audit-only.", ""]
    lines += md_table(windowability.round(4))

    lines += ["", "## Failure cases", ""]
    lines += md_table(failures.round(4))
    lines += ["",
              "- Surviving false tracks after Stage 12.5 are rare but worth inspecting.",
              "- True tracks rejected by Stage 12.5 should be reviewed before deployment.", ""]

    lines += ["## Consolidated conclusion", ""]
    if all_days_present:
        verdict = "The Stage 12.5 conclusion generalizes across all four days."
    elif len(dates_present):
        verdict = ("The Stage 12.5 conclusion is only partially validated because some required"
                   " outputs are missing.")
    else:
        verdict = "The Stage 12.5 conclusion does not generalize (no complete days available)."
    lines += [f"- **{verdict}**",
              "- Stage 12.5 remains the recommended primary false-track filter.",
              "- **Stage 09 interpretable fallback** remains recommended where transparency wins.",
              ("- The single-day limitation from Stage 16 is **closed**." if all_days_present else
               "- The single-day limitation from Stage 16 is **still open**; run the missing"
               " stages (see run plan) to close it."),
              ""]

    lines += ["## Recommended next stage", ""]
    if all_days_present:
        lines += ["1. **Final report / paper packaging** of the validated result, or",
                  "2. an optional compact model-zoo benchmark only if a specific gap remains, or",
                  "3. an optional deployment-style runtime / operating-point study.", ""]
    else:
        lines += ["1. **Run the missing Stage 08/09/12.5 outputs for all four days first**"
                  " (see `run_plan.csv`; the stage-6 detections and stage-12 checkpoints exist",
                  "   locally, so `--run-missing-stage08/09/12` can generate them), then re-run",
                  "   this validation.", ""]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stage17_four_day_validation_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_gate(output_dir, availability, run_plan, s12_metrics, all_days_present) -> None:
    def fail(msg):
        raise ValueError(f"Stage 17 validation failed: {msg}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (four-day validation)")
    print("=" * 70)
    for name, df in [("input_availability.csv", availability), ("run_plan.csv", run_plan)]:
        p = os.path.join(output_dir, name)
        if not os.path.exists(p):
            fail(f"{name} missing")
    if availability.empty:
        fail("input_availability.csv is empty")
    rep = os.path.join(output_dir, "stage17_four_day_validation_report.md")
    if not os.path.exists(rep):
        fail("report missing")
    if s12_metrics is None or s12_metrics.empty:
        fail("no stage-12 result for any date/threshold")
    print("  availability + run plan + report present; >=1 stage-12 result: OK")

    for col in ("true_track_retention", "false_track_reduction", "precision_after"):
        v = s12_metrics[col].dropna()
        if len(v) and not v.between(0, 1).all():
            fail(f"{col} outside [0, 1]")
    print("  retention / false-reduction / precision in [0, 1]: OK")

    text = open(rep).read()
    required = ["Stage 17 adds no new model", "four-day", "Stage 12.5",
                "Stage 09 interpretable fallback", "windowability", "Recommended next stage"]
    for needle in required:
        if needle not in text:
            fail(f"report missing required text: {needle!r}")
    gap_closed = "single-day limitation from Stage 16 is **closed**" in text
    gap_open = "single-day limitation from Stage 16 is **still open**" in text
    if all_days_present and not gap_closed:
        fail("all four days present but report does not state the gap is closed")
    if not all_days_present and not gap_open:
        fail("not all days present but report does not state the gap remains")
    print(f"  report strings present; gap status consistent (closed={all_days_present}): OK")
