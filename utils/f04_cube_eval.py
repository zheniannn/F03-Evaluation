"""Stage 22 helpers: evaluate F04 cube-derived CFAR detections inside F03.

Stage 22 adds no model and no new physics. It imports an F04 Stage-21 export
(cube-derived CFAR point detections), adapts it to what F03's stage-08 discovery
and schema expect, runs the existing stage-08 tracker and stage-12.5 sequence-prior
scorer over it, and compares the outcome with the original F02 point-detection
result.

Two things this module exists to keep honest:

1. **CFAR scale is not dB.** F04's operating point is ``threshold_scale = 6.0``, a
   *linear multiplier* on the local CFAR noise estimate (~7.78 dB). F03's
   ``--threshold-db`` and its ``detections_<date>_thr_<token>dB.csv`` discovery
   pattern read that token as a **dB SNR threshold**. The token is reused purely as a
   compatibility label; every artifact this module writes carries the warning.

2. **The exported timestamp column is unusable as-is.** F04's Stage 21 writer uses
   ``float_format="%.6g"``, which rounds epoch seconds (~1.65e9) to 6 significant
   digits: 200 distinct frame times collapse onto 2 values. Stage 08 is immune (it
   takes dt from ``--frame-period-s``) but stage 12's window features differentiate
   position with respect to ``timestamp``, so a collapsed clock would silently
   corrupt every velocity feature. The importer reconstructs the clock from
   ``frame_id`` and refuses to proceed unless the reconstruction agrees with the
   rounded original. See ``reconstruct_timestamps``.
"""

import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.common import REPO_ROOT, md_table, threshold_to_token

SOURCE = "f04_cube_cfar"

ALIAS_WARNING = ("thr_6p0dB is a compatibility filename token only; it represents F04 CFAR "
                 "scale 6.0, not a 6 dB SNR threshold.")

NOT_RAW = ("F04 detections come from a synthetic range-Doppler-azimuth intensity cube. "
           "This is not raw RF/IQ, not a full radar waveform simulation, and not measured "
           "radar data. Target/clutter labels are evaluation-only.")

# What F03's stage-08 actually reads (scripts/08_run_kalman_baseline.py DETECTION_USECOLS).
STAGE08_REQUIRED = ["frame_id", "timestamp", "detection_id", "is_target", "trajectory_id",
                    "meas_range_m", "meas_azimuth_rad", "meas_elevation_rad",
                    "truth_range_m", "truth_azimuth_rad", "truth_elevation_rad"]

# Additionally used downstream (stage 09 joins snr_db by detection_id).
STAGE09_EXTRA = ["snr_db"]


def scale_to_db(scale: float) -> float:
    """CFAR linear threshold multiplier -> dB. 6.0 -> 7.782 dB (NOT 6 dB)."""
    return float(10.0 * np.log10(scale))


def alias_warning(token: str, scale: float) -> str:
    """The warning every stage-22 artifact must carry. `6p0` renders as `6`, not `6.0`."""
    from utils.common import token_to_threshold
    nominal = f"{token_to_threshold(token):g}"
    return (f"thr_{token}dB is a compatibility filename token only; it represents F04 CFAR "
            f"scale {scale:g}, not a {nominal} dB SNR threshold. "
            f"CFAR scale {scale:g} is {scale_to_db(scale):.2f} dB above the local noise estimate.")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Import + adapt
# =============================================================================

def validate_schema(df: pd.DataFrame, required: Optional[List[str]] = None
                    ) -> Tuple[bool, List[str]]:
    required = list(required or STAGE08_REQUIRED)
    missing = [c for c in required if c not in df.columns]
    return (not missing), missing


def reconstruct_timestamps(df: pd.DataFrame, frame_period_s: float
                           ) -> Tuple[pd.Series, Dict]:
    """Recover a per-frame clock from ``frame_id``, and prove it matches the original.

    F04's cube simulator defines ``frame_id = floor(timestamp / scan_period_s)``, so
    ``frame_id * scan_period_s`` is the frame's start time. The exported ``timestamp``
    survives only to 6 significant digits, so we cannot compare exactly -- instead we
    require the reconstruction to round to the *same* 6-significant-digit value the
    export carries. That is a real check: a wrong period or a wrong frame_id convention
    would fail it.
    """
    fid = df["frame_id"].to_numpy(dtype=np.int64)
    recon = fid.astype(np.float64) * float(frame_period_s)
    orig = df["timestamp"].to_numpy(dtype=np.float64)

    def six_sig(a):
        return np.array([float(f"{v:.6g}") for v in a])

    n_unique_orig = int(pd.unique(orig).size)
    n_unique_recon = int(pd.unique(recon).size)
    n_frames = int(pd.unique(fid).size)

    # compare on frame representatives, not all 262k rows
    reps = pd.DataFrame({"fid": fid, "orig": orig, "recon": recon}).groupby("fid").first()
    agree = bool(np.all(six_sig(reps["recon"].to_numpy()) == six_sig(reps["orig"].to_numpy())))
    max_abs = float(np.max(np.abs(reps["recon"].to_numpy() - reps["orig"].to_numpy())))

    info = {
        "timestamp_reconstructed": True,
        "frame_period_s": float(frame_period_s),
        "unique_timestamps_original": n_unique_orig,
        "unique_timestamps_reconstructed": n_unique_recon,
        "frames": n_frames,
        "reconstruction_matches_rounded_original": agree,
        "max_abs_timestamp_shift_s": max_abs,
        "degenerate_original_clock": n_unique_orig < n_frames,
    }
    return pd.Series(recon, index=df.index), info


def import_f04_export(export_path: str, detections_dir: str, date: str, cfar_type: str,
                      threshold_scale: float, cap: int, alias_token: str,
                      frame_period_s: float = 10.0, max_frames: Optional[int] = None,
                      overwrite: bool = False) -> Dict:
    """Copy + adapt the F04 export into F03's detections dir, and write the alias metadata.

    The original F04 export is never modified. Adaptations are recorded in the returned
    dict under ``adaptations`` and echoed into the alias metadata JSON.
    """
    if not os.path.exists(export_path):
        raise SystemExit(f"F04 export not found: {export_path}")
    ensure_dir(detections_dir)

    imported_name = os.path.basename(export_path)
    imported_path = os.path.join(detections_dir, imported_name)
    alias_name = f"detections_{date}_thr_{alias_token}dB.csv"
    alias_path = os.path.join(detections_dir, alias_name)
    meta_path = os.path.join(detections_dir, f"f04_cube_alias_metadata_{date}.json")

    if os.path.exists(alias_path) and not overwrite:
        raise SystemExit(f"Alias already exists (pass --overwrite): {alias_path}")

    df = pd.read_csv(export_path, dtype={"trajectory_id": str}, low_memory=False)
    ok, missing = validate_schema(df)
    adaptations: List[str] = []

    if max_frames is not None:
        keep = np.sort(df["frame_id"].unique())[:max_frames]
        if len(keep) < df["frame_id"].nunique():
            adaptations.append(f"restricted to the first {len(keep)} frames (--max-frames)")
        df = df[df["frame_id"].isin(keep)].copy()

    # --- adaptation 1: repair the collapsed clock -------------------------
    ts_info = {}
    if "timestamp" in df.columns and "frame_id" in df.columns:
        recon, ts_info = reconstruct_timestamps(df, frame_period_s)
        if ts_info["degenerate_original_clock"]:
            if not ts_info["reconstruction_matches_rounded_original"]:
                raise SystemExit(
                    "F04 export timestamps are degenerate "
                    f"({ts_info['unique_timestamps_original']} unique values for "
                    f"{ts_info['frames']} frames) and the frame_id-based reconstruction does "
                    "NOT agree with the rounded original. Refusing to guess a clock. "
                    "Fix F04's Stage 21 writer (float_format='%.6g' destroys epoch seconds).")
            adaptations.append(
                f"reconstructed `timestamp` as frame_id * {frame_period_s:g}s: the F04 export "
                f"carries only {ts_info['unique_timestamps_original']} distinct timestamps for "
                f"{ts_info['frames']} frames (6-significant-digit float_format rounding). "
                "Stage 12 window features differentiate position w.r.t. timestamp, so the "
                "original column would corrupt every velocity feature")
            df["timestamp"] = recon.to_numpy()
        else:
            ts_info["timestamp_reconstructed"] = False

    # --- adaptation 2: clutter trajectory_id must be blank, not the string "nan" ---
    if "trajectory_id" in df.columns:
        n_nanstr = int((df["trajectory_id"].astype(str).str.lower() == "nan").sum())
        if n_nanstr:
            df.loc[df["trajectory_id"].astype(str).str.lower() == "nan", "trajectory_id"] = np.nan
            adaptations.append(f"normalized {n_nanstr:,} literal 'nan' trajectory_id strings to "
                               "empty (clutter)")

    frames = int(df["frame_id"].nunique())
    n_target = int((df["is_target"] == 1).sum())
    n_clutter = int((df["is_target"] == 0).sum())

    df.to_csv(imported_path, index=False)
    # alias is a real copy, not a symlink: some tools resolve symlinks inconsistently and
    # the file must survive being read by a subprocess with a different cwd.
    shutil.copyfile(imported_path, alias_path)

    meta = {
        "source": SOURCE,
        "original_file": os.path.abspath(export_path),
        "imported_file": os.path.abspath(imported_path),
        "alias_file": alias_name,
        "cfar_type": cfar_type,
        "threshold_scale": float(threshold_scale),
        "cap": int(cap),
        "alias_token": alias_token,
        "alias_warning": alias_warning(alias_token, threshold_scale),
        "threshold_scale_in_db": round(scale_to_db(threshold_scale), 4),
        "frames": frames,
        "rows": int(len(df)),
        "target_detections": n_target,
        "clutter_detections": n_clutter,
        "schema_valid": bool(ok),
        "missing_columns": missing,
        "adaptations": adaptations,
        "timestamp_check": ts_info,
        "not_raw": NOT_RAW,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    meta["alias_path"] = alias_path
    meta["metadata_path"] = meta_path
    meta["detections"] = df
    return meta


def write_import_summary(meta: Dict, date: str, path: str) -> str:
    row = {
        "date": date,
        "source_file": os.path.basename(meta["original_file"]),
        "imported_file": os.path.basename(meta["imported_file"]),
        "alias_file": meta["alias_file"],
        "rows": meta["rows"],
        "frames": meta["frames"],
        "target_detections": meta["target_detections"],
        "clutter_detections": meta["clutter_detections"],
        "cfar_type": meta["cfar_type"],
        "threshold_scale": meta["threshold_scale"],
        "cap": meta["cap"],
        "alias_token": meta["alias_token"],
        "schema_valid": meta["schema_valid"],
        "notes": "; ".join(meta["adaptations"]) if meta["adaptations"] else "no adaptation needed",
    }
    ensure_dir(os.path.dirname(path))
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


# =============================================================================
# Command construction (never executed by the self-test)
# =============================================================================

def build_stage08_command(args, alias_threshold: float) -> List[str]:
    return [sys.executable, os.path.join(REPO_ROOT, "scripts", "08_run_kalman_baseline.py"),
            "--detections-dir", args.detections_dir,
            "--truth-dir", args.truth_dir,
            "--tracks-dir", args.tracks_dir,
            "--report-dir", os.path.join(args.report_dir, "stage08_kalman_f04_cube"),
            "--threshold-db", f"{alias_threshold:g}",
            "--date", args.date,
            "--max-frames", str(args.max_frames),
            "--overwrite"]


def build_stage09_command(args, alias_threshold: float) -> List[str]:
    return [sys.executable, os.path.join(REPO_ROOT, "scripts", "09_score_tracks_physics.py"),
            "--tracks-dir", args.tracks_dir,
            "--detections-dir", args.detections_dir,
            "--scored-tracks-dir", os.path.join(args.report_dir, "generated",
                                                "scored_tracks_f04_cube"),
            "--report-dir", os.path.join(args.report_dir, "stage09_physics_f04_cube"),
            "--threshold-db", f"{alias_threshold:g}",
            "--date", args.date,
            "--overwrite"]


def build_stage12_command(args, alias_threshold: float, calibration_mode: str,
                          calibration_output: str) -> List[str]:
    """Stage-12.5 scoring over the F04 tracks.

    ``--calibration-tracks-dir`` is pinned to the F04 tracks. Its default is the canonical
    F02 ``data/active/tracks_kalman``, which *also* contains a ``thr_6p0dB`` file -- so
    omitting the flag would silently build the noise-matched calibration from F02
    point-detection tracks while scoring F04 cube-CFAR tracks, and succeed without warning.
    That is precisely the noise mismatch stage 12.5 exists to remove.

    ``--calibration-output`` is pinned inside the stage-22 report dir so the canonical
    stage-12 calibration artifact can never be overwritten (stage 17.5 hardening).
    """
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts",
                                        "12_score_tracks_sequence_prior.py"),
           "--tracks-dir", args.tracks_dir,
           "--models-dir", args.models_dir,
           "--stage09-dir", os.path.join(args.report_dir, "stage09_physics_f04_cube")
           if args.run_stage09 else args.stage09_dir,
           "--stage11-dir", args.stage11_dir,
           "--report-dir", os.path.join(args.report_dir, "stage12_sequence_f04_cube"),
           "--detections-dir", args.detections_dir,
           "--threshold-db", f"{alias_threshold:g}",
           "--date", args.date,
           "--calibration-mode", calibration_mode,
           "--score-threshold", f"{args.score_threshold:g}",
           "--calibration-output", calibration_output,
           "--overwrite"]
    if calibration_mode == "track_purity":
        cmd += ["--calibration-tracks-dir", args.tracks_dir,
                "--calibration-date", args.date,
                "--calibration-threshold-db", f"{alias_threshold:g}"]
    if getattr(args, "model", None):
        cmd += ["--model"] + list(args.model)
    return cmd


def run_command(cmd: List[str], label: str, log_path: Optional[str] = None) -> Dict:
    assert cmd[0] == sys.executable, "internal subprocess must use sys.executable"
    print(f"\n>>> {label}\n  " + " ".join(cmd) + "\n", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "w") as f:
            f.write(" ".join(cmd) + "\n\n" + out)
    tail = "\n".join(out.strip().splitlines()[-25:])
    print(tail, flush=True)
    return {"label": label, "returncode": proc.returncode, "ok": proc.returncode == 0,
            "output": out, "tail": tail, "command": " ".join(cmd)}


# =============================================================================
# Result extraction
# =============================================================================

def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df if len(df) else None
        except Exception:
            return None
    return None


def extract_stage08_summary(report_dir: str, meta: Dict, date: str, alias_token: str,
                            ok: bool, note: str = "") -> pd.DataFrame:
    base = {"date": date, "alias_token": alias_token, "cfar_type": meta["cfar_type"],
            "threshold_scale": meta["threshold_scale"], "cap": meta["cap"]}
    cols = ["confirmed_tracks", "true_tracks", "false_tracks", "track_detection_rate",
            "target_assignment_rate", "clutter_assignment_rate", "mean_position_rmse_m",
            "median_position_rmse_m"]
    df = _read_csv(os.path.join(report_dir, "stage08_kalman_f04_cube",
                                "kalman_baseline_summary.csv"))
    if df is None:
        row = {**base, **{c: np.nan for c in cols},
               "notes": note or "stage 08 did not run or produced no summary"}
        return pd.DataFrame([row])
    r = df.iloc[0]
    row = {**base}
    row["confirmed_tracks"] = r.get("tracks_confirmed", np.nan)
    for c in cols[1:]:
        row[c] = r.get(c, np.nan)
    notes = [note] if note else []
    notes.append(f"alias thr {alias_token}dB = F04 CFAR scale {meta['threshold_scale']:g} "
                 f"({scale_to_db(meta['threshold_scale']):.2f} dB), not a dB SNR threshold")
    if not ok:
        notes.append("stage 08 exited nonzero")
    row["notes"] = "; ".join(notes)
    return pd.DataFrame([row])


def extract_stage12_summary(report_dir: str, meta: Dict, date: str, alias_token: str,
                            calibration_mode: str, score_threshold: float,
                            ok: bool, note: str = "") -> pd.DataFrame:
    base = {"date": date, "alias_token": alias_token, "cfar_type": meta["cfar_type"],
            "threshold_scale": meta["threshold_scale"], "cap": meta["cap"]}
    cols = ["stage08_true_tracks", "stage08_false_tracks", "stage12_kept_true_tracks",
            "stage12_kept_false_tracks", "true_track_retention", "false_track_reduction",
            "precision_before", "precision_after"]
    df = _read_csv(os.path.join(report_dir, "stage12_sequence_f04_cube",
                                "sequence_metrics_by_model_threshold.csv"))
    if df is None:
        return pd.DataFrame([{**base, "model": "n/a", "calibration_mode": calibration_mode,
                              "score_threshold": score_threshold,
                              **{c: np.nan for c in cols},
                              "notes": note or "stage 12 did not run or produced no metrics"}])
    rows = []
    for _, r in df.iterrows():
        row = {**base, "model": r.get("model"),
               "calibration_mode": r.get("calibration_mode", calibration_mode),
               "score_threshold": score_threshold}
        for c in cols:
            row[c] = r.get(c, np.nan)
        notes = [note] if note else []
        if not np.isfinite(row.get("false_track_reduction", np.nan)):
            notes.append("false_track_reduction undefined (no false tracks in denominator)")
        if not ok:
            notes.append("stage 12 exited nonzero")
        row["notes"] = "; ".join(notes)
        rows.append(row)
    return pd.DataFrame(rows)


def build_windowability_audit(report_dir: str, s08: pd.DataFrame, s12: pd.DataFrame,
                              model: str = "mlp_dae") -> pd.DataFrame:
    """Reconcile stage-08's track counts with stage-12's scoring denominator.

    Stage 12 scores only tracks that yield at least one window; its
    ``stage08_true_tracks`` / ``stage08_false_tracks`` columns are therefore the
    *windowable* subset, not what stage 08 confirmed. On F04 cube tracks the gap is
    large enough to invert the headline: a filter that removes 26.8% of the tracks it
    sees removes far less of what the tracker actually emitted, because most false
    tracks are too short to window.

    Reported both ways, because the pipeline-level number depends on a policy choice
    stage 12 does not make: what happens to a confirmed track the filter never scored.
    """
    if not len(s08) or not len(s12):
        return pd.DataFrame()
    sel = s12[s12["model"] == model]
    if not len(sel):
        return pd.DataFrame()
    r8, r12 = s08.iloc[0], sel.iloc[0]

    def f(v):
        return float(v) if v is not None and np.isfinite(float(v)) else np.nan

    all_true, all_false = f(r8.get("true_tracks")), f(r8.get("false_tracks"))
    win_true, win_false = f(r12.get("stage08_true_tracks")), f(r12.get("stage08_false_tracks"))
    kept_true, kept_false = (f(r12.get("stage12_kept_true_tracks")),
                             f(r12.get("stage12_kept_false_tracks")))
    unscored_true, unscored_false = all_true - win_true, all_false - win_false

    rows = [
        {"denominator": "stage12_windowable_only",
         "description": "tracks stage 12.5 actually scored (>=1 window) -- the scorer's own "
                        "denominator, and what f04_stage12_summary.csv reports",
         "true_tracks": win_true, "false_tracks": win_false,
         "kept_true_tracks": kept_true, "kept_false_tracks": kept_false,
         "true_track_retention": kept_true / win_true if win_true else np.nan,
         "false_track_reduction": 1 - kept_false / win_false if win_false else np.nan,
         "notes": "comparable to the F02 stage-12/17 figures, which use the same denominator"},
        {"denominator": "stage08_all_unscored_kept",
         "description": "all confirmed stage-08 tracks; tracks the filter never scored pass "
                        "through unfiltered",
         "true_tracks": all_true, "false_tracks": all_false,
         "kept_true_tracks": kept_true + unscored_true,
         "kept_false_tracks": kept_false + unscored_false,
         "true_track_retention": (kept_true + unscored_true) / all_true if all_true else np.nan,
         "false_track_reduction": (1 - (kept_false + unscored_false) / all_false
                                   if all_false else np.nan),
         "notes": "pipeline-level effect if an unscored confirmed track is retained"},
        {"denominator": "stage08_all_unscored_dropped",
         "description": "all confirmed stage-08 tracks; tracks the filter never scored are "
                        "discarded",
         "true_tracks": all_true, "false_tracks": all_false,
         "kept_true_tracks": kept_true, "kept_false_tracks": kept_false,
         "true_track_retention": kept_true / all_true if all_true else np.nan,
         "false_track_reduction": 1 - kept_false / all_false if all_false else np.nan,
         "notes": "pipeline-level effect if an unscored confirmed track is dropped; buys "
                  "false-track suppression by destroying true-track retention"},
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "unscored_false_tracks", unscored_false)
    df.insert(0, "unscored_true_tracks", unscored_true)
    df.to_csv(os.path.join(report_dir, "f04_windowability_audit.csv"), index=False)
    return df


def build_f02_comparison(stage12_f04: pd.DataFrame, stage12_dir: str, four_day_dir: str,
                         date: str, alias_token: str, meta: Dict,
                         model: str = "mlp_dae") -> pd.DataFrame:
    """F04 cube-CFAR (200 frames, one operating point) vs the F02 point-detection results.

    Deliberately kept as three clearly-labelled rows rather than a single delta: the runs
    differ in detection source, frame count, day count and threshold semantics at once, so
    no single difference is attributable.
    """
    rows: List[Dict] = []
    op = (f"cfar={meta['cfar_type']} scale={meta['threshold_scale']:g} "
          f"({scale_to_db(meta['threshold_scale']):.2f} dB) cap={meta['cap']}")

    f04 = stage12_f04[stage12_f04["model"] == model]
    if len(f04):
        r = f04.iloc[0]
        rows.append({
            "comparison_scope": "f04_cube_cfar_200_frames_1_day",
            "method": f"stage12.5 {model}", "source": "F04 synthetic radar cube -> CA-CFAR",
            "date": date, "threshold_or_operating_point": op,
            "true_retention": r.get("true_track_retention"),
            "false_reduction": r.get("false_track_reduction"),
            "false_tracks_kept": r.get("stage12_kept_false_tracks"),
            "true_tracks": r.get("stage08_true_tracks"),
            "false_tracks": r.get("stage08_false_tracks"),
            "notes": "small structured-clutter stress test; NOT a replacement for the "
                     "four-day F02 result. true_tracks/false_tracks are the WINDOWABLE "
                     "denominator; see f04_windowability_audit.csv for the full stage-08 "
                     "population"})

    # F02 single-day stage-12.5, matched on the numeric token only (6 dB) -- a coincidence
    # of naming, not of meaning. Included because it is the closest single-day F02 point.
    s12 = _read_csv(os.path.join(stage12_dir, "sequence_metrics_by_model_threshold.csv"))
    if s12 is not None and "model" in s12.columns:
        m = s12[(s12["model"] == model) & (np.isclose(s12.get("threshold_db", np.nan), 6.0))]
        if len(m):
            r = m.iloc[0]
            rows.append({
                "comparison_scope": "f02_point_detection_1_day_thr_6dB",
                "method": f"stage12.5 {model}",
                "source": "F02 point-detection simulator", "date": date,
                "threshold_or_operating_point": "threshold_db=6.0 (a real 6 dB SNR threshold)",
                "true_retention": r.get("true_track_retention"),
                "false_reduction": r.get("false_track_reduction"),
                "false_tracks_kept": r.get("stage12_kept_false_tracks"),
                "true_tracks": r.get("stage08_true_tracks"),
                "false_tracks": r.get("stage08_false_tracks"),
                "notes": "numeric token 6 matches the F04 alias by coincidence only; F04's 6 "
                         "is a linear CFAR scale, this 6 is dB"})

    # F02 four-day pooled headline (stage 17)
    overall = _read_csv(os.path.join(four_day_dir, "four_day_summary_overall.csv"))
    if overall is not None and "model" in overall.columns:
        m = overall[overall["model"] == model]
        if len(m):
            r = m.iloc[0]
            rows.append({
                "comparison_scope": "f02_point_detection_4_day_pooled",
                "method": f"stage12.5 {model}",
                "source": "F02 point-detection simulator",
                "date": str(r.get("dates_included", "4 days")),
                "threshold_or_operating_point": str(r.get("thresholds_included", "multi-dB")),
                "true_retention": r.get("pooled_true_retention"),
                "false_reduction": r.get("pooled_false_reduction"),
                "false_tracks_kept": r.get("stage12_kept_false_tracks"),
                "true_tracks": r.get("stage08_true_tracks"),
                "false_tracks": r.get("stage08_false_tracks"),
                "notes": "headline F02 result: four days, multiple dB thresholds, pooled"})
    return pd.DataFrame(rows)


def build_key_findings(meta: Dict, s08: pd.DataFrame, s12: pd.DataFrame,
                       cmp_df: pd.DataFrame, stage08_ok: bool, stage12_ok: bool,
                       calibration_mode: str, calibration_note: str,
                       model: str = "mlp_dae",
                       audit: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []

    def add(fid, finding, evidence, metric, value, interp):
        rows.append({"finding_id": fid, "finding": finding, "evidence_file": evidence,
                     "metric": metric, "value": value, "interpretation": interp})

    add("F1", "F04 cube-derived CFAR export imported into F03", "f04_import_summary.csv",
        "rows_imported", meta["rows"],
        f"{meta['frames']} frames, {meta['target_detections']:,} target / "
        f"{meta['clutter_detections']:,} clutter detections; schema_valid="
        f"{meta['schema_valid']}")

    add("F2", "Stage 08 Kalman tracking ran on F04 cube detections",
        "f04_stage08_summary.csv", "stage08_ran", bool(stage08_ok),
        ("confirmed=%s true=%s false=%s" % (s08.iloc[0].get("confirmed_tracks"),
                                            s08.iloc[0].get("true_tracks"),
                                            s08.iloc[0].get("false_tracks")))
        if stage08_ok and len(s08) else "stage 08 did not complete; see run log")

    sel = s12[s12["model"] == model]
    if stage12_ok and len(sel):
        r = sel.iloc[0]
        add("F3", "Stage 12.5 sequence-prior scoring ran on F04 tracks",
            "f04_stage12_summary.csv", "stage12_ran", True,
            f"model={model}, calibration={r.get('calibration_mode')}")
        add("F4", "Stage 12.5 false-track suppression under cube-CFAR detections "
                  "(windowable denominator)",
            "f04_stage12_summary.csv", "false_track_reduction",
            r.get("false_track_reduction"),
            _interpret_reduction(r.get("false_track_reduction"),
                                 r.get("stage08_false_tracks")))
        add("F5", "Stage 12.5 true-track retention under cube-CFAR detections",
            "f04_stage12_summary.csv", "true_track_retention",
            r.get("true_track_retention"),
            _interpret_retention(r.get("true_track_retention")))
    else:
        add("F3", "Stage 12.5 sequence-prior scoring on F04 tracks",
            "f04_stage12_summary.csv", "stage12_ran", bool(stage12_ok),
            calibration_note or "stage 12 did not complete; see run log")

    if len(s08):
        ft = s08.iloc[0].get("false_tracks")
        add("F6", "Are cube-CFAR false tracks harder than F02 point-detection false tracks?",
            "f04_cube_vs_f02_point_comparison.csv", "stage08_false_tracks", ft,
            _interpret_hardness(cmp_df, model))

    add("F7", "CFAR threshold scale is not a dB SNR threshold",
        "f04_cube_alias_metadata_<date>.json", "threshold_scale_vs_db",
        f"scale {meta['threshold_scale']:g} = {scale_to_db(meta['threshold_scale']):.2f} dB",
        meta["alias_warning"])

    add("F8", "Calibration provenance for the stage 12.5 run",
        "f04_stage12_summary.csv", "calibration_mode", calibration_mode,
        calibration_note or ("noise-matched calibration built from the F04 cube tracks "
                             "themselves (not the canonical F02 tracks)"))

    if audit is not None and len(audit):
        w = audit[audit["denominator"] == "stage12_windowable_only"].iloc[0]
        k = audit[audit["denominator"] == "stage08_all_unscored_kept"].iloc[0]
        unscored = float(audit["unscored_false_tracks"].iloc[0])
        add("F9", "Most stage-08 false tracks are never scored by stage 12.5",
            "f04_windowability_audit.csv", "unscored_false_tracks", unscored,
            f"stage 08 confirmed {k['false_tracks']:,.0f} false tracks but only "
            f"{w['false_tracks']:,.0f} are windowable (>=1 window), so "
            f"{unscored:,.0f} are never seen by the filter. The {w['false_track_reduction']:.1%} "
            "reduction is over the windowable subset only")
        add("F10", "Pipeline-level false-track suppression over ALL stage-08 false tracks",
            "f04_windowability_audit.csv", "false_track_reduction_unscored_kept",
            k["false_track_reduction"],
            f"if unscored confirmed tracks pass through, stage 12.5 removes only "
            f"{k['false_track_reduction']:.1%} of the false tracks the tracker emitted. "
            "Dropping unscored tracks instead would raise suppression but collapse true-track "
            "retention -- stage 12.5 does not define this policy, so both are reported")
    return pd.DataFrame(rows)


def _interpret_reduction(v, n_false) -> str:
    if v is None or not np.isfinite(v):
        return ("undefined: stage 08 produced no windowable false tracks in the denominator "
                f"(stage08 false tracks = {n_false}); no suppression claim can be made")
    suffix = " (denominator: windowable scored tracks only)"
    if v >= 0.9:
        return (f"stage 12.5 removes {v:.1%} of false tracks -- suppression persists under "
                f"cube-CFAR{suffix}")
    if v >= 0.5:
        return (f"stage 12.5 removes {v:.1%} of false tracks -- weaker than the F02 "
                f"point-detection result{suffix}")
    return (f"stage 12.5 removes only {v:.1%} of false tracks -- structured CFAR clutter is "
            f"substantially harder than F02 point clutter{suffix}")


def _interpret_retention(v) -> str:
    if v is None or not np.isfinite(v):
        return "undefined: no windowable true tracks"
    if v >= 0.9:
        return f"true tracks are preserved ({v:.1%}); the filter is not simply discarding everything"
    return f"true-track retention is only {v:.1%}; the filter is losing real targets"


def _interpret_hardness(cmp_df: pd.DataFrame, model: str) -> str:
    if cmp_df is None or not len(cmp_df):
        return "no comparison rows available"
    f04 = cmp_df[cmp_df["comparison_scope"].str.startswith("f04")]
    f02 = cmp_df[cmp_df["comparison_scope"] == "f02_point_detection_4_day_pooled"]
    if not len(f04) or not len(f02):
        return "F02 reference unavailable; hardness not assessed"
    a = f04.iloc[0].get("false_reduction")
    b = f02.iloc[0].get("false_reduction")
    if not (np.isfinite(a) and np.isfinite(b)):
        return ("one side's false_reduction is undefined; the two runs are not comparable on "
                "this metric")
    delta = a - b
    d = ("harder" if delta < -0.02 else "easier" if delta > 0.02 else "comparable")
    return (f"F04 false_reduction {a:.4f} vs F02 four-day pooled {b:.4f} (delta {delta:+.4f}): "
            f"cube-CFAR false tracks look {d}. Different detection source, frame count and "
            "threshold semantics -- suggestive, not a controlled comparison")


# =============================================================================
# Plots
# =============================================================================

def make_plots(plots_dir: str, det: pd.DataFrame, s08: pd.DataFrame, s12: pd.DataFrame,
               cmp_df: pd.DataFrame, model: str = "mlp_dae",
               audit: Optional[pd.DataFrame] = None) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(plots_dir)
    written = []

    def save(fig, name):
        p = os.path.join(plots_dir, name)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 1. detections per frame
    per = det.groupby("frame_id")["is_target"].agg(target="sum", total="size")
    per["clutter"] = per["total"] - per["target"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    idx = np.arange(len(per))
    ax.plot(idx, per["clutter"], lw=0.9, label=f"clutter (mean {per['clutter'].mean():.0f})")
    ax.plot(idx, per["target"], lw=0.9, label=f"target (mean {per['target'].mean():.0f})")
    ax.set_xlabel("frame index")
    ax.set_ylabel("detections")
    ax.set_title("F04 cube-derived CFAR detections per frame (evaluation-only labels)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.13, "Synthetic radar-cube CFAR output, not raw RF/IQ.",
             ha="center", fontsize=7, style="italic")
    save(fig, "f04_detection_counts.png")

    # 2. stage-08 track counts
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    if len(s08):
        r = s08.iloc[0]
        keys = ["confirmed_tracks", "true_tracks", "false_tracks"]
        vals = [float(r.get(k, np.nan)) for k in keys]
        finite = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar([k.replace("_", "\n") for k in keys], finite,
                      color=["#4C72B0", "#55A868", "#C44E52"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{int(v):,}" if np.isfinite(v) else "n/a",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("tracks")
    ax.set_title("Stage 08 tracks on F04 cube detections")
    ax.grid(alpha=0.3, axis="y")
    save(fig, "f04_track_counts.png")

    # 3. stage-12 filter effect -- includes the never-scored false tracks, which the
    #    scorer's own metrics table silently excludes from its denominator
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    sel = s12[s12["model"] == model] if len(s12) else s12
    if len(sel):
        r = sel.iloc[0]
        keys = [("stage08_true_tracks", "windowable\ntrue"),
                ("stage12_kept_true_tracks", "stage 12.5\nkept true"),
                ("stage08_false_tracks", "windowable\nfalse"),
                ("stage12_kept_false_tracks", "stage 12.5\nkept false")]
        vals = [float(r.get(k, np.nan)) for k, _ in keys]
        colors = ["#55A868", "#8FCFA1", "#C44E52", "#E39A9C"]
        labels = [lbl for _, lbl in keys]
        if audit is not None and len(audit):
            vals.append(float(audit["unscored_false_tracks"].iloc[0]))
            labels.append("false, never\nscored")
            colors.append("#7F7F7F")
        finite = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(labels, finite, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{int(v):,}" if np.isfinite(v) else "n/a",
                    ha="center", va="bottom", fontsize=9)
        ax.set_yscale("symlog")
    ax.set_ylabel("tracks (symlog)")
    ax.set_title(f"Stage 12.5 filter effect on F04 cube tracks ({model})")
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.5, -0.08, "The grey bar is confirmed false tracks with no window: stage 12.5 "
                         "never scores them, so they are absent from its reduction denominator.",
             ha="center", fontsize=7, style="italic")
    save(fig, "f04_stage12_filter_effect.png")

    # 4. F04 vs F02
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    if cmp_df is not None and len(cmp_df):
        labels = [s.replace("_", "\n") for s in cmp_df["comparison_scope"]]
        x = np.arange(len(labels))
        w = 0.36
        ret = [float(v) if pd.notna(v) else np.nan for v in cmp_df["true_retention"]]
        red = [float(v) if pd.notna(v) else np.nan for v in cmp_df["false_reduction"]]
        ax.bar(x - w / 2, [0 if not np.isfinite(v) else v for v in ret], w,
               label="true-track retention", color="#55A868")
        ax.bar(x + w / 2, [0 if not np.isfinite(v) else v for v in red], w,
               label="false-track reduction", color="#C44E52")
        for i, (a, b) in enumerate(zip(ret, red)):
            ax.text(i - w / 2, (a if np.isfinite(a) else 0), "n/a" if not np.isfinite(a)
                    else f"{a:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w / 2, (b if np.isfinite(b) else 0), "n/a" if not np.isfinite(b)
                    else f"{b:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_title("F04 cube-CFAR vs F02 point-detection (NOT directly comparable)")
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.5, -0.06,
             "Different detection source, frame count, day count and threshold semantics. "
             "Indicative only.", ha="center", fontsize=7, style="italic")
    save(fig, "f04_vs_f02_comparison.png")
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(report_dir: str, meta: Dict, date: str, s08: pd.DataFrame, s12: pd.DataFrame,
                 cmp_df: pd.DataFrame, findings: pd.DataFrame, stage08_ok: bool,
                 stage12_ok: bool, stage09_ok: Optional[bool], calibration_mode: str,
                 calibration_note: str, gate_lines: List[str], plots: List[str],
                 model: str = "mlp_dae", audit: Optional[pd.DataFrame] = None) -> str:
    ensure_dir(report_dir)
    scale = meta["threshold_scale"]
    tok = meta["alias_token"]
    lines = [
        "# Stage 22 Evaluation of F04 Cube-Derived CFAR Detections in F03",
        "",
        "## Status",
        "",
        "- Stage 22 evaluates **F04 cube-derived CFAR detections** using the existing F03 "
        "tracking/scoring pipeline.",
        f"- This is a small **{meta['frames']}-frame, single-day, single-operating-point "
        "stress test**.",
        "- It **adds no new model** and retrains nothing; stage-08 and stage-12.5 code is "
        "unchanged.",
        f"- **{NOT_RAW}**",
        f"- **F04 CFAR scale is not the same as F03 threshold dB.** Scale {scale:g} is a linear "
        f"multiplier on the local CFAR noise estimate, i.e. {scale_to_db(scale):.2f} dB above "
        f"it -- *not* a {scale:g} dB SNR threshold.",
        "",
        "## Input operating point",
        "",
        "| field | value |",
        "|---|---|",
        f"| date | {date} |",
        f"| frames | {meta['frames']} |",
        f"| cfar_type | `{meta['cfar_type']}` |",
        f"| threshold_scale | {scale:g} (= {scale_to_db(scale):.2f} dB above local noise) |",
        f"| cap (`max_detections_per_frame`) | {meta['cap']} |",
        "| Stage 20.5 recommendation | `B_balanced`, the only candidate with "
        "`stage21_recommended = True`; its cap does not bind |",
        f"| detections imported | {meta['rows']:,} "
        f"({meta['target_detections']:,} target / {meta['clutter_detections']:,} clutter) |",
        "",
        "The cap not binding matters: a binding cap truncates each frame strongest-first and "
        "turns Pd and false-alarm rate into lower bounds rather than measurements. Stage 21 "
        f"verified the observed maximum was 1,496 detections/frame against the cap of "
        f"{meta['cap']}.",
        "",
        "## Import and aliasing",
        "",
        f"- **Original F04 export** (never modified): `{meta['original_file']}`",
        f"- **Imported into F03 as**: `{os.path.basename(meta['imported_file'])}`",
        f"- **Compatibility alias**: `{meta['alias_file']}`",
        f"- **Alias metadata**: `{os.path.basename(meta['metadata_path'])}`",
        "",
        f"> **{meta['alias_warning']}**",
        "",
        "F03's stage-08 discovers detection files only via "
        "`detections_<date>_thr_<token>dB.csv` and filters them with a numeric "
        "`--threshold-db`. The alias exists purely to satisfy that parser. Nothing in F03 "
        "interprets the token beyond matching it.",
        "",
    ]

    if meta["adaptations"]:
        lines += ["### Adaptations applied on import", ""]
        lines += [f"{i+1}. {a}" for i, a in enumerate(meta["adaptations"])]
        lines += [""]
        ts = meta.get("timestamp_check", {})
        if ts.get("timestamp_reconstructed"):
            lines += [
                "The clock repair deserves emphasis. The F04 Stage 21 writer serializes with "
                "`float_format=\"%.6g\"`, which rounds epoch seconds (~1.65e9) to six "
                f"significant digits: the export carries only "
                f"**{ts['unique_timestamps_original']} distinct timestamps for "
                f"{ts['frames']} frames**. Stage 08 is immune (it takes `dt` from "
                "`--frame-period-s`), but stage 12's window features differentiate position "
                "with respect to `timestamp`, so the original column would have silently "
                "corrupted every velocity feature. Stage 22 reconstructs "
                f"`timestamp = frame_id * {ts['frame_period_s']:g}s` -- F04's cube simulator "
                "defines `frame_id = floor(timestamp / scan_period_s)` -- and verifies the "
                "reconstruction rounds to the same six-significant-digit value the export "
                f"carries (agreement: {ts['reconstruction_matches_rounded_original']}). "
                "**This is a real bug in F04's Stage 21 writer and should be fixed there** "
                "(exclude `timestamp` from the float format, or write it as an integer). "
                "Stage 22 works around it without modifying F04.",
                "",
            ]
    else:
        lines += ["No schema adaptation was required.", ""]

    lines += ["## Stage 08 Kalman tracking on F04 detections", ""]
    lines += [f"Stage 08 ran: **{'yes' if stage08_ok else 'no'}**.", ""]
    lines += md_table(s08.drop(columns=["notes"], errors="ignore")) + [""]
    if len(s08) and str(s08.iloc[0].get("notes", "")):
        lines += [f"Notes: {s08.iloc[0]['notes']}", ""]

    lines += ["## Stage 12.5 sequence-prior scoring on F04 tracks", ""]
    lines += [f"Stage 12.5 ran: **{'yes' if stage12_ok else 'no'}**. "
              f"Calibration mode: **`{calibration_mode}`**.", ""]
    if calibration_note:
        lines += [f"> {calibration_note}", ""]
    lines += [
        "The noise-matched (`track_purity`) calibration was built from the **F04 cube tracks "
        "themselves**, by pinning `--calibration-tracks-dir` at the F04 track directory. This "
        "is not a detail: that flag defaults to the canonical F02 `data/active/tracks_kalman`, "
        "which happens to contain a `thr_6p0dB` file too, so omitting it would have quietly "
        "calibrated on F02 point-detection tracks while scoring F04 cube-CFAR tracks -- "
        "exactly the noise mismatch stage 12.5 exists to remove. The calibration JSON is "
        "written inside this stage's report directory; the canonical stage-12 calibration is "
        "untouched.",
        "",
    ]
    if len(s12):
        lines += md_table(s12.drop(columns=["notes"], errors="ignore")) + [""]

    if audit is not None and len(audit):
        w = audit[audit["denominator"] == "stage12_windowable_only"].iloc[0]
        k = audit[audit["denominator"] == "stage08_all_unscored_kept"].iloc[0]
        d = audit[audit["denominator"] == "stage08_all_unscored_dropped"].iloc[0]
        lines += [
            "### Which false tracks? (the denominator matters here)",
            "",
            "Stage 12 scores only tracks that yield at least one window. Its "
            "`stage08_true_tracks` / `stage08_false_tracks` columns are therefore the "
            "**windowable subset**, not what stage 08 confirmed. On F04 cube tracks that gap "
            "is not a footnote:",
            "",
            f"- stage 08 confirmed **{k['false_tracks']:,.0f}** false tracks",
            f"- only **{w['false_tracks']:,.0f}** of them are windowable, so "
            f"**{audit['unscored_false_tracks'].iloc[0]:,.0f} false tracks are never scored by "
            "the filter at all**",
            "",
            "The headline reduction is over the windowable subset. The pipeline-level number "
            "depends on a policy stage 12.5 does not define -- what happens to a confirmed "
            "track the filter never scored -- so both branches are reported:",
            "",
        ]
        lines += md_table(audit.drop(columns=["notes", "description"], errors="ignore")) + [""]
        lines += [
            f"Retaining unscored tracks gives a pipeline false-track reduction of "
            f"**{k['false_track_reduction']:.1%}** at **{k['true_track_retention']:.1%}** true "
            f"retention. Dropping them gives **{d['false_track_reduction']:.1%}** reduction but "
            f"collapses true retention to **{d['true_track_retention']:.1%}** -- it buys "
            "suppression by throwing away most real targets. Neither branch reproduces the F02 "
            "result.",
            "",
            "The F02 stage-12/17 figures use the same windowable denominator, so the "
            "`stage12_windowable_only` row is the like-for-like comparison. What differs is how "
            "*much* of the track population is windowable: cube-CFAR clutter produces a large "
            "population of short, unwindowable false tracks that the F02 point-detection "
            "simulator did not.",
            "",
        ]

    if stage09_ok is not None:
        lines += ["## Stage 09 hand-physics scoring (optional)", "",
                  f"Stage 09 ran: **{'yes' if stage09_ok else 'no'}**.", ""]

    lines += [
        "## Comparison with original F02 point-detection experiment",
        "",
        "The two experiments differ in **detection source, frame count, day count and "
        "threshold semantics simultaneously**, so no single difference between them is "
        "attributable to any one cause.",
        "",
        f"- The F02/F03 headline result is a **four-day, multi-threshold point-detection** "
        "experiment, where the threshold really is an SNR threshold in dB.",
        f"- The F04 result is a **{meta['frames']}-frame, one-day, one-operating-point "
        "structured cube-CFAR stress test**, where the `6` in the filename is a linear CFAR "
        "scale.",
        "",
        "**The F04 cube-derived result is a small structured-clutter stress test, not a direct "
        "replacement for the four-day F02 result.**",
        "",
    ]
    if cmp_df is not None and len(cmp_df):
        lines += md_table(cmp_df.drop(columns=["notes"], errors="ignore")) + [""]

    lines += ["## Interpretation", ""]
    sel = s12[s12["model"] == model] if len(s12) else s12
    lines += [f"- **Did Stage 08 run?** {'Yes.' if stage08_ok else 'No -- see the run log.'}"]
    lines += [f"- **Did Stage 12.5 run?** {'Yes.' if stage12_ok else 'No -- see the run log.'}"]
    if stage12_ok and len(sel):
        r = sel.iloc[0]
        lines += [f"- **Did Stage 12.5 suppress F04 false tracks?** "
                  f"{_interpret_reduction(r.get('false_track_reduction'), r.get('stage08_false_tracks'))}"]
        if audit is not None and len(audit):
            k = audit[audit["denominator"] == "stage08_all_unscored_kept"].iloc[0]
            lines += [f"  Measured against **all** stage-08 false tracks (unscored ones passing "
                      f"through), the suppression is only "
                      f"**{k['false_track_reduction']:.1%}**."]
        lines += [f"- **Were true tracks preserved?** "
                  f"{_interpret_retention(r.get('true_track_retention'))}"]
    lines += [f"- **Are F04 CFAR false tracks harder or easier?** {_interpret_hardness(cmp_df, model)}",
              ""]
    if stage12_ok and len(sel) and audit is not None and len(audit):
        lines += [
            "**Answer to the research question.** The stage-12.5 sequence-prior filter does "
            "**not** carry over. On F02 point detections it removed 99.4% of false tracks at "
            "97.3% true-track retention; on cube-derived CA-CFAR detections at the same "
            "windowable denominator it removes "
            f"{sel.iloc[0].get('false_track_reduction'):.1%} at "
            f"{sel.iloc[0].get('true_track_retention'):.1%} retention, and most cube-CFAR false "
            "tracks are too short to be scored at all. Structured CFAR clutter produces "
            "false tracks that both evade the filter's window requirement and, when scored, "
            "look far more target-like than the F02 simulator's independent point clutter. "
            "This is a negative result for transfer, not a bug: nothing was retrained, and the "
            "filter is being asked to generalize across detection statistics it never saw.",
            "",
        ]

    lines += [
        "## Limitations",
        "",
        f"- Only **{meta['frames']} frames**, one day.",
        "- A **single CFAR operating point**; no sweep.",
        "- **Synthetic cube abstraction**: injected point-target energy in a "
        "range-Doppler-azimuth intensity cube.",
        "- **No raw RF/IQ**, no waveform or PRF modelling.",
        "- **No elevation grid** in F04: clutter carries `meas_elevation_rad = 0` and labelled "
        "targets borrow the nearest truth elevation, so elevation is not measured.",
        "- F03 was built around **F02 point-detection filenames** and required an alias to "
        "ingest F04 output at all.",
        f"- The `thr_{tok}dB` alias token is a **compatibility label**, not a dB threshold.",
        "- Target/clutter labels are **evaluation-only**; neither the tracker nor the scorer "
        "sees them.",
        "",
        "## Recommended next stage",
        "",
    ]
    if stage08_ok and stage12_ok:
        lines += [
            "Stage 08 and Stage 12.5 both ran end-to-end on cube-derived detections, so the "
            "adapter is sound. **Stage 23 should run an F04 operating-point sweep through "
            "F03**, using 2-3 of the CFAR operating points Stage 20.5 recommended "
            "(`A_high_pd` scale 4 / cap 10000, `B_balanced` scale 6 / cap 2000, "
            "`C_conservative` scale 12 / cap 2000), so false-track difficulty can be traced "
            "against detector aggressiveness rather than inferred from one point.",
            "",
            "Two specific questions Stage 23 should answer, both raised by this run:",
            "",
            "1. **Does the unwindowable-false-track population shrink at a stricter CFAR "
            "threshold?** If most cube-CFAR false tracks are short because clutter detections "
            "are dense and transient, `C_conservative` should thin them out. If it does not, "
            "the tracker's confirmation logic -- not the filter -- is what needs attention.",
            "2. **Is the transfer gap a calibration gap or a training-distribution gap?** The "
            "filter here was calibrated on F04 tracks but *trained* on F02 trajectories. A "
            "stage-23 run that retrains the sequence prior on cube-derived tracks would "
            "separate the two. That is a training stage and is deliberately out of Stage 22's "
            "scope.",
        ]
    else:
        lines += [
            "The pipeline did **not** complete end-to-end on cube-derived detections. "
            "**Stage 23 should fix the F03 adapter/tracker compatibility before broader "
            "experiments** -- a sweep over more operating points would only multiply the same "
            "failure.",
        ]
    lines += ["", "## Validation gate", ""] + [f"- {g}" for g in gate_lines] + [""]
    if plots:
        lines += ["## Plots", ""] + [f"- `plots/{os.path.basename(p)}`" for p in plots] + [""]
    lines += ["---", "", f"> {NOT_RAW}", "",
              f"> {meta['alias_warning']}", ""]

    path = os.path.join(report_dir, "stage22_f04_cube_eval_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# =============================================================================
# Validation gate
# =============================================================================

def run_gate(report_dir: str, detections_dir: str, meta: Dict, date: str, report_path: str,
             ran_stage08: bool, ran_stage12: bool, stage08_ok: bool, stage12_ok: bool,
             canonical_calibration_sha: Optional[str],
             canonical_calibration_path: str) -> List[str]:
    out = []

    def chk(label, ok, detail=""):
        out.append(f"{label}: {'OK' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}")
        return ok

    print("\n" + "=" * 70)
    print("VALIDATION GATE (stage 22 F04 cube evaluation)")
    print("=" * 70)

    imp = os.path.join(report_dir, "f04_import_summary.csv")
    df = _read_csv(imp)
    chk("import summary exists and is nonempty", df is not None)
    meta_path = os.path.join(detections_dir, f"f04_cube_alias_metadata_{date}.json")
    has_meta = os.path.exists(meta_path)
    chk("alias metadata JSON exists", has_meta)
    warn_ok = False
    if has_meta:
        with open(meta_path) as f:
            j = json.load(f)
        warn_ok = "not a" in j.get("alias_warning", "") and "dB" in j.get("alias_warning", "")
    chk("alias warning present in metadata", warn_ok)
    chk("imported detection CSV exists", os.path.exists(meta["imported_file"]))
    chk("compatibility alias CSV exists", os.path.exists(meta["alias_path"]))
    chk("required F03 stage-08 input columns present", meta["schema_valid"],
        "" if meta["schema_valid"] else f"missing {meta['missing_columns']}")

    ts = meta.get("timestamp_check", {})
    if ts.get("timestamp_reconstructed"):
        chk("reconstructed timestamps agree with the rounded original",
            bool(ts.get("reconstruction_matches_rounded_original")))
        chk("reconstructed clock has one timestamp per frame",
            ts.get("unique_timestamps_reconstructed") == ts.get("frames"),
            f"{ts.get('unique_timestamps_reconstructed')} unique vs {ts.get('frames')} frames")

    if ran_stage08:
        s8 = os.path.join(report_dir, "stage08_kalman_f04_cube", "kalman_baseline_report.md")
        chk("stage 08 report/metrics exist (or failure documented)",
            os.path.exists(s8) or not stage08_ok,
            "" if os.path.exists(s8) else "stage 08 failed; documented in report + run log")
    if ran_stage12:
        s12 = os.path.join(report_dir, "stage12_sequence_f04_cube",
                           "sequence_metrics_by_model_threshold.csv")
        chk("stage 12 report/metrics exist (or failure documented)",
            os.path.exists(s12) or not stage12_ok,
            "" if os.path.exists(s12) else "stage 12 failed; documented in report + run log")

    text = open(report_path).read() if os.path.exists(report_path) else ""
    needles = ["not raw RF/IQ", "CFAR scale", "not the same as", "Stage 12.5", "Stage 23"]
    miss = [n for n in needles if n not in text]
    chk("report contains the required statements", not miss, f"missing {miss}" if miss else "")

    if canonical_calibration_sha is not None:
        now = _sha256(canonical_calibration_path)
        chk("canonical stage-12 calibration file unchanged", now == canonical_calibration_sha,
            "" if now == canonical_calibration_sha else "CANONICAL CALIBRATION WAS MODIFIED")

    big = _large_staged_csvs()
    chk("no large CSVs staged for commit", not big, f"{big}" if big else "")

    for line in out:
        print("  " + line)
    return out


def _sha256(path: str) -> Optional[str]:
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _large_staged_csvs(limit_bytes: int = 5_000_000) -> List[str]:
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    big = []
    for name in out.stdout.split():
        p = os.path.join(REPO_ROOT, name)
        if name.endswith(".csv") and os.path.exists(p) and os.path.getsize(p) > limit_bytes:
            big.append(name)
    return big
