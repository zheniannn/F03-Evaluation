"""Stage 22: evaluate F04 cube-derived CFAR detections with the F03 stack.

Research question: does the stage-12.5 learned sequence-prior filter still suppress false
tracks when detections come from structured radar-cube CFAR output instead of the simpler
F02 point-detection simulator?

Stage 22 is an evaluation/adaptation stage. It retrains nothing, adds no model, changes no
stage-08 or stage-12 logic, and never modifies F04. It imports one F04 Stage-21 export,
adapts it to F03's detection schema/filename contract, runs stage 08 then stage 12.5 over
it, and writes compact comparison artifacts.

**F04 `threshold_scale = 6.0` is a linear CFAR noise multiplier (~7.78 dB), NOT a 6 dB SNR
threshold.** The `thr_6p0dB` filename token is a compatibility label only.

Usage:
    python scripts/22_evaluate_f04_cube_detections.py --self-test
    python scripts/22_evaluate_f04_cube_detections.py \
        --f04-export ../F04-RADAR-CUBE/data/active/f03_exports/f04_cube_detections_2022-06-06_cfar_ca_scale_6p0_cap_2000.csv \
        --date 2022-06-06 --run-stage08 --run-stage12 --overwrite
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT, token_to_threshold
from utils.f04_cube_eval import (ALIAS_WARNING, NOT_RAW, _sha256, alias_warning,
                                 build_f02_comparison, build_key_findings,
                                 build_stage08_command, build_stage09_command,
                                 build_stage12_command, build_windowability_audit, ensure_dir,
                                 extract_stage08_summary, extract_stage12_summary,
                                 import_f04_export, make_plots, run_command, run_gate,
                                 scale_to_db, validate_schema, write_import_summary,
                                 write_report)

CANONICAL_CALIBRATION = os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors",
                                     "calibration", "sequence_track_calibration.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 22: evaluate F04 cube-derived CFAR detections in F03 "
                    "(no retraining; CFAR scale is not dB).")
    p.add_argument("--f04-export", type=str, default=None,
                   help="Path to the F04 stage-21 export CSV. Required unless --self-test.")
    p.add_argument("--truth-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    p.add_argument("--detections-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "f04_cube_detections"))
    p.add_argument("--tracks-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman_f04_cube"))
    p.add_argument("--report-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage22_f04_cube_eval"))
    p.add_argument("--models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--stage12-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"),
                   help="Canonical stage-12 reports, read only for the F02 comparison.")
    p.add_argument("--stage09-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    p.add_argument("--stage11-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage11_adsb_prior_scoring"))
    p.add_argument("--four-day-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage17_four_day_validation"))

    p.add_argument("--date", type=str, default="2022-06-06")
    p.add_argument("--cfar-type", type=str, default="ca")
    p.add_argument("--threshold-scale", type=float, default=6.0)
    p.add_argument("--cap", type=int, default=2000)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--frame-period-s", type=float, default=10.0)

    p.add_argument("--alias-token", type=str, default="6p0",
                   help="F03 compatibility filename token ONLY. Not a dB threshold.")
    p.add_argument("--model", type=str, nargs="+", default=["mlp_dae"],
                   help="Sequence-prior models to score with (stage 14 selected mlp_dae).")
    p.add_argument("--primary-model", type=str, default="mlp_dae")
    p.add_argument("--score-threshold", type=float, default=0.5)

    p.add_argument("--run-stage08", action="store_true")
    p.add_argument("--run-stage09", action="store_true")
    p.add_argument("--run-stage12", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def run(args) -> dict:
    ensure_dir(args.report_dir)
    ensure_dir(os.path.join(args.report_dir, "plots"))
    ensure_dir(os.path.join(args.report_dir, "generated"))
    logs = os.path.join(args.report_dir, "generated")

    scale = args.threshold_scale
    alias_thr = token_to_threshold(args.alias_token)

    print("=" * 70)
    print("Stage 22: F04 cube-derived CFAR detections through the F03 stack")
    print("=" * 70)
    print(f"operating point: cfar={args.cfar_type} scale={scale:g} cap={args.cap} "
          f"frames<={args.max_frames}")
    print(f"WARNING: CFAR scale {scale:g} = {scale_to_db(scale):.2f} dB above local noise. "
          f"The alias token thr_{args.alias_token}dB is a filename label, NOT a dB threshold.")

    canonical_sha = _sha256(CANONICAL_CALIBRATION)

    # --- import + adapt ---------------------------------------------------
    meta = import_f04_export(args.f04_export, args.detections_dir, args.date, args.cfar_type,
                             scale, args.cap, args.alias_token,
                             frame_period_s=args.frame_period_s, max_frames=args.max_frames,
                             overwrite=args.overwrite or args.skip_existing)
    det = meta.pop("detections")
    print(f"\nimported {meta['rows']:,} detections over {meta['frames']} frames "
          f"({meta['target_detections']:,} target / {meta['clutter_detections']:,} clutter)")
    for a in meta["adaptations"]:
        print(f"  ADAPTED: {a}")
    ok, missing = validate_schema(det)
    if not ok:
        raise SystemExit(f"Imported F04 export is missing stage-08 columns: {missing}")

    write_import_summary(meta, args.date, os.path.join(args.report_dir,
                                                       "f04_import_summary.csv"))

    # --- stage 08 ---------------------------------------------------------
    stage08_ok, stage08_note = False, ""
    if args.run_stage08:
        r = run_command(build_stage08_command(args, alias_thr), "Stage 08 Kalman tracking",
                        os.path.join(logs, "stage08_run.log"))
        stage08_ok = r["ok"]
        if not stage08_ok:
            stage08_note = f"stage 08 exited {r['returncode']}; see generated/stage08_run.log"
    else:
        stage08_note = "stage 08 not requested (--run-stage08 omitted)"

    s08 = extract_stage08_summary(args.report_dir, meta, args.date, args.alias_token,
                                  stage08_ok, stage08_note)
    s08.to_csv(os.path.join(args.report_dir, "f04_stage08_summary.csv"), index=False)

    # --- stage 09 (optional) ---------------------------------------------
    stage09_ok = None
    if args.run_stage09:
        if not stage08_ok:
            stage09_ok = False
            print("\nskipping stage 09: stage 08 produced no tracks")
        else:
            r = run_command(build_stage09_command(args, alias_thr),
                            "Stage 09 hand-physics scoring",
                            os.path.join(logs, "stage09_run.log"))
            stage09_ok = r["ok"]
            if not stage09_ok:
                print("\nstage 09 failed on F04 tracks -- documented and skipped "
                      "(schema incompatibility is an expected outcome here)")

    # --- stage 12.5 -------------------------------------------------------
    stage12_ok, stage12_note = False, ""
    calibration_mode = "track_purity"
    calib_out = os.path.join(args.report_dir, "generated_calibration",
                             f"sequence_track_calibration_f04_cube_{args.date}.json")
    if args.run_stage12:
        if not stage08_ok:
            stage12_note = "stage 12.5 skipped: stage 08 produced no F04 tracks to score"
            print("\n" + stage12_note)
        else:
            r = run_command(build_stage12_command(args, alias_thr, "track_purity", calib_out),
                            "Stage 12.5 sequence-prior scoring (track_purity calibration)",
                            os.path.join(logs, "stage12_run.log"))
            stage12_ok = r["ok"]
            if not stage12_ok:
                # Do not hide calibration failure: report it, then try clean_truth explicitly.
                stage12_note = (f"track_purity calibration FAILED on the F04 tracks "
                                f"(exit {r['returncode']}). ")
                print(f"\n{stage12_note}Falling back to clean_truth calibration and "
                      "reporting the limitation.")
                calibration_mode = "clean_truth"
                r2 = run_command(
                    build_stage12_command(args, alias_thr, "clean_truth", calib_out),
                    "Stage 12.5 sequence-prior scoring (clean_truth fallback)",
                    os.path.join(logs, "stage12_fallback_run.log"))
                stage12_ok = r2["ok"]
                stage12_note += (
                    "Fell back to clean_truth calibration, which is NOT noise-matched to the "
                    "cube-CFAR detections; retention/reduction figures below are therefore "
                    "not a like-for-like stage-12.5 result."
                    if stage12_ok else
                    f"clean_truth fallback also failed (exit {r2['returncode']}). "
                    "Stage 12.5 calibration could not be performed on this subset.")
    else:
        stage12_note = "stage 12.5 not requested (--run-stage12 omitted)"

    s12 = extract_stage12_summary(args.report_dir, meta, args.date, args.alias_token,
                                  calibration_mode, args.score_threshold, stage12_ok,
                                  stage12_note)
    s12.to_csv(os.path.join(args.report_dir, "f04_stage12_summary.csv"), index=False)

    # --- comparison, findings, plots, report -------------------------------
    audit = build_windowability_audit(args.report_dir, s08, s12, args.primary_model)

    cmp_df = build_f02_comparison(s12, args.stage12_dir, args.four_day_dir, args.date,
                                  args.alias_token, meta, args.primary_model)
    cmp_df.to_csv(os.path.join(args.report_dir, "f04_cube_vs_f02_point_comparison.csv"),
                  index=False)

    findings = build_key_findings(meta, s08, s12, cmp_df, stage08_ok, stage12_ok,
                                  calibration_mode, stage12_note, args.primary_model, audit)
    findings.to_csv(os.path.join(args.report_dir, "stage22_key_findings.csv"), index=False)

    plots = make_plots(os.path.join(args.report_dir, "plots"), det, s08, s12, cmp_df,
                       args.primary_model, audit)

    report = write_report(args.report_dir, meta, args.date, s08, s12, cmp_df, findings,
                          stage08_ok, stage12_ok, stage09_ok, calibration_mode, stage12_note,
                          ["(pending)"], plots, args.primary_model, audit)
    gate = run_gate(args.report_dir, args.detections_dir, meta, args.date, report,
                    args.run_stage08, args.run_stage12, stage08_ok, stage12_ok,
                    canonical_sha, CANONICAL_CALIBRATION)
    report = write_report(args.report_dir, meta, args.date, s08, s12, cmp_df, findings,
                          stage08_ok, stage12_ok, stage09_ok, calibration_mode, stage12_note,
                          gate, plots, args.primary_model, audit)

    print(f"\nreport: {os.path.abspath(report)}")
    print(f"stage 08 ran: {stage08_ok}    stage 12.5 ran: {stage12_ok} "
          f"(calibration={calibration_mode})")
    print("\nkey findings:")
    print(findings[["finding_id", "metric", "value"]].to_string(index=False))
    return {"meta": meta, "s08": s08, "s12": s12, "cmp": cmp_df, "findings": findings,
            "report": report, "stage08_ok": stage08_ok, "stage12_ok": stage12_ok}


# =============================================================================
# Self-test
# =============================================================================

def _fake_export(path: str, date: str = "2022-01-01", frames: int = 6) -> None:
    """A miniature F04-style export, including the 6-significant-digit timestamp defect."""
    rng = np.random.default_rng(0)
    rows = []
    did = 0
    for k in range(frames):
        fid = 165447360 + k
        ts = fid * 10.0
        for j in range(2):                       # targets
            r = 20_000.0 + 500 * k + 100 * j
            rows.append(dict(date=date, frame_id=fid, timestamp=ts, detection_id=did,
                             source="f04_cube_cfar", scenario="f04_cube_cfar", is_target=1,
                             trajectory_id=f"traj_{j}", meas_range_m=r,
                             meas_azimuth_rad=0.4 + 0.01 * k + j, meas_elevation_rad=0.02,
                             meas_radial_velocity_mps=40.0, snr_db=15.0,
                             x_m=r * 0.6, y_m=r * 0.8, z_m=400.0,
                             truth_x_m=r * 0.6, truth_y_m=r * 0.8, truth_z_m=400.0,
                             truth_range_m=r, truth_azimuth_rad=0.4 + 0.01 * k + j,
                             truth_elevation_rad=0.02))
            did += 1
        for _ in range(3):                       # clutter
            r = float(rng.uniform(5_000, 60_000))
            rows.append(dict(date=date, frame_id=fid, timestamp=ts, detection_id=did,
                             source="f04_cube_cfar", scenario="f04_cube_cfar", is_target=0,
                             trajectory_id=np.nan, meas_range_m=r,
                             meas_azimuth_rad=float(rng.uniform(0, 6.28)),
                             meas_elevation_rad=0.0, meas_radial_velocity_mps=-3.0,
                             snr_db=9.0, x_m=r * 0.5, y_m=r * 0.5, z_m=0.0,
                             truth_x_m=np.nan, truth_y_m=np.nan, truth_z_m=np.nan,
                             truth_range_m=np.nan, truth_azimuth_rad=np.nan,
                             truth_elevation_rad=np.nan))
            did += 1
    ensure_dir(os.path.dirname(path))
    # reproduce F04's lossy writer: this is what makes the timestamp repair necessary
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def self_test() -> None:
    date = "2022-01-01"
    with tempfile.TemporaryDirectory() as tmp:
        export = os.path.join(tmp, "src", "f04_cube_detections_x.csv")
        _fake_export(export, date)

        args = parse_args([])
        args.f04_export = export
        args.detections_dir = os.path.join(tmp, "det")
        args.tracks_dir = os.path.join(tmp, "tracks")
        args.report_dir = os.path.join(tmp, "reports")
        args.date = date
        args.max_frames = 6
        args.overwrite = True
        args.run_stage08 = args.run_stage09 = args.run_stage12 = False

        # the lossy export really is degenerate before repair
        raw = pd.read_csv(export)
        assert raw["timestamp"].nunique() < raw["frame_id"].nunique(), \
            "fixture should reproduce the collapsed-clock defect"

        meta = import_f04_export(export, args.detections_dir, date, "ca", 6.0, 2000, "6p0",
                                 frame_period_s=10.0, max_frames=6, overwrite=True)
        det = meta.pop("detections")

        assert os.path.exists(meta["metadata_path"]), "alias metadata missing"
        with open(meta["metadata_path"]) as f:
            j = json.load(f)
        assert "not a" in j["alias_warning"] and "6 dB" in j["alias_warning"], \
            f"alias warning must say it is not a 6 dB threshold: {j['alias_warning']!r}"
        assert os.path.exists(meta["alias_path"]), "alias CSV missing"
        assert validate_schema(det)[0], "imported schema invalid"

        # the repair worked: one timestamp per frame, and it matches the rounded original
        ts = meta["timestamp_check"]
        assert ts["timestamp_reconstructed"], "timestamp should have been reconstructed"
        assert ts["reconstruction_matches_rounded_original"], "reconstruction disagrees"
        assert det["timestamp"].nunique() == det["frame_id"].nunique() == 6, \
            "repaired clock must have one timestamp per frame"

        write_import_summary(meta, date, os.path.join(args.report_dir,
                                                      "f04_import_summary.csv"))
        assert os.path.exists(os.path.join(args.report_dir, "f04_import_summary.csv")), \
            "import summary missing"

        # commands are built, never executed
        c8 = build_stage08_command(args, 6.0)
        c9 = build_stage09_command(args, 6.0)
        c12 = build_stage12_command(args, 6.0, "track_purity",
                                    os.path.join(args.report_dir, "generated_calibration",
                                                 "cal.json"))
        for c, name in ((c8, "stage08"), (c9, "stage09"), (c12, "stage12")):
            assert c[0] == sys.executable, f"{name} command must start with sys.executable"
            assert c[0] != "python", f"{name} must not invoke a bare `python`"
        assert "--calibration-tracks-dir" in c12 and args.tracks_dir in c12, \
            "stage 12 must calibrate on the F04 tracks, not the canonical F02 tracks"
        assert args.report_dir in c12[c12.index("--calibration-output") + 1], \
            "stage 12 calibration output must stay inside the stage-22 report dir"

        s08 = extract_stage08_summary(args.report_dir, meta, date, "6p0", False, "not run")
        s12 = extract_stage12_summary(args.report_dir, meta, date, "6p0", "track_purity",
                                      0.5, False, "not run")

        # windowability audit: synthetic stage-08/stage-12 rows where 90 of 100 false tracks
        # are too short to window, so the filter never sees them
        fake08 = pd.DataFrame([{"true_tracks": 50.0, "false_tracks": 100.0}])
        fake12 = pd.DataFrame([{"model": "mlp_dae", "stage08_true_tracks": 40.0,
                                "stage08_false_tracks": 10.0,
                                "stage12_kept_true_tracks": 38.0,
                                "stage12_kept_false_tracks": 2.0}])
        audit = build_windowability_audit(args.report_dir, fake08, fake12)
        assert len(audit) == 3, "audit must report all three denominators"
        assert float(audit["unscored_false_tracks"].iloc[0]) == 90.0, "unscored false count"
        win = audit[audit["denominator"] == "stage12_windowable_only"].iloc[0]
        keep = audit[audit["denominator"] == "stage08_all_unscored_kept"].iloc[0]
        assert abs(win["false_track_reduction"] - 0.8) < 1e-9, "windowable reduction 1-2/10"
        assert abs(keep["false_track_reduction"] - 0.08) < 1e-9, \
            "pipeline reduction must be 1-(2+90)/100 = 0.08, far below the windowable figure"

        cmp_df = build_f02_comparison(s12, os.path.join(tmp, "nope"),
                                      os.path.join(tmp, "nope"), date, "6p0", meta)
        findings = build_key_findings(meta, s08, s12, cmp_df, False, False, "track_purity",
                                      "not run", "mlp_dae", audit)
        plots = make_plots(os.path.join(args.report_dir, "plots"), det, s08, s12, cmp_df,
                           "mlp_dae", audit)
        assert len(plots) == 4, f"expected 4 plots, got {len(plots)}"

        report = write_report(args.report_dir, meta, date, s08, s12, cmp_df, findings,
                              False, False, None, "track_purity", "not run", ["gate: n/a"],
                              plots, "mlp_dae", audit)
        text = open(report).read()
        for needle in ["CFAR scale", "not raw RF/IQ", "not the same as", "Stage 12.5",
                       "Stage 23"]:
            assert needle in text, f"report missing {needle!r}"

        print(f"\nself-test: imported {meta['rows']} detections over {meta['frames']} frames; "
              f"clock repaired ({ts['unique_timestamps_original']} -> "
              f"{ts['unique_timestamps_reconstructed']} unique timestamps); "
              f"{len(plots)} plots; commands use {os.path.basename(sys.executable)}")
    print("\nStage 22 F04 cube evaluation self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.f04_export:
        raise SystemExit("--f04-export is required (or pass --self-test)")
    run(args)
    print("\n22_evaluate_f04_cube_detections completed successfully.")


if __name__ == "__main__":
    main()
