"""Stage 17.5: regression checks for the three bugs found during four-day validation.

These are guards, not research. Each check pins one failure mode that actually
happened, so it cannot silently return:

  R1  every internal Python subprocess uses sys.executable (never a bare `python`,
      which does not exist on some machines);
  R2  orchestrators pass --calibration-output inside their OWN report directory,
      so a per-day rerun can never overwrite the canonical stage-12 calibration;
  R3  a zero false-track denominator yields an UNDEFINED (NaN) false-track
      reduction, explicitly labelled as such -- not 0, not 1, and never
      misattributed to missing stage-09 data;
  R4  large generated per-track score CSVs are git-ignored.

Usage:
    python scripts/17p5_regression_checks.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import (REPO_ROOT, NO_WINDOWABLE_FALSE, safe_reduction, summarize_defined,
                          undefined_reason)

CANONICAL_CALIB_DIR = os.path.join("reports", "stage12_sequence_priors", "calibration")
PASSED, FAILED = [], []


def check(name, fn):
    try:
        detail = fn()
        PASSED.append(name)
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"  FAIL  {name} -- {exc}")


# =============================================================================
# R1: internal subprocess calls must use sys.executable
# =============================================================================

def _capture_commands(module_path, module_name, build):
    """Import a script and capture the command lists it hands to subprocess.run."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved_argv

    captured = []

    class _Result:
        returncode = 0

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        return _Result()

    real_run = mod.subprocess.run
    mod.subprocess.run = fake_run
    try:
        build(mod)
    finally:
        mod.subprocess.run = real_run
    return captured


def r1_stage17_uses_sys_executable():
    def build(mod):
        args = mod.parse_args()
        args.output_dir = tempfile.mkdtemp()
        args.tracks_dir = args.detections_dir = args.truth_dir = tempfile.mkdtemp()
        args.models_dir = tempfile.mkdtemp()
        args.run_missing_stage08 = args.run_missing_stage09 = args.run_missing_stage12 = True
        avail = pd.DataFrame([
            {"date": "2022-06-13", "threshold_db": -5.0, "stage08_metrics_available": False,
             "stage09_metrics_available": False, "stage12_mlp_available": False,
             "stage12_gru_available": False}])
        mod.run_missing(args, avail)

    cmds = _capture_commands(os.path.join(REPO_ROOT, "scripts", "17_four_day_validation.py"),
                             "s17", build)
    assert cmds, "no subprocess commands were captured"
    for c in cmds:
        assert c[0] == sys.executable, f"command does not use sys.executable: {c[0]!r}"
        assert c[0] not in ("python", "python3"), f"bare interpreter name: {c[0]!r}"
    return f"{len(cmds)} stage-17 commands all use sys.executable"


def r1_no_bare_python_in_source():
    """Static guard: no `["python", ...]` command construction anywhere in scripts/utils."""
    offenders = []
    self_name = os.path.basename(__file__)   # this checker holds the patterns as literals
    for root in ("scripts", "utils"):
        for fn in sorted(os.listdir(os.path.join(REPO_ROOT, root))):
            if not fn.endswith(".py") or fn == self_name:
                continue
            path = os.path.join(REPO_ROOT, root, fn)
            for i, line in enumerate(open(path), 1):
                stripped = line.strip()
                if stripped.startswith("#") or "f\"python " in line or 'f"python' in stripped:
                    continue  # documentation strings for run plans are fine
                if '["python"' in line or "['python'" in line or '["python3"' in line:
                    offenders.append(f"{root}/{fn}:{i}")
    assert not offenders, f"bare python in command list: {offenders}"
    return "no `[\"python\", ...]` command construction in scripts/ or utils/"


# =============================================================================
# R2: orchestrators must not target the canonical calibration directory
# =============================================================================

def r2_stage17_calibration_output_is_sandboxed():
    out_dir = os.path.join(REPO_ROOT, "reports", "stage17_four_day_validation")

    def build(mod):
        args = mod.parse_args()
        args.output_dir = out_dir
        args.tracks_dir = args.detections_dir = args.truth_dir = tempfile.mkdtemp()
        args.models_dir = tempfile.mkdtemp()
        args.run_missing_stage08 = args.run_missing_stage09 = False
        args.run_missing_stage12 = True
        avail = pd.DataFrame([
            {"date": "2022-06-13", "threshold_db": -5.0, "stage08_metrics_available": True,
             "stage09_metrics_available": True, "stage12_mlp_available": False,
             "stage12_gru_available": False}])
        mod.run_missing(args, avail)

    cmds = _capture_commands(os.path.join(REPO_ROOT, "scripts", "17_four_day_validation.py"),
                             "s17b", build)
    s12 = [c for c in cmds if any("12_score_tracks_sequence_prior" in x for x in c)]
    assert s12, "no stage-12 command was constructed"
    for c in s12:
        assert "--calibration-output" in c, "stage-12 command must pass --calibration-output"
        cal = c[c.index("--calibration-output") + 1]
        assert os.path.abspath(cal).startswith(os.path.abspath(out_dir)), \
            f"calibration path escapes the stage-17 output dir: {cal}"
        assert CANONICAL_CALIB_DIR not in os.path.normpath(cal), \
            f"calibration path targets the CANONICAL stage-12 dir: {cal}"
    return f"{len(s12)} stage-12 command(s) write calibration inside the stage-17 output dir"


def r2_stage16_calibration_output_is_sandboxed():
    src = open(os.path.join(REPO_ROOT, "scripts", "16_robustness_ablation.py")).read()
    assert "--calibration-output" in src, \
        "stage-16 run-missing must pass --calibration-output"
    assert "generated_calibration" in src, \
        "stage-16 calibration must go to its own generated_calibration dir"
    return "stage-16 run-missing sandboxes its calibration output"


def r2_stage12_default_follows_report_dir():
    """The scorer's default calibration path must derive from --report-dir, not be hardcoded."""
    import importlib.util
    path = os.path.join(REPO_ROOT, "scripts", "12_score_tracks_sequence_prior.py")
    spec = importlib.util.spec_from_file_location("s12", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved

    class A:
        report_dir = "/tmp/some_other_report_dir"
        calibration_output = None
    resolved = mod.resolve_calibration_output(A())
    assert resolved.startswith("/tmp/some_other_report_dir"), \
        f"default calibration must follow --report-dir, got {resolved}"
    assert CANONICAL_CALIB_DIR not in os.path.normpath(resolved), \
        f"omitting --calibration-output must NOT target the canonical dir: {resolved}"

    class B:
        report_dir = "/tmp/x"
        calibration_output = "/tmp/explicit/cal.json"
    assert mod.resolve_calibration_output(B()) == "/tmp/explicit/cal.json", \
        "explicit --calibration-output must win"
    return "omitting --calibration-output follows --report-dir (canonical dir not hardcoded)"


# =============================================================================
# R3: zero false-track denominator => undefined, not 0/1, not "stage-09 missing"
# =============================================================================

def r3_safe_reduction_is_nan_on_zero_denominator():
    assert np.isnan(safe_reduction(0, 0)), "0/0 must be NaN"
    assert np.isnan(safe_reduction(0, 5)), "zero denominator must be NaN"
    assert np.isnan(safe_reduction(float("nan"), 1)), "non-finite denominator must be NaN"
    assert safe_reduction(100, 1) == 0.99, "well-defined case must still compute"
    assert safe_reduction(10, 0) == 1.0, "kept=0 with real denominator is a genuine 1.0"
    return "zero denominator -> NaN; kept=0 with real denominator -> 1.0"


def r3_undefined_reason_is_not_missing_stage09():
    reason = undefined_reason(0, windowable=True)
    assert "windowable" in reason.lower(), f"reason must mention windowable: {reason!r}"
    assert "undefined" in reason.lower(), f"reason must say undefined: {reason!r}"
    assert "stage-09" not in reason.lower() and "stage09" not in reason.lower(), \
        f"zero-denominator must NOT be blamed on stage-09: {reason!r}"
    assert undefined_reason(5) == "", "well-defined cell has no undefined reason"
    return "undefined reason names the zero denominator, never stage-09"


def r3_fallback_table_labels_undefined_cells():
    """Toy data with zero windowable false tracks must be flagged undefined, not a s09 gap."""
    from utils import four_day_validation as fdv
    s09 = pd.DataFrame([{"date": "2022-06-13", "threshold_db": 6.0,
                         "stage09_true_retention": 0.98, "stage09_false_reduction": 0.47,
                         "stage09_precision_after": 0.99, "stage09_kept_true_tracks": 10,
                         "stage09_kept_false_tracks": 1}])
    s12 = pd.DataFrame([{"date": "2022-06-13", "threshold_db": 6.0, "model": "mlp_dae",
                         "stage08_true_tracks": 100, "stage08_false_tracks": 0,
                         "stage12_kept_true_tracks": 97, "stage12_kept_false_tracks": 0,
                         "true_track_retention": 0.97,
                         "false_track_reduction": np.nan,      # zero denominator
                         "precision_after": 1.0, "windowable_false_tracks": 0}])
    out = fdv.interpretable_fallback(s09, s12, [6.0], "mlp_dae")
    assert len(out) == 1
    r = out.iloc[0]
    assert not r["false_reduction_defined"], "cell must be marked undefined"
    assert r["false_reduction_denominator"] == 0, "denominator must be recorded as 0"
    assert "windowable" in r["undefined_reason"].lower(), \
        f"undefined_reason must cite the denominator: {r['undefined_reason']!r}"
    assert "unavailable" not in r["interpretation"].lower(), \
        f"must NOT be labelled stage-09 unavailable: {r['interpretation']!r}"
    assert np.isnan(r["stage12_minus_stage09_false_reduction"]), "gain must stay NaN"
    return "zero-denominator cell => defined=False, reason cites windowable, not a s09 gap"


def r3_aggregates_exclude_undefined_and_count_them():
    n_def, n_undef = summarize_defined([1.0, np.nan, 0.5, np.nan])
    assert (n_def, n_undef) == (2, 2), f"got {(n_def, n_undef)}"
    # a mean over defined cells only must ignore the NaNs, not treat them as 0
    mean = pd.Series([1.0, np.nan, 0.5]).mean()
    assert abs(mean - 0.75) < 1e-12, f"mean must skip NaN, got {mean}"
    return "aggregates count undefined cells and average only the defined ones"


# =============================================================================
# R4: large generated artifacts are git-ignored
# =============================================================================

def _ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT)
    return r.returncode == 0


def r4_large_generated_files_are_ignored():
    must_ignore = [
        "reports/stage17_four_day_validation/generated/stage12/2022-06-13/sequence_track_scores.csv",
        "reports/stage17_four_day_validation/generated/stage09/2022-06-13/physics_track_scores.csv",
        "reports/stage12_sequence_priors/sequence_track_scores.csv",
        "data/active/tracks_kalman/tracks_2022-06-13_thr_0p0dB.csv",
        "models/sequence_priors/mlp_dae.pt",
    ]
    missed = [p for p in must_ignore if not _ignored(p)]
    assert not missed, f"large artifacts NOT git-ignored: {missed}"
    return f"{len(must_ignore)} large-artifact patterns are git-ignored"


def r4_compact_reports_are_not_ignored():
    must_track = [
        "reports/stage17_four_day_validation/four_day_summary_overall.csv",
        "reports/stage17_four_day_validation/stage17_key_findings.csv",
        "reports/stage12_sequence_priors/calibration/sequence_track_calibration.json",
    ]
    wrongly = [p for p in must_track if _ignored(p)]
    assert not wrongly, f"compact reports wrongly git-ignored: {wrongly}"
    return "compact summaries and calibration JSON remain tracked"


# =============================================================================
# R5: stage-16 must date itself from stage-12 evidence, not stage-08 coverage
# =============================================================================

def r5_stage16_dates_follow_stage12_evidence():
    """Stage 08 may cover more days than stage 12 was scored on; stage 16 audits
    stage 12, so it must not inherit stage-08's wider coverage and claim multi-day."""
    from utils import robustness_analysis as ra
    with tempfile.TemporaryDirectory() as tmp:
        s08 = os.path.join(tmp, "kalman_metrics_by_day.csv")
        s12 = os.path.join(tmp, "sequence_track_scores.csv")
        pd.DataFrame({"date": ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"],
                      "threshold_db": [-5.0] * 4}).to_csv(s08, index=False)
        pd.DataFrame({"date": ["2022-06-06"], "threshold_db": [-5.0]}).to_csv(s12, index=False)
        dates = ra.detect_dates({"stage08_metrics": s08, "stage12_scores": s12})
        assert dates == ["2022-06-06"], \
            f"stage-16 must date from stage-12 evidence (got {dates}); stage-08 has 4 days"
        # fallback when no stage-12 scores exist
        fallback = ra.detect_dates({"stage08_metrics": s08, "stage12_scores": "/nonexistent"})
        assert len(fallback) == 4, "with no stage-12 scores, fall back to stage-08 coverage"
    return "stage-16 dates from stage-12 scores (1 day), not stage-08 coverage (4 days)"


def main():
    print("=" * 70)
    print("STAGE 17.5 REGRESSION CHECKS")
    print("=" * 70)
    print("\nR1 -- internal subprocess calls use sys.executable")
    check("R1a stage-17 run-missing uses sys.executable", r1_stage17_uses_sys_executable)
    check("R1b no bare `python` command lists in source", r1_no_bare_python_in_source)

    print("\nR2 -- canonical stage-12 calibration cannot be clobbered")
    check("R2a stage-17 sandboxes --calibration-output", r2_stage17_calibration_output_is_sandboxed)
    check("R2b stage-16 sandboxes --calibration-output", r2_stage16_calibration_output_is_sandboxed)
    check("R2c stage-12 default follows --report-dir", r2_stage12_default_follows_report_dir)

    print("\nR3 -- zero false-track denominator is undefined, not 0/1/missing-stage-09")
    check("R3a safe_reduction NaN on zero denominator", r3_safe_reduction_is_nan_on_zero_denominator)
    check("R3b undefined_reason never blames stage-09", r3_undefined_reason_is_not_missing_stage09)
    check("R3c fallback table labels undefined cells", r3_fallback_table_labels_undefined_cells)
    check("R3d aggregates exclude and count undefined", r3_aggregates_exclude_undefined_and_count_them)

    print("\nR4 -- large generated artifacts are git-ignored")
    check("R4a large per-track CSVs / checkpoints ignored", r4_large_generated_files_are_ignored)
    check("R4b compact reports still tracked", r4_compact_reports_are_not_ignored)

    print("\nR5 -- stage-16 dates itself from stage-12 evidence, not stage-08 coverage")
    check("R5a stage-16 detect_dates follows stage-12", r5_stage16_dates_follow_stage12_evidence)

    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name, err in FAILED:
            print(f"  FAILED: {name}\n    {err}")
        raise SystemExit(1)
    print("\nStage 17.5 regression checks passed.")


if __name__ == "__main__":
    main()
