"""Entry point: stage 14 -- unified benchmark and operating-point selection.

Consolidates the compact report CSVs from stages 07/08/09/11/12.5/13 into one
comparison: method inventory, unified operating-point metrics, best method per
detection threshold, matched-retention and matched-false-reduction comparisons,
a Pareto frontier, descriptive rankings, a runtime inventory, and optional
failure-case candidates. Adds NO new model and retrains nothing.

Usage:
    python scripts/14_benchmark_methods.py --date 2022-06-06 --overwrite
    python scripts/14_benchmark_methods.py --self-test
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils import method_benchmark as mb


def parse_args():
    p = argparse.ArgumentParser(description="Stage 14 unified method benchmark.")
    p.add_argument("--stage07-dir", default=os.path.join(REPO_ROOT, "reports", "stage07_threshold_only"))
    p.add_argument("--stage08-dir", default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline"))
    p.add_argument("--stage09-dir", default=os.path.join(REPO_ROOT, "reports", "stage09_physics_scoring"))
    p.add_argument("--stage11-dir", default=os.path.join(REPO_ROOT, "reports", "stage11_adsb_prior_scoring"))
    p.add_argument("--stage12-dir", default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    p.add_argument("--stage13-dir", default=os.path.join(REPO_ROOT, "reports", "stage13_vae_prior"))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "reports", "stage14_method_benchmark"))
    p.add_argument("--date", default="2022-06-06")
    p.add_argument("--target-retention", type=float, default=0.97)
    p.add_argument("--target-false-reduction", type=float, default=0.95)
    p.add_argument("--min-true-retention", type=float, default=0.95)
    p.add_argument("--max-methods-per-threshold", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real reports needed) and exit.")
    return p.parse_args()


def run_benchmark(args) -> dict:
    out = args.output_dir
    key = os.path.join(out, "unified_method_metrics.csv")
    if os.path.exists(key) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key}")
    os.makedirs(out, exist_ok=True)

    dirs = {"stage07": args.stage07_dir, "stage08": args.stage08_dir,
            "stage09": args.stage09_dir, "stage11": args.stage11_dir,
            "stage12": args.stage12_dir, "stage13": args.stage13_dir}

    unified, inventory = mb.build_unified(dirs, args.date, out)
    inventory.to_csv(os.path.join(out, "method_inventory.csv"), index=False)
    print("input inventory:")
    print(inventory[["stage", "method_family", "available", "rows_loaded"]].to_string(index=False))

    if unified.empty:
        raise SystemExit("No usable track-level benchmark inputs found -- nothing to benchmark.")
    unified.to_csv(key, index=False, float_format="%.6g")

    best_by_thr = mb.best_method_by_threshold(unified, args.min_true_retention)
    matched_ret = mb.matched_retention_comparison(unified, args.target_retention)
    matched_fr = mb.matched_false_reduction_comparison(unified, args.target_false_reduction)
    pareto = mb.pareto_frontier(unified)
    ranks = mb.rankings(unified)
    runtime = mb.runtime_inventory(dirs)
    recs = mb.global_recommendations(unified, best_by_thr)

    best_by_thr.to_csv(os.path.join(out, "best_method_by_threshold.csv"), index=False)
    matched_ret.to_csv(os.path.join(out, "matched_retention_comparison.csv"), index=False)
    matched_fr.to_csv(os.path.join(out, "matched_false_reduction_comparison.csv"), index=False)
    pareto.to_csv(os.path.join(out, "pareto_frontier.csv"), index=False)
    ranks.to_csv(os.path.join(out, "rankings.csv"), index=False)
    runtime.to_csv(os.path.join(out, "runtime_inventory.csv"), index=False)

    failures, note = mb.failure_case_candidates(dirs, args.date, out)
    if len(failures):
        failures.to_csv(os.path.join(out, "failure_case_candidates.csv"), index=False)
    else:
        with open(os.path.join(out, "failure_case_candidates.csv"), "w") as f:
            f.write("note\n")
            f.write(f"\"{note or 'no failure-case candidates found'}\"\n")

    if not args.no_plots:
        mb.make_plots(unified, best_by_thr, pareto, ranks, os.path.join(out, "plots"))

    report = mb.write_report(out, args.date, inventory, unified, best_by_thr, matched_ret,
                             matched_fr, pareto, ranks, runtime, recs, note)
    mb.run_gate(out, inventory, unified, best_by_thr, pareto)

    print(f"\nreport: {os.path.abspath(report)}")
    print("\nglobal recommendations:")
    for k, v in recs.items():
        print(f"  {k}: {v}")
    print("\nbest method by threshold:")
    print(best_by_thr[["threshold_db", "best_method_id", "true_track_retention",
                       "false_track_reduction", "precision_after"]].to_string(index=False))
    return {"unified": unified, "inventory": inventory, "best_by_thr": best_by_thr,
            "pareto": pareto, "recs": recs}


# =============================================================================
# Self-test
# =============================================================================

def _write(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _make_mini_reports(root):
    thr = [-5.0, 0.0]
    # Stage 08: counts baseline
    _write(os.path.join(root, "s08", "kalman_metrics_by_day.csv"), pd.DataFrame({
        "date": ["2022-06-06", "2022-06-06"], "threshold_db": thr,
        "true_tracks": [1000, 900], "false_tracks": [200, 100]}))

    def sweep(kept_true, kept_false):
        rows = []
        for i, t in enumerate(thr):
            for st in (0.3, 0.5, 0.7):
                rows.append({"threshold_db": t, "score_threshold": st,
                             "kept_tracks": kept_true[i] + kept_false[i],
                             "kept_true_tracks": kept_true[i], "kept_false_tracks": kept_false[i],
                             "true_track_retention": kept_true[i] / (1000 if i == 0 else 900),
                             "false_track_retention": kept_false[i] / (200 if i == 0 else 100),
                             "false_track_reduction": 1 - kept_false[i] / (200 if i == 0 else 100),
                             "precision_after": kept_true[i] / (kept_true[i] + kept_false[i])})
        return pd.DataFrame(rows)

    def counts(prefix):
        return pd.DataFrame({"threshold_db": thr,
                             "stage08_true_tracks": [1000, 900],
                             "stage08_false_tracks": [200, 100]})

    # Stage 09 hand physics: decent (keeps 950/900 true, removes to 60/20 false)
    _write(os.path.join(root, "s09", "physics_metrics_by_threshold.csv"), counts("s09"))
    _write(os.path.join(root, "s09", "physics_filter_sweep.csv"),
           sweep([970, 880], [60, 20]))
    # Stage 11 adsb: weaker false reduction
    _write(os.path.join(root, "s11", "adsb_prior_metrics_by_threshold.csv"), counts("s11"))
    _write(os.path.join(root, "s11", "adsb_prior_filter_sweep.csv"),
           sweep([990, 895], [180, 90]))
    # Stage 12 mlp: DOMINATES (keeps more true AND removes more false than s09)
    s12sweep = sweep([980, 890], [10, 3]).assign(model="mlp_dae")
    s12sweep = s12sweep[["model", "threshold_db", "score_threshold", "kept_tracks",
                         "kept_true_tracks", "kept_false_tracks", "true_track_retention",
                         "false_track_retention", "false_track_reduction", "precision_after"]]
    _write(os.path.join(root, "s12", "sequence_filter_sweep.csv"), s12sweep)
    _write(os.path.join(root, "s12", "sequence_metrics_by_model_threshold.csv"),
           counts("s12").assign(model="mlp_dae", calibration_mode="track_purity"))
    # Stage 13 vae: slightly worse than s12
    v = sweep([975, 885], [15, 5])
    for variant in ("reconstruction", "elbo"):
        vv = v.assign(variant=variant)
        _write(os.path.join(root, "s13", f"_tmp_{variant}.csv"), vv)
    v13 = pd.concat([pd.read_csv(os.path.join(root, "s13", "_tmp_reconstruction.csv")),
                     pd.read_csv(os.path.join(root, "s13", "_tmp_elbo.csv"))], ignore_index=True)
    v13 = v13[["variant", "threshold_db", "score_threshold", "kept_tracks", "kept_true_tracks",
               "kept_false_tracks", "true_track_retention", "false_track_retention",
               "false_track_reduction", "precision_after"]]
    _write(os.path.join(root, "s13", "vae_filter_sweep.csv"), v13)
    _write(os.path.join(root, "s13", "vae_metrics_by_threshold.csv"),
           pd.concat([counts("s13").assign(variant="reconstruction"),
                      counts("s13").assign(variant="elbo")], ignore_index=True))
    os.remove(os.path.join(root, "s13", "_tmp_reconstruction.csv"))
    os.remove(os.path.join(root, "s13", "_tmp_elbo.csv"))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_mini_reports(tmp)
        args = parse_args()
        args.stage07_dir = os.path.join(tmp, "s07_missing")
        args.stage08_dir = os.path.join(tmp, "s08")
        args.stage09_dir = os.path.join(tmp, "s09")
        args.stage11_dir = os.path.join(tmp, "s11")
        args.stage12_dir = os.path.join(tmp, "s12")
        args.stage13_dir = os.path.join(tmp, "s13")
        args.output_dir = os.path.join(tmp, "out")
        args.date = "2022-06-06"
        args.overwrite = True
        out = run_benchmark(args)

        o = args.output_dir
        for f_ in ["unified_method_metrics.csv", "best_method_by_threshold.csv",
                   "pareto_frontier.csv", "stage14_method_benchmark_report.md"]:
            assert os.path.exists(os.path.join(o, f_)), f"missing {f_}"

        best = out["best_by_thr"]
        assert (best["best_method_id"] == "stage12_mlp_dae_track_calibrated").all(), \
            f"expected stage12 mlp to win, got {best['best_method_id'].tolist()}"
        pareto = out["pareto"]
        assert pareto["is_pareto"].any(), "no Pareto point"
        assert "stage12_mlp_dae_track_calibrated" in \
            pareto[pareto["is_pareto"]]["method_id"].values, "s12 mlp should be on the frontier"

        text = open(os.path.join(o, "stage14_method_benchmark_report.md")).read()
        for needle in ["Stage 14 does not introduce a new model", "operating-point selection",
                       "Stage 12.5", "Stage 13 VAE", "Stage 15"]:
            assert needle in text, f"report missing expected text: {needle!r}"

        print("\nbest method by threshold (toy data):")
        print(best[["threshold_db", "best_method_id", "false_track_reduction"]].to_string(index=False))

    print("\nStage 14 method benchmark self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_benchmark(args)
    print("\n14_benchmark_methods completed successfully.")


if __name__ == "__main__":
    main()
