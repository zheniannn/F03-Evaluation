"""Entry point: stage 18 -- final report / paper package.

Packages the completed stage 01-17.5 pipeline into a paper-ready technical report,
an Overleaf LaTeX package, final tables, and an extensive figure set. Adds NO new
model, runs NO new experiment, retrains nothing, and changes no scientific result.
The hardened stage-17/17.5 four-day artifacts are the source of truth for headline
metrics; stale single-day values are never promoted to the headline.

Scope note carried through every artifact: the radar model is a POINT-DETECTION
simulation. "Pseudo range-Doppler" figures are scatter plots of point detections,
not raw range-Doppler intensity heatmaps.

Usage:
    python scripts/18_build_final_report.py --overwrite
    python scripts/18_build_final_report.py --self-test
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils import final_report_package as frp

DEFAULT_TITLE = "ADS-B-Guided Weak-Target Radar Tracking Under Low SNR and Clutter"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 18 final report / paper package.")
    p.add_argument("--reports-root", default=os.path.join(REPO_ROOT, "reports"))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "reports",
                                                        "stage18_final_package"))
    p.add_argument("--detections-dir",
                   default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    p.add_argument("--tracks-dir", default=os.path.join(REPO_ROOT, "data", "active",
                                                        "tracks_kalman"))
    p.add_argument("--project-title", default=DEFAULT_TITLE)
    p.add_argument("--author", default="Your Name")
    p.add_argument("--institution", default="Your Institution")
    p.add_argument("--include-overleaf", action="store_true", default=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Build a tiny package from synthetic mini reports and exit.")
    return p.parse_args()


def _det(args, token):
    return os.path.join(args.detections_dir, f"detections_2022-06-06_thr_{token}dB.csv")


def build_figures(args, src, hm, checklist) -> list:
    """Build every figure; return manifest entries (never raises on missing input)."""
    figs_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    f = lambda n: os.path.join(figs_dir, n)
    entries = []

    def add(idx, filename, shows, source_stage, result):
        path, ftype, notes = result
        entries.append({"index": idx, "filename": filename, "shows": shows,
                        "source_stage": source_stage, "figure_type": ftype,
                        "data_driven": ftype == frp.DATA_DRIVEN, "notes": notes,
                        "path": os.path.relpath(path, args.output_dir) if path else None})

    add(1, "01_pipeline_diagram.png", "Three-repo pipeline, stages 01-18", "all",
        frp.build_pipeline_diagram(f("01_pipeline_diagram.png")))
    add(2, "02_radar_simulation_concept.png", "Radar geometry and relocation annulus", "05/06",
        frp.build_radar_simulation_concept(f("02_radar_simulation_concept.png")))
    add(3, "03_pseudo_range_doppler_frame_low_threshold.png",
        "Pseudo range-Doppler point detections, low threshold (-5 dB)", "06",
        frp.build_pseudo_range_doppler_frame(_det(args, "m5p0"),
                                             f("03_pseudo_range_doppler_frame_low_threshold.png"),
                                             "low threshold, -5 dB"))
    add(4, "04_pseudo_range_doppler_frame_high_threshold.png",
        "Pseudo range-Doppler point detections, high threshold (12 dB)", "06",
        frp.build_pseudo_range_doppler_frame(_det(args, "12p0"),
                                             f("04_pseudo_range_doppler_frame_high_threshold.png"),
                                             "high threshold, 12 dB"))
    add(5, "05_range_azimuth_frame_low_threshold.png",
        "Range-azimuth point detections, low threshold (-5 dB)", "06",
        frp.build_range_azimuth_frame(_det(args, "m5p0"),
                                      f("05_range_azimuth_frame_low_threshold.png"),
                                      "low threshold, -5 dB"))
    add(6, "06_range_azimuth_frame_high_threshold.png",
        "Range-azimuth point detections, high threshold (12 dB)", "06",
        frp.build_range_azimuth_frame(_det(args, "12p0"),
                                      f("06_range_azimuth_frame_high_threshold.png"),
                                      "high threshold, 12 dB"))
    add(7, "07_threshold_tradeoff_stage07.png", "Detection-threshold tradeoff (Pd vs clutter)",
        "07", frp.build_threshold_tradeoff_plot(src, f("07_threshold_tradeoff_stage07.png")))
    add(8, "08_kalman_tracking_effect_stage08.png",
        "Tracking recovers trajectories as frame Pd falls", "07/08",
        frp.build_kalman_effect_plot(src, f("08_kalman_tracking_effect_stage08.png")))
    add(9, "09_stage09_physics_filter_effect.png", "False tracks before/after hand physics", "09",
        frp.build_stage09_effect_plot(src, f("09_stage09_physics_filter_effect.png")))
    add(10, "10_stage12_calibration_effect.png", "Clean-truth vs noise-matched calibration",
        "12.5/16", frp.build_calibration_effect_plot(src, f("10_stage12_calibration_effect.png")))
    add(11, "11_method_ladder_comparison.png", "Retention vs false reduction across methods",
        "14/17", frp.build_method_ladder_plot(src, hm, f("11_method_ladder_comparison.png")))

    r12, r13 = frp.build_four_day_validation_plots(
        src, f("12_four_day_validation_retention.png"),
        f("13_four_day_validation_false_reduction.png"))
    add(12, "12_four_day_validation_retention.png", "True-track retention by day", "17", r12)
    add(13, "13_four_day_validation_false_reduction.png", "False-track reduction by day", "17", r13)

    add(14, "14_false_tracks_before_after_stage12.png",
        "False tracks before/after Stage 12.5 (815 -> 5)", "17",
        frp.build_false_tracks_before_after_plot(src, f("14_false_tracks_before_after_stage12.png")))
    add(15, "15_mlp_vs_gru_four_day.png", "MLP vs GRU across four days", "17",
        frp.build_model_comparison_plot(src, f("15_mlp_vs_gru_four_day.png")))
    add(16, "16_stage09_vs_stage12_interpretable_comparison.png",
        "Learned sequence prior vs interpretable fallback", "17",
        frp.build_stage09_vs_stage12_plot(
            src, f("16_stage09_vs_stage12_interpretable_comparison.png")))
    add(17, "17_windowability_caveat.png", "Windowable false-track fraction by threshold",
        "16/17", frp.build_windowability_plot(src, f("17_windowability_caveat.png")))

    r18, r19 = frp.build_failure_case_plots(
        src, args.tracks_dir, f("18_failure_case_surviving_false_track.png"),
        f("19_failure_case_rejected_true_track.png"))
    add(18, "18_failure_case_surviving_false_track.png",
        "A false track that survived Stage 12.5", "14/17", r18)
    add(19, "19_failure_case_rejected_true_track.png",
        "A true track rejected by Stage 12.5", "14/17", r19)

    r20, r21 = frp.build_diffusion_example_plots(
        src, f("20_diffusion_denoising_example.png"), f("21_diffusion_gap_filling_example.png"))
    add(20, "20_diffusion_denoising_example.png", "Kalman vs diffusion-denoised window", "15", r20)
    add(21, "21_diffusion_gap_filling_example.png",
        "Interpolation vs diffusion gap filling", "15", r21)

    add(22, "22_final_method_pareto.png", "Non-dominated operating points", "14",
        frp.build_pareto_plot(src, f("22_final_method_pareto.png")))
    add(23, "23_final_precision_by_threshold.png", "Precision after filtering by threshold",
        "17", frp.build_precision_by_threshold_plot(src, f("23_final_precision_by_threshold.png")))
    add(24, "24_final_score_distributions.png", "Sequence-prior score, true vs false tracks",
        "12.5", frp.build_score_distribution_plot(src, f("24_final_score_distributions.png")))
    add(25, "25_reproducibility_pipeline_checklist.png", "Reproducibility checklist", "17.5",
        frp.build_reproducibility_checklist_plot(
            checklist, f("25_reproducibility_pipeline_checklist.png")))
    return entries


def run(args) -> dict:
    out = args.output_dir
    rep_path = os.path.join(out, "final_report.md")
    if os.path.exists(rep_path) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {rep_path}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)

    src = frp.Sources(args.reports_root)
    if src.missing:
        print("missing source artifacts (recorded in the manifest, figures degrade to "
              "placeholders):")
        for k in src.missing:
            print(f"  - {k}: {src.paths[k]}")
    else:
        print("all expected source artifacts present.")

    hm, notes = frp.headline_metrics(src)
    print(f"\nheadline (four-day, source of truth = stage 17): "
          f"{hm['true_tracks']:,} true / {hm['false_tracks']:,} false tracks; "
          f"mlp_dae retention {hm['mlp_true_retention']:.3f}, "
          f"reduction {hm['mlp_false_reduction']:.4f}, kept {hm['mlp_false_kept']}")

    tables = {
        "final_results_summary": frp.build_final_results_summary(src, hm),
        "final_method_comparison": frp.build_final_method_comparison(src, hm),
        "final_ablation_summary": frp.build_final_ablation_summary(src),
        "final_reproducibility_checklist": frp.build_final_reproducibility_checklist(src),
        "final_limitations": frp.build_final_limitations(),
        "final_contributions": frp.build_final_contributions(),
    }
    outputs = []
    for name, df in tables.items():
        p = os.path.join(out, f"{name}.csv")
        df.to_csv(p, index=False)
        outputs.append(p)

    # paper-facing table copies
    table_map = {
        "table_threshold_vs_tracking.csv": tables["final_results_summary"],
        "table_stage09_to_stage17_methods.csv": tables["final_method_comparison"],
        "table_four_day_validation.csv": (src.get("s17_by_day")
                                          if src.get("s17_by_day") is not None
                                          else tables["final_method_comparison"]),
        "table_ablation_summary.csv": tables["final_ablation_summary"],
        "table_reproducibility.csv": tables["final_reproducibility_checklist"],
    }
    for name, df in table_map.items():
        p = os.path.join(out, "tables", name)
        df.to_csv(p, index=False)
        outputs.append(p)

    figures = [] if args.no_plots else build_figures(
        args, src, hm, tables["final_reproducibility_checklist"])
    outputs += [os.path.join(out, "figures", f["filename"]) for f in figures]

    rep = frp.write_final_report(out, args.project_title, hm, tables, figures, src, notes)
    outputs.append(rep)

    if args.include_overleaf and not args.no_plots:
        ol = frp.write_overleaf_package(out, args.project_title, args.author,
                                        args.institution, hm, tables,
                                        os.path.join(out, "figures"))
        outputs += ol

    man = frp.write_manifest(out, src, hm, figures, outputs, notes)
    outputs.append(man)

    if not args.no_plots:
        frp.run_gate(out, figures, tables)

    kinds = {}
    for f in figures:
        kinds[f["figure_type"]] = kinds.get(f["figure_type"], 0) + 1
    print(f"\nreport:   {os.path.abspath(rep)}")
    print(f"manifest: {os.path.abspath(man)}")
    print(f"figures:  {len(figures)} ({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))})")
    return {"figures": figures, "tables": tables, "report": rep, "manifest": man, "hm": hm}


# =============================================================================
# Self-test
# =============================================================================

def _w(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _make_mini_reports(root):
    thr = [-5.0, 0.0]
    _w(os.path.join(root, "stage07_threshold_only", "threshold_overall.csv"), pd.DataFrame({
        "threshold_db": [-5.0, 0.0, 6.0], "empirical_pd": [0.79, 0.62, 0.41],
        "false_alarm_per_frame": [46.0, 5.0, 0.2]}))
    _w(os.path.join(root, "stage08_kalman_baseline", "kalman_metrics_by_day.csv"), pd.DataFrame({
        "date": ["2022-06-06"] * 2, "threshold_db": thr, "tracks_confirmed": [3000, 3100],
        "true_tracks": [2000, 2100], "false_tracks": [200, 60]}))
    _w(os.path.join(root, "stage09_physics_scoring", "physics_metrics_by_threshold.csv"),
       pd.DataFrame({"threshold_db": thr, "stage08_true_tracks": [2000, 2100],
                     "stage08_false_tracks": [200, 60], "stage09_kept_true_tracks": [1940, 2060],
                     "stage09_kept_false_tracks": [80, 22],
                     "true_track_retention": [0.97, 0.98],
                     "false_track_reduction": [0.6, 0.63], "precision_after": [0.96, 0.99]}))
    _w(os.path.join(root, "stage11_adsb_prior_scoring", "adsb_prior_metrics_by_threshold.csv"),
       pd.DataFrame({"threshold_db": thr, "true_track_retention": [1.0, 1.0],
                     "false_track_reduction": [0.01, 0.0], "precision_after": [0.86, 0.96]}))
    _w(os.path.join(root, "stage12_sequence_priors", "sequence_metrics_by_model_threshold.csv"),
       pd.DataFrame({"model": ["mlp_dae"] * 2, "threshold_db": thr,
                     "stage08_true_tracks": [2000, 2100], "stage08_false_tracks": [200, 60],
                     "stage12_kept_true_tracks": [1960, 2070],
                     "stage12_kept_false_tracks": [2, 1],
                     "true_track_retention": [0.98, 0.985],
                     "false_track_reduction": [0.99, 0.983],
                     "precision_after": [0.999, 0.999], "calibration_mode": ["track_purity"] * 2}))
    _w(os.path.join(root, "stage12_sequence_priors", "calibration",
                    "sequence_calibration_comparison.csv"),
       pd.DataFrame({"model": ["mlp_dae"] * 4, "threshold_db": thr * 2,
                     "calibration_mode": ["clean_truth"] * 2 + ["track_purity"] * 2,
                     "true_track_retention": [0.11, 0.12, 0.98, 0.985],
                     "false_track_reduction": [1.0, 1.0, 0.99, 0.983],
                     "precision_after": [1.0, 1.0, 0.999, 0.999]}))
    _w(os.path.join(root, "stage13_vae_prior", "vae_metrics_by_threshold.csv"),
       pd.DataFrame({"variant": ["elbo"] * 2, "threshold_db": thr,
                     "true_track_retention": [0.983, 0.977],
                     "false_track_reduction": [0.98, 0.93], "precision_after": [0.999, 0.999]}))
    _w(os.path.join(root, "stage14_method_benchmark", "pareto_frontier.csv"), pd.DataFrame({
        "threshold_db": [-5.0] * 3, "method_id": ["stage12_mlp_dae_track_calibrated",
                                                  "stage09_hand_physics", "kalman_only"],
        "method_family": ["sequence_autoencoder", "hand_physics", "kalman_only"],
        "score_threshold": [0.5, 0.5, np.nan],
        "true_track_retention": [0.98, 0.97, 1.0],
        "false_track_reduction": [0.99, 0.6, 0.0],
        "precision_after": [0.999, 0.96, 0.9], "is_pareto": [True, True, True]}))
    _w(os.path.join(root, "stage14_method_benchmark", "failure_case_candidates.csv"),
       pd.DataFrame({"case_type": ["false_survives_s12", "true_rejected_s12"],
                     "date": ["2022-06-06"] * 2, "threshold_db": [0.0, -5.0],
                     "track_id": [1, 2], "score_stage12": [1.0, 0.0],
                     "median_range_m": [1600.0, 4000.0]}))
    _w(os.path.join(root, "stage15_diffusion_denoising", "diffusion_gap_filling_metrics.csv"),
       pd.DataFrame({"threshold_db": thr, "gap_mode": ["block"] * 2,
                     "interp_mse": [0.96, 0.93], "diffusion_mse": [0.86, 0.84],
                     "improvement_ratio": [0.10, 0.10]}))
    _w(os.path.join(root, "stage15_diffusion_denoising", "diffusion_residual_by_threshold.csv"),
       pd.DataFrame({"threshold_db": thr, "true_track_retention": [0.88, 0.85],
                     "false_track_reduction": [0.42, 0.63], "precision_after": [0.99, 0.999]}))
    _w(os.path.join(root, "stage16_robustness", "calibration_ablation.csv"), pd.DataFrame({
        "model": ["mlp_dae"] * 4, "threshold_db": thr * 2,
        "calibration_mode": ["clean_truth"] * 2 + ["track_purity"] * 2,
        "true_track_retention": [0.11, 0.12, 0.98, 0.985],
        "false_track_reduction": [1.0, 1.0, 0.99, 0.983]}))
    _w(os.path.join(root, "stage16_robustness", "model_ablation_mlp_gru_tcn.csv"), pd.DataFrame({
        "row_type": ["aggregate"] * 2, "model": ["mlp_dae", "gru_ae"],
        "mean_true_retention": [0.978, 0.977], "mean_false_reduction": [0.984, 0.980],
        "overall_rank": [1, 2]}))
    _w(os.path.join(root, "stage17_four_day_validation", "four_day_summary_overall.csv"),
       pd.DataFrame({"model": ["mlp_dae", "gru_ae"],
                     "dates_included": ["2022-06-06,2022-06-13"] * 2,
                     "thresholds_included": ["-5,0"] * 2,
                     "stage08_true_tracks": [323808, 323808], "stage08_false_tracks": [815, 815],
                     "stage12_kept_true_tracks": [315000, 314800],
                     "stage12_kept_false_tracks": [5, 17],
                     "pooled_true_retention": [0.973, 0.972],
                     "pooled_false_reduction": [0.9939, 0.9791],
                     "pooled_precision_after": [0.99999, 0.99995]}))
    _w(os.path.join(root, "stage17_four_day_validation", "four_day_summary_by_day.csv"),
       pd.DataFrame({"date": ["2022-06-06", "2022-06-13"] * 2,
                     "model": ["mlp_dae"] * 2 + ["gru_ae"] * 2,
                     "stage08_true_tracks": [160000, 163808] * 2,
                     "stage08_false_tracks": [400, 415] * 2,
                     "stage12_kept_true_tracks": [156000, 159000] * 2,
                     "stage12_kept_false_tracks": [2, 3, 8, 9],
                     "pooled_true_retention": [0.975, 0.971, 0.974, 0.970],
                     "pooled_false_reduction": [0.995, 0.993, 0.980, 0.978],
                     "pooled_precision_after": [0.9999, 0.9999, 0.9999, 0.9999]}))
    _w(os.path.join(root, "stage17_four_day_validation", "four_day_summary_by_threshold.csv"),
       pd.DataFrame({"threshold_db": thr * 2, "model": ["mlp_dae"] * 2 + ["gru_ae"] * 2,
                     "pooled_true_retention": [0.975, 0.971, 0.974, 0.970],
                     "pooled_false_reduction": [0.995, 0.993, 0.980, 0.978],
                     "pooled_precision_after": [0.9999, 0.9999, 0.9999, 0.9999]}))
    _w(os.path.join(root, "stage17_four_day_validation", "model_comparison_mlp_vs_gru.csv"),
       pd.DataFrame({"date": ["ALL"] * 3,
                     "metric": ["true_track_retention", "false_track_reduction",
                                "precision_after"],
                     "mlp_dae": [0.973, 0.9939, 0.99999], "gru_ae": [0.972, 0.9791, 0.99995],
                     "winner": ["mlp_dae"] * 3}))
    _w(os.path.join(root, "stage17_four_day_validation",
                    "interpretable_fallback_comparison.csv"),
       pd.DataFrame({"date": ["2022-06-06"] * 2, "threshold_db": thr,
                     "stage09_true_retention": [0.973, 0.985],
                     "stage09_false_reduction": [0.788, 0.637],
                     "stage09_precision_after": [0.95, 0.98],
                     "best_stage12_model": ["mlp_dae"] * 2,
                     "stage12_true_retention": [0.987, 0.981],
                     "stage12_false_reduction": [1.0, 0.9375],
                     "stage12_precision_after": [1.0, 0.99995],
                     "stage12_minus_stage09_true_retention": [0.014, -0.004],
                     "stage12_minus_stage09_false_reduction": [0.212, 0.30],
                     "stage12_minus_stage09_precision": [0.05, 0.02],
                     "false_reduction_defined": [True, True],
                     "false_reduction_denominator": [164, 16],
                     "undefined_reason": ["", ""],
                     "interpretation": ["stage-12 wins on false reduction"] * 2}))
    _w(os.path.join(root, "stage17_four_day_validation", "windowability_four_day_audit.csv"),
       pd.DataFrame({"date": ["2022-06-06"] * 3, "threshold_db": [-5.0, 0.0, 12.0],
                     "windowable_fraction_false": [0.106, 0.063, 0.0],
                     "windowable_false_tracks": [164, 16, 0],
                     "high_threshold_caveat": [False, False, True]}))
    _w(os.path.join(root, "stage17_four_day_validation", "four_day_stage08_context.csv"),
       pd.DataFrame({"date": ["2022-06-06"] * 2, "threshold_db": thr,
                     "stage08_true_tracks": [160000, 163808],
                     "stage08_false_tracks": [400, 415],
                     "stage08_confirmed_tracks": [160400, 164223]}))
    with open(os.path.join(root, "stage17_four_day_validation",
                           "stage17p5_repro_hardening.md"), "w") as f:
        f.write("# Stage 17.5 Reproducibility Hardening\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reports = os.path.join(tmp, "reports")
        _make_mini_reports(reports)
        args = parse_args()
        args.reports_root = reports
        args.output_dir = os.path.join(tmp, "out")
        args.detections_dir = os.path.join(tmp, "no_detections")   # force RD placeholder
        args.tracks_dir = os.path.join(tmp, "no_tracks")
        args.project_title = DEFAULT_TITLE
        args.overwrite = True
        out = run(args)

        o = args.output_dir
        assert os.path.exists(os.path.join(o, "final_report.md")), "final_report.md missing"
        assert os.path.exists(os.path.join(o, "overleaf", "main.tex")), "main.tex missing"
        figs = [f for f in os.listdir(os.path.join(o, "figures")) if f.endswith(".png")]
        assert len(figs) >= 5, f"expected >= 5 figures in self-test, got {len(figs)}"
        man_path = os.path.join(o, "stage18_final_package_manifest.json")
        assert os.path.exists(man_path), "manifest missing"
        import json
        man = json.load(open(man_path))
        assert isinstance(man.get("figures"), list) and man["figures"], "manifest has no figures"

        report = open(os.path.join(o, "final_report.md")).read()
        assert "Final figure guide" in report, "report missing figure guide"
        assert "noise-matched calibration" in report, "report missing noise-matched calibration"

        tex = open(os.path.join(o, "overleaf", "main.tex")).read()
        assert r"\bibliography" in tex, "main.tex missing \\bibliography"
        assert "figures/" in tex, "main.tex references no figures"

        # the RD figure must be an explicitly-reasoned placeholder here
        rd = [f for f in man["figures"] if "pseudo_range_doppler" in f["filename"]]
        assert rd and all(f["figure_type"] == frp.PLACEHOLDER and f["notes"] for f in rd), \
            "pseudo-RD placeholders must record a reason when detections are absent"

        print(f"\nself-test figures: {len(figs)}; "
              f"pseudo-RD recorded as placeholder with reason: "
              f"{rd[0]['notes']!r}")
    print("\nStage 18 final report package self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)
    print("\n18_build_final_report completed successfully.")


if __name__ == "__main__":
    main()
