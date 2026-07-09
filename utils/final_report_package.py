"""Stage 18: final report / paper package.

Packages the completed stage 01-17.5 pipeline into a paper-ready technical
report, an Overleaf LaTeX package, final tables, and an extensive figure set.

This module adds NO new model and runs NO new experiment. It reads the compact,
hardened artifacts from stages 07-17.5 (the stage-17/17.5 four-day validation is
the source of truth for headline metrics) and renders them.

Every figure builder returns (path | None, figure_type, notes) and degrades
gracefully: a missing input yields a labelled placeholder plus a manifest entry
recording *why*, never a crash and never a fabricated number.

IMPORTANT SCOPE NOTE carried through every artifact: the radar simulation is a
POINT-DETECTION simulation. It is not raw RF/IQ and not a gridded range-Doppler
intensity simulation. "Pseudo range-Doppler" figures are scatter plots of point
detections in (radial velocity, range) space, not radar intensity heatmaps.
"""

import json
import os
import shutil
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table

# figure_type vocabulary recorded in the manifest
DATA_DRIVEN = "data-driven"
SCHEMATIC = "schematic"
COPIED = "copied"
PLACEHOLDER = "placeholder"

PSEUDO_RD_CAPTION = ("This is a pseudo range-Doppler point-detection visualization, "
                     "not a raw range-Doppler intensity heatmap.")

# Headline numbers are read from stage-17 artifacts; these are the documented
# fallbacks used only when an artifact is missing (and flagged as such).
HEADLINE_FALLBACK = {
    "true_tracks": 323808, "false_tracks": 815,
    "mlp_true_retention": 0.973, "mlp_false_reduction": 0.9939, "mlp_false_kept": 5,
    "gru_true_retention": 0.972, "gru_false_reduction": 0.9791, "gru_false_kept": 17,
}


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"  WARNING: could not read {path}: {exc}")
        return None


def _warn(msg):
    print(f"  WARNING: {msg}")


def _placeholder(path: str, title: str, reason: str) -> str:
    """A figure that says, in plain text, why it has no data."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=13, weight="bold",
            wrap=True)
    ax.text(0.5, 0.38, "\n".join(textwrap.wrap(f"Placeholder: {reason}", 72)),
            ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =============================================================================
# Source loading (stage-17/17.5 first, per the source-of-truth priority)
# =============================================================================

class Sources:
    """All compact artifacts stage 18 reads, loaded once and reported on."""

    def __init__(self, reports_root: str):
        self.root = reports_root
        j = lambda *p: os.path.join(reports_root, *p)
        self.paths = {
            "s07_overall": j("stage07_threshold_only", "threshold_overall.csv"),
            "s08_metrics": j("stage08_kalman_baseline", "kalman_metrics_by_day.csv"),
            "s07_vs_s08": j("stage08_kalman_baseline", "stage07_vs_stage08_2022-06-06.csv"),
            "s09_metrics": j("stage09_physics_scoring", "physics_metrics_by_threshold.csv"),
            "s11_metrics": j("stage11_adsb_prior_scoring", "adsb_prior_metrics_by_threshold.csv"),
            "s12_metrics": j("stage12_sequence_priors", "sequence_metrics_by_model_threshold.csv"),
            "s12_calib_cmp": j("stage12_sequence_priors", "calibration",
                               "sequence_calibration_comparison.csv"),
            "s12_scores": j("stage12_sequence_priors", "sequence_track_scores.csv"),
            "s13_metrics": j("stage13_vae_prior", "vae_metrics_by_threshold.csv"),
            "s14_unified": j("stage14_method_benchmark", "unified_method_metrics.csv"),
            "s14_pareto": j("stage14_method_benchmark", "pareto_frontier.csv"),
            "s14_failures": j("stage14_method_benchmark", "failure_case_candidates.csv"),
            "s15_resid": j("stage15_diffusion_denoising", "diffusion_residual_by_threshold.csv"),
            "s15_gap": j("stage15_diffusion_denoising", "diffusion_gap_filling_metrics.csv"),
            "s15_denoise": j("stage15_diffusion_denoising", "diffusion_track_denoising_metrics.csv"),
            "s16_model_abl": j("stage16_robustness", "model_ablation_mlp_gru_tcn.csv"),
            "s16_calib_abl": j("stage16_robustness", "calibration_ablation.csv"),
            "s16_window": j("stage16_robustness", "windowability_audit.csv"),
            "s17_overall": j("stage17_four_day_validation", "four_day_summary_overall.csv"),
            "s17_by_day": j("stage17_four_day_validation", "four_day_summary_by_day.csv"),
            "s17_by_thr": j("stage17_four_day_validation", "four_day_summary_by_threshold.csv"),
            "s17_model_cmp": j("stage17_four_day_validation", "model_comparison_mlp_vs_gru.csv"),
            "s17_fallback": j("stage17_four_day_validation", "interpretable_fallback_comparison.csv"),
            "s17_window": j("stage17_four_day_validation", "windowability_four_day_audit.csv"),
            "s17_findings": j("stage17_four_day_validation", "stage17_key_findings.csv"),
            "s17_s08_ctx": j("stage17_four_day_validation", "four_day_stage08_context.csv"),
            "s17p5_note": j("stage17_four_day_validation", "stage17p5_repro_hardening.md"),
            "s15_plots": j("stage15_diffusion_denoising", "plots"),
        }
        self.data = {k: read_csv_safe(v) if v.endswith(".csv") else None
                     for k, v in self.paths.items()}
        self.missing = [k for k, v in self.paths.items()
                        if not os.path.exists(v)]

    def get(self, key) -> Optional[pd.DataFrame]:
        return self.data.get(key)

    def present(self, key) -> bool:
        return os.path.exists(self.paths.get(key, ""))


def headline_metrics(src: Sources) -> Tuple[Dict, List[str]]:
    """Four-day headline numbers, read from stage-17 (never stale single-day)."""
    notes = []
    m = dict(HEADLINE_FALLBACK)
    ov = src.get("s17_overall")
    if ov is not None and len(ov):
        for model, prefix in (("mlp_dae", "mlp"), ("gru_ae", "gru")):
            r = ov[ov["model"] == model]
            if len(r):
                r = r.iloc[0]
                m[f"{prefix}_true_retention"] = float(r["pooled_true_retention"])
                m[f"{prefix}_false_reduction"] = float(r["pooled_false_reduction"])
                m[f"{prefix}_false_kept"] = int(r["stage12_kept_false_tracks"])
                m["true_tracks"] = int(r["stage08_true_tracks"])
                m["false_tracks"] = int(r["stage08_false_tracks"])
        m["dates"] = str(ov.iloc[0].get("dates_included", ""))
        m["thresholds"] = str(ov.iloc[0].get("thresholds_included", "-5,0,3,6"))
    else:
        notes.append("stage-17 four_day_summary_overall.csv missing; "
                     "headline metrics fall back to documented published values")
    m["selected_method"] = "stage12_mlp_dae_track_calibrated"
    return m, notes


# =============================================================================
# Final tables
# =============================================================================

def build_final_results_summary(src: Sources, hm: Dict) -> pd.DataFrame:
    rows = []

    def add(cid, claim, stage, evidence, metric, value, interp):
        rows.append({"claim_id": cid, "claim": claim, "stage": stage,
                     "evidence_file": evidence, "metric": metric, "value": value,
                     "interpretation": interp})

    s07 = src.get("s07_overall")
    if s07 is not None and len(s07):
        lo = s07.sort_values("threshold_db").iloc[0]
        hi = s07.sort_values("threshold_db").iloc[-1]
        add("C1", "Threshold-only detection exhibits a frame-level tradeoff", "07",
            "stage07_threshold_only/threshold_overall.csv",
            "empirical_pd / false_alarm_per_frame at low vs high threshold",
            f"{lo['empirical_pd']:.3f}@{lo['false_alarm_per_frame']:.1f} -> "
            f"{hi['empirical_pd']:.3f}@{hi['false_alarm_per_frame']:.2f}",
            "lowering the threshold buys detections at the cost of heavy clutter")
    else:
        add("C1", "Threshold-only detection exhibits a frame-level tradeoff", "07",
            "MISSING", "-", "n/a", "stage-07 artifact unavailable")

    ctx = src.get("s17_s08_ctx")
    if ctx is not None and len(ctx):
        broad = int(ctx["stage08_false_tracks"].sum())
        add("C2", "Kalman tracking recovers trajectories but leaves false tracks", "08",
            "stage17_four_day_validation/four_day_stage08_context.csv",
            "false tracks over 4 days at -5..6 dB "
            "(stage-08 own labels / strict+windowable evaluation denominator)",
            f"{broad:,} / {hm['false_tracks']:,}",
            "temporal association restores trajectories yet clutter still forms false tracks. "
            "The two counts are DIFFERENT populations: stage 08 labels every confirmed track "
            "(purity >= 0.5), whereas the filters are evaluated on the stricter, windowable "
            "subset (purity >= 0.8 and long enough to window). All filter metrics in this "
            "package use the second denominator.")

    fb = src.get("s17_fallback")
    if fb is not None and len(fb):
        d = fb[fb["false_reduction_defined"]] if "false_reduction_defined" in fb.columns else fb
        add("C3", "Hand physics scoring is strong and interpretable", "09",
            "stage17_four_day_validation/interpretable_fallback_comparison.csv",
            "mean stage-09 false reduction (defined cells)",
            round(float(d["stage09_false_reduction"].mean()), 4),
            "transparent rule-based scoring removes a large share of false tracks")

    s11 = src.get("s11_metrics")
    if s11 is not None and len(s11):
        add("C4", "Empirical marginal ADS-B priors are discriminative but insufficient", "11",
            "stage11_adsb_prior_scoring/adsb_prior_metrics_by_threshold.csv",
            "mean false reduction at score 0.5",
            round(float(s11["false_track_reduction"].mean()), 4),
            "marginal priors alone do not beat hand physics")

    add("C5", "Noise-matched deterministic sequence autoencoders are the strongest method",
        "12.5 / 17", "stage17_four_day_validation/four_day_summary_overall.csv",
        "pooled false reduction / true retention (4 days)",
        f"{hm['mlp_false_reduction']:.4f} / {hm['mlp_true_retention']:.3f}",
        f"mlp_dae keeps {hm['mlp_false_kept']} of {hm['false_tracks']:,} false tracks")

    s13 = src.get("s13_metrics")
    if s13 is not None and len(s13):
        add("C6", "The VAE does not beat the deterministic autoencoders", "13",
            "stage13_vae_prior/vae_metrics_by_threshold.csv",
            "mean VAE false reduction vs stage-12.5",
            round(float(s13["false_track_reduction"].mean()), 4),
            "probabilistic latent model matches but does not exceed stage 12.5")

    s15 = src.get("s15_gap")
    if s15 is not None and len(s15):
        add("C7", "Diffusion helps regularization/gap filling but not primary filtering", "15",
            "stage15_diffusion_denoising/diffusion_gap_filling_metrics.csv",
            "mean gap-fill improvement over linear interpolation",
            round(float(s15["improvement_ratio"].mean()), 4),
            "useful as a denoiser/regularizer; secondary as a classifier")

    add("C8", "Four-day validation confirms stage 12.5 generalizes", "17",
        "stage17_four_day_validation/stage17_four_day_validation_report.md",
        "days with complete results", 4,
        "the single-day limitation from stage 16 is closed")

    add("C9", "Reproducibility hardening completed", "17.5",
        "stage17_four_day_validation/stage17p5_repro_hardening.md",
        "regression checks passing", 12,
        "sys.executable, calibration protection, undefined-cell handling, ignore rules")
    return pd.DataFrame(rows)


def build_final_method_comparison(src: Sources, hm: Dict) -> pd.DataFrame:
    rows = []

    def add(method, stage, typ, role, tr, fr, kept, prec, scope, concl):
        rows.append({"method": method, "stage": stage, "type": typ, "primary_role": role,
                     "true_retention": tr, "false_reduction": fr, "false_tracks_kept": kept,
                     "precision_after": prec, "scope": scope, "conclusion": concl})

    # The baseline must use the SAME denominator the filters are scored against:
    # the strict (purity >= 0.8), windowable false-track population (815 over four days),
    # NOT stage-08's own broader purity>=0.5 count over all confirmed tracks (34,773).
    add("Kalman only", "08", "classical tracker", "baseline / denominator",
        1.0, 0.0, hm["false_tracks"], np.nan, "four-day (-5..6 dB)",
        "no filtering; keeps all false tracks and so defines the denominator "
        "(strict, windowable population)")

    fb = src.get("s17_fallback")
    if fb is not None and len(fb):
        d = fb[fb["false_reduction_defined"]] if "false_reduction_defined" in fb.columns else fb
        add("Hand physics scoring", "09", "rule-based", "interpretable fallback",
            round(float(d["stage09_true_retention"].mean()), 4),
            round(float(d["stage09_false_reduction"].mean()), 4), np.nan,
            round(float(d["stage09_precision_after"].mean()), 4), "four-day (-5..6 dB)",
            "strong, transparent, no training; beaten on false reduction by stage 12.5")

    s11 = src.get("s11_metrics")
    if s11 is not None and len(s11):
        add("ADS-B marginal priors", "11", "empirical prior", "evidence, not sufficient",
            round(float(s11["true_track_retention"].mean()), 4),
            round(float(s11["false_track_reduction"].mean()), 4), np.nan,
            round(float(s11["precision_after"].mean()), 4), "ONE-DAY (2022-06-06)",
            "discriminative but does not clearly beat stage 09")

    ov = src.get("s17_overall")
    for model, label, prefix in (("mlp_dae", "Sequence AE (MLP-DAE), track-calibrated", "mlp"),
                                 ("gru_ae", "Sequence AE (GRU-AE), track-calibrated", "gru")):
        prec = np.nan
        if ov is not None and len(ov):
            r = ov[ov["model"] == model]
            if len(r):
                prec = round(float(r.iloc[0]["pooled_precision_after"]), 6)
        add(label, "12.5", "learned sequence prior",
            "PRIMARY false-track filter" if model == "mlp_dae" else "strong alternative",
            round(hm[f"{prefix}_true_retention"], 4), round(hm[f"{prefix}_false_reduction"], 4),
            hm[f"{prefix}_false_kept"], prec, "four-day (-5..6 dB)",
            "selected method" if model == "mlp_dae" else "close second; best at low thresholds")

    s13 = src.get("s13_metrics")
    if s13 is not None and len(s13):
        for variant in sorted(s13["variant"].unique()):
            g = s13[s13["variant"] == variant]
            add(f"VAE prior ({variant})", "13", "probabilistic latent prior", "not selected",
                round(float(g["true_track_retention"].mean()), 4),
                round(float(g["false_track_reduction"].mean()), 4), np.nan,
                round(float(g["precision_after"].mean()), 4), "ONE-DAY (2022-06-06)",
                "matches but does not beat stage 12.5")

    s15 = src.get("s15_resid")
    if s15 is not None and len(s15):
        add("Diffusion residual classifier", "15", "DDPM denoiser residual",
            "secondary / regularizer",
            round(float(s15["true_track_retention"].mean()), 4),
            round(float(s15["false_track_reduction"].mean()), 4), np.nan,
            round(float(s15["precision_after"].mean()), 4), "ONE-DAY (2022-06-06)",
            "useful for denoising/gap filling; clearly worse as a filter")
    return pd.DataFrame(rows)


def build_final_ablation_summary(src: Sources) -> pd.DataFrame:
    rows = []

    def add(abl, comp, winner, ev, interp):
        rows.append({"ablation": abl, "comparison": comp, "winner": winner,
                     "evidence": ev, "interpretation": interp})

    ca = src.get("s16_calib_abl")
    if ca is not None and len(ca):
        ct = ca[ca["calibration_mode"] == "clean_truth"]["true_track_retention"].mean()
        tp = ca[ca["calibration_mode"] == "track_purity"]["true_track_retention"].mean()
        add("Calibration", "clean-truth vs track-purity (noise-matched)", "track_purity",
            "stage16_robustness/calibration_ablation.csv",
            f"true retention {ct:.3f} -> {tp:.3f}: clean-truth quantiles under-retain noisy "
            "tracks; noise-matched calibration fixes the domain shift")

    ma = src.get("s16_model_abl")
    if ma is not None and len(ma) and "row_type" in ma.columns:
        agg = ma[ma["row_type"] == "aggregate"].sort_values("overall_rank")
        if len(agg):
            add("Model family", "MLP-DAE vs GRU-AE vs TCN-AE", str(agg.iloc[0]["model"]),
                "stage16_robustness/model_ablation_mlp_gru_tcn.csv",
                "MLP and GRU are close and strong; TCN retains fewer true tracks")

    fb = src.get("s17_fallback")
    if fb is not None and len(fb):
        d = fb[fb["false_reduction_defined"]] if "false_reduction_defined" in fb.columns else fb
        wins = int((d["stage12_minus_stage09_false_reduction"] > 0.01).sum())
        undef = len(fb) - len(d)
        add("Learned vs interpretable", "stage 09 hand physics vs stage 12.5", "stage 12.5",
            "stage17_four_day_validation/interpretable_fallback_comparison.csv",
            f"stage 12 removes more false tracks in {wins}/{len(d)} defined cells "
            f"({undef} undefined); stage 09 retains slightly more true tracks and stays the "
            "interpretable fallback")

    add("Probabilistic latent", "stage 12.5 vs stage 13 VAE", "stage 12.5",
        "stage13_vae_prior/vae_prior_report.md",
        "the VAE matches but does not beat the deterministic autoencoder; ELBO adds nothing "
        "over reconstruction error")

    add("Generative denoiser", "stage 12.5 vs stage 15 diffusion residual", "stage 12.5",
        "stage15_diffusion_denoising/diffusion_denoising_report.md",
        "diffusion regularizes tracks and modestly improves gap filling, but is clearly worse "
        "as a primary false-track classifier")

    add("Windowability caveat", "high thresholds (9/12 dB) vs -5/0/3/6 dB",
        "n/a (denominator effect)", "stage17_four_day_validation/windowability_four_day_audit.csv",
        "sequence methods only score windowable tracks; at 9/12 dB there are ~0 windowable "
        "false tracks, so false-reduction is undefined and cross-method comparison is not "
        "apples-to-apples there")
    return pd.DataFrame(rows)


def build_final_reproducibility_checklist(src: Sources) -> pd.DataFrame:
    hardened = src.present("s17p5_note")
    rows = [
        ("Deterministic SHA-256 derived seeding in stages 05/06/10", "yes",
         "F02 stage-05/06 relocation + stage-10 reservoir sampling",
         "full-pipeline rerun reproduced every count and metric exactly"),
        ("All stage validation gates green", "yes",
         "each stage script prints a VALIDATION GATE block",
         "gates fail loudly rather than emitting inconsistent reports"),
        ("Stage 17.5 regression checks pass", "yes" if hardened else "unknown",
         "scripts/17p5_regression_checks.py",
         "12 assertions incl. a negative control for the bare-`python` guard"),
        ("Canonical stage-12 calibration protected", "yes" if hardened else "unknown",
         "reports/stage17_four_day_validation/stage17p5_repro_hardening.md",
         "--calibration-output defaults to <report-dir>/calibration/; orchestrators sandbox it"),
        ("Internal subprocesses use sys.executable", "yes" if hardened else "unknown",
         "scripts/16_robustness_ablation.py, scripts/17_four_day_validation.py",
         "a bare `python` does not exist on all machines"),
        ("Undefined false-reduction handled explicitly", "yes" if hardened else "unknown",
         "utils/common.py safe_reduction/undefined_reason",
         "zero denominator -> NaN + reason, never 0/1 and never blamed on stage 09"),
        ("Large CSVs / checkpoints git-ignored", "yes", ".gitignore + git check-ignore",
         "track files, per-track score CSVs and .pt checkpoints are never committed"),
        ("Four-day validation rerun succeeded", "yes" if src.present("s17_overall") else "no",
         "reports/stage17_four_day_validation/",
         "stage-08/09/12.5 generated for all four days; gate enforced gap closed"),
        ("Code committed and pushed through the final stage", "yes",
         "git log / remote origin",
         "each stage committed with compact reports; large artifacts excluded"),
    ]
    return pd.DataFrame(rows, columns=["item", "status", "evidence", "notes"])


def build_final_limitations() -> pd.DataFrame:
    rows = [
        ("Point-detection radar simulation, not raw RF/IQ",
         "Detection-level realism only; no waveform, sidelobe, or receiver effects",
         "A raw RF/range-Doppler intensity simulation would test the detector itself"),
        ("Pseudo range-Doppler figures are point-detection scatter plots",
         "They must not be read as radar intensity maps",
         "Generate true RD heatmaps only if gridded intensity data is simulated"),
        ("Synthetic relocation of trajectories near the radar",
         "Motion shape is real ADS-B; radar geometry/engagement is synthetic",
         "Deploy against a real radar site with co-located ADS-B truth"),
        ("OpenSky ADS-B-derived fixed-wing GA focus",
         "Conclusions are scoped to light general-aviation motion",
         "Extend to rotorcraft, UAS, and manoeuvring military profiles"),
        ("High-threshold windowability caveat (9/12 dB)",
         "Sequence methods have ~0 windowable false tracks there; reduction is undefined",
         "Compare only at -5/0/3/6 dB, or use a length-matched denominator"),
        ("Track-purity calibration uses truth labels to SELECT calibration tracks",
         "The score itself never sees labels, but calibration-set selection does",
         "Use a purity proxy (e.g. self-consistency) to make calibration fully unsupervised"),
        ("Learned-method evidence is on simulated detections, not real radar returns",
         "Absolute numbers will not transfer directly to a fielded radar",
         "Validate on recorded radar detections with ADS-B truth"),
        ("Diffusion and VAE were not tuned exhaustively",
         "Their negative results are for the configurations tested, not the families",
         "A scoped sweep could revisit them if a specific gap emerges"),
        ("Runtime / deployment not optimized",
         "Throughput and latency for an operational system are unmeasured",
         "A deployment-style runtime and operating-point study"),
    ]
    return pd.DataFrame(rows, columns=["limitation", "impact", "mitigation_or_future_work"])


def build_final_contributions() -> pd.DataFrame:
    rows = [
        (1, "An ADS-B-grounded weak-target radar simulation and evaluation framework "
            "(real trajectories -> radar truth -> noisy point detections -> evaluation ladder)",
         "integration / framework",
         "F01 stages 01-04, F02 stages 05-06, F03 stages 07-17.5"),
        (2, "Synthetic relocation that preserves ADS-B motion shape while controlling radar "
            "geometry (per-trajectory sha256-seeded anchors)", "methodological",
         "F02 stage 05 (--relocate-near-radar), reports/relocated_experiment_audit.md"),
        (3, "A baseline ladder from threshold-only detection through Kalman tracking, hand "
            "physics, empirical priors, sequence autoencoders, VAE, and diffusion",
         "systematic evaluation",
         "reports/stage14_method_benchmark/stage14_method_benchmark_report.md"),
        (4, "Noise-matched (track-purity) calibration of a learned sequence prior, which "
            "converts a well-discriminating but miscalibrated score into a usable filter",
         "methodological (key)",
         "reports/stage12_sequence_priors/ + stage16_robustness/calibration_ablation.csv"),
        (5, "Four-day evidence that trajectory-shape scoring suppresses clutter-induced false "
            "tracks while retaining true aircraft tracks",
         "empirical result",
         "reports/stage17_four_day_validation/four_day_summary_overall.csv"),
        (6, "A reproducible multi-repo pipeline with deterministic outputs, per-stage "
            "validation gates, and regression-checked hardening",
         "engineering / reproducibility",
         "scripts/17p5_regression_checks.py, stage17p5_repro_hardening.md"),
    ]
    return pd.DataFrame(rows, columns=["contribution_id", "contribution", "novelty_level",
                                       "supporting_evidence"])


# =============================================================================
# Figures 01-02: schematics
# =============================================================================

def build_pipeline_diagram(path: str) -> Tuple[str, str, str]:
    repos = [
        ("F01-Preprocessing", ["01 whitelist", "02 filtering", "03 cleaning", "04 resampling"]),
        ("F02-Radar", ["05 relocated radar truth", "06 noisy/cluttered detections"]),
        ("F03-Evaluation", ["07 threshold-only", "08 Kalman tracking", "09 hand physics",
                            "10 ADS-B priors", "11 marginal priors",
                            "12/12.5 sequence AE + calibration", "13 VAE",
                            "14 benchmark", "15 diffusion", "16 robustness",
                            "17 four-day validation", "17.5 hardening",
                            "18 final package"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    x = [0.05, 0.38, 0.71]
    box_w = 0.26
    for i, (repo, stages) in enumerate(repos):
        ax.text(x[i] + box_w / 2, 0.96, repo, ha="center", va="top", fontsize=11,
                weight="bold")
        y = 0.90
        for s in stages:
            ax.add_patch(plt.Rectangle((x[i], y - 0.045), box_w, 0.04, fill=False, lw=1.0))
            ax.text(x[i] + box_w / 2, y - 0.025, s, ha="center", va="center", fontsize=8)
            if s != stages[-1]:
                ax.annotate("", xy=(x[i] + box_w / 2, y - 0.055),
                            xytext=(x[i] + box_w / 2, y - 0.045),
                            arrowprops=dict(arrowstyle="->", lw=0.8))
            y -= 0.062
        if i < len(repos) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.005, 0.5), xytext=(x[i] + box_w + 0.005, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.5, 0.02,
            "Real ADS-B trajectories -> cleaned GA trajectories -> relocated radar truth -> "
            "noisy point detections -> evaluation ladder",
            ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Three-repo pipeline: ADS-B to weak-target radar evaluation", fontsize=13)
    return _save(fig, path), SCHEMATIC, "hand-drawn box/arrow diagram of stages 01-18"


def build_radar_simulation_concept(path: str) -> Tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    th = np.linspace(0, 2 * np.pi, 400)
    for r, ls, lbl in [(10, "--", "10 km inner anchor radius"),
                       (80, "--", "80 km outer anchor radius"),
                       (100, ":", "100 km clutter support")]:
        ax.plot(r * np.cos(th), r * np.sin(th), linestyle=ls, lw=1.2, label=lbl)
    ax.plot([0], [0], marker="^", markersize=13, label="radar (origin)")
    rng = np.random.default_rng(18)
    for k in range(7):
        r0 = rng.uniform(10, 80)
        a0 = rng.uniform(0, 2 * np.pi)
        hdg = rng.uniform(0, 2 * np.pi)
        t = np.linspace(0, 130, 60)
        x = r0 * np.cos(a0) + t * np.cos(hdg)
        y = r0 * np.sin(a0) + t * np.sin(hdg)
        ax.plot(x, y, lw=1.0, alpha=0.85)
        ax.plot(x[0], y[0], marker="o", markersize=3.5)
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_title("Radar simulation concept: relocated ADS-B trajectories\n"
                 "(anchors in a 10-80 km annulus; long tracks may drift beyond 100 km)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, lw=0.4)
    ax.text(0.02, 0.02, "Noise, missed detections and clutter are injected in Stage 06",
            transform=ax.transAxes, fontsize=7, style="italic")
    return _save(fig, path), SCHEMATIC, "conceptual geometry; trajectory shapes illustrative"


# =============================================================================
# Figures 03-06: pseudo range-Doppler / range-azimuth point-detection frames
# =============================================================================

def _load_busy_frame(det_path: str, max_rows: int = 3_000_000):
    """Load a detection file's key columns and return the busiest frame's rows."""
    cols = ["frame_id", "is_target", "meas_range_m", "meas_radial_velocity_mps",
            "meas_azimuth_deg"]
    df = pd.read_csv(det_path, usecols=cols, nrows=max_rows)
    if df.empty:
        return None
    counts = df.groupby("frame_id").size()
    # a frame with plenty of detections AND at least a few targets
    tgt = df[df["is_target"] == 1].groupby("frame_id").size()
    cand = counts[counts.index.isin(tgt[tgt >= 3].index)] if len(tgt) else counts
    if cand.empty:
        cand = counts
    frame = int(cand.idxmax())
    return df[df["frame_id"] == frame], frame


def build_pseudo_range_doppler_frame(det_path: str, path: str, label: str) -> Tuple[str, str, str]:
    if not det_path or not os.path.exists(det_path):
        _warn(f"stage-06 detections unavailable for pseudo-RD ({label})")
        return (_placeholder(path, f"Pseudo range-Doppler point-detection frame ({label})",
                             "Stage 06 detection file unavailable; this figure requires "
                             "data/active/sim_detections_relocated/ (large, git-ignored)."),
                PLACEHOLDER, "stage-06 detection file unavailable")
    try:
        frame_df, frame = _load_busy_frame(det_path)
    except Exception as exc:
        _warn(f"could not read {det_path}: {exc}")
        return (_placeholder(path, f"Pseudo range-Doppler point-detection frame ({label})",
                             f"Could not read stage-06 detections: {exc}"),
                PLACEHOLDER, f"read error: {exc}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for is_t, name, marker, alpha in ((0, "clutter", ".", 0.25), (1, "target", "o", 0.9)):
        g = frame_df[frame_df["is_target"] == is_t]
        if len(g):
            ax.scatter(g["meas_radial_velocity_mps"], g["meas_range_m"] / 1000.0,
                       s=(6 if is_t == 0 else 30), alpha=alpha, marker=marker, label=name)
    ax.set_xlabel("radial velocity (m/s)")
    ax.set_ylabel("range (km)")
    ax.set_title(f"Pseudo range-Doppler point-detection frame ({label})\n"
                 f"2022-06-06, frame {frame}, {len(frame_df):,} detections")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)
    ax.text(0.01, -0.16, PSEUDO_RD_CAPTION, transform=ax.transAxes, fontsize=7,
            style="italic")
    n_t = int((frame_df["is_target"] == 1).sum())
    return (_save(fig, path), DATA_DRIVEN,
            f"stage-06 detections, frame {frame}: {len(frame_df)} detections, {n_t} targets; "
            "labels used for visualization only")


def build_range_azimuth_frame(det_path: str, path: str, label: str) -> Tuple[str, str, str]:
    if not det_path or not os.path.exists(det_path):
        _warn(f"stage-06 detections unavailable for range-azimuth ({label})")
        return (_placeholder(path, f"Range-azimuth point-detection frame ({label})",
                             "Stage 06 detection file unavailable."),
                PLACEHOLDER, "stage-06 detection file unavailable")
    try:
        frame_df, frame = _load_busy_frame(det_path)
    except Exception as exc:
        return (_placeholder(path, f"Range-azimuth point-detection frame ({label})",
                             f"Could not read stage-06 detections: {exc}"),
                PLACEHOLDER, f"read error: {exc}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for is_t, name, marker, alpha in ((0, "clutter", ".", 0.25), (1, "target", "o", 0.9)):
        g = frame_df[frame_df["is_target"] == is_t]
        if len(g):
            ax.scatter(g["meas_azimuth_deg"], g["meas_range_m"] / 1000.0,
                       s=(6 if is_t == 0 else 30), alpha=alpha, marker=marker, label=name)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("range (km)")
    ax.set_title(f"Range-azimuth point-detection frame ({label})\n"
                 f"2022-06-06, frame {frame}, {len(frame_df):,} detections")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)
    n_t = int((frame_df["is_target"] == 1).sum())
    return (_save(fig, path), DATA_DRIVEN,
            f"stage-06 detections, frame {frame}: {len(frame_df)} detections, {n_t} targets")


# =============================================================================
# Figures 07-11: ladder context
# =============================================================================

def build_threshold_tradeoff_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s07_overall")
    if df is None or df.empty:
        return (_placeholder(path, "Detection-threshold tradeoff (Stage 07)",
                             "stage07_threshold_only/threshold_overall.csv unavailable"),
                PLACEHOLDER, "stage-07 artifact missing")
    df = df.sort_values("threshold_db")
    pd_n = df["empirical_pd"]
    fa = df["false_alarm_per_frame"]
    fa_n = (fa - fa.min()) / (fa.max() - fa.min()) if fa.max() > fa.min() else fa * 0
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df["threshold_db"], pd_n, marker="o", label="empirical $P_d$ (0-1)")
    ax.plot(df["threshold_db"], fa_n, marker="s", linestyle="--",
            label=f"false alarms/frame (normalized, max={fa.max():.1f})")
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("normalized trend (0-1)")
    ax.set_title("Stage 07: the detection-threshold tradeoff\n"
                 "low threshold -> more weak targets AND far more clutter")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "both series normalized to [0,1]; FA/frame max annotated"


def build_kalman_effect_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    cmp_df = src.get("s07_vs_s08")
    s07 = src.get("s07_overall")
    if cmp_df is None and s07 is None:
        return (_placeholder(path, "Stage 07 vs Stage 08: tracking recovers trajectories",
                             "stage07/stage08 comparison artifacts unavailable"),
                PLACEHOLDER, "stage-07/08 artifacts missing")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    note = ""
    if s07 is not None and len(s07):
        g = s07.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["empirical_pd"], marker="o", label="Stage 07 frame $P_d$")
    if cmp_df is not None and len(cmp_df):
        col = next((c for c in cmp_df.columns if "track_detection_rate" in c), None)
        thr = "threshold_db" if "threshold_db" in cmp_df.columns else None
        if col and thr:
            g = cmp_df.sort_values(thr)
            ax.plot(g[thr], g[col], marker="s", linestyle="--",
                    label="Stage 08 track detection rate")
            note = "stage07_vs_stage08 comparison CSV"
    ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("rate")
    ax.set_title("Temporal tracking recovers trajectories even as frame $P_d$ falls")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    ax.text(0.01, -0.17, "Frame-level $P_d$ and track-level detection rate measure different "
            "things and must not be equated.", transform=ax.transAxes, fontsize=7,
            style="italic")
    return _save(fig, path), DATA_DRIVEN, note or "stage-07 curve only"


def build_stage09_effect_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s09_metrics")
    if df is None or df.empty:
        return (_placeholder(path, "Stage 09 physics filter effect",
                             "stage09 physics_metrics_by_threshold.csv unavailable"),
                PLACEHOLDER, "stage-09 artifact missing")
    g = df.sort_values("threshold_db")
    x = np.arange(len(g)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, g["stage08_false_tracks"], w, label="false tracks before (Stage 08)")
    ax.bar(x + w / 2, g["stage09_kept_false_tracks"], w, label="false tracks after (Stage 09)")
    ax.set_xticks(x); ax.set_xticklabels([f"{t:g}" for t in g["threshold_db"]])
    ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("false tracks")
    ax.set_yscale("symlog")
    ax.set_title("Stage 09 hand physics: a strong, interpretable false-track filter")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "symlog y-axis; one-day (2022-06-06) stage-09 metrics"


def build_calibration_effect_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s16_calib_abl")
    if df is None or df.empty:
        df = src.get("s12_calib_cmp")   # never `or` on DataFrames: truthiness is ambiguous
    if df is None or df.empty:
        return (_placeholder(path, "Noise-matched calibration effect",
                             "calibration ablation artifact unavailable"),
                PLACEHOLDER, "calibration ablation missing")
    agg = df.groupby("calibration_mode").agg(
        retention=("true_track_retention", "mean"),
        reduction=("false_track_reduction", "mean")).reset_index()
    x = np.arange(len(agg)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, agg["retention"], w, label="true-track retention")
    ax.bar(x + w / 2, agg["reduction"], w, label="false-track reduction")
    ax.set_xticks(x); ax.set_xticklabels(agg["calibration_mode"])
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
    ax.set_title("Why noise-matched (track-purity) calibration was essential\n"
                 "clean-truth quantiles collapse true-track retention")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    for xi, v in zip(x - w / 2, agg["retention"]):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    return _save(fig, path), DATA_DRIVEN, "means across models and thresholds at score 0.5"


def build_method_ladder_plot(src: Sources, hm: Dict, path: str) -> Tuple[str, str, str]:
    cmp_df = build_final_method_comparison(src, hm)
    d = cmp_df[np.isfinite(cmp_df["true_retention"]) & np.isfinite(cmp_df["false_reduction"])]
    d = d[d["method"] != "Kalman only"]
    if d.empty:
        return (_placeholder(path, "Method ladder comparison", "no comparable methods found"),
                PLACEHOLDER, "no method rows")
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in d.iterrows():
        ax.scatter(r["true_retention"], r["false_reduction"], s=70)
        ax.annotate(f"{r['method']}\n({r['scope'].split()[0]})",
                    (r["true_retention"], r["false_reduction"]),
                    textcoords="offset points", xytext=(7, -4), fontsize=7)
    ax.set_xlabel("true-track retention"); ax.set_ylabel("false-track reduction")
    ax.set_title("Method ladder: retention vs false-track suppression\n"
                 "(stage 12.5 four-day; other methods one-day, as labelled)")
    ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "scope labels distinguish four-day from one-day methods"


# =============================================================================
# Figures 12-17: four-day validation and caveats
# =============================================================================

def build_four_day_validation_plots(src: Sources, ret_path: str,
                                    fr_path: str) -> List[Tuple[str, str, str]]:
    df = src.get("s17_by_day")
    out = []
    if df is None or df.empty:
        for p, t in ((ret_path, "True-track retention by day"),
                     (fr_path, "False-track reduction by day")):
            out.append((_placeholder(p, t, "stage-17 four_day_summary_by_day.csv unavailable"),
                        PLACEHOLDER, "stage-17 by-day artifact missing"))
        return out
    for p, col, title in ((ret_path, "pooled_true_retention", "True-track retention by day"),
                          (fr_path, "pooled_false_reduction", "False-track reduction by day")):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for model, g in df.groupby("model"):
            g = g.sort_values("date")
            ax.plot(g["date"], g[col], marker="o", label=model)
        ax.set_xlabel("date"); ax.set_ylabel(col.replace("pooled_", "").replace("_", " "))
        ax.set_title(f"{title} (thresholds -5/0/3/6 dB)")
        ax.legend(fontsize=8); ax.grid(True, lw=0.4)
        plt.xticks(rotation=15)
        lo = df[col].min()
        ax.set_ylim(max(0, lo - 0.03), 1.005)
        out.append((_save(fig, p), DATA_DRIVEN, "pooled across thresholds within each day"))
    return out


def build_false_tracks_before_after_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s17_by_day")
    if df is None or df.empty:
        return (_placeholder(path, "False tracks before/after Stage 12.5",
                             "stage-17 by-day artifact unavailable"),
                PLACEHOLDER, "stage-17 by-day missing")
    mlp = df[df["model"] == "mlp_dae"].sort_values("date")
    if mlp.empty:
        mlp = df.sort_values("date")
    x = np.arange(len(mlp)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, mlp["stage08_false_tracks"], w, label="Stage 08 false tracks (before)")
    ax.bar(x + w / 2, mlp["stage12_kept_false_tracks"], w, label="kept after Stage 12.5")
    for xi, v in zip(x - w / 2, mlp["stage08_false_tracks"]):
        ax.text(xi, v + 2, f"{int(v)}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, mlp["stage12_kept_false_tracks"]):
        ax.text(xi, v + 2, f"{int(v)}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mlp["date"], rotation=15)
    ax.set_ylabel("false tracks"); ax.set_xlabel("date")
    tot_b = int(mlp["stage08_false_tracks"].sum()); tot_a = int(mlp["stage12_kept_false_tracks"].sum())
    ax.set_title(f"False tracks before and after Stage 12.5 (mlp_dae)\n"
                 f"four-day total: {tot_b} -> {tot_a}")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, f"four-day totals {tot_b} -> {tot_a}"


def build_model_comparison_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s17_model_cmp")
    if df is None or df.empty:
        return (_placeholder(path, "MLP vs GRU (four-day)",
                             "stage-17 model_comparison_mlp_vs_gru.csv unavailable"),
                PLACEHOLDER, "stage-17 model comparison missing")
    agg = df[df["date"] == "ALL"]
    agg = agg[agg["metric"].isin(["true_track_retention", "false_track_reduction",
                                  "precision_after"])]
    if agg.empty:
        return (_placeholder(path, "MLP vs GRU (four-day)", "no aggregate rows"),
                PLACEHOLDER, "no ALL rows")
    x = np.arange(len(agg)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, agg["mlp_dae"], w, label="mlp_dae")
    ax.bar(x + w / 2, agg["gru_ae"], w, label="gru_ae")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_", " ") for m in agg["metric"]], fontsize=8)
    ax.set_ylim(0.9, 1.005); ax.set_ylabel("aggregate mean (four-day)")
    ax.set_title("MLP vs GRU over four days: MLP wins overall, GRU is close")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "y-axis zoomed to 0.90-1.00 to show the small gap"


def build_stage09_vs_stage12_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s17_fallback")
    if df is None or df.empty:
        return (_placeholder(path, "Stage 09 vs Stage 12.5",
                             "stage-17 interpretable_fallback_comparison.csv unavailable"),
                PLACEHOLDER, "stage-17 fallback missing")
    d = df[df["false_reduction_defined"]] if "false_reduction_defined" in df.columns else df
    g = d.groupby("threshold_db").agg(s09=("stage09_false_reduction", "mean"),
                                      s12=("stage12_false_reduction", "mean")).reset_index()
    x = np.arange(len(g)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, g["s09"], w, label="Stage 09 hand physics")
    ax.bar(x + w / 2, g["s12"], w, label="Stage 12.5 mlp_dae")
    ax.set_xticks(x); ax.set_xticklabels([f"{t:g}" for t in g["threshold_db"]])
    ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("false-track reduction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learned sequence prior vs interpretable fallback\n"
                 "(four-day means over defined cells)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    n_undef = len(df) - len(d)
    return (_save(fig, path), DATA_DRIVEN,
            f"means over defined cells only; {n_undef} undefined (zero-denominator) cells excluded")


def build_windowability_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s17_window")
    if df is None or df.empty:
        df = src.get("s16_window")
    if df is None or df.empty or "windowable_fraction_false" not in getattr(df, "columns", []):
        return (_placeholder(path, "Windowability caveat",
                             "windowability audit artifact unavailable"),
                PLACEHOLDER, "windowability audit missing")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if "date" in df.columns and df["date"].nunique() > 1:
        for date, g in df.groupby("date"):
            g = g.sort_values("threshold_db")
            ax.plot(g["threshold_db"], g["windowable_fraction_false"], marker="o", label=str(date))
    else:
        g = df.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["windowable_fraction_false"], marker="o",
                label="windowable false fraction")
    ax.set_xlabel("detection threshold (dB)")
    ax.set_ylabel("windowable fraction of false tracks")
    ax.set_title("Windowability caveat: at high thresholds almost no false track\n"
                 "is long enough to window, so false-reduction is undefined")
    ax.legend(fontsize=7); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "explains why 9/12 dB stay audit-only"


# =============================================================================
# Figures 18-19: failure cases (need the large track CSVs; degrade gracefully)
# =============================================================================

def _load_track_xy(tracks_dir: str, date: str, threshold_db: float, track_id: int):
    def tok(t):
        s = f"{abs(t):g}".replace(".", "p")
        if float(t).is_integer():
            s = f"{int(abs(t))}p0"
        return ("m" if t < 0 else "") + s
    path = os.path.join(tracks_dir, f"tracks_{date}_thr_{tok(threshold_db)}dB.csv")
    if not os.path.exists(path):
        return None, path
    keep = []
    for chunk in pd.read_csv(path, usecols=["track_id", "timestamp", "x_m", "y_m"],
                             chunksize=1_000_000):
        m = chunk[chunk["track_id"] == track_id]
        if len(m):
            keep.append(m)
    if not keep:
        return None, path
    return pd.concat(keep).sort_values("timestamp"), path


def _failure_case_plot(src, tracks_dir, path, case_type, title, subtitle):
    fc = src.get("s14_failures")
    if fc is None or fc.empty or "case_type" not in getattr(fc, "columns", []):
        return (_placeholder(path, title, "stage-14 failure_case_candidates.csv unavailable"),
                PLACEHOLDER, "failure candidates missing")
    cand = fc[fc["case_type"] == case_type]
    if cand.empty:
        return (_placeholder(path, title,
                             f"no '{case_type}' candidate exists in the compact artifacts"),
                PLACEHOLDER, f"no {case_type} candidate")
    r = cand.iloc[0]
    if not tracks_dir or not os.path.isdir(tracks_dir):
        return (_placeholder(path, title,
                             "per-track trajectory requires data/active/tracks_kalman/ "
                             "(large, git-ignored) which is unavailable"),
                PLACEHOLDER, "tracks_kalman unavailable")
    try:
        xy, tpath = _load_track_xy(tracks_dir, str(r["date"]), float(r["threshold_db"]),
                                   int(r["track_id"]))
    except Exception as exc:
        return (_placeholder(path, title, f"could not read track file: {exc}"),
                PLACEHOLDER, f"track read error: {exc}")
    if xy is None or xy.empty:
        return (_placeholder(path, title,
                             f"track {int(r['track_id'])} not found in {os.path.basename(tpath)}"),
                PLACEHOLDER, "track id not found")

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy["x_m"] / 1000.0, xy["y_m"] / 1000.0,
                    c=(xy["timestamp"] - xy["timestamp"].min()), s=16)
    ax.plot(xy["x_m"] / 1000.0, xy["y_m"] / 1000.0, lw=0.7, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="time since track start (s)")
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    s12 = r.get("score_stage12", np.nan)
    ax.set_title(f"{title}\ntrack {int(r['track_id'])}, {r['date']}, "
                 f"{float(r['threshold_db']):g} dB, stage-12.5 score {s12:.3f}")
    ax.grid(True, lw=0.4)
    ax.text(0.01, -0.14, subtitle, transform=ax.transAxes, fontsize=7, style="italic")
    return (_save(fig, path), DATA_DRIVEN,
            f"track {int(r['track_id'])} from {os.path.basename(tpath)}; "
            f"stage-12.5 score {s12}")


def build_failure_case_plots(src: Sources, tracks_dir: str, surviving_path: str,
                             rejected_path: str) -> List[Tuple[str, str, str]]:
    a = _failure_case_plot(
        src, tracks_dir, surviving_path, "false_survives_s12",
        "Failure case: a FALSE track that survived Stage 12.5",
        "Likely failure mode: a smooth clutter chain whose windows reconstruct like real motion.")
    b = _failure_case_plot(
        src, tracks_dir, rejected_path, "true_rejected_s12",
        "Failure case: a TRUE track rejected by Stage 12.5",
        "Possible causes: unusual manoeuvre, short/noisy segment, or a calibration edge case.")
    return [a, b]


# =============================================================================
# Figures 20-21: diffusion examples (copied from stage 15 when present)
# =============================================================================

def _copy_stage_plot(src_plot: str, dst: str, title: str,
                     reason: str) -> Tuple[str, str, str]:
    if src_plot and os.path.exists(src_plot):
        shutil.copyfile(src_plot, dst)
        return dst, COPIED, f"copied from {src_plot}"
    return _placeholder(dst, title, reason), PLACEHOLDER, reason


def build_diffusion_example_plots(src: Sources, denoise_path: str,
                                  gap_path: str) -> List[Tuple[str, str, str]]:
    plots = src.paths.get("s15_plots", "")
    a = _copy_stage_plot(os.path.join(plots, "example_denoised_track.png"), denoise_path,
                         "Diffusion denoising example",
                         "stage-15 example_denoised_track.png unavailable")
    b = _copy_stage_plot(os.path.join(plots, "gap_filling_error_comparison.png"), gap_path,
                         "Diffusion gap-filling example",
                         "stage-15 gap_filling_error_comparison.png unavailable")
    return [a, b]


# =============================================================================
# Figures 22-25: final summary figures
# =============================================================================

def build_pareto_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    df = src.get("s14_pareto")
    if df is None or df.empty:
        return (_placeholder(path, "Final method Pareto frontier",
                             "stage-14 pareto_frontier.csv unavailable"),
                PLACEHOLDER, "stage-14 pareto missing")
    thr0 = sorted(df["threshold_db"].unique())[0]
    g = df[df["threshold_db"] == thr0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    dom = g[~g["is_pareto"]]
    par = g[g["is_pareto"]].sort_values("true_track_retention")
    ax.scatter(dom["true_track_retention"], dom["false_track_reduction"], s=16, alpha=0.35,
               label="dominated")
    ax.plot(par["true_track_retention"], par["false_track_reduction"], marker="o",
            label="Pareto frontier")
    for _, r in par.iterrows():
        ax.annotate(r["method_id"].replace("stage12_", "s12:").replace("stage09_", "s09:"),
                    (r["true_track_retention"], r["false_track_reduction"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6)
    ax.set_xlabel("true-track retention"); ax.set_ylabel("false-track reduction")
    ax.set_title(f"Non-dominated operating points at {thr0:g} dB (Stage 14)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, f"threshold {thr0:g} dB (worst clutter regime)"


def build_precision_by_threshold_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    by_thr = src.get("s17_by_thr")
    fb = src.get("s17_fallback")
    if by_thr is None or by_thr.empty:
        return (_placeholder(path, "Precision after filtering by threshold",
                             "stage-17 four_day_summary_by_threshold.csv unavailable"),
                PLACEHOLDER, "stage-17 by-threshold missing")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, g in by_thr.groupby("model"):
        g = g.sort_values("threshold_db")
        ax.plot(g["threshold_db"], g["pooled_precision_after"], marker="o",
                label=f"Stage 12.5 {model}")
    if fb is not None and len(fb):
        g = fb.groupby("threshold_db")["stage09_precision_after"].mean().reset_index()
        ax.plot(g["threshold_db"], g["stage09_precision_after"], marker="s", linestyle=":",
                label="Stage 09 hand physics")
    ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("precision after filtering")
    ax.set_title("Precision after filtering (four-day)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return _save(fig, path), DATA_DRIVEN, "stage-12.5 four-day pooled; stage-09 four-day mean"


def build_score_distribution_plot(src: Sources, path: str) -> Tuple[str, str, str]:
    scores_path = src.paths.get("s12_scores", "")
    if not os.path.exists(scores_path):
        return (_placeholder(path, "Stage 12.5 score distributions",
                             "per-track sequence_track_scores.csv is large and git-ignored; "
                             "regenerate stage 12 scoring locally to build this figure"),
                PLACEHOLDER, "per-track score CSV unavailable")
    try:
        df = pd.read_csv(scores_path, usecols=["model", "sequence_prior_score", "is_true_track"])
        df = df[df["model"] == "mlp_dae"]
    except Exception as exc:
        return (_placeholder(path, "Stage 12.5 score distributions", f"read error: {exc}"),
                PLACEHOLDER, f"read error: {exc}")
    df = df[np.isfinite(df["sequence_prior_score"])]
    if df.empty:
        return (_placeholder(path, "Stage 12.5 score distributions", "no finite scores"),
                PLACEHOLDER, "no finite scores")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(df.loc[df["is_true_track"], "sequence_prior_score"], bins=bins, density=True,
            histtype="step", lw=2, label="true tracks")
    ax.hist(df.loc[~df["is_true_track"], "sequence_prior_score"], bins=bins, density=True,
            histtype="step", lw=2, linestyle="--", label="false tracks")
    ax.axvline(0.5, lw=0.8, linestyle=":", label="keep threshold 0.5")
    ax.set_xlabel("sequence-prior score (mlp_dae, track-calibrated)")
    ax.set_ylabel("density")
    ax.set_title("Stage 12.5 score distributions (2022-06-06)")
    ax.legend(fontsize=8); ax.grid(True, lw=0.4)
    return (_save(fig, path), DATA_DRIVEN,
            f"{len(df):,} scored tracks from the (git-ignored) per-track score CSV")


def build_reproducibility_checklist_plot(checklist: pd.DataFrame, path: str) -> Tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axis("off")
    ax.set_title("Reproducibility checklist (Stage 17.5 hardened)", fontsize=13)
    y = 0.93
    for _, r in checklist.iterrows():
        mark = "[x]" if str(r["status"]).lower() == "yes" else "[ ]"
        ax.text(0.02, y, mark, fontsize=11, family="monospace")
        ax.text(0.08, y, str(r["item"]), fontsize=9, va="center")
        y -= 0.098
    ax.text(0.02, 0.02, "Checked by scripts/17p5_regression_checks.py (12 assertions) and the "
            "per-stage validation gates.", fontsize=7, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return _save(fig, path), DATA_DRIVEN, "rendered from final_reproducibility_checklist.csv"


# =============================================================================
# Markdown final report
# =============================================================================

def write_final_report(out_dir: str, title: str, hm: Dict, tables: Dict[str, pd.DataFrame],
                       figures: List[Dict], src: Sources, notes: List[str]) -> str:
    def fig_line(fn, alt):
        return f"![{alt}](figures/{fn})"

    tt, ft = hm["true_tracks"], hm["false_tracks"]
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Across all four ADS-B days at detection thresholds -5/0/3/6 dB, the Stage 08 Kalman "
        f"baseline produced **{tt:,} true** and **{ft:,} false** confirmed tracks. The selected "
        f"method — a **track-calibrated deterministic sequence autoencoder** (Stage 12.5, "
        f"`mlp_dae`) — retained **{hm['mlp_true_retention']*100:.1f}%** of true tracks while "
        f"removing **{hm['mlp_false_reduction']*100:.2f}%** of false tracks, keeping only "
        f"**{hm['mlp_false_kept']} of {ft:,}**.",
        "",
        "> **Final claim.** A track-calibrated deterministic sequence autoencoder trained on "
        "ADS-B-derived aircraft trajectory windows suppresses low-threshold clutter-induced "
        "false tracks across four days while retaining true aircraft tracks.",
        "",
        f"`gru_ae` is a close second ({hm['gru_true_retention']*100:.1f}% retention, "
        f"{hm['gru_false_reduction']*100:.2f}% reduction, {hm['gru_false_kept']} false tracks "
        "kept) and is the stronger choice at the lowest threshold. Stage 09 hand-designed "
        "physics scoring remains the recommended interpretable fallback.",
        "",
        "## Problem",
        "",
        "A radar detector must pick a detection threshold, and neither end of that dial is safe:",
        "",
        "- a **high threshold** suppresses clutter but loses weak targets;",
        "- a **low threshold** recovers weak targets but floods the tracker with clutter, which "
        "  associates into large numbers of false tracks.",
        "",
        "Per-frame reasoning cannot resolve this, because a clutter point and a weak target look "
        "alike in a single frame. **Trajectory-level reasoning can**: real aircraft motion has a "
        "shape over time that clutter chains do not reproduce. This project asks how far that "
        "idea can be pushed, and what it costs.",
        "",
        "## What the project does",
        "",
        "```text",
        "Real OpenSky ADS-B trajectories",
        "  -> cleaned fixed-wing GA trajectories        (F01, stages 01-04)",
        "  -> relocated radar-coordinate truth          (F02, stage 05)",
        "  -> noisy / cluttered radar point detections  (F02, stage 06)",
        "  -> threshold / Kalman / physics / learned-prior evaluation  (F03, stages 07-17.5)",
        "```",
        "",
        "**Scope, stated plainly:**",
        "",
        "- There is **no raw RF/IQ** anywhere in this project.",
        "- There is **no true range-Doppler intensity simulation**.",
        "- The radar model is a **point-detection simulation**: per-scan detections with "
        "  range/azimuth/elevation/radial-velocity measurement error, an SNR-dependent "
        "  probability of detection, and Poisson clutter.",
        f"- Consequently, the \"pseudo range-Doppler\" figures below are **scatter plots of point "
        f"detections** in (radial velocity, range) space. {PSEUDO_RD_CAPTION}",
        "",
        "## Data and simulation pipeline",
        "",
        "**F01-Preprocessing** turns raw OpenSky ADS-B into uniform 10 s fixed-wing GA "
        "trajectories: aircraft-type whitelisting (01), filtering (02), cleaning (03) and "
        "resampling onto a common time grid (04).",
        "",
        "**F02-Radar** converts those trajectories into radar observables. Stage 05 builds WGS84 "
        "radar truth and can **relocate** trajectories so their start anchors fall in a 10-80 km "
        "annulus around a synthetic radar — preserving the real ADS-B motion shape while placing "
        "the engagement geometry under experimental control (per-trajectory sha256-seeded "
        "anchors, so the relocation is deterministic). Stage 06 simulates point detections: "
        "R^-4 SNR decay, a logistic probability of detection, per-component measurement noise, "
        "and Poisson clutter.",
        "",
        fig_line("01_pipeline_diagram.png", "Pipeline diagram"),
        "",
        fig_line("02_radar_simulation_concept.png", "Radar simulation concept"),
        "",
        fig_line("03_pseudo_range_doppler_frame_low_threshold.png",
                 "Pseudo range-Doppler point-detection frame, low threshold"),
        "",
        f"*{PSEUDO_RD_CAPTION}*",
        "",
        "## Evaluation ladder",
        "",
        "| stage | method | role |",
        "|---|---|---|",
        "| 07 | threshold-only detection | frame-level tradeoff, no tracking |",
        "| 08 | constant-velocity Kalman + greedy gated NN | track-level baseline / denominator |",
        "| 09 | hand-designed physics scoring | interpretable fallback |",
        "| 10 | empirical ADS-B motion priors | prior construction (no scoring) |",
        "| 11 | empirical marginal-prior scoring | evidence, not sufficient alone |",
        "| 12 / 12.5 | sequence autoencoders + noise-matched calibration | **selected method** |",
        "| 13 | VAE trajectory prior | does not beat 12.5 |",
        "| 14 | unified benchmark + operating-point selection | method choice |",
        "| 15 | diffusion denoiser | regularization / gap filling, not filtering |",
        "| 16 | robustness + ablations | stability of the choice |",
        "| 17 / 17.5 | four-day validation + reproducibility hardening | generalization |",
        "",
        fig_line("07_threshold_tradeoff_stage07.png", "Threshold tradeoff"),
        "",
        fig_line("08_kalman_tracking_effect_stage08.png", "Tracking effect"),
        "",
        "Frame-level $P_d$ and track-level detection rate measure different things; the ladder "
        "keeps them separate throughout.",
        "",
        "## Final selected method",
        "",
        "**Stage 12.5 — a deterministic sequence autoencoder with noise-matched calibration.**",
        "",
        "1. Trajectory windows (length 20, stride 5) are origin-shifted and heading-rotated, so "
        "   the model learns motion *shape*, not absolute position or bearing.",
        "2. An MLP denoising autoencoder is trained on **clean** stage-05 truth windows.",
        "3. A track is scored by the **median per-window reconstruction error** of its Kalman "
        "   posterior.",
        "4. Crucially, the error-to-score mapping is calibrated on **high-purity noisy Stage 08 "
        "   tracks**, not on clean truth — this is the **noise-matched calibration** step.",
        "",
        "Step 4 is what makes the method usable. Calibrated against clean truth, the score "
        "separates true from false tracks superbly but is *miscalibrated*: genuine tracks are "
        "Kalman posteriors over noisy measurements, so they reconstruct worse than clean truth "
        "and collapse toward score 0. Retention at threshold 0.5 was ~0.08. Re-anchoring the "
        "p50/p99 band on noisy high-purity tracks lifted retention to ~0.96 with essentially no "
        "loss of false-track suppression.",
        "",
        fig_line("10_stage12_calibration_effect.png", "Calibration effect"),
        "",
        "The score itself never sees truth labels. Labels are used only to *select* the "
        "calibration set and to evaluate — a limitation recorded explicitly below.",
        "",
        "## Results",
        "",
        "### Headline claims",
        "",
    ]
    lines += md_table(tables["final_results_summary"])
    lines += ["", "### Method comparison", ""]
    lines += md_table(tables["final_method_comparison"].round(4))
    lines += ["",
              "Scope is labelled per row: Stage 12.5 numbers are **four-day**; methods evaluated "
              "on a single day are marked `ONE-DAY`. Four-day values are never replaced by stale "
              "single-day values.",
              "",
              "> **Which false tracks?** Two false-track populations appear in this project and "
              "must not be conflated. Stage 08 labels *every* confirmed track by its own criterion "
              "(purity >= 0.5), giving 34,773 false tracks over four days and four thresholds. The "
              "filters are evaluated on the stricter, **windowable** subset (purity >= 0.8 and long "
              f"enough to form a window): **{ft:,} false tracks**. Every retention/reduction number "
              "in this package — including the headline — uses that second denominator, and the "
              "Kalman-only row is the no-filter baseline on the same population.",
              "",
              fig_line("11_method_ladder_comparison.png", "Method ladder"),
              "",
              fig_line("12_four_day_validation_retention.png", "Four-day retention"),
              "",
              fig_line("13_four_day_validation_false_reduction.png", "Four-day false reduction"),
              "",
              fig_line("14_false_tracks_before_after_stage12.png", "False tracks before/after"),
              "",
              f"The {ft:,} -> {hm['mlp_false_kept']} reduction is the central empirical result.",
              ""]

    lines += ["## Ablations", ""]
    lines += md_table(tables["final_ablation_summary"])
    lines += ["",
              fig_line("15_mlp_vs_gru_four_day.png", "MLP vs GRU"),
              "",
              fig_line("16_stage09_vs_stage12_interpretable_comparison.png",
                       "Stage 09 vs Stage 12.5"),
              "",
              fig_line("17_windowability_caveat.png", "Windowability caveat"),
              "",
              "**Windowability caveat.** Sequence methods only score tracks long enough to window "
              "(>= window_len points, >= 5 hits). At 9/12 dB almost no false track is that long, "
              "so the false-reduction denominator is zero and the metric is **undefined** — not "
              "0, not 1. Those thresholds are kept audit-only, and undefined cells are excluded "
              "from aggregates and counted separately.",
              ""]

    lines += ["## Reproducibility", ""]
    lines += md_table(tables["final_reproducibility_checklist"])
    lines += ["",
              fig_line("25_reproducibility_pipeline_checklist.png", "Reproducibility checklist"),
              "",
              "Stage 17.5 hardened the pipeline after defects surfaced during the four-day run: "
              "internal subprocesses now use `sys.executable`; the canonical Stage 12 calibration "
              "can no longer be overwritten by a per-day rerun; zero-denominator false-reduction "
              "cells are explicitly undefined rather than mislabelled; and 12 regression checks "
              "(including a negative control) guard all of it. The scientific results were "
              "unchanged by that pass — only labels, paths and diagnostic columns.",
              ""]

    lines += ["## Limitations", ""]
    lines += md_table(tables["final_limitations"])
    lines += [""]

    lines += ["## Novelty and contribution", "",
              "The novelty is **not** in Kalman filtering, autoencoders, or radar thresholding "
              "individually. The contribution is an ADS-B-grounded weak-target radar evaluation "
              "framework and a noise-matched learned trajectory-shape scoring method that "
              "suppresses low-threshold clutter-induced false tracks while retaining true "
              "aircraft tracks.",
              ""]
    lines += md_table(tables["final_contributions"])
    lines += [""]

    lines += ["## Final figure guide", "",
              "Every figure, its source stage, and whether it is data-driven, schematic, copied "
              "from an earlier stage, or a placeholder:",
              "",
              "| # | filename | shows | source stage | type | notes |",
              "|---|---|---|---|---|---|"]
    for f in figures:
        lines.append(f"| {f['index']:02d} | `{f['filename']}` | {f['shows']} | "
                     f"{f['source_stage']} | {f['figure_type']} | {f['notes']} |")
    lines += [""]

    lines += ["## Recommended next work", "",
              "1. **Deployment-style runtime / operating-point study** — throughput, latency, and "
              "   the score threshold an operator would actually choose.",
              "2. **Raw radar / range-Doppler simulation** — replace the point-detection model "
              "   with a gridded intensity simulation, which would let the detector itself be "
              "   evaluated (and make true RD heatmaps meaningful).",
              "3. **Broader model zoo only if a specific gap is identified** — Stage 14 showed "
              "   most of the comparison space is already resolved.",
              "4. **Real radar data**, with co-located ADS-B truth, to test transfer.",
              "",
              "## Appendix: Commands", "",
              "```bash",
              "# selected-method scoring (stage 12.5)",
              "python scripts/12_score_tracks_sequence_prior.py \\",
              "  --calibration-mode track_purity --calibration-threshold-db 3 6 9 12 \\",
              "  --threshold-db -5 0 3 6 --date 2022-06-06 --overwrite",
              "",
              "# unified benchmark and operating-point selection",
              "python scripts/14_benchmark_methods.py --overwrite",
              "",
              "# four-day validation (consolidation; add --run-missing-* to generate days)",
              "python scripts/17_four_day_validation.py \\",
              "  --date 2022-06-06 2022-06-13 2022-06-20 2022-06-27 \\",
              "  --threshold-db -5 0 3 6 --overwrite",
              "",
              "# reproducibility guards",
              "python scripts/17p5_regression_checks.py",
              "",
              "# this package",
              "python scripts/18_build_final_report.py --overwrite",
              "```",
              ""]

    if src.missing:
        lines += ["## Missing source artifacts", "",
                  "The following expected inputs were not found; affected figures are marked "
                  "`placeholder` in the guide above and in the manifest:", ""]
        for k in src.missing:
            lines.append(f"- `{k}` -> `{src.paths[k]}`")
        lines += [""]
    if notes:
        lines += ["## Build notes", ""] + [f"- {n}" for n in notes] + [""]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Overleaf package
# =============================================================================

OVERLEAF_FIGURES = [
    ("01_pipeline_diagram.png", "Three-repo pipeline from real ADS-B trajectories to "
     "weak-target radar evaluation.", "fig:pipeline"),
    ("02_radar_simulation_concept.png", "Radar simulation concept: relocated ADS-B "
     "trajectories anchored in a 10--80\\,km annulus around the radar.", "fig:concept"),
    ("03_pseudo_range_doppler_frame_low_threshold.png",
     "Pseudo range-Doppler point-detection frame at the low threshold ($-5$\\,dB). "
     "This is a point-detection pseudo range-Doppler visualization, not a raw radar "
     "intensity map.", "fig:rdlow"),
    ("07_threshold_tradeoff_stage07.png", "The detection-threshold tradeoff: lowering the "
     "threshold recovers weak targets but multiplies clutter.", "fig:tradeoff"),
    ("08_kalman_tracking_effect_stage08.png", "Temporal tracking recovers trajectories even "
     "as frame-level $P_d$ falls.", "fig:tracking"),
    ("10_stage12_calibration_effect.png", "Noise-matched (track-purity) calibration is what "
     "makes the learned score usable.", "fig:calib"),
    ("11_method_ladder_comparison.png", "Method ladder: true-track retention against "
     "false-track reduction.", "fig:ladder"),
    ("12_four_day_validation_retention.png", "True-track retention across all four days.",
     "fig:fourdayret"),
    ("14_false_tracks_before_after_stage12.png", "False tracks before and after Stage~12.5.",
     "fig:beforeafter"),
    ("17_windowability_caveat.png", "Windowability caveat: at high thresholds almost no false "
     "track is long enough to window, so false-track reduction is undefined.", "fig:window"),
]


def write_overleaf_package(out_dir: str, title: str, author: str, institution: str,
                           hm: Dict, tables: Dict[str, pd.DataFrame],
                           figures_dir: str) -> List[str]:
    ol = os.path.join(out_dir, "overleaf")
    olf = os.path.join(ol, "figures")
    os.makedirs(olf, exist_ok=True)

    # copy every generated figure so any \includegraphics resolves
    copied = []
    if os.path.isdir(figures_dir):
        for fn in sorted(os.listdir(figures_dir)):
            if fn.endswith(".png"):
                shutil.copyfile(os.path.join(figures_dir, fn), os.path.join(olf, fn))
                copied.append(fn)

    tt, ft = hm["true_tracks"], hm["false_tracks"]
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

    def figure_block(fn, caption, label):
        return "\n".join([
            r"\begin{figure}[htbp]", r"  \centering",
            rf"  \includegraphics[width=0.92\linewidth]{{figures/{fn}}}",
            rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
            r"\end{figure}", ""])

    def latex_table(df, caption, label, cols=None, maxrows=12):
        d = df if cols is None else df[cols]
        d = d.head(maxrows)
        spec = "l" * len(d.columns)
        head = " & ".join(esc(c) for c in d.columns) + r" \\"
        body = "\n".join(" & ".join(esc(v) for v in row) + r" \\"
                         for row in d.astype(str).values)
        return "\n".join([
            r"\begin{table}[htbp]", r"  \centering", r"  \small",
            rf"  \begin{{tabular}}{{{spec}}}", r"    \hline", "    " + head, r"    \hline",
            "    " + body.replace("\n", "\n    "), r"    \hline", r"  \end{tabular}",
            rf"  \caption{{{caption}}}", rf"  \label{{{label}}}", r"\end{table}", ""])

    body = []
    body.append(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{siunitx}
""")
    body.append(rf"\title{{{esc(title)}}}")
    body.append(rf"\author{{{esc(author)} \\ {esc(institution)}}}")
    body.append(r"\date{\today}")
    body.append(r"\begin{document}")
    body.append(r"\maketitle")

    body.append(r"\begin{abstract}")
    body.append(
        f"Radar detection forces a threshold tradeoff: a high threshold suppresses clutter but "
        f"loses weak targets, while a low threshold recovers weak targets and floods the tracker "
        f"with clutter-induced false tracks. We build an ADS-B-grounded evaluation framework in "
        f"which real fixed-wing general-aviation trajectories from OpenSky are cleaned, relocated "
        f"into a synthetic radar's coordinate frame, and converted into noisy, cluttered radar "
        f"\\emph{{point detections}}. On this framework we compare a ladder of methods: "
        f"threshold-only detection, a constant-velocity Kalman tracker, hand-designed physics "
        f"scoring, empirical ADS-B motion priors, learned sequence autoencoders, a variational "
        f"autoencoder, and a diffusion denoiser. Our selected method is a deterministic sequence "
        f"autoencoder scored by reconstruction error over origin- and heading-normalized "
        f"trajectory windows, with an error-to-score mapping calibrated on high-purity "
        f"\\emph{{noisy}} tracks rather than clean truth. This noise-matched calibration is "
        f"essential: without it, true-track retention collapses. Over four days and thresholds "
        f"$-5$ to $6$\\,dB, the Kalman baseline yields \\num{{{tt}}} true and \\num{{{ft}}} false "
        f"tracks; our method retains \\SI{{{hm['mlp_true_retention']*100:.1f}}}{{\\percent}} of "
        f"true tracks while removing "
        f"\\SI{{{hm['mlp_false_reduction']*100:.2f}}}{{\\percent}} of false tracks, keeping only "
        f"{hm['mlp_false_kept']} of \\num{{{ft}}}. We emphasise scope: this is a point-detection "
        f"simulation, not a raw RF or range-Doppler intensity simulation.")
    body.append(r"\end{abstract}")

    body.append(r"\section{Introduction}")
    body.append(
        "A radar detector's threshold sets an unavoidable operating point. Raising it removes "
        "clutter at the cost of weak targets; lowering it recovers weak targets but presents the "
        "tracker with many clutter detections, which associate into false tracks. Single-frame "
        "reasoning cannot separate a weak target from a clutter point. Trajectory-level reasoning "
        "can, because aircraft motion has temporal structure that clutter chains do not "
        "reproduce. This paper asks how much of the false-track burden trajectory-shape reasoning "
        "can remove, how it must be calibrated, and whether the result survives across days.")
    body.append(figure_block(*OVERLEAF_FIGURES[0]))

    body.append(r"\section{Related work}")
    body.append(
        "Recursive state estimation for tracking dates to the Kalman filter~\\cite{kalman1960}. "
        "Track-before-detect approaches integrate energy across frames before thresholding to "
        "recover weak targets~\\cite{tbd}. Our detections derive from the OpenSky "
        "network~\\cite{opensky}, a widely used source of real ADS-B trajectories, and "
        "trajectory-prediction work has established that ADS-B motion is highly "
        "structured~\\cite{adsbpred}. Our scoring uses reconstruction error from an autoencoder "
        "as an anomaly signal~\\cite{aeanomaly}, and we additionally evaluate a variational "
        "autoencoder~\\cite{vae} and a denoising diffusion model~\\cite{ddpm}. Our contribution "
        "is not any one of these components but their integration into an ADS-B-grounded "
        "weak-target evaluation framework, together with the noise-matched calibration that "
        "makes the learned score usable on noisy tracks.")

    body.append(r"\section{Data and simulation pipeline}")
    body.append(
        "Real OpenSky ADS-B tracks are whitelisted to fixed-wing general aviation, filtered, "
        "cleaned, and resampled onto a uniform \\SI{10}{\\second} grid. Trajectories are then "
        "\\emph{relocated}: each trajectory's start anchor is placed in a \\SIrange{10}{80}{\\km} "
        "annulus around a synthetic radar using a per-trajectory SHA-256-derived seed, preserving "
        "the real motion shape while bringing the engagement geometry under experimental control.")
    body.append(figure_block(*OVERLEAF_FIGURES[1]))

    body.append(r"\section{Radar point-detection simulation}")
    body.append(
        "The radar model produces \\emph{point detections} per scan: an $R^{-4}$ SNR decay, a "
        "logistic probability of detection, per-component measurement noise on range, azimuth, "
        "elevation and radial velocity, and Poisson clutter. \\textbf{There is no raw RF/IQ and "
        "no gridded range-Doppler intensity simulation.} Accordingly, the range-Doppler figure "
        "below is a scatter plot of point detections, not a radar intensity map.")
    body.append(figure_block(*OVERLEAF_FIGURES[2]))
    body.append(figure_block(*OVERLEAF_FIGURES[3]))

    body.append(r"\section{Method ladder}")
    body.append(
        "We evaluate, in order: threshold-only detection (frame level); a constant-velocity "
        "Kalman tracker with greedy gated nearest-neighbour association, which defines the "
        "true/false-track denominator; hand-designed physics scoring; empirical marginal ADS-B "
        "motion priors; learned sequence autoencoders; a VAE trajectory prior; and a diffusion "
        "denoiser. The tracker never sees truth labels; labels enter only in evaluation.")
    body.append(figure_block(*OVERLEAF_FIGURES[4]))
    body.append(figure_block(*OVERLEAF_FIGURES[6]))

    body.append(r"\section{Sequence-prior calibration}")
    body.append(
        "Trajectory windows are origin-shifted and heading-rotated so the model learns motion "
        "\\emph{shape}. A denoising autoencoder is trained on clean truth windows, and a track is "
        "scored by the median per-window reconstruction error of its Kalman posterior. Mapping "
        "that error to a score using \\emph{clean-truth} quantiles is a domain shift: genuine "
        "tracks are posteriors over noisy measurements and reconstruct worse than clean truth, so "
        "they collapse toward score zero and true-track retention falls to roughly $0.08$. "
        "Re-anchoring the $p_{50}/p_{99}$ band on high-purity \\emph{noisy} tracks restores "
        "retention to roughly $0.96$ with essentially no loss of false-track suppression. We call "
        "this \\emph{noise-matched calibration}; it is the decisive step.")
    body.append(figure_block(*OVERLEAF_FIGURES[5]))

    body.append(r"\section{Results}")
    body.append(
        f"Across four days at $-5/0/3/6$\\,dB the Kalman baseline yields \\num{{{tt}}} true and "
        f"\\num{{{ft}}} false confirmed tracks. The selected method retains "
        f"\\SI{{{hm['mlp_true_retention']*100:.1f}}}{{\\percent}} of true tracks and removes "
        f"\\SI{{{hm['mlp_false_reduction']*100:.2f}}}{{\\percent}} of false tracks, keeping "
        f"{hm['mlp_false_kept']} of \\num{{{ft}}}. Results are stable across days "
        f"(Fig.~\\ref{{fig:fourdayret}}, Fig.~\\ref{{fig:beforeafter}}).")
    body.append(figure_block(*OVERLEAF_FIGURES[7]))
    body.append(figure_block(*OVERLEAF_FIGURES[8]))
    mc = tables["final_method_comparison"][["method", "stage", "true_retention",
                                            "false_reduction", "scope"]].copy()
    for c in ("true_retention", "false_reduction"):
        mc[c] = mc[c].map(lambda v: "--" if not np.isfinite(v) else f"{v:.4f}")
    body.append(latex_table(mc, "Method comparison. Stage~12.5 rows are four-day; rows marked "
                                "ONE-DAY were evaluated on a single day.", "tab:methods"))

    body.append(r"\section{Ablations}")
    body.append(
        "Calibration dominates: clean-truth versus track-purity calibration is the difference "
        "between an unusable and a usable filter. Among model families, the MLP autoencoder is "
        "best overall and the GRU is a close second, while the TCN retains fewer true tracks. The "
        "VAE does not beat the deterministic autoencoder, and the diffusion residual is clearly "
        "worse as a classifier, although diffusion does regularize tracks and modestly improves "
        "gap filling. Hand-designed physics scoring remains a strong interpretable fallback.")
    body.append(
        "A denominator caveat applies at high thresholds: sequence methods only score tracks long "
        "enough to window, and at $9$/$12$\\,dB essentially no false track is that long, so "
        "false-track reduction is \\emph{undefined} rather than zero or one "
        "(Fig.~\\ref{fig:window}).")
    body.append(figure_block(*OVERLEAF_FIGURES[9]))

    body.append(r"\section{Limitations}")
    lim = tables["final_limitations"]["limitation"].tolist()
    body.append(r"\begin{itemize}")
    for l in lim:
        body.append(rf"  \item {esc(l)}")
    body.append(r"\end{itemize}")

    body.append(r"\section{Conclusion}")
    body.append(
        "The novelty is not in Kalman filtering, autoencoders, or radar thresholding "
        "individually. The contribution is an ADS-B-grounded weak-target radar evaluation "
        "framework and a noise-matched learned trajectory-shape scoring method that suppresses "
        "low-threshold clutter-induced false tracks while retaining true aircraft tracks, "
        "demonstrated across four days.")

    body.append(r"\bibliographystyle{plain}")
    body.append(r"\bibliography{references}")
    body.append(r"\end{document}")

    tex_path = os.path.join(ol, "main.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(body))

    bib = r"""@article{kalman1960,
  author  = {Kalman, Rudolf E.},
  title   = {A New Approach to Linear Filtering and Prediction Problems},
  journal = {Journal of Basic Engineering},
  volume  = {82},
  number  = {1},
  pages   = {35--45},
  year    = {1960}
}

@article{opensky,
  author  = {Sch{\"a}fer, Matthias and Strohmeier, Martin and Lenders, Vincent and
             Martinovic, Ivan and Wilhelm, Matthias},
  title   = {Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research},
  journal = {Proceedings of the 13th IEEE/ACM International Symposium on Information
             Processing in Sensor Networks (IPSN)},
  pages   = {83--94},
  year    = {2014},
  note    = {The OpenSky Network, \url{https://opensky-network.org}}
}

@misc{tbd,
  title = {Track-before-detect processing for weak-target radar tracking},
  note  = {Generic reference for track-before-detect approaches, in which energy is
           integrated across frames prior to thresholding so that weak targets are not
           lost by a per-frame detection threshold. Replace with the specific
           track-before-detect reference used in your literature review.},
  year  = {2024}
}

@misc{adsbpred,
  title = {ADS-B-based aircraft trajectory prediction},
  note  = {Generic reference for trajectory prediction from ADS-B data, establishing that
           ADS-B-derived aircraft motion is highly structured and predictable. Replace
           with the specific ADS-B trajectory-prediction reference used in your review.},
  year  = {2024}
}

@misc{aeanomaly,
  title = {Autoencoder reconstruction error for anomaly detection},
  note  = {Generic reference for using autoencoder reconstruction error as an anomaly
           score. Replace with the specific autoencoder anomaly-detection reference used
           in your review.},
  year  = {2024}
}

@misc{vae,
  author = {Kingma, Diederik P. and Welling, Max},
  title  = {Auto-Encoding Variational Bayes},
  year   = {2013},
  note   = {arXiv:1312.6114}
}

@misc{ddpm,
  author = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  title  = {Denoising Diffusion Probabilistic Models},
  year   = {2020},
  note   = {arXiv:2006.11239}
}
"""
    bib_path = os.path.join(ol, "references.bib")
    with open(bib_path, "w") as f:
        f.write(bib)

    readme = f"""# Overleaf package

Upload this whole `overleaf/` directory to Overleaf (or compile locally).

```
main.tex          complete LaTeX article
references.bib    bibliography (see the caveat below)
figures/          {len(copied)} PNG figures, all referenced paths resolve
```

Compile with `pdflatex -> bibtex -> pdflatex -> pdflatex`, or just press Recompile
in Overleaf (it runs bibtex automatically).

## Figure scope caveat

`figures/03_pseudo_range_doppler_frame_low_threshold.png` and its high-threshold
counterpart are **point-detection pseudo range-Doppler visualizations, not raw radar
intensity maps.** The simulation produces per-scan point detections, not gridded
range-Doppler intensity. The caption in `main.tex` states this; please keep it.

## Bibliography caveat

`references.bib` contains **exact** entries for Kalman (1960), the OpenSky network,
Kingma \\& Welling (VAE), and Ho et al. (DDPM). Three entries -- `tbd`, `adsbpred`
and `aeanomaly` -- are deliberate **placeholder `@misc` entries** with descriptive
titles and notes, because the specific references were not fixed in this project's
documentation. Replace them with the concrete citations from your literature review
before submission. No citation here was invented to look authoritative.

## Author fields

`main.tex` currently reads:

```
\\author{{{esc(author)} \\\\ {esc(institution)}}}
```

Set `--author` and `--institution` when running `scripts/18_build_final_report.py`,
or edit `main.tex` directly.
"""
    rm_path = os.path.join(ol, "README_OVERLEAF.md")
    with open(rm_path, "w") as f:
        f.write(readme)
    return [tex_path, bib_path, rm_path] + [os.path.join(olf, c) for c in copied]


# =============================================================================
# Manifest + validation gate
# =============================================================================

def write_manifest(out_dir: str, src: Sources, hm: Dict, figures: List[Dict],
                   outputs: List[str], notes: List[str]) -> str:
    manifest = {
        "created_by": "Stage 18",
        "source_reports": sorted({os.path.dirname(p) for p in src.paths.values()}),
        "missing_source_artifacts": [{"key": k, "path": src.paths[k]} for k in src.missing],
        "outputs": sorted(os.path.relpath(o, out_dir) for o in outputs),
        "headline_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in hm.items()},
        "selected_method": hm.get("selected_method"),
        "scope_note": ("Point-detection radar simulation. No raw RF/IQ and no gridded "
                       "range-Doppler intensity simulation. " + PSEUDO_RD_CAPTION),
        "figures": figures,
        "notes": notes,
    }
    path = os.path.join(out_dir, "stage18_final_package_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def run_gate(out_dir: str, figures: List[Dict], tables: Dict[str, pd.DataFrame]) -> None:
    def fail(msg):
        raise ValueError(f"Stage 18 validation failed: {msg}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (final package)")
    print("=" * 70)

    rep = os.path.join(out_dir, "final_report.md")
    tex = os.path.join(out_dir, "overleaf", "main.tex")
    bib = os.path.join(out_dir, "overleaf", "references.bib")
    for p, name in ((rep, "final_report.md"), (tex, "overleaf/main.tex"),
                    (bib, "overleaf/references.bib")):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            fail(f"{name} missing or empty")
    print("  final_report.md, overleaf/main.tex, references.bib present and nonempty: OK")

    for name, df in tables.items():
        p = os.path.join(out_dir, f"{name}.csv")
        if not os.path.exists(p) or df is None or df.empty:
            fail(f"{name}.csv missing or empty")
    print(f"  {len(tables)} final summary CSVs present and nonempty: OK")

    figs_dir = os.path.join(out_dir, "figures")
    n_figs = len([f for f in os.listdir(figs_dir)]) if os.path.isdir(figs_dir) else 0
    if n_figs < 20:
        fail(f"expected >= 20 figures, found {n_figs}")
    print(f"  {n_figs} figure files present (>= 20): OK")

    rd = [f for f in figures if "pseudo_range_doppler" in f["filename"]]
    if not rd:
        fail("no pseudo range-Doppler figure entry in the manifest")
    for f in rd:
        if f["figure_type"] == PLACEHOLDER and not f["notes"]:
            fail("pseudo range-Doppler placeholder must record a reason")
    kinds = {f["figure_type"] for f in rd}
    print(f"  pseudo range-Doppler figures present ({', '.join(sorted(kinds))}) "
          "with recorded provenance: OK")

    tex_src = open(tex).read()
    refd = [fn for fn, _, _ in OVERLEAF_FIGURES if f"figures/{fn}" in tex_src]
    if len(refd) < 10:
        fail(f"overleaf/main.tex references only {len(refd)} figures (need >= 10)")
    olf = os.path.join(out_dir, "overleaf", "figures")
    missing = [fn for fn in refd if not os.path.exists(os.path.join(olf, fn))]
    if missing:
        fail(f"overleaf/figures/ missing referenced PNGs: {missing}")
    print(f"  overleaf/main.tex references {len(refd)} figures, all present in "
          "overleaf/figures/: OK")

    report = open(rep).read()
    required = ["323,808", "815", "0.973", "0.9939", "Stage 12.5",
                "noise-matched calibration", "Stage 17.5", "Final figure guide"]
    missing = [r for r in required if r not in report]
    if missing:
        fail(f"final_report.md missing required strings: {missing}")
    print("  final_report.md contains all required headline strings: OK")

    for needle in [r"\begin{abstract}", r"\section{Results}", r"\section{Limitations}",
                   r"\bibliography"]:
        if needle not in tex_src:
            fail(f"overleaf/main.tex missing {needle!r}")
    print("  overleaf/main.tex has abstract, results, limitations, bibliography: OK")

    man_path = os.path.join(out_dir, "stage18_final_package_manifest.json")
    man = json.load(open(man_path))
    if len(man.get("figures", [])) != len(figures):
        fail("manifest does not list all figures")
    print(f"  manifest lists all {len(figures)} figures: OK")
