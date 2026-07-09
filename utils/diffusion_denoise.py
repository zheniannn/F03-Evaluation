"""Stage 15: diffusion denoising / gap-filling evaluation + residual scoring.

Three tasks over normalized trajectory windows built from stage-8 tracks:

  1. denoising true tracks   -- can single-step diffusion recover a corrupted
     true-track window, and does it regularize (smooth) the Kalman posterior?
  2. gap filling             -- mask timesteps, linearly interpolate, then let
     diffusion refine; compare to interpolation alone.
  3. residual/anomaly score  -- median denoising residual as a SECONDARY
     plausibility score, calibrated in the noisy-track domain (stage-12.5
     track-purity), compared against stage 12.5 / 13.

Truth labels are used only for evaluation and for track-purity calibration,
never for the denoising operation itself.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table
from utils.diffusion_models import POS_IDX, VEL_IDX, denoise_windows, forward_corrupt
from utils.vae_prior_score import score_from_band, calibration_quantiles  # noqa: F401 (reuse)


# =============================================================================
# Small numeric helpers
# =============================================================================

def _rmse(a: np.ndarray, b: np.ndarray, idx: List[int], mask: Optional[np.ndarray] = None) -> float:
    d = (a[:, :, idx] - b[:, :, idx]) ** 2
    if mask is not None:
        m = mask[:, :, None]
        tot = (d * m).sum()
        cnt = m.sum() * len(idx)
        return float(np.sqrt(tot / cnt)) if cnt else np.nan
    return float(np.sqrt(d.mean()))


def _smoothness(w: np.ndarray) -> np.ndarray:
    """Per-window position 2nd-difference energy (lower = smoother)."""
    p = w[:, :, POS_IDX]
    d2 = p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]
    return (d2 ** 2).mean(axis=(1, 2))


def sample_rows(X: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(X) <= cap:
        return X
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(X), size=cap, replace=False))
    return X[keep]


# =============================================================================
# Task 1: denoising true tracks
# =============================================================================

def denoising_metrics_for_run(model, sched, X: np.ndarray, denoise_t: int,
                              batch_size: int, device: str, seed: int) -> Dict:
    """Synthetic-corruption recovery (RMSE, normalized space) + direct smoothness
    regularization of the actual Kalman windows."""
    import torch
    if not len(X):
        return {}
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(Xt.shape, generator=gen)
    Xn, _ = forward_corrupt(Xt, denoise_t, sched.to("cpu"), eps=eps)
    Xn = Xn.numpy()

    X0 = denoise_windows(model, Xn, denoise_t, sched, batch_size, device)   # recover corruption
    X0_direct = denoise_windows(model, X, denoise_t, sched, batch_size, device)  # regularize Kalman

    kal_pos = _rmse(Xn, X, POS_IDX)
    dif_pos = _rmse(X0, X, POS_IDX)
    kal_vel = _rmse(Xn, X, VEL_IDX)
    dif_vel = _rmse(X0, X, VEL_IDX)
    sm_before = float(_smoothness(X).mean())
    sm_after = float(_smoothness(X0_direct).mean())
    return {
        "n_windows": len(X),
        "kalman_position_rmse_norm": kal_pos, "diffusion_position_rmse_norm": dif_pos,
        "rmse_improvement_ratio": (1 - dif_pos / kal_pos) if kal_pos else np.nan,
        "kalman_velocity_rmse_norm": kal_vel, "diffusion_velocity_rmse_norm": dif_vel,
        "velocity_improvement_ratio": (1 - dif_vel / kal_vel) if kal_vel else np.nan,
        "smoothness_before": sm_before, "smoothness_after": sm_after,
        "smoothness_improvement_ratio": (1 - sm_after / sm_before) if sm_before else np.nan,
    }


# =============================================================================
# Task 2: gap filling
# =============================================================================

def _make_gap_mask(n, T, mode, frac, rng):
    mask = np.zeros((n, T), dtype=bool)
    if mode == "block":
        blk = 3
        starts = rng.integers(1, max(2, T - blk), size=n)
        for i, s in enumerate(starts):
            mask[i, s:s + blk] = True
    else:  # random
        k = max(1, int(round(frac * (T - 1))))
        for i in range(n):
            idx = rng.choice(np.arange(1, T), size=min(k, T - 1), replace=False)
            mask[i, idx] = True
    return mask


def _linear_fill(X, mask):
    """Linear-interpolate masked timesteps per window/feature along time."""
    filled = X.copy()
    n, T, F = X.shape
    tgrid = np.arange(T)
    for i in range(n):
        obs = ~mask[i]
        if obs.sum() < 2:
            continue
        for f in range(F):
            filled[i, mask[i], f] = np.interp(tgrid[mask[i]], tgrid[obs], X[i, obs, f])
    return filled


def gap_filling_for_run(model, sched, X: np.ndarray, denoise_t: int, batch_size: int,
                        device: str, seed: int) -> List[Dict]:
    if not len(X):
        return []
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    rows = []
    for mode, frac in [("block", 3.0 / T), ("random", 0.2)]:
        mask = _make_gap_mask(len(X), T, mode, frac, rng)
        interp = _linear_fill(X, mask)
        refined = denoise_windows(model, interp, denoise_t, sched, batch_size, device)
        interp_mse = float((((interp - X) ** 2) * mask[:, :, None]).sum()
                           / max(mask.sum() * X.shape[2], 1))
        diff_mse = float((((refined - X) ** 2) * mask[:, :, None]).sum()
                         / max(mask.sum() * X.shape[2], 1))
        rows.append({"gap_mode": mode, "gap_fraction": round(float(mask.mean()), 4),
                     "n_windows": len(X), "interp_mse": interp_mse, "diffusion_mse": diff_mse,
                     "improvement_ratio": (1 - diff_mse / interp_mse) if interp_mse else np.nan})
    return rows


# =============================================================================
# Task 3: residual score
# =============================================================================

def residual_per_track(model, sched, normed: np.ndarray, gids: np.ndarray, denoise_t: int,
                       batch_size: int, device: str) -> Dict:
    """Per-track median/p90 single-step denoising residual (MSE(denoise(x), x))."""
    if not len(normed):
        return {}
    X0 = denoise_windows(model, normed, denoise_t, sched, batch_size, device)
    resid = ((X0 - normed) ** 2).mean(axis=(1, 2))
    out = {}
    for gid, idx in pd.Series(np.arange(len(gids))).groupby(np.asarray(gids, dtype=object)):
        r = resid[idx.to_numpy()]
        out[gid] = {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)),
                    "n": len(r), "all": r}
    return out


def aggregate_residual_by_threshold(scores: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    rows = []
    for thr, g in scores.groupby("threshold_db"):
        scored = g[np.isfinite(g["diffusion_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        kept = scored[scored["diffusion_score"] >= score_threshold]
        kt = int(kept["is_true_track"].sum())
        kf = len(kept) - kt
        rows.append({
            "threshold_db": thr, "stage08_confirmed_tracks": len(scored),
            "stage08_true_tracks": len(true), "stage08_false_tracks": len(false),
            "kept_tracks": len(kept), "kept_true_tracks": kt, "kept_false_tracks": kf,
            "true_track_retention": kt / len(true) if len(true) else np.nan,
            "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
            "precision_after": kt / len(kept) if len(kept) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("threshold_db").reset_index(drop=True)


def residual_sweep(scores: pd.DataFrame, sweep_thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for thr, g in scores.groupby("threshold_db"):
        scored = g[np.isfinite(g["diffusion_score"])]
        true = scored[scored["is_true_track"]]
        false = scored[~scored["is_true_track"]]
        for st in sweep_thresholds:
            kept = scored[scored["diffusion_score"] >= st]
            kt = int(kept["is_true_track"].sum())
            kf = len(kept) - kt
            rows.append({
                "threshold_db": thr, "score_threshold": st, "kept_tracks": len(kept),
                "kept_true_tracks": kt, "kept_false_tracks": kf,
                "true_track_retention": kt / len(true) if len(true) else np.nan,
                "false_track_reduction": 1 - kf / len(false) if len(false) else np.nan,
                "precision_after": kt / len(kept) if len(kept) else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Comparison with stage 12.5 / 13
# =============================================================================

def _stage_best(csv_path, ret_col, red_col, group_col) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if group_col not in df.columns:
        return None
    # pick the group (model/variant) with the best mean false reduction at >=0.95 retention
    agg = df.groupby(group_col).agg(tr=(ret_col, "mean"), fr=(red_col, "mean")).reset_index()
    ok = agg[agg["tr"] >= 0.95]
    best = (ok.sort_values("fr", ascending=False).iloc[0][group_col] if len(ok)
            else agg.sort_values("fr", ascending=False).iloc[0][group_col])
    return df[df[group_col] == best].copy(), best


def build_comparison(by_thr_resid: pd.DataFrame, task1: pd.DataFrame, task2: pd.DataFrame,
                     stage12_dir: str, stage13_dir: str) -> pd.DataFrame:
    s12 = _stage_best(os.path.join(stage12_dir, "stage08_vs_stage09_vs_stage11_vs_stage12.csv"),
                      "stage12_true_retention", "stage12_false_reduction", "model")
    s13 = _stage_best(os.path.join(stage13_dir,
                                   "stage08_vs_stage09_vs_stage11_vs_stage12_vs_stage13.csv"),
                      "stage13_true_retention", "stage13_false_reduction", "stage13_variant")
    t1 = task1.groupby("threshold_db")["rmse_improvement_ratio"].mean()
    t2 = task2.groupby("threshold_db")["improvement_ratio"].mean()

    rows = []
    for _, r in by_thr_resid.iterrows():
        thr = r["threshold_db"]
        row = {"threshold_db": thr, "stage12_best_model": None,
               "stage12_true_retention": np.nan, "stage12_false_reduction": np.nan,
               "stage13_best_variant": None, "stage13_true_retention": np.nan,
               "stage13_false_reduction": np.nan,
               "stage15_diffusion_true_retention": r["true_track_retention"],
               "stage15_diffusion_false_reduction": r["false_track_reduction"],
               "stage15_denoise_rmse_improvement": float(t1.get(thr, np.nan)),
               "stage15_gap_fill_improvement": float(t2.get(thr, np.nan)),
               "notes": "diffusion residual is a SECONDARY score"}
        if s12 is not None:
            df12, best12 = s12
            m = df12[np.isclose(df12["threshold_db"], thr)]
            if len(m):
                row["stage12_best_model"] = best12
                row["stage12_true_retention"] = float(m["stage12_true_retention"].iloc[0])
                row["stage12_false_reduction"] = float(m["stage12_false_reduction"].iloc[0])
        if s13 is not None:
            df13, best13 = s13
            m = df13[np.isclose(df13["threshold_db"], thr)]
            if len(m):
                row["stage13_best_variant"] = best13
                row["stage13_true_retention"] = float(m["stage13_true_retention"].iloc[0])
                row["stage13_false_reduction"] = float(m["stage13_false_reduction"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold_db").reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================

def make_plots(training_summary, val_denoise, task1, task2, scores, comparison,
               example, runtime, plots_dir) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []

    if training_summary is not None and len(training_summary):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(training_summary["epoch"], training_summary["train_loss"], marker="o", label="train")
        ax.plot(training_summary["epoch"], training_summary["val_loss"], marker="s",
                linestyle="--", label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("noise-prediction MSE")
        ax.set_title("Diffusion training loss")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "diffusion_loss_curve.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    if len(task1):
        t = task1.sort_values("threshold_db")
        x = np.arange(len(t)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - w / 2, t["kalman_position_rmse_norm"], w, label="noisy (corrupted)")
        ax.bar(x + w / 2, t["diffusion_position_rmse_norm"], w, label="diffusion denoised")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in t["threshold_db"]])
        ax.set_xlabel("detection threshold (dB)"); ax.set_ylabel("position RMSE (normalized)")
        ax.set_title("Denoising RMSE: corrupted vs diffusion-recovered")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "denoising_rmse_comparison.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    if len(task2):
        agg = task2.groupby("gap_mode").agg(interp=("interp_mse", "mean"),
                                            diff=("diffusion_mse", "mean")).reset_index()
        x = np.arange(len(agg)); w = 0.38
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - w / 2, agg["interp"], w, label="linear interpolation")
        ax.bar(x + w / 2, agg["diff"], w, label="diffusion refine")
        ax.set_xticks(x); ax.set_xticklabels(agg["gap_mode"])
        ax.set_xlabel("gap mode"); ax.set_ylabel("MSE at masked steps (normalized)")
        ax.set_title("Gap filling: interpolation vs diffusion")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "gap_filling_error_comparison.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    if len(scores):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.linspace(0, 1, 41)
        s = scores[np.isfinite(scores["diffusion_score"])]
        ax.hist(s.loc[s["is_true_track"], "diffusion_score"], bins=bins, density=True,
                histtype="step", linewidth=2, label="true")
        ax.hist(s.loc[~s["is_true_track"], "diffusion_score"], bins=bins, density=True,
                histtype="step", linewidth=2, linestyle="--", label="false")
        ax.set_xlabel("diffusion residual score"); ax.set_ylabel("density")
        ax.set_title("Diffusion residual score (solid true, dashed false)")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "diffusion_residual_true_false.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    if example is not None:
        noisy, denoised = example["input"], example["denoised"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(noisy[:, 0], noisy[:, 1], marker="o", label="Kalman posterior (input)")
        ax.plot(denoised[:, 0], denoised[:, 1], marker="s", linestyle="--",
                label="diffusion denoised")
        ax.set_xlabel("dx (normalized)"); ax.set_ylabel("dy (normalized)")
        ax.set_title("Example true-track window: Kalman vs diffusion")
        ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, "example_denoised_track.png"); fig.savefig(p, dpi=150)
        plt.close(fig); written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp = runtime[np.isfinite(runtime["seconds"])] if len(runtime) else runtime
    if len(comp):
        ax.bar(comp["component"], comp["seconds"])
        ax.set_ylabel("seconds")
    else:
        ax.text(0.5, 0.5, "runtime not recorded", ha="center", va="center")
    ax.set_title("Diffusion runtime components")
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    p = os.path.join(plots_dir, "diffusion_runtime_summary.png"); fig.savefig(p, dpi=150)
    plt.close(fig); written.append(p)
    return written


# =============================================================================
# Report
# =============================================================================

def write_report(report_dir, date, manifest, val_denoise, task1, task2, scores,
                 by_thr, comparison, calib, calib_meta, runtime, denoise_t, score_threshold) -> str:
    s = scores[np.isfinite(scores["diffusion_score"])]
    true_med = s.loc[s["is_true_track"], "diffusion_score"].median() if len(s) else np.nan
    false_med = s.loc[~s["is_true_track"], "diffusion_score"].median() if len(s) else np.nan
    mean_rmse_impr = task1["rmse_improvement_ratio"].mean() if len(task1) else np.nan
    mean_gap_impr = task2["improvement_ratio"].mean() if len(task2) else np.nan
    mean_sm_impr = task1["smoothness_improvement_ratio"].mean() if len(task1) else np.nan

    denoise_verdict = ("diffusion recovers synthetic corruption (positive RMSE improvement)"
                       if np.isfinite(mean_rmse_impr) and mean_rmse_impr > 0 else
                       "diffusion does NOT reduce corruption RMSE on these windows")
    gap_verdict = ("diffusion refinement beats linear interpolation at masked steps"
                   if np.isfinite(mean_gap_impr) and mean_gap_impr > 0 else
                   "diffusion refinement does NOT beat linear interpolation")
    class_verdict = ("the diffusion residual separates true and false tracks"
                     if np.isfinite(true_med) and np.isfinite(false_med) and true_med > false_med
                     else "the diffusion residual does not clearly separate true and false tracks")

    lines = [
        "# Stage 15 Diffusion Trajectory Denoising Study",
        "",
        "## Status",
        "",
        "- Stage 15 trains a lightweight diffusion denoiser over normalized",
        "  trajectory windows.",
        "- Stage 15 is primarily a denoising/gap-filling study, **not a new",
        "  primary false-track classifier**.",
        f"- Results are for {date} and thresholds "
        f"{', '.join(f'{t:g}' for t in sorted(scores['threshold_db'].unique()))}"
        " unless more data are processed.",
        "- **Stage 12.5 remains the current best false-track suppression method",
        "  unless diffusion clearly beats it.**",
        "- This is not the full model zoo.",
        "",
        "## Motivation",
        "",
        "- Stage 12.5 already nearly solves false-track suppression (a single",
        "  false track survived it in the stage-14 benchmark).",
        "- Diffusion may still help with noisy-track regularization, gap filling,",
        "  and rare-maneuver reconstruction -- so it is evaluated on those tasks,",
        "  with keep/reject classification kept explicitly secondary.",
        "",
        "## Training data",
        "",
        f"- Source: `{manifest.get('data_source', 'unknown')}`",
        f"- Train dates: {', '.join(manifest.get('train_dates', []))};"
        f" holdout: {', '.join(manifest.get('holdout_dates', []))}",
        f"- Train windows: {manifest.get('n_train_windows', 0):,};"
        f" validation windows: {manifest.get('n_val_windows', 0):,}",
        f"- Window length {manifest.get('window_len')} @ stride {manifest.get('stride')};"
        f" features {', '.join(manifest.get('features', []))}.",
        f"- Normalizer: {manifest.get('normalizer_source', 'unknown')}",
        "",
        "## Model",
        "",
        "- DDPM-style **noise prediction**: corrupt a clean window at a random",
        "  timestep, predict the injected noise, MSE loss.",
        f"- Temporal 1D-conv residual denoiser: hidden {manifest.get('hidden_dim')},"
        f" {manifest.get('num_blocks')} residual blocks, sinusoidal timestep embedding.",
        f"- Noise schedule: {manifest.get('num_diffusion_steps')} steps, linear beta"
        f" {manifest.get('beta_start')}..{manifest.get('beta_end')}.",
        f"- Evaluation uses single-step denoising (Mode A) at denoise_t = {denoise_t}.",
        "",
        "## Validation denoising",
        "",
        "Synthetic-noise recovery on clean holdout windows (improvement_ratio ="
        " 1 - denoised_mse / noisy_mse):",
        "",
    ]
    lines += md_table(val_denoise.round(6))
    lines += ["", "## True-track denoising", "",
              "Normalized-space metrics (per-window meter-space truth alignment is out of",
              "scope for v1). RMSE columns measure recovery of synthetic corruption of the",
              "Kalman windows; smoothness columns measure direct regularization of the",
              "actual Kalman posterior windows.", ""]
    lines += md_table(task1.round(6))
    lines += [f"", f"- Mean RMSE improvement ratio: {mean_rmse_impr:.4f}"
              f" -- {denoise_verdict}.",
              f"- Mean smoothness improvement ratio: {mean_sm_impr:.4f}"
              " (positive = diffusion smooths the Kalman track).", ""]

    lines += ["## Gap-filling", "",
              "Mask timesteps, linearly interpolate, then refine with diffusion; MSE at",
              "the masked steps (normalized space):", ""]
    lines += md_table(task2.round(6))
    lines += [f"", f"- Mean gap-fill improvement ratio: {mean_gap_impr:.4f} -- {gap_verdict}.", ""]

    lines += ["## Residual score as classifier", "",
              f"Calibration: track-purity on high-purity true tracks"
              f" (dates {', '.join(calib_meta.get('calibration_dates', []))};"
              f" thresholds {', '.join(f'{t:g}' for t in calib_meta.get('calibration_thresholds', []))} dB);"
              f" p50 {calib.get('error_p50', float('nan')):.5f} -> score 1,"
              f" p99 {calib.get('error_p99', float('nan')):.5f} -> score 0.",
              f"- Median residual score: true {true_med:.3f} vs false {false_med:.3f}"
              f" -- {class_verdict}.",
              "- **This is a secondary diagnostic**, not the primary filter.", ""]
    lines += md_table(by_thr.round(4))

    lines += ["", "## Comparison with Stage 12.5 and Stage 13", ""]
    lines += md_table(comparison.round(4))
    s12_fr = comparison["stage12_false_reduction"].mean()
    s15_fr = comparison["stage15_diffusion_false_reduction"].mean()
    beats = np.isfinite(s12_fr) and np.isfinite(s15_fr) and s15_fr >= s12_fr
    lines += ["",
              (f"- As a classifier the diffusion residual (mean false reduction {s15_fr:.3f})"
               f" {'matches/beats' if beats else 'does NOT beat'} the best stage-12.5 model"
               f" (mean {s12_fr:.3f}).") if np.isfinite(s12_fr) else
              "- Stage 12.5 comparison unavailable.", ""]

    lines += ["## Runtime and complexity", ""]
    lines += md_table(runtime)
    lines += ["",
              "- A DDPM denoiser is heavier than a deterministic autoencoder (extra",
              "  timestep conditioning and, if used, iterative sampling). Single-step",
              "  Mode A keeps inference cheap.", ""]

    conclusion = (
        "Diffusion is useful diagnostically (it recovers synthetic corruption and can"
        " smooth tracks) but does NOT improve enough over the deterministic stage-12.5"
        " autoencoders to replace them as the primary false-track filter."
        if not beats else
        "Diffusion improves denoising/gap filling and is competitive as a filter; it is"
        " worth keeping for regularization.")
    lines += ["## Conclusion", "", f"- {conclusion}", "",
              "## Recommended next stage", "",
              "Stage 16 should be either:",
              "1. a compact **model-zoo benchmark** including only promising families, or",
              "2. a **robustness/ablation study** across all four days and clutter/noise",
              "   levels.", ""]

    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "diffusion_denoising_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# Validation gates
# =============================================================================

def run_eval_gate(report_dir, task1, task2, scores, by_thr, calib) -> None:
    def fail(msg):
        raise ValueError(f"Stage 15 evaluation validation failed: {msg}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (evaluation)")
    print("=" * 70)
    for name, path in [("diffusion_track_denoising_metrics.csv", "task1"),
                       ("diffusion_gap_filling_metrics.csv", "task2"),
                       ("diffusion_residual_scores.csv", "scores")]:
        p = os.path.join(report_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            fail(f"{name} missing or empty")
    print("  denoising / gap-filling / residual files present and nonempty: OK")

    if not np.isfinite(task1[["kalman_position_rmse_norm", "diffusion_position_rmse_norm"]]
                       .to_numpy()).all():
        fail("denoising metrics not finite")
    if not np.isfinite(task2[["interp_mse", "diffusion_mse"]].to_numpy()).all():
        fail("gap-filling metrics not finite")
    print("  denoising and gap-filling metrics finite: OK")

    sc = scores[np.isfinite(scores["diffusion_score"])]
    if len(sc) and not sc["diffusion_score"].between(0, 1).all():
        fail("diffusion_score outside [0, 1]")
    if not sc["keep_diffusion_score"].isin([True, False]).all():
        fail("keep_diffusion_score not boolean")
    if not (calib.get("n_calibration_tracks", 0) > 0 and calib.get("n_calibration_windows", 0) > 0):
        fail("calibration tracks/windows must be > 0")
    print("  scores in [0,1], keep boolean, calibration nonempty: OK")

    for _, r in by_thr.iterrows():
        for col in ("true_track_retention", "false_track_reduction"):
            v = r[col]
            if np.isfinite(v) and not (0 <= v <= 1):
                fail(f"{col} outside [0,1]")
    print("  retention / false-reduction in [0,1]: OK")

    mean_rmse = task1["rmse_improvement_ratio"].mean()
    mean_gap = task2["improvement_ratio"].mean()
    true_med = sc.loc[sc["is_true_track"], "diffusion_score"].median() if len(sc) else np.nan
    false_med = sc.loc[~sc["is_true_track"], "diffusion_score"].median() if len(sc) else np.nan
    print(f"  (report-only) mean denoise RMSE improvement {mean_rmse:.4f}; "
          f"mean gap-fill improvement {mean_gap:.4f}")
    print(f"  (report-only) residual median true {true_med:.3f} vs false {false_med:.3f} "
          + ("-- separable" if np.isfinite(true_med) and true_med > false_med else
             "-- NOT clearly separable"))
