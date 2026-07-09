"""Entry point: stage 15 step A -- train a trajectory-window diffusion denoiser.

Trains a lightweight DDPM noise-prediction denoiser over origin/heading-
normalized trajectory windows from stage-5 relocated truth (stage-4 fallback),
holding out whole days for validation. Reuses the stage-12 normalizer when
compatible so stages 12/13/15 share the same window space. Validation checks
that the model actually denoises synthetic corruption at several noise levels.

Usage:
    python scripts/15_train_diffusion_denoiser.py --holdout-date 2022-06-27 --overwrite
    python scripts/15_train_diffusion_denoiser.py --self-test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.sequence_windows import (
    WindowConfig, apply_normalizer, build_windows_by_group, discover_stage04_files,
    discover_truth_files, fit_normalizer, load_normalizer, load_stage04_for_windows,
    load_truth_for_windows, sample_windows_deterministic, save_normalizer, split_train_val_by_date,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the stage-15 diffusion trajectory denoiser.")
    p.add_argument("--truth-dir", default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    p.add_argument("--stage04-dir", default=os.path.join(REPO_ROOT, "data", "active", "trajectories_10s"))
    p.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models", "diffusion_denoisers"))
    p.add_argument("--sequence-models-dir", default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--report-dir", default=os.path.join(REPO_ROOT, "reports", "stage15_diffusion_denoising"))
    p.add_argument("--window-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--features", default="dx,dy,dz,vx,vy,vz,speed,vertical_speed,turn_rate")
    p.add_argument("--holdout-date", nargs="*", default=["2022-06-27"])
    p.add_argument("--max-train-windows", type=int, default=200_000)
    p.add_argument("--max-val-windows", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-diffusion-steps", type=int, default=100)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return p.parse_args()


def build_dataset(truth_dir, stage04_dir, wcfg, holdout_dates, max_train, max_val, seed):
    files = discover_truth_files(truth_dir)
    loader, source = load_truth_for_windows, truth_dir
    if not files:
        files = discover_stage04_files(stage04_dir)
        loader, source = load_stage04_for_windows, stage04_dir
    if not files:
        raise SystemExit(f"No training data found in {truth_dir} or {stage04_dir}")
    train_files, val_files = split_train_val_by_date(files, holdout_dates)
    if not train_files:
        raise SystemExit("No training files left after holdout split")
    if not val_files:
        raise SystemExit("No holdout/validation files -- pass a --holdout-date present in the data")

    def collect(file_list, cap, off):
        parts = []
        for date, path in file_list:
            print(f"  [{date}] building windows from {os.path.basename(path)} ...", flush=True)
            parts.append(build_windows_by_group(loader(path), "group_id", wcfg))
        w = np.concatenate(parts) if parts else np.empty((0, wcfg.window_len, len(wcfg.features)),
                                                          np.float32)
        return sample_windows_deterministic(w, cap, seed + off)

    return (collect(train_files, max_train, 0), collect(val_files, max_val, 1), source,
            [d for d, _ in train_files], [d for d, _ in val_files])


def resolve_normalizer(args, wcfg, train_w):
    seq_norm = os.path.join(args.sequence_models_dir, "normalizer.json")
    if os.path.exists(seq_norm):
        norm = load_normalizer(seq_norm)
        if norm.get("features") == list(wcfg.features) and \
                int(norm.get("window_len", -1)) == wcfg.window_len:
            print(f"reusing stage-12 normalizer: {seq_norm}")
            return norm, f"stage12:{seq_norm}"
        print("stage-12 normalizer incompatible; fitting fresh")
    return fit_normalizer(train_w, wcfg), "stage15:fitted"


def validation_denoising(model, sched, val_n, args, device) -> pd.DataFrame:
    """Corrupt clean holdout windows at several levels; check diffusion recovers them."""
    import torch
    from utils.diffusion_models import denoise_windows, forward_corrupt
    rows = []
    Xt = torch.from_numpy(np.ascontiguousarray(val_n, dtype=np.float32))
    for dt in [5, 10, 20, 50]:
        if dt >= sched.num_steps:
            continue
        gen = torch.Generator().manual_seed(999 + dt)
        eps = torch.randn(Xt.shape, generator=gen)
        Xn, _ = forward_corrupt(Xt, dt, sched.to("cpu"), eps=eps)
        Xn = Xn.numpy()
        X0 = denoise_windows(model, Xn, dt, sched, args.batch_size, device)
        noisy_mse = float(((Xn - val_n) ** 2).mean())
        denoised_mse = float(((X0 - val_n) ** 2).mean())
        rows.append({"denoise_t": dt, "n_val_windows": len(val_n), "noisy_mse": noisy_mse,
                     "denoised_mse": denoised_mse,
                     "improvement_ratio": (1 - denoised_mse / noisy_mse) if noisy_mse else np.nan})
    return pd.DataFrame(rows)


def run_training(args) -> dict:
    from utils.diffusion_models import (NoiseSchedule, TemporalDiffusionDenoiser, make_loader,
                                        resolve_device, save_diffusion, train_diffusion)
    import matplotlib
    matplotlib.use("Agg")

    manifest_path = os.path.join(args.models_dir, "diffusion_training_manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {manifest_path}")

    wcfg = WindowConfig(window_len=args.window_len, stride=args.stride,
                        features=args.features.split(","))
    device = resolve_device(args.device)
    print(f"device: {device}")

    train_w, val_w, source, train_dates, val_dates = build_dataset(
        args.truth_dir, args.stage04_dir, wcfg, args.holdout_date,
        args.max_train_windows, args.max_val_windows, args.seed)
    print(f"windows: train {len(train_w):,}, val {len(val_w):,} "
          f"(train {train_dates}, holdout {val_dates})")

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    normalizer, norm_source = resolve_normalizer(args, wcfg, train_w)
    save_normalizer(os.path.join(args.models_dir, "normalizer.json"), normalizer)
    train_n = apply_normalizer(train_w, normalizer)
    val_n = apply_normalizer(val_w, normalizer)
    input_dim = train_n.shape[-1]

    sched = NoiseSchedule(args.num_diffusion_steps, args.beta_start, args.beta_end)
    cfg = {"device": device, "epochs": args.epochs, "learning_rate": args.learning_rate,
           "weight_decay": args.weight_decay, "seed": args.seed, "input_dim": input_dim,
           "window_len": wcfg.window_len, "hidden_dim": args.hidden_dim,
           "num_blocks": args.num_blocks}

    print("\ntraining diffusion denoiser ...", flush=True)
    model = TemporalDiffusionDenoiser(input_dim, wcfg.window_len, args.hidden_dim, args.num_blocks)
    history = train_diffusion(model, make_loader(train_n, args.batch_size, True),
                              make_loader(val_n, args.batch_size, False), sched, cfg)

    ckpt = os.path.join(args.models_dir, "diffusion_denoiser.pt")
    save_diffusion(ckpt, model, cfg)
    with open(os.path.join(args.models_dir, "diffusion_config.json"), "w") as f:
        json.dump({"window_len": wcfg.window_len, "stride": wcfg.stride, "features": wcfg.features,
                   "input_dim": input_dim, "hidden_dim": args.hidden_dim,
                   "num_blocks": args.num_blocks, "num_diffusion_steps": args.num_diffusion_steps,
                   "beta_start": args.beta_start, "beta_end": args.beta_end}, f, indent=1)

    summary = pd.DataFrame({"epoch": history["epoch"], "train_loss": history["train_loss"],
                            "val_loss": history["val_loss"],
                            "learning_rate": args.learning_rate, "device": device})
    summary.to_csv(os.path.join(args.report_dir, "diffusion_training_summary.csv"), index=False)

    val_denoise = validation_denoising(model, sched, val_n, args, device)
    val_denoise.to_csv(os.path.join(args.report_dir, "diffusion_validation_denoising.csv"),
                       index=False)

    manifest = {"created_by": "Stage 15 diffusion training", "data_source": source,
                "train_dates": train_dates, "holdout_dates": val_dates, "features": wcfg.features,
                "window_len": wcfg.window_len, "stride": wcfg.stride, "seed": args.seed,
                "n_train_windows": int(len(train_n)), "n_val_windows": int(len(val_n)),
                "checkpoint": ckpt, "normalizer": os.path.join(args.models_dir, "normalizer.json"),
                "normalizer_source": norm_source, "device": device, "epochs": args.epochs,
                "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                "num_diffusion_steps": args.num_diffusion_steps, "beta_start": args.beta_start,
                "beta_end": args.beta_end, "best_val_loss": history["best_val_loss"],
                "validation_denoising": os.path.join(args.report_dir,
                                                     "diffusion_validation_denoising.csv"),
                "max_train_windows": args.max_train_windows}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    # loss curve plot
    import matplotlib.pyplot as plt
    plots_dir = os.path.join(args.report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(summary["epoch"], summary["train_loss"], marker="o", label="train")
    ax.plot(summary["epoch"], summary["val_loss"], marker="s", linestyle="--", label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("noise-prediction MSE")
    ax.set_title("Stage 15 diffusion training loss")
    ax.legend(fontsize=8); ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "diffusion_loss_curve.png"), dpi=150)
    plt.close(fig)

    # ---- training validation gate --------------------------------------
    def fail(msg):
        raise ValueError(f"Stage 15 training validation failed: {msg}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (training)")
    print("=" * 70)
    if not train_dates:
        fail("no training files processed")
    if not val_dates:
        fail("no validation/holdout files processed")
    if not os.path.exists(ckpt):
        fail("checkpoint not saved")
    for f_ in ["diffusion_config.json", "diffusion_training_manifest.json"]:
        if not os.path.exists(os.path.join(args.models_dir, f_)):
            fail(f"{f_} missing")
    for f_ in ["diffusion_training_summary.csv", "diffusion_validation_denoising.csv"]:
        pth = os.path.join(args.report_dir, f_)
        if not os.path.exists(pth) or os.path.getsize(pth) == 0:
            fail(f"{f_} missing or empty")
    if not np.isfinite(summary[["train_loss", "val_loss"]].to_numpy()).all():
        fail("loss values not finite")
    if not (val_denoise["improvement_ratio"] > 0).any():
        fail("diffusion did not denoise (no positive improvement at any noise level)")
    print(f"  {len(train_dates)} train + {len(val_dates)} holdout files; checkpoint, config, "
          "manifest, summaries: OK")
    print("  losses finite; denoised MSE < noisy MSE for >= 1 noise level: OK")

    print("\ntraining summary (last epoch):")
    print(summary.tail(1).to_string(index=False))
    print("\nvalidation denoising:")
    print(val_denoise.to_string(index=False))
    return {"summary": summary, "val_denoise": val_denoise, "manifest": manifest}


def make_synthetic_truth(path, date, n_traj=8, n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traj):
        t = 1_000.0 + 10.0 * np.arange(n)
        speed = 55.0 + rng.normal(0, 3)
        heading = np.radians(rng.uniform(0, 360)) + np.cumsum(np.radians(rng.normal(0, 0.5, n)))
        vx, vy, vz = speed * np.sin(heading), speed * np.cos(heading), rng.normal(0, 1, n)
        rows.append(pd.DataFrame({
            "trajectory_id": f"{date}_t{k}", "timestamp": t,
            "east_m": 20_000 + np.cumsum(vx) * 10.0, "north_m": 15_000 + np.cumsum(vy) * 10.0,
            "up_m": 1_000 + np.cumsum(vz) * 10.0, "ve_mps": vx, "vn_mps": vy, "vu_mps": vz}))
    pd.concat(rows).to_csv(path, index=False)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        truth_dir = os.path.join(tmp, "truth")
        os.makedirs(truth_dir)
        for i, date in enumerate(["2022-01-01", "2022-01-02"]):
            make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"), date, seed=i)
        args = parse_args()
        args.truth_dir = truth_dir
        args.sequence_models_dir = os.path.join(tmp, "seq_missing")
        args.models_dir = os.path.join(tmp, "models")
        args.report_dir = os.path.join(tmp, "reports")
        args.holdout_date = ["2022-01-02"]
        args.window_len, args.stride = 10, 2
        args.epochs, args.batch_size = 20, 64
        args.hidden_dim, args.num_blocks = 64, 3
        args.num_diffusion_steps = 50
        args.overwrite = True
        out = run_training(args)
        for f_ in ["diffusion_denoiser.pt", "diffusion_config.json",
                   "diffusion_training_manifest.json", "normalizer.json"]:
            assert os.path.exists(os.path.join(args.models_dir, f_)), f"missing {f_}"
        vd = out["val_denoise"]
        assert np.isfinite(vd["denoised_mse"].to_numpy()).all(), "denoised MSE not finite"
        assert (vd["improvement_ratio"] > 0).any(), \
            "expected denoised MSE < noisy MSE for at least one easy noise level"
    print("\nStage 15 diffusion training self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_training(args)
    print("\n15_train_diffusion_denoiser completed successfully.")


if __name__ == "__main__":
    main()
