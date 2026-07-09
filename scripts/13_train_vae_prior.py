"""Entry point: stage 13 step A -- train a sequence VAE on clean truth windows.

Trains a fully-connected variational autoencoder over origin/heading-normalized
trajectory windows from stage-5 relocated truth (stage-4 fallback), holding out
whole days for validation/calibration. Reuses the stage-12 window pipeline and,
when compatible, the stage-12 normalizer so stage 12 and 13 are directly
comparable. Not diffusion (stage 14), not the full model zoo.

Usage:
    python scripts/13_train_vae_prior.py --holdout-date 2022-06-27 --overwrite
    python scripts/13_train_vae_prior.py --self-test
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
    WindowConfig,
    apply_normalizer,
    build_windows_by_group,
    discover_stage04_files,
    discover_truth_files,
    fit_normalizer,
    load_stage04_for_windows,
    load_truth_for_windows,
    load_normalizer,
    sample_windows_deterministic,
    save_normalizer,
    split_train_val_by_date,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the stage-13 sequence VAE on clean truth windows.")
    p.add_argument("--truth-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    p.add_argument("--stage04-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "trajectories_10s"))
    p.add_argument("--models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "vae_priors"))
    p.add_argument("--sequence-models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "sequence_priors"),
                   help="Stage-12 models dir; its normalizer.json is reused when compatible.")
    p.add_argument("--report-dir", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage13_vae_prior"))
    p.add_argument("--window-len", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--features", type=str,
                   default="dx,dy,dz,vx,vy,vz,speed,vertical_speed,turn_rate")
    p.add_argument("--holdout-date", type=str, nargs="*", default=["2022-06-27"])
    p.add_argument("--max-train-windows", type=int, default=200_000)
    p.add_argument("--max-val-windows", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--kl-anneal-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
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
        raise SystemExit("No holdout/validation files -- pass at least one --holdout-date "
                         "that exists in the data")

    def collect(file_list, cap, seed_offset):
        parts = []
        for date, path in file_list:
            print(f"  [{date}] building windows from {os.path.basename(path)} ...", flush=True)
            parts.append(build_windows_by_group(loader(path), "group_id", wcfg))
        windows = (np.concatenate(parts) if parts else
                   np.empty((0, wcfg.window_len, len(wcfg.features)), np.float32))
        return sample_windows_deterministic(windows, cap, seed + seed_offset)

    train_w = collect(train_files, max_train, 0)
    val_w = collect(val_files, max_val, 1)
    return train_w, val_w, source, [d for d, _ in train_files], [d for d, _ in val_files]


def resolve_normalizer(args, wcfg, train_w):
    """Reuse the stage-12 normalizer when compatible; else fit a fresh one."""
    seq_norm = os.path.join(args.sequence_models_dir, "normalizer.json")
    if os.path.exists(seq_norm):
        norm = load_normalizer(seq_norm)
        if (norm.get("features") == list(wcfg.features)
                and int(norm.get("window_len", -1)) == wcfg.window_len):
            print(f"reusing stage-12 normalizer: {seq_norm}")
            return norm, f"stage12:{seq_norm}"
        print("stage-12 normalizer incompatible (features/window_len differ); fitting fresh")
    norm = fit_normalizer(train_w, wcfg)
    return norm, "stage13:fitted"


def run_training(args) -> dict:
    from utils.vae_models import (SequenceVAE, make_loader, resolve_device, save_vae, train_vae,
                                  vae_window_metrics)
    from utils.vae_prior_score import make_training_plot

    manifest_path = os.path.join(args.models_dir, "vae_training_manifest.json")
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
          f"(train dates {train_dates}, holdout {val_dates})")

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    normalizer, norm_source = resolve_normalizer(args, wcfg, train_w)
    save_normalizer(os.path.join(args.models_dir, "normalizer.json"), normalizer)
    train_n = apply_normalizer(train_w, normalizer)
    val_n = apply_normalizer(val_w, normalizer)
    input_dim = train_n.shape[-1]

    cfg = {"device": device, "epochs": args.epochs, "beta": args.beta,
           "kl_anneal_epochs": args.kl_anneal_epochs, "learning_rate": args.learning_rate,
           "weight_decay": args.weight_decay, "seed": args.seed,
           "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
           "input_dim": input_dim, "window_len": wcfg.window_len}

    print("\ntraining sequence VAE ...", flush=True)
    model = SequenceVAE(input_dim, wcfg.window_len, args.hidden_dim, args.latent_dim)
    history = train_vae(model, make_loader(train_n, args.batch_size, True),
                        make_loader(val_n, args.batch_size, False), cfg)

    ckpt = os.path.join(args.models_dir, "vae_prior.pt")
    save_vae(ckpt, model, cfg)
    with open(os.path.join(args.models_dir, "vae_config.json"), "w") as f:
        json.dump({"window_len": wcfg.window_len, "stride": wcfg.stride,
                   "features": wcfg.features, "input_dim": input_dim,
                   "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
                   "beta": args.beta, "kl_anneal_epochs": args.kl_anneal_epochs}, f, indent=1)

    metrics = vae_window_metrics(model, val_n, args.batch_size, device)
    recon = metrics["recon_error"]
    kl = metrics["kl"]
    recon_row = {"error_p50": float(np.percentile(recon, 50)),
                 "error_p75": float(np.percentile(recon, 75)),
                 "error_p90": float(np.percentile(recon, 90)),
                 "error_p95": float(np.percentile(recon, 95)),
                 "error_p99": float(np.percentile(recon, 99)),
                 "error_mean": float(recon.mean()), "error_std": float(recon.std()),
                 "kl_p50": float(np.percentile(kl, 50)), "kl_p90": float(np.percentile(kl, 90)),
                 "kl_mean": float(kl.mean()), "n_val_windows": int(len(recon))}
    recon_df = pd.DataFrame([recon_row])[["n_val_windows", "error_p50", "error_p75",
                                          "error_p90", "error_p95", "error_p99",
                                          "error_mean", "error_std", "kl_p50", "kl_p90", "kl_mean"]]
    recon_df.to_csv(os.path.join(args.report_dir, "vae_validation_reconstruction.csv"), index=False)

    summary_df = pd.DataFrame([{
        "train_windows": len(train_n), "val_windows": len(val_n), "epochs": args.epochs,
        "best_val_loss": history["best_val_loss"],
        "final_train_recon": history["train_recon"][-1],
        "final_val_recon": history["val_recon"][-1],
        "final_train_kl": history["train_kl"][-1], "final_val_kl": history["val_kl"][-1],
        "beta": args.beta, "kl_anneal_epochs": args.kl_anneal_epochs,
        "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim, "device": device,
        "window_len": wcfg.window_len, "stride": wcfg.stride,
        "features": ",".join(wcfg.features)}])
    summary_df.to_csv(os.path.join(args.report_dir, "vae_training_summary.csv"), index=False)

    manifest = {
        "created_by": "Stage 13 VAE training", "data_source": source,
        "train_dates": train_dates, "holdout_dates": val_dates,
        "features": wcfg.features, "window_len": wcfg.window_len, "stride": wcfg.stride,
        "seed": args.seed, "n_train_windows": int(len(train_n)),
        "n_val_windows": int(len(val_n)), "checkpoint": ckpt,
        "normalizer": os.path.join(args.models_dir, "normalizer.json"),
        "normalizer_source": norm_source,
        "validation_reconstruction": os.path.join(args.report_dir,
                                                  "vae_validation_reconstruction.csv"),
        "device": device, "epochs": args.epochs, "beta": args.beta,
        "kl_anneal_epochs": args.kl_anneal_epochs, "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim, "max_train_windows": args.max_train_windows,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    make_training_plot(history, os.path.join(args.report_dir, "plots"))

    # ---- training validation gate --------------------------------------
    def fail(message: str) -> None:
        raise ValueError(f"Stage 13 training validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (training)")
    print("=" * 70)
    if not train_dates:
        fail("no training files processed")
    if not val_dates:
        fail("no validation/holdout files processed")
    if not os.path.exists(ckpt):
        fail("VAE checkpoint not saved")
    for f_ in ["vae_config.json", "vae_training_manifest.json"]:
        if not os.path.exists(os.path.join(args.models_dir, f_)):
            fail(f"{f_} missing")
    for f_ in ["vae_training_summary.csv", "vae_validation_reconstruction.csv"]:
        pth = os.path.join(args.report_dir, f_)
        if not os.path.exists(pth) or os.path.getsize(pth) == 0:
            fail(f"{f_} missing or empty")
    if not (np.isfinite(recon_row["error_p50"]) and np.isfinite(recon_row["error_p99"])
            and recon_row["error_p99"] >= recon_row["error_p50"]):
        fail("validation reconstruction quantiles invalid")
    if not np.isfinite(recon_row["kl_mean"]) or recon_row["kl_mean"] < 0:
        fail("validation KL not finite/nonnegative")
    print(f"  {len(train_dates)} train + {len(val_dates)} holdout files; checkpoint, config, "
          "manifest, summaries: OK")
    print("  validation reconstruction finite/ordered; KL finite and nonnegative: OK")
    if recon_row["kl_mean"] < 1e-4:
        print("  WARNING: mean validation KL is near zero -- possible posterior collapse.")

    print("\ntraining summary:")
    print(summary_df[["train_windows", "val_windows", "best_val_loss", "final_val_recon",
                      "final_val_kl"]].to_string(index=False))
    print("\nvalidation reconstruction/KL:")
    print(recon_df.to_string(index=False))
    return {"summary": summary_df, "recon": recon_df, "manifest": manifest, "history": history}


def make_synthetic_truth(path: str, date: str, n_traj: int = 8, n: int = 80, seed: int = 0) -> None:
    """Smooth synthetic GA-like trajectories in a stage-5-truth-like schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traj):
        t = 1_000.0 + 10.0 * np.arange(n)
        speed = 55.0 + rng.normal(0, 3)
        heading = np.radians(rng.uniform(0, 360)) + np.cumsum(np.radians(rng.normal(0, 0.5, n)))
        vx = speed * np.sin(heading)
        vy = speed * np.cos(heading)
        vz = rng.normal(0, 1, n)
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
        args.sequence_models_dir = os.path.join(tmp, "seq_missing")  # force fresh normalizer
        args.models_dir = os.path.join(tmp, "models")
        args.report_dir = os.path.join(tmp, "reports")
        args.holdout_date = ["2022-01-02"]
        args.window_len, args.stride = 10, 5
        args.epochs, args.batch_size = 2, 64
        args.hidden_dim, args.latent_dim = 32, 4
        args.overwrite = True

        out = run_training(args)
        for f_ in ["vae_prior.pt", "vae_config.json", "vae_training_manifest.json",
                   "normalizer.json"]:
            assert os.path.exists(os.path.join(args.models_dir, f_)), f"missing {f_}"
        for f_ in ["vae_training_summary.csv", "vae_validation_reconstruction.csv"]:
            assert os.path.exists(os.path.join(args.report_dir, f_)), f"missing {f_}"
        recon = out["recon"].iloc[0]
        assert np.isfinite(recon["error_p50"]) and np.isfinite(recon["error_p99"]), \
            "validation reconstruction not finite"
        assert np.isfinite(recon["kl_mean"]) and recon["kl_mean"] >= 0, \
            "validation KL not finite/nonnegative"

    print("\nStage 13 VAE training self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_training(args)
    print("\n13_train_vae_prior completed successfully.")


if __name__ == "__main__":
    main()
