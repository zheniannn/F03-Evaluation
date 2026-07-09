"""Entry point: stage 12 step A -- train trajectory-window autoencoders on
clean truth windows.

Trains up to three denoising autoencoders (mlp_dae, gru_ae, tcn_ae) on
origin/heading-normalized trajectory windows from stage-5 relocated truth
(preferred) or stage-4 trajectories (fallback), holding out whole days for
validation/calibration. Not a VAE (stage 13), not diffusion (stage 14).

Usage:
    python scripts/12_train_sequence_priors.py --holdout-date 2022-06-27 --overwrite
    python scripts/12_train_sequence_priors.py --self-test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.sequence_windows import (
    WindowConfig,
    build_windows_by_group,
    discover_stage04_files,
    discover_truth_files,
    fit_normalizer,
    apply_normalizer,
    load_stage04_for_windows,
    load_truth_for_windows,
    sample_windows_deterministic,
    save_normalizer,
    split_train_val_by_date,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train stage-12 sequence-prior autoencoders on clean truth windows.")
    parser.add_argument("--truth-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "radar_truth_relocated"))
    parser.add_argument("--stage04-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "trajectories_10s"))
    parser.add_argument("--models-dir", type=str,
                        default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors"))
    parser.add_argument("--model", type=str, nargs="+", default=["mlp_dae", "gru_ae", "tcn_ae"])
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--features", type=str,
                        default="dx,dy,dz,vx,vy,vz,speed,vertical_speed,turn_rate")
    parser.add_argument("--holdout-date", type=str, nargs="*", default=["2022-06-27"])
    parser.add_argument("--max-train-windows", type=int, default=500_000)
    parser.add_argument("--max-val-windows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def build_dataset(truth_dir: str, stage04_dir: str, wcfg: WindowConfig,
                  holdout_dates, max_train: int, max_val: int, seed: int):
    """Windows from stage-5 truth (preferred) or stage-4 (fallback), split by date."""
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
            df = loader(path)
            parts.append(build_windows_by_group(df, "group_id", wcfg))
        windows = np.concatenate(parts) if parts else np.empty((0, wcfg.window_len, 9), np.float32)
        return sample_windows_deterministic(windows, cap, seed + seed_offset)

    train_w = collect(train_files, max_train, 0)
    val_w = collect(val_files, max_val, 1)
    return train_w, val_w, source, [d for d, _ in train_files], [d for d, _ in val_files]


def run_training(args) -> dict:
    from utils.sequence_models import (get_model, make_loader, reconstruction_errors,
                                       resolve_device, save_model, train_autoencoder)
    from utils.sequence_prior_score import calibrate_score_from_validation_errors
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = os.path.join(args.models_dir, "training_manifest.json")
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

    normalizer = fit_normalizer(train_w, wcfg)
    save_normalizer(os.path.join(args.models_dir, "normalizer.json"), normalizer)
    with open(os.path.join(args.models_dir, "window_config.json"), "w") as f:
        json.dump({"window_len": wcfg.window_len, "stride": wcfg.stride,
                   "features": wcfg.features}, f, indent=1)
    train_n = apply_normalizer(train_w, normalizer)
    val_n = apply_normalizer(val_w, normalizer)

    model_cfg = {"hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
                 "num_layers": args.num_layers, "device": device, "epochs": args.epochs,
                 "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                 "noise_std": args.noise_std, "seed": args.seed}
    input_dim = train_n.shape[-1]

    summary_rows, recon_rows, histories, checkpoints = [], [], {}, {}
    for name in args.model:
        print(f"\ntraining {name} ...", flush=True)
        model = get_model(name, input_dim, wcfg.window_len, model_cfg)
        history = train_autoencoder(model, make_loader(train_n, args.batch_size, True),
                                    make_loader(val_n, args.batch_size, False), model_cfg)
        ckpt = os.path.join(args.models_dir, f"{name}.pt")
        save_model(ckpt, model, {**model_cfg, "model": name, "input_dim": input_dim,
                                 "window_len": wcfg.window_len})
        checkpoints[name] = ckpt
        histories[name] = history

        errors = reconstruction_errors(model, val_n, args.batch_size, device)
        cal = calibrate_score_from_validation_errors(errors)
        recon_rows.append({"model": name, **cal})
        summary_rows.append({
            "model": name, "train_windows": len(train_n), "val_windows": len(val_n),
            "epochs": args.epochs, "best_val_loss": history["best_val_loss"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "device": device, "window_len": wcfg.window_len, "stride": wcfg.stride,
            "features": ",".join(wcfg.features),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.report_dir, "training_summary.csv"), index=False)
    recon_df = pd.DataFrame(recon_rows)[["model", "n_val_windows", "error_p50", "error_p75",
                                         "error_p90", "error_p95", "error_p99",
                                         "error_mean", "error_std"]]
    recon_df.to_csv(os.path.join(args.report_dir, "validation_reconstruction.csv"), index=False)

    manifest = {
        "created_by": "Stage 12 training",
        "data_source": source,
        "train_dates": train_dates, "holdout_dates": val_dates,
        "models": args.model, "features": wcfg.features,
        "window_len": wcfg.window_len, "stride": wcfg.stride, "seed": args.seed,
        "n_train_windows": int(len(train_n)), "n_val_windows": int(len(val_n)),
        "checkpoints": checkpoints,
        "normalizer": os.path.join(args.models_dir, "normalizer.json"),
        "validation_reconstruction": os.path.join(args.report_dir,
                                                  "validation_reconstruction.csv"),
        "device": device, "epochs": args.epochs,
        "max_train_windows": args.max_train_windows,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    plots_dir = os.path.join(args.report_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in histories.items():
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], label=f"{name} train")
        ax.plot(ep, h["val_loss"], linestyle="--", label=f"{name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title("Stage 12 reconstruction loss curves")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "reconstruction_loss_curves.png"), dpi=150)
    plt.close(fig)

    # ---- training validation gate --------------------------------------
    def fail(message: str) -> None:
        raise ValueError(f"Stage 12 training validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE (training)")
    print("=" * 70)
    if not train_dates:
        fail("no training files processed")
    if not val_dates:
        fail("no validation files processed")
    for f_ in ["normalizer.json", "window_config.json"]:
        if not os.path.exists(os.path.join(args.models_dir, f_)):
            fail(f"{f_} missing")
    if not any(os.path.exists(c) for c in checkpoints.values()):
        fail("no model checkpoint saved")
    for f_ in ["training_summary.csv", "validation_reconstruction.csv"]:
        p = os.path.join(args.report_dir, f_)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            fail(f"{f_} missing or empty")
    for _, r in recon_df.iterrows():
        if not (np.isfinite(r["error_p50"]) and np.isfinite(r["error_p99"])
                and r["error_p99"] >= r["error_p50"]):
            fail(f"{r['model']}: validation error quantiles invalid")
    print(f"  {len(train_dates)} train + {len(val_dates)} holdout files; normalizer, config, "
          f"{len(checkpoints)} checkpoints, summaries: OK")
    print("  validation error p50/p99 finite and ordered: OK")

    print("\ntraining summary:")
    print(summary_df[["model", "train_windows", "val_windows", "best_val_loss",
                      "final_val_loss"]].to_string(index=False))
    print("\nvalidation reconstruction quantiles:")
    print(recon_df.to_string(index=False))
    return {"summary": summary_df, "recon": recon_df, "manifest": manifest}


def make_synthetic_truth(path: str, date: str, n_traj: int = 6, n: int = 60,
                         seed: int = 0) -> None:
    """Smooth synthetic GA-like trajectories in a stage-5-truth-like schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traj):
        t = 1_000.0 + 10.0 * np.arange(n)
        speed = 55.0 + rng.normal(0, 3)
        heading = np.radians(rng.uniform(0, 360)) + np.cumsum(
            np.radians(rng.normal(0, 0.5, n)))            # gentle wander
        vx = speed * np.sin(heading)
        vy = speed * np.cos(heading)
        vz = rng.normal(0, 1, n)
        x = 20_000 + np.cumsum(vx) * 10.0
        y = 15_000 + np.cumsum(vy) * 10.0
        z = 1_000 + np.cumsum(vz) * 10.0
        rows.append(pd.DataFrame({
            "trajectory_id": f"{date}_t{k}", "timestamp": t,
            "east_m": x, "north_m": y, "up_m": z,
            "ve_mps": vx, "vn_mps": vy, "vu_mps": vz,
        }))
    pd.concat(rows).to_csv(path, index=False)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        truth_dir = os.path.join(tmp, "truth")
        os.makedirs(truth_dir)
        for i, date in enumerate(["2022-01-01", "2022-01-02"]):
            make_synthetic_truth(os.path.join(truth_dir, f"radar_truth_{date}.csv"),
                                 date, seed=i)

        args = parse_args()
        args.truth_dir = truth_dir
        args.models_dir = os.path.join(tmp, "models")
        args.report_dir = os.path.join(tmp, "reports")
        args.holdout_date = ["2022-01-02"]
        args.window_len, args.stride = 10, 5
        args.epochs, args.batch_size = 2, 64
        args.hidden_dim, args.latent_dim, args.num_layers = 16, 4, 1
        args.overwrite = True

        out = run_training(args)
        for name in ["mlp_dae", "gru_ae", "tcn_ae"]:
            assert os.path.exists(os.path.join(args.models_dir, f"{name}.pt")), f"missing {name}.pt"
        for f_ in ["normalizer.json", "window_config.json", "training_manifest.json"]:
            assert os.path.exists(os.path.join(args.models_dir, f_)), f"missing {f_}"
        recon = out["recon"]
        assert np.isfinite(recon[["error_p50", "error_p99"]].to_numpy()).all(), \
            "validation reconstruction errors not finite"

    print("\nStage 12 training self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_training(args)
    print("\n12_train_sequence_priors completed successfully.")


if __name__ == "__main__":
    main()
