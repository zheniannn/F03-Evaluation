"""Entry point: stage 10 -- learn empirical ADS-B motion priors from the
F01 stage-4 fixed-wing GA trajectories.

Fits transparent histogram priors (speed, |acceleration|, vector
acceleration, |turn rate|, |vertical speed|) and writes them as JSON under
models/motion_priors/, plus compact reports and plots. Stage 10 only LEARNS
priors -- no track scoring (that is stage 11), no neural networks, no VAE,
and nothing in stages 7-9 changes.

Usage:
    python scripts/10_learn_adsb_motion_priors.py --overwrite
    python scripts/10_learn_adsb_motion_priors.py --holdout-date 2022-06-27 --overwrite
    python scripts/10_learn_adsb_motion_priors.py --self-test
"""

import argparse
import os
import sys

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.motion_priors import (
    MotionPriorConfig,
    learn_motion_priors,
    make_plots,
    run_validation_gate,
    self_test,
    write_model_files,
    write_reports,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn empirical ADS-B motion priors from stage-4 trajectories (no track scoring).")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "active", "trajectories_10s"))
    parser.add_argument("--models-dir", type=str,
                        default=os.path.join(REPO_ROOT, "models", "motion_priors"))
    parser.add_argument("--report-dir", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage10_adsb_motion_priors"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--sample-per-feature", type=int, default=1_000_000,
                        help="Reservoir cap per feature for quantiles/correlation "
                             "(counts and histograms stay exact).")
    parser.add_argument("--hist-bins", type=int, default=200)
    parser.add_argument("--min-speed-mps", type=float, default=5.0,
                        help="Drop near-stationary samples from prior fitting (default: 5).")
    parser.add_argument("--max-speed-mps", type=float, default=160.0,
                        help="Drop implausible/noisy speed samples from fitting (default: 160).")
    parser.add_argument("--max-abs-vertical-speed-mps", type=float, default=40.0)
    parser.add_argument("--max-abs-turn-rate-deg-s", type=float, default=30.0)
    parser.add_argument("--max-accel-mps2", type=float, default=15.0)
    parser.add_argument("--train-date", type=str, nargs="*", default=None,
                        help="Use only these dates for training (default: all available).")
    parser.add_argument("--holdout-date", type=str, nargs="*", default=None,
                        help="Report these dates' statistics but exclude them from the fit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a tiny synthetic end-to-end check (no real data needed) and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    cfg = MotionPriorConfig(
        input_dir=args.input_dir,
        models_dir=args.models_dir,
        report_dir=args.report_dir,
        chunksize=args.chunksize,
        sample_per_feature=args.sample_per_feature,
        hist_bins=args.hist_bins,
        min_speed_mps=args.min_speed_mps,
        max_speed_mps=args.max_speed_mps,
        max_abs_vertical_speed_mps=args.max_abs_vertical_speed_mps,
        max_abs_turn_rate_deg_s=args.max_abs_turn_rate_deg_s,
        max_accel_mps2=args.max_accel_mps2,
        train_dates=args.train_date,
        holdout_dates=args.holdout_date,
        overwrite=args.overwrite,
        no_plots=args.no_plots,
    )

    key_output = os.path.join(cfg.models_dir, "motion_prior_manifest.json")
    if os.path.exists(key_output) and not cfg.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to regenerate): {key_output}")

    state = learn_motion_priors(cfg)
    priors = write_model_files(state, cfg)
    report_path = write_reports(state, priors, cfg)
    if not cfg.no_plots:
        make_plots(priors, os.path.join(cfg.report_dir, "plots"))
    run_validation_gate(state, priors, cfg)

    print(f"\npriors:  {os.path.abspath(cfg.models_dir)}")
    print(f"report:  {os.path.abspath(report_path)}")
    print("\nfitted features:")
    for name, prior in priors.items():
        q = prior["quantiles"]
        print(f"  {name:<26} n={prior['n_samples_used']:>11,}  "
              f"p50={q['p50']:.3f}  p95={q['p95']:.3f}  p99={q['p99']:.3f} {prior['units']}")

    print("\n10_learn_adsb_motion_priors completed successfully.")


if __name__ == "__main__":
    main()
