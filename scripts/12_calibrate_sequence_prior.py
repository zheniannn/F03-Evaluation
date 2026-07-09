"""Stage 12.5 (standalone): build a noise-matched calibration for the stage-12
sequence-prior scores from high-purity stage-8 true tracks, without scoring.

This produces the same calibration artifacts that
``12_score_tracks_sequence_prior.py --calibration-mode track_purity`` builds
internally, so the scoring script can consume the JSON afterwards. The stage-12
autoencoder weights are never touched -- only the reconstruction-error quantiles
that define the score band are (re)computed. Truth labels are used ONLY to select
the calibration tracks.

Usage:
    python scripts/12_calibrate_sequence_prior.py \
        --calibration-tracks-dir data/active/tracks_kalman \
        --calibration-date 2022-06-06 \
        --calibration-threshold-db 3 6 9 12 \
        --calibration-min-target-fraction 0.95 --calibration-min-purity 0.95 \
        --calibration-output reports/stage12_sequence_priors/calibration/sequence_track_calibration.json
"""

import argparse
import importlib.util
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import REPO_ROOT
from utils.sequence_windows import WindowConfig, load_normalizer
from utils.sequence_prior_score import write_calibration_files


def _load_scoring_module():
    spec = importlib.util.spec_from_file_location(
        "score12", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "12_score_tracks_sequence_prior.py"))
    score12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score12)
    return score12


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a track-purity (noise-matched) calibration for stage-12 scores.")
    p.add_argument("--calibration-tracks-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "tracks_kalman"))
    p.add_argument("--models-dir", type=str,
                   default=os.path.join(REPO_ROOT, "models", "sequence_priors"))
    p.add_argument("--detections-dir", type=str,
                   default=os.path.join(REPO_ROOT, "data", "active", "sim_detections_relocated"))
    p.add_argument("--model", type=str, nargs="+", default=["mlp_dae", "gru_ae", "tcn_ae"])
    p.add_argument("--calibration-date", type=str, nargs="*", default=None)
    p.add_argument("--calibration-threshold-db", type=float, nargs="+", default=None)
    p.add_argument("--calibration-min-target-fraction", type=float, default=0.95)
    p.add_argument("--calibration-min-purity", type=float, default=0.95)
    p.add_argument("--calibration-max-false-tracks", type=int, default=0)
    p.add_argument("--min-track-hits", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window-len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--calibration-output", type=str,
                   default=os.path.join(REPO_ROOT, "reports", "stage12_sequence_priors",
                                        "calibration", "sequence_track_calibration.json"),
                   help="Where the calibration JSON is written. This standalone tool "
                        "DELIBERATELY defaults to the canonical stage-12 calibration path -- "
                        "that is its purpose. Orchestrators must not call it without setting "
                        "this to a path inside their own report directory.")
    # attributes the shared builder reads but that have no meaning standalone
    p.add_argument("--date", type=str, nargs="*", default=None)
    p.add_argument("--threshold-db", type=float, nargs="+", default=None)
    return p.parse_args()


def main():
    from utils.sequence_models import load_model, resolve_device

    args = parse_args()
    score12 = _load_scoring_module()
    device = resolve_device(args.device)

    normalizer = load_normalizer(os.path.join(args.models_dir, "normalizer.json"))
    with open(os.path.join(args.models_dir, "window_config.json")) as f:
        window_config = json.load(f)
    wcfg = WindowConfig(window_len=args.window_len or window_config["window_len"],
                        stride=args.stride or window_config["stride"],
                        features=window_config["features"])

    models = {}
    input_dim = len(wcfg.features)
    for name in args.model:
        ckpt = os.path.join(args.models_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Model checkpoint missing: {ckpt} -- run training first")
        models[name] = load_model(ckpt, name, input_dim, wcfg.window_len, device)

    cal_track, meta = score12.build_track_purity_calibration(
        models, normalizer, wcfg, args, device)
    calib_dir = os.path.dirname(args.calibration_output)
    json_path, csv_path = write_calibration_files(calib_dir, cal_track, meta)

    print(f"\ncalibration JSON: {json_path}")
    print(f"calibration CSV:  {csv_path}")
    print("\nper-model calibration band (p50 -> p99, n_tracks / n_windows):")
    for m, d in cal_track.items():
        print(f"  {m:<10} {d['error_p50']:.5f} -> {d['error_p99']:.5f}  "
              f"({d['n_calibration_tracks']} tracks / {d['n_calibration_windows']} windows)")
    print("\n12_calibrate_sequence_prior completed successfully.")


if __name__ == "__main__":
    main()
