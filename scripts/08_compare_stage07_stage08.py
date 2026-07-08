"""Join stage-7 frame-level metrics with stage-8 track-level metrics into a
side-by-side comparison table (CSV + Markdown).

The two stages measure DIFFERENT things: stage 7's empirical Pd is the
per-frame probability that a single truth sample produces a detection;
stage 8's track detection rate is the fraction of trackable trajectories
recovered by at least one true track. They must not be equated -- the
comparison shows whether temporal continuity recovers trajectories despite
missed detections and clutter.

Usage:
    python scripts/08_compare_stage07_stage08.py --date 2022-06-06 \
        --output-prefix reports/stage08_kalman_baseline/stage07_vs_stage08_2022-06-06
    python scripts/08_compare_stage07_stage08.py --self-test
"""

import argparse
import os
import sys
import tempfile

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# stage07 source column -> comparison column
STAGE07_COLUMNS = {
    "empirical_pd": "stage07_empirical_pd",
    "false_alarm_per_frame": "stage07_false_alarm_per_frame",
    "target_detections": "stage07_target_detections",
    "clutter_detections": "stage07_clutter_detections",
}
STAGE08_COLUMNS = {
    "track_detection_rate": "stage08_track_detection_rate",
    "tracks_confirmed": "stage08_confirmed_tracks",
    "true_tracks": "stage08_confirmed_true_tracks",
    "false_tracks": "stage08_confirmed_false_tracks",
    "target_assignment_rate": "stage08_target_assignment_rate",
    "clutter_assignment_rate": "stage08_clutter_assignment_rate",
    "mean_track_purity": "stage08_mean_track_purity",
    "median_track_purity": "stage08_median_track_purity",
    "mean_position_rmse_m": "stage08_mean_position_rmse_m",
    "median_position_rmse_m": "stage08_median_position_rmse_m",
}


def build_comparison(stage07_path: str, stage08_path: str, date: str = None) -> pd.DataFrame:
    s7 = pd.read_csv(stage07_path)
    s8 = pd.read_csv(stage08_path)
    if date is not None:
        s7 = s7[s7["date"] == date]
        s8 = s8[s8["date"] == date]

    s7 = s7[["date", "threshold_db"] + list(STAGE07_COLUMNS)].rename(columns=STAGE07_COLUMNS)
    s8 = s8[["date", "threshold_db"] + list(STAGE08_COLUMNS)].rename(columns=STAGE08_COLUMNS)
    merged = s7.merge(s8, on=["date", "threshold_db"], how="inner")
    if merged.empty:
        raise SystemExit("No overlapping (date, threshold_db) rows between stage 7 and stage 8 inputs")
    return merged.sort_values(["date", "threshold_db"]).reset_index(drop=True)


def write_outputs(df: pd.DataFrame, output_prefix: str, date: str = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    df.to_csv(output_prefix + ".csv", index=False)

    cols = list(df.columns)
    lines = [
        "# Stage 07 vs Stage 08 comparison" + (f" — {date}" if date else ""),
        "",
        "- **Stage 07** is frame-level threshold-only detection: `empirical_pd`",
        "  is the per-frame probability that one truth sample yields a detection.",
        "- **Stage 08** is track-level detect-then-track evaluation:",
        "  `track_detection_rate` is the fraction of trackable trajectories",
        "  recovered by at least one true track.",
        "- **Do not directly equate frame Pd with track detection rate** — they",
        "  measure different things at different granularities.",
        "- The comparison shows whether temporal continuity recovers",
        "  trajectories despite missed detections and clutter: a threshold with",
        "  modest frame Pd can still achieve near-complete trajectory coverage",
        "  because a track only needs enough hits to confirm and survive",
        "  coasting, not a detection every frame.",
        "",
    ]
    if date:
        lines += [f"These results are for {date} only and should be treated as a",
                  "one-day baseline before the full 4-day sweep.", ""]

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---:"] * len(cols)) + "|"
    lines += [header, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not float(v).is_integer():
                cells.append(f"{v:.4f}")
            elif isinstance(v, float):
                cells.append(f"{int(v):,}")
            else:
                cells.append(f"{v:,}" if isinstance(v, int) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    with open(output_prefix + ".md", "w") as f:
        f.write("\n".join(lines))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s7 = pd.DataFrame([
            dict(date="2022-01-01", threshold_db=0.0, empirical_pd=0.6,
                 false_alarm_per_frame=0.8, target_detections=6, clutter_detections=4),
            dict(date="2022-01-01", threshold_db=6.0, empirical_pd=0.3,
                 false_alarm_per_frame=0.2, target_detections=3, clutter_detections=1),
        ])
        s8 = pd.DataFrame([
            dict(date="2022-01-01", threshold_db=0.0, track_detection_rate=1.0,
                 tracks_confirmed=3, true_tracks=2, false_tracks=1,
                 target_assignment_rate=0.9, clutter_assignment_rate=0.1,
                 mean_track_purity=0.95, median_track_purity=0.95,
                 mean_position_rmse_m=40.0, median_position_rmse_m=38.0),
            dict(date="2022-01-01", threshold_db=6.0, track_detection_rate=0.5,
                 tracks_confirmed=1, true_tracks=1, false_tracks=0,
                 target_assignment_rate=0.6, clutter_assignment_rate=0.0,
                 mean_track_purity=1.0, median_track_purity=1.0,
                 mean_position_rmse_m=45.0, median_position_rmse_m=45.0),
        ])
        p7 = os.path.join(tmp, "s7.csv")
        p8 = os.path.join(tmp, "s8.csv")
        s7.to_csv(p7, index=False)
        s8.to_csv(p8, index=False)

        prefix = os.path.join(tmp, "cmp")
        df = build_comparison(p7, p8, date="2022-01-01")
        write_outputs(df, prefix, date="2022-01-01")

        assert os.path.exists(prefix + ".csv") and os.path.exists(prefix + ".md")
        expected = ["date", "threshold_db"] + list(STAGE07_COLUMNS.values()) + list(STAGE08_COLUMNS.values())
        assert list(df.columns) == expected, f"unexpected columns: {list(df.columns)}"
        assert len(df) == 2
        text = open(prefix + ".md").read()
        assert "Do not directly equate" in text and "temporal continuity" in text

    print("Stage 07 vs Stage 08 comparison self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join stage-7 and stage-8 metrics into a comparison table.")
    parser.add_argument("--stage07", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage07_threshold_only",
                                             "threshold_by_day.csv"))
    parser.add_argument("--stage08", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline",
                                             "kalman_metrics_by_day.csv"))
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--output-prefix", type=str,
                        default=os.path.join(REPO_ROOT, "reports", "stage08_kalman_baseline",
                                             "stage07_vs_stage08"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    df = build_comparison(args.stage07, args.stage08, args.date)
    write_outputs(df, args.output_prefix, args.date)
    print(f"written: {args.output_prefix}.csv and .md\n")
    show = df[["date", "threshold_db", "stage07_empirical_pd", "stage07_false_alarm_per_frame",
               "stage08_track_detection_rate", "stage08_confirmed_true_tracks",
               "stage08_confirmed_false_tracks", "stage08_clutter_assignment_rate",
               "stage08_median_position_rmse_m"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
