"""Shared helpers for the F03 evaluation stages.

Everything here existed as near-identical copies inside the stage modules;
this module is the single home for repo-root resolution, stage-6 detection
filename parsing, and Markdown table rendering.
"""

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DETECTION_PATTERN = re.compile(r"detections_(\d{4}-\d{2}-\d{2})_thr_(.+)dB\.csv$")


def repo_path(*parts: str) -> str:
    """Absolute path under the F03-Evaluation repo root."""
    return os.path.join(REPO_ROOT, *parts)


def threshold_to_token(threshold_db: float) -> str:
    """Filename-safe threshold token, matching stage 6: 6.0 -> '6p0', -5.0 -> 'm5p0'."""
    return f"{threshold_db:.1f}".replace("-", "m").replace(".", "p")


def token_to_threshold(token: str) -> float:
    """Inverse of threshold_to_token: 'm5p0' -> -5.0."""
    return float(token.replace("m", "-").replace("p", "."))


def parse_detection_filename(path: str) -> Optional[Tuple[str, float]]:
    """detections_2022-06-06_thr_m5p0dB.csv -> ('2022-06-06', -5.0), else None."""
    m = DETECTION_PATTERN.search(os.path.basename(path))
    if not m:
        return None
    return m.group(1), token_to_threshold(m.group(2))


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> List[str]:
    """Render a DataFrame as GitHub-flavored Markdown table lines.

    Non-integral floats use float_fmt; integral floats and ints get
    thousands separators; everything else is str()."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---:"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float) and np.isfinite(v) and not float(v).is_integer():
                cells.append(float_fmt.format(v))
            elif isinstance(v, float) and np.isfinite(v):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines
