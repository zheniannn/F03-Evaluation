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


# =============================================================================
# Denominator-safe metric helpers (stage 17.5 hardening)
#
# False-track reduction is 1 - kept_false / total_false. When a run has NO
# false tracks in its denominator (e.g. high detection thresholds, or no
# *windowable* false tracks for the sequence methods), the reduction is
# UNDEFINED -- it is not 0 (a total failure) and not 1 (a perfect filter),
# and it is emphatically not evidence that some other stage's data is
# missing. These helpers keep that distinction explicit everywhere.
# =============================================================================

NO_FALSE_TRACKS = "undefined: no false tracks in denominator"
NO_WINDOWABLE_FALSE = "undefined: no windowable false tracks for this cell"


def safe_reduction(before: float, after: float) -> float:
    """1 - after/before, or NaN when the denominator is zero / non-finite."""
    try:
        before = float(before)
        after = float(after)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(before) or not np.isfinite(after) or before == 0:
        return np.nan
    return 1.0 - after / before


def safe_ratio(numerator: float, denominator: float) -> float:
    """numerator/denominator, or NaN when the denominator is zero / non-finite."""
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(denominator) or not np.isfinite(numerator) or denominator == 0:
        return np.nan
    return numerator / denominator


def undefined_reason(false_denominator: float, windowable: bool = False) -> str:
    """Why a false-reduction cell is undefined ('' when it is well defined)."""
    try:
        d = float(false_denominator)
    except (TypeError, ValueError):
        return NO_FALSE_TRACKS
    if not np.isfinite(d) or d == 0:
        return NO_WINDOWABLE_FALSE if windowable else NO_FALSE_TRACKS
    return ""


def summarize_defined(values) -> Tuple[int, int]:
    """(n_defined, n_undefined) for an iterable of possibly-NaN metric values."""
    arr = np.asarray(list(values), dtype=float)
    finite = int(np.isfinite(arr).sum())
    return finite, int(len(arr) - finite)
