"""Stage 12: trajectory-window construction and normalization.

Turns per-trajectory (or per-track) state sequences into fixed-length,
origin/heading-normalized feature windows for the sequence autoencoders:

  * features per time step: dx, dy, dz (position relative to the window's
    first point), vx, vy, vz (ENU velocity), speed, vertical_speed (= vz),
    turn_rate (wrapped heading difference / dt, in DEG/S);
  * each window is translated so its first position is the origin and
    rotated about the vertical axis so the initial horizontal velocity
    points along +x -- the models learn motion SHAPE, not absolute
    location or bearing (skipped when the initial horizontal speed is
    below MIN_HEADING_SPEED_MPS; then the first sufficiently fast step is
    used, or no rotation at all);
  * finally standardized with the TRAINING-set mean/std (normalizer.json).

The same code path builds training windows from stage-5 truth (grouped by
trajectory_id) and scoring windows from stage-8 tracks (grouped by
track_id), so train and score representations cannot drift apart.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

FEATURES = ["dx", "dy", "dz", "vx", "vy", "vz", "speed", "vertical_speed", "turn_rate"]
MIN_HEADING_SPEED_MPS = 5.0     # below this, don't trust the heading for rotation

TRUTH_PATTERN = re.compile(r"radar_truth_(\d{4}-\d{2}-\d{2})\.csv$")
STAGE04_PATTERN = re.compile(r"states_(\d{4}-\d{2}-\d{2})_conventionalGA_trajectories_10s\.csv$")


@dataclass
class WindowConfig:
    window_len: int = 20
    stride: int = 5
    features: List[str] = field(default_factory=lambda: list(FEATURES))


# =============================================================================
# Discovery / parsing
# =============================================================================

def parse_date_from_filename(path: str) -> Optional[str]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else None


def parse_threshold_from_filename(path: str) -> Optional[float]:
    m = re.search(r"_thr_(.+)dB\.csv$", os.path.basename(path))
    return float(m.group(1).replace("m", "-").replace("p", ".")) if m else None


def discover_truth_files(truth_dir: str) -> List[Tuple[str, str]]:
    out = []
    if os.path.isdir(truth_dir):
        for name in sorted(os.listdir(truth_dir)):
            m = TRUTH_PATTERN.search(name)
            if m:
                out.append((m.group(1), os.path.join(truth_dir, name)))
    return out


def discover_stage04_files(stage04_dir: str) -> List[Tuple[str, str]]:
    out = []
    if os.path.isdir(stage04_dir):
        for name in sorted(os.listdir(stage04_dir)):
            m = STAGE04_PATTERN.search(name)
            if m:
                out.append((m.group(1), os.path.join(stage04_dir, name)))
    return out


def discover_track_files(tracks_dir: str):
    """Stage-8 runs, both schemas -- delegates to the stage-9 loader helpers."""
    from utils.track_physics_score import discover_stage08_runs
    return discover_stage08_runs(tracks_dir)


# =============================================================================
# Loading
# =============================================================================

def load_truth_for_windows(path: str) -> pd.DataFrame:
    """Stage-5 truth -> canonical (group_id, timestamp, x, y, z, vx, vy, vz).
    Velocities come from the truth's ENU velocity columns when present, else
    finite differences per trajectory."""
    header = list(pd.read_csv(path, nrows=0).columns)
    pos = {"x": "east_m" if "east_m" in header else "x_m",
           "y": "north_m" if "north_m" in header else "y_m",
           "z": "up_m" if "up_m" in header else "z_m"}
    vel = {"vx": "ve_mps", "vy": "vn_mps", "vz": "vu_mps"}
    have_vel = all(c in header for c in vel.values())

    usecols = ["trajectory_id", "timestamp"] + list(pos.values())
    usecols += list(vel.values()) if have_vel else []
    df = pd.read_csv(path, usecols=usecols, dtype={"trajectory_id": str})
    out = pd.DataFrame({
        "group_id": df["trajectory_id"], "timestamp": df["timestamp"],
        "x": df[pos["x"]], "y": df[pos["y"]], "z": df[pos["z"]],
    })
    if have_vel:
        out["vx"], out["vy"], out["vz"] = df[vel["vx"]], df[vel["vy"]], df[vel["vz"]]
    else:
        out = compute_motion_columns(out)
    return out


def load_stage04_for_windows(path: str) -> pd.DataFrame:
    """Fallback training source: stage-4 trajectories (lat/lon/alt) mapped to
    a local flat-earth frame per file, velocities by finite differences."""
    header = list(pd.read_csv(path, nrows=0).columns)
    lat = "lat_smooth" if "lat_smooth" in header else "lat_interp"
    lon = "lon_smooth" if "lon_smooth" in header else "lon_interp"
    alt = "alt_smooth" if "alt_smooth" in header else "alt_interp"
    df = pd.read_csv(path, usecols=["trajectory_id", "timestamp", lat, lon, alt],
                     dtype={"trajectory_id": str})
    lat0 = float(df[lat].median())
    r = 6_371_000.0
    out = pd.DataFrame({
        "group_id": df["trajectory_id"], "timestamp": df["timestamp"],
        "x": r * np.cos(np.radians(lat0)) * np.radians(df[lon]),
        "y": r * np.radians(df[lat]),
        "z": df[alt],
    })
    return compute_motion_columns(out)


def load_tracks_for_windows(path: str, date: str, threshold_db: float,
                            detections_dir: str) -> pd.DataFrame:
    """Stage-8 tracks (either schema) -> canonical frame plus the
    evaluation-only label columns, via the stage-9 loader."""
    from utils.track_physics_score import load_run
    df = load_run(path, date, threshold_db, detections_dir)
    df = df.rename(columns={"track_id": "group_id"})
    return df


def compute_motion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add vx/vy/vz by per-group finite differences (np.gradient) when the
    source has positions only."""
    t = df["timestamp"].to_numpy(dtype=float)
    out = {c: np.full(len(df), np.nan) for c in ("vx", "vy", "vz")}
    coords = {c: df[c].to_numpy(dtype=float) for c in ("x", "y", "z")}
    for _, pos in df.groupby("group_id", sort=False).indices.items():
        if len(pos) >= 3:
            for vc, xc in (("vx", "x"), ("vy", "y"), ("vz", "z")):
                out[vc][pos] = np.gradient(coords[xc][pos], t[pos])
        elif len(pos) == 2:
            dt = t[pos[1]] - t[pos[0]]
            if dt > 0:
                for vc, xc in (("vx", "x"), ("vy", "y"), ("vz", "z")):
                    out[vc][pos] = (coords[xc][pos[1]] - coords[xc][pos[0]]) / dt
    for c, v in out.items():
        df[c] = v
    return df


# =============================================================================
# Window construction
# =============================================================================

def _group_feature_matrix(t, x, y, z, vx, vy, vz) -> np.ndarray:
    """Per-time-step raw features for one group (before per-window origin
    shift and rotation): columns match FEATURES with dx/dy/dz still absolute."""
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    heading = np.degrees(np.arctan2(vx, vy))              # x=east, y=north
    dh = (np.diff(heading) + 180.0) % 360.0 - 180.0
    dt = np.diff(t)
    turn = np.zeros(len(t))
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(dt > 0, dh / dt, 0.0)
    turn[1:] = step
    if len(turn) > 1:
        turn[0] = turn[1]
    hspeed = np.sqrt(vx**2 + vy**2)
    turn[np.r_[hspeed[0], hspeed[:-1]] < MIN_HEADING_SPEED_MPS] = 0.0
    return np.column_stack([x, y, z, vx, vy, vz, speed, vz, turn])


def normalize_window_origin_heading(window: np.ndarray) -> np.ndarray:
    """Shift the window so the first position is the origin and rotate the
    horizontal plane so the initial heading points along +x. Operates on the
    raw feature layout produced by _group_feature_matrix."""
    w = window.copy()
    w[:, 0:3] -= w[0, 0:3]                                # dx, dy, dz

    vx0, vy0 = w[0, 3], w[0, 4]
    if np.hypot(vx0, vy0) < MIN_HEADING_SPEED_MPS:
        fast = np.hypot(w[:, 3], w[:, 4]) >= MIN_HEADING_SPEED_MPS
        idx = int(np.argmax(fast)) if fast.any() else -1
        if idx >= 0:
            vx0, vy0 = w[idx, 3], w[idx, 4]
        else:
            return w                                      # no reliable heading: no rotation
    ang = np.arctan2(vy0, vx0)                            # rotate so velocity -> +x
    c, s = np.cos(-ang), np.sin(-ang)
    for cx, cy in ((0, 1), (3, 4)):                       # positions and velocities
        px, py = w[:, cx].copy(), w[:, cy].copy()
        w[:, cx] = c * px - s * py
        w[:, cy] = s * px + c * py
    return w


def build_windows_by_group(df: pd.DataFrame, group_col: str, cfg: WindowConfig,
                           max_windows: Optional[int] = None,
                           rng: Optional[np.random.Generator] = None,
                           return_group_index: bool = False):
    """Slide windows over every group. Returns float32 array (N, L, F) --
    origin/heading-normalized but NOT yet standardized -- plus, optionally,
    the group id of each window. Groups shorter than window_len yield none."""
    L, S = cfg.window_len, cfg.stride
    t_all = df["timestamp"].to_numpy(dtype=float)
    cols = {c: df[c].to_numpy(dtype=float) for c in ("x", "y", "z", "vx", "vy", "vz")}

    windows: List[np.ndarray] = []
    group_ids: List = []
    for gid, pos in df.groupby(group_col, sort=False).indices.items():
        if len(pos) < L:
            continue
        order = np.argsort(t_all[pos], kind="mergesort")
        idx = pos[order]
        feats = _group_feature_matrix(t_all[idx], *[cols[c][idx] for c in
                                                    ("x", "y", "z", "vx", "vy", "vz")])
        if not np.isfinite(feats).all():
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        for start in range(0, len(idx) - L + 1, S):
            windows.append(normalize_window_origin_heading(feats[start:start + L]))
            group_ids.append(gid)

    if not windows:
        empty = np.empty((0, L, len(cfg.features)), dtype=np.float32)
        return (empty, []) if return_group_index else empty
    arr = np.stack(windows).astype(np.float32)
    if max_windows is not None and len(arr) > max_windows:
        rng = rng or np.random.default_rng(0)
        keep = rng.choice(len(arr), size=max_windows, replace=False)
        keep.sort()
        arr = arr[keep]
        group_ids = [group_ids[i] for i in keep]
    return (arr, group_ids) if return_group_index else arr


# =============================================================================
# Normalizer (training-set standardization)
# =============================================================================

def fit_normalizer(windows: np.ndarray, cfg: WindowConfig) -> Dict:
    flat = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)
    return {"features": list(cfg.features),
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "window_len": cfg.window_len, "stride": cfg.stride,
            "origin_heading_normalized": True}


def apply_normalizer(windows: np.ndarray, normalizer: Dict) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    return (windows - mean) / std


def save_normalizer(path: str, normalizer: Dict) -> None:
    with open(path, "w") as f:
        json.dump(normalizer, f, indent=1)


def load_normalizer(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Misc
# =============================================================================

def split_train_val_by_date(files: List[Tuple[str, str]],
                            holdout_dates: Optional[List[str]]):
    holdout = set(holdout_dates or [])
    train = [(d, p) for d, p in files if d not in holdout]
    val = [(d, p) for d, p in files if d in holdout]
    return train, val


def sample_windows_deterministic(windows: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(windows) <= max_n:
        return windows
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(windows), size=max_n, replace=False)
    keep.sort()
    return windows[keep]
