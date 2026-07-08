"""Stage 8: constant-velocity Kalman detect-then-track baseline.

Runs a classical multi-target tracker over one stage-6 detection stream:

  * measurements are the NOISY spherical detections (meas_range/azimuth/
    elevation), converted to local Cartesian ENU for tracking;
  * per track, a 6-state constant-velocity Kalman filter
    (position + velocity, discrete white-noise acceleration process model);
  * per frame, GREEDY gated nearest-neighbor association between predicted
    track positions and detections (closest pair first, each track and
    detection used at most once, pairs beyond the gate rejected);
  * track management: unassigned detections spawn tentative tracks; a track
    is confirmed after `confirm_hits` total hits and deleted after
    `max_misses` consecutive misses (it coasts on prediction in between).

The tracker NEVER sees truth labels: is_target / trajectory_id / the truth
position are carried through to the output rows purely so
utils/track_eval.py can score the result afterwards (pos_error_m is the
distance between the POSTERIOR track position and the truth position of
the associated target detection -- evaluation only, never fed back).

Design notes (baseline simplifications, deliberate):
  * measurement noise is folded into an isotropic per-detection position
    sigma = sqrt(sigma_range^2 + (range * sigma_angle)^2) -- the true
    spherical noise is anisotropic, but this keeps R diagonal;
  * association distance is Euclidean to the predicted position (not
    Mahalanobis) with a fixed gate in metres;
  * radial velocity is not used. All of these are the first things stage 9+
    should improve.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class KalmanTrackerConfig:
    """All stage-8 tracker tunables in one place (populated from the CLI)."""
    frame_period_s: float = 10.0     # radar scan period (stage-6 frame spacing)
    gate_m: float = 2500.0           # association gate around the predicted position
    q_accel_mps2: float = 1.0        # process-noise acceleration std (GA maneuvering)
    sigma_range_m: float = 75.0      # stage-6 range noise (for R)
    sigma_angle_deg: float = 0.15    # stage-6 az/el noise (for R)
    init_speed_std_mps: float = 60.0 # velocity uncertainty for newborn tracks
    confirm_hits: int = 3            # hits needed to confirm a tentative track
    max_misses: int = 3              # consecutive misses before deletion


# Output-row schema; rows are built as tuples (not dicts) so a multi-million
# state run stays within memory.
ROW_COLUMNS = [
    "track_id", "frame_id", "timestamp",
    "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "confirmed", "hits", "misses",
    "detection_id", "assoc_is_target", "assoc_trajectory_id", "pos_error_m",
]


def spherical_to_cartesian(range_m, azimuth_rad, elevation_rad):
    """Radar spherical (range, az, el) -> local ENU Cartesian (x=E, y=N, z=U).
    Azimuth is compass-style (0 = north, pi/2 = east), matching stage 5/6."""
    r = np.asarray(range_m, dtype=float)
    az = np.asarray(azimuth_rad, dtype=float)
    el = np.asarray(elevation_rad, dtype=float)
    cos_el = np.cos(el)
    return r * cos_el * np.sin(az), r * cos_el * np.cos(az), r * np.sin(el)


class Track:
    """One constant-velocity Kalman track. State: [x y z vx vy vz]."""

    __slots__ = ("track_id", "x", "P", "hits", "misses", "confirmed")

    def __init__(self, track_id: int, position: np.ndarray, sigma_pos: float,
                 cfg: KalmanTrackerConfig):
        self.track_id = track_id
        self.x = np.concatenate([position, np.zeros(3)])
        self.P = np.diag([sigma_pos**2] * 3 + [cfg.init_speed_std_mps**2] * 3)
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, sigma_pos: float) -> None:
        # H = [I3 0]: position-only measurement
        S = self.P[:3, :3] + np.eye(3) * sigma_pos**2
        K = self.P[:, :3] @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:3])
        self.P = self.P - K @ self.P[:3, :]
        self.hits += 1
        self.misses = 0


def _cv_matrices(dt: float, q_accel: float):
    """Constant-velocity F and discrete white-noise-acceleration Q (6x6)."""
    F = np.eye(6)
    F[:3, 3:] = np.eye(3) * dt
    q = q_accel**2
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.eye(3) * (dt**4 / 4) * q
    Q[:3, 3:] = Q[3:, :3] = np.eye(3) * (dt**3 / 2) * q
    Q[3:, 3:] = np.eye(3) * dt**2 * q
    return F, Q


def run_tracker(detections: pd.DataFrame, cfg: KalmanTrackerConfig) -> pd.DataFrame:
    """Track one detection stream. `detections` must have: frame_id,
    timestamp, detection_id, meas_range_m, meas_azimuth_rad,
    meas_elevation_rad (+ is_target / trajectory_id and, when available,
    truth_range/azimuth/elevation, carried for evaluation only).
    Returns one row per (frame, live track); see ROW_COLUMNS."""
    F, Q = _cv_matrices(cfg.frame_period_s, cfg.q_accel_mps2)
    sigma_angle = np.radians(cfg.sigma_angle_deg)

    x, y, z = spherical_to_cartesian(detections["meas_range_m"],
                                     detections["meas_azimuth_rad"],
                                     detections["meas_elevation_rad"])
    det = detections.assign(_x=x, _y=y, _z=z)
    det = det.assign(_sigma=np.sqrt(cfg.sigma_range_m**2
                                    + (det["meas_range_m"].to_numpy() * sigma_angle)**2))
    # Truth positions (targets only; NaN for clutter) -- evaluation-only.
    if "truth_range_m" in det.columns:
        tx, ty, tz = spherical_to_cartesian(det["truth_range_m"],
                                            det["truth_azimuth_rad"],
                                            det["truth_elevation_rad"])
    else:
        tx = ty = tz = np.full(len(det), np.nan)
    det = det.assign(_tx=tx, _ty=ty, _tz=tz)

    tracks: List[Track] = []
    rows: List[tuple] = []
    next_id = 0

    for frame_id, frame in det.groupby("frame_id", sort=True):
        timestamp = float(frame["timestamp"].iloc[0])
        positions = frame[["_x", "_y", "_z"]].to_numpy()
        truth_pos = frame[["_tx", "_ty", "_tz"]].to_numpy()
        sigmas = frame["_sigma"].to_numpy()
        det_ids = frame["detection_id"].to_numpy()
        is_target = frame["is_target"].to_numpy()
        traj_ids = frame["trajectory_id"].to_numpy()

        # --- predict all live tracks ------------------------------------
        for t in tracks:
            t.predict(F, Q)

        # --- greedy gated nearest-neighbor association -------------------
        assigned_track: Dict[int, int] = {}   # track index -> detection index
        assigned_det: set = set()
        if tracks and len(frame):
            pred = np.stack([t.x[:3] for t in tracks])
            dists = np.linalg.norm(pred[:, None, :] - positions[None, :, :], axis=2)
            order = np.argsort(dists, axis=None)
            for flat in order:
                ti, di = np.unravel_index(flat, dists.shape)
                if dists[ti, di] > cfg.gate_m:
                    break                      # all remaining pairs are farther
                if ti in assigned_track or di in assigned_det:
                    continue
                assigned_track[ti] = di
                assigned_det.add(di)

        def state_row(t: Track, di: Optional[int]) -> tuple:
            if di is None:
                return (t.track_id, frame_id, timestamp,
                        t.x[0], t.x[1], t.x[2], t.x[3], t.x[4], t.x[5],
                        int(t.confirmed), t.hits, t.misses, -1, -1, "", np.nan)
            err = (float(np.linalg.norm(t.x[:3] - truth_pos[di]))
                   if np.isfinite(truth_pos[di]).all() else np.nan)
            return (t.track_id, frame_id, timestamp,
                    t.x[0], t.x[1], t.x[2], t.x[3], t.x[4], t.x[5],
                    int(t.confirmed), t.hits, t.misses,
                    int(det_ids[di]), int(is_target[di]), traj_ids[di], err)

        # --- update / coast / spawn --------------------------------------
        survivors: List[Track] = []
        for ti, t in enumerate(tracks):
            if ti in assigned_track:
                di = assigned_track[ti]
                t.update(positions[di], sigmas[di])
                if t.hits >= cfg.confirm_hits:
                    t.confirmed = True
                rows.append(state_row(t, di))
                survivors.append(t)
            else:
                t.misses += 1
                if t.misses <= cfg.max_misses:
                    rows.append(state_row(t, None))
                    survivors.append(t)
                # else: deleted silently
        tracks = survivors

        for di in range(len(frame)):
            if di not in assigned_det:
                t = Track(next_id, positions[di], sigmas[di], cfg)
                next_id += 1
                rows.append(state_row(t, di))
                tracks.append(t)

    return pd.DataFrame(rows, columns=ROW_COLUMNS)
