"""Stage 10: learn empirical ADS-B motion priors from stage-4 trajectories.

Fits data-derived motion plausibility distributions (speed, |acceleration|,
vector acceleration, |turn rate|, |vertical speed|) from the cleaned,
uniformly resampled fixed-wing GA trajectories produced by F01 stage 4.
The priors are histogram densities + quantiles written as JSON, intended to
replace the hand-tuned penalty knees of stage 9.

Stage 10 ONLY learns priors: no track scoring (stage 11), no neural
networks, no VAE. Nothing from stages 7-9 is modified.

Accuracy/memory design: files are read in chunks with trajectory carry-over
so per-trajectory vertical speed is exact; per-feature sample COUNTS and
HISTOGRAMS are exact (fixed bin ranges given by the fitting filters);
QUANTILES and the joint-prior sample come from a seeded streaming reservoir
(capped at --sample-per-feature), which is statistically fair across days
and disclosed in the manifest.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common import md_table

STAGE04_PATTERN = re.compile(r"states_(\d{4}-\d{2}-\d{2})_conventionalGA_trajectories_10s\.csv$")

QUANTILE_KEYS = {
    "p001": 0.1, "p01": 1, "p05": 5, "p10": 10, "p25": 25, "p50": 50,
    "p75": 75, "p90": 90, "p95": 95, "p99": 99, "p995": 99.5, "p999": 99.9,
}

LOG_LIKELIHOOD_FLOOR = 1e-12
RESERVOIR_SEED = 20220606
DAY_RESERVOIR_CAP = 200_000
JOINT_RESERVOIR_CAP = 500_000

# Stage-9 hand-tuned knees, for the report-only comparison.
STAGE09_KNEES = [
    ("speed_mps p95", "good <= 110, bad >= 160 m/s"),
    ("accel p95", "good <= 3.0, bad >= 8.0 m/s^2"),
    ("turn rate p95", "good <= 5, bad >= 15 deg/s"),
    ("vertical speed p95", "good <= 10, bad >= 25 m/s"),
]


@dataclass
class MotionPriorConfig:
    """All stage-10 tunables in one place (populated from the CLI)."""
    input_dir: str
    models_dir: str
    report_dir: str
    chunksize: int = 1_000_000
    sample_per_feature: int = 1_000_000
    hist_bins: int = 200
    min_speed_mps: float = 5.0        # drop near-stationary samples from fitting
    max_speed_mps: float = 160.0      # drop implausible/noisy speed samples
    max_abs_vertical_speed_mps: float = 40.0
    max_abs_turn_rate_deg_s: float = 30.0
    max_accel_mps2: float = 15.0
    train_dates: Optional[List[str]] = None
    holdout_dates: Optional[List[str]] = None
    overwrite: bool = False
    no_plots: bool = False


def feature_definitions(cfg: MotionPriorConfig) -> Dict[str, Dict]:
    """Fitted features: units + [lo, hi] fitting filter (also the exact
    histogram range)."""
    return {
        "speed_mps": {"units": "m/s", "lo": cfg.min_speed_mps, "hi": cfg.max_speed_mps},
        "accel_abs_mps2": {"units": "m/s^2", "lo": 0.0, "hi": cfg.max_accel_mps2},
        "accel_vector_mps2": {"units": "m/s^2", "lo": 0.0, "hi": cfg.max_accel_mps2},
        "turn_rate_abs_deg_s": {"units": "deg/s", "lo": 0.0, "hi": cfg.max_abs_turn_rate_deg_s},
        "vertical_speed_abs_mps": {"units": "m/s", "lo": 0.0, "hi": cfg.max_abs_vertical_speed_mps},
    }


SIGNED_REPORT_FEATURES = {          # reported in feature summary, no prior JSON
    "vertical_speed_mps": "m/s",
    "turn_rate_deg_s": "deg/s",
    "accel_mps2": "m/s^2",
}


# =============================================================================
# Discovery / loading
# =============================================================================

def parse_stage04_filename(path: str) -> Optional[str]:
    """Return the YYYY-MM-DD date from a stage-4 trajectories filename, else None."""
    m = STAGE04_PATTERN.search(os.path.basename(path))
    return m.group(1) if m else None


def discover_stage04_files(input_dir: str) -> List[Tuple[str, str]]:
    """Sorted (date, path) pairs for stage-4 trajectory CSVs in input_dir."""
    out = []
    if not os.path.isdir(input_dir):
        return out
    for name in sorted(os.listdir(input_dir)):
        date = parse_stage04_filename(name)
        if date:
            out.append((date, os.path.join(input_dir, name)))
    return out


def load_stage04_columns(path: str) -> Dict[str, Optional[str]]:
    """Inspect the header and choose which columns to use.

    Requires trajectory_id, timestamp, an altitude column, and speed_mps.
    Prefers *_smooth positions and vector acceleration when available."""
    columns = list(pd.read_csv(path, nrows=0).columns)

    alt_col = next((c for c in ("alt_smooth", "alt_interp") if c in columns), None)
    missing = [c for c in ("trajectory_id", "timestamp") if c not in columns]
    if alt_col is None:
        missing.append("alt_smooth/alt_interp")
    if "speed_mps" not in columns:
        missing.append("speed_mps")
    if missing:
        raise ValueError(f"Stage-4 file {path} is missing required column(s): {missing}")

    return {
        "alt": alt_col,
        "speed": "speed_mps",
        "accel": "accel_mps2" if "accel_mps2" in columns else None,
        "accel_vector": "accel_vector_mps2" if "accel_vector_mps2" in columns else None,
        "turn_rate": "turn_rate_deg_s" if "turn_rate_deg_s" in columns else None,
    }


def iter_complete_trajectories(path: str, usecols: List[str], chunksize: int):
    """Yield chunks whose trajectories are COMPLETE: the possibly-truncated
    last trajectory of each raw chunk is carried into the next one, so
    per-trajectory gradients are exact."""
    carry = None
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"trajectory_id": str},
                             chunksize=chunksize):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_id = chunk["trajectory_id"].iloc[-1]
        tail_mask = chunk["trajectory_id"] == last_id
        # hold back the final trajectory unless it is the only one in the chunk
        if tail_mask.all():
            carry = chunk
            continue
        carry = chunk[tail_mask].copy()
        yield chunk[~tail_mask]
    if carry is not None and len(carry):
        yield carry


# =============================================================================
# Feature computation
# =============================================================================

def compute_motion_features_chunk(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Row-aligned motion features for one complete-trajectory chunk.

    Vertical speed is np.gradient(alt, timestamp) per trajectory (>= 3
    points; plain diff-based slope for 2 points; NaN for singletons)."""
    alt = df[cols["alt"]].to_numpy(dtype=float)
    t = df["timestamp"].to_numpy(dtype=float)
    vspeed = np.full(len(df), np.nan)

    for _, pos in df.groupby("trajectory_id", sort=False).indices.items():
        if len(pos) >= 3:
            vspeed[pos] = np.gradient(alt[pos], t[pos])
        elif len(pos) == 2:
            dt = t[pos[1]] - t[pos[0]]
            if dt > 0:
                vspeed[pos] = (alt[pos[1]] - alt[pos[0]]) / dt

    out = pd.DataFrame({
        "speed_mps": df[cols["speed"]].to_numpy(dtype=float),
        "vertical_speed_mps": vspeed,
        "vertical_speed_abs_mps": np.abs(vspeed),
    })
    if cols["accel"]:
        out["accel_mps2"] = df[cols["accel"]].to_numpy(dtype=float)
        out["accel_abs_mps2"] = np.abs(out["accel_mps2"])
    if cols["accel_vector"]:
        out["accel_vector_mps2"] = df[cols["accel_vector"]].to_numpy(dtype=float)
    if cols["turn_rate"]:
        out["turn_rate_deg_s"] = df[cols["turn_rate"]].to_numpy(dtype=float)
        out["turn_rate_abs_deg_s"] = np.abs(out["turn_rate_deg_s"])
    return out


def apply_feature_filters(values: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """Drop non-finite values and values outside [lo, hi]. Returns
    (kept_values, {'n_nonfinite': ..., 'n_out_of_range': ...})."""
    finite = np.isfinite(values)
    v = values[finite]
    in_range = (v >= lo) & (v <= hi)
    return v[in_range], {"n_nonfinite": int((~finite).sum()),
                         "n_out_of_range": int((~in_range).sum())}


# =============================================================================
# Streaming reservoir (Algorithm R, batched, seeded)
# =============================================================================

class Reservoir:
    """Fixed-capacity uniform sample over a stream, deterministic via seed."""

    def __init__(self, cap: int, seed: int, ndim: int = 1):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((cap,) if ndim == 1 else (cap, ndim))
        self.filled = 0
        self.seen = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n == 0:
            return
        take = min(self.cap - self.filled, n)
        if take > 0:
            self.data[self.filled:self.filled + take] = values[:take]
            self.filled += take
            self.seen += take
            values = values[take:]
            n -= take
        if n > 0:
            # batched Algorithm R: element i (1-based global index seen+i)
            # replaces a random slot with probability cap / (seen + i)
            idx = self.seen + np.arange(1, n + 1)
            accept = self.rng.random(n) < (self.cap / idx)
            k = int(accept.sum())
            if k:
                slots = self.rng.integers(0, self.cap, k)
                self.data[slots] = values[accept]
            self.seen += n

    def values(self) -> np.ndarray:
        return self.data[:self.filled]


def reservoir_or_capped_sample(existing: Optional[Reservoir], new_values: np.ndarray,
                               max_n: int, rng_seed: int) -> Reservoir:
    """Convenience wrapper: create-if-needed, then stream new values in."""
    if existing is None:
        existing = Reservoir(max_n, rng_seed)
    existing.add(new_values)
    return existing


# =============================================================================
# Prior fitting / evaluation / IO
# =============================================================================

def fit_empirical_prior(sample: np.ndarray, hist_counts: np.ndarray, bin_edges: np.ndarray,
                        feature_name: str, units: str, n_total: int, n_used: int,
                        drop_filters: Dict) -> Dict:
    """Assemble the prior JSON: quantiles from the reservoir sample, density
    from the EXACT histogram counts."""
    widths = np.diff(bin_edges)
    total = hist_counts.sum()
    density = hist_counts / (total * widths) if total > 0 else np.zeros_like(widths)
    quantiles = {k: float(np.percentile(sample, q)) for k, q in QUANTILE_KEYS.items()} \
        if len(sample) else {k: float("nan") for k in QUANTILE_KEYS}
    return {
        "feature": feature_name,
        "units": units,
        "n_samples_total": int(n_total),
        "n_samples_used": int(n_used),
        "n_samples_dropped": int(n_total - n_used),
        "drop_filters": drop_filters,
        "quantiles": quantiles,
        "histogram": {
            "bin_edges": [float(x) for x in bin_edges],
            "density": [float(x) for x in density],
            "counts": [int(x) for x in hist_counts],
        },
        "log_likelihood_floor": LOG_LIKELIHOOD_FLOOR,
        "created_by": "Stage 10",
    }


def evaluate_empirical_logpdf(values, prior: Dict) -> np.ndarray:
    """Approximate log-density of values under a fitted prior (floor applied
    outside the histogram support and in empty bins)."""
    v = np.asarray(values, dtype=float)
    edges = np.asarray(prior["histogram"]["bin_edges"])
    density = np.asarray(prior["histogram"]["density"])
    idx = np.searchsorted(edges, v, side="right") - 1
    inside = (idx >= 0) & (idx < len(density)) & np.isfinite(v)
    dens = np.where(inside, density[np.clip(idx, 0, len(density) - 1)], 0.0)
    return np.log(np.maximum(dens, prior["log_likelihood_floor"]))


def write_prior_json(path: str, prior: Dict) -> None:
    with open(path, "w") as f:
        json.dump(prior, f, indent=1)


def load_prior_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_joint_motion_prior(sample_df: pd.DataFrame, cfg: MotionPriorConfig) -> Dict:
    """Simple joint summary for stage 11: correlations, robust stats, and a
    covariance on log1p-transformed nonnegative features. Not a model."""
    features = list(sample_df.columns)
    corr = sample_df.corr()
    log_features = [c for c in features]
    logged = np.log1p(sample_df[log_features].clip(lower=0.0))
    return {
        "features": features,
        "n_samples": int(len(sample_df)),
        "correlation": {"features": features,
                        "matrix": [[float(x) for x in row] for row in corr.to_numpy()]},
        "log1p_covariance": {"features": log_features,
                             "matrix": [[float(x) for x in row] for row in np.cov(logged.T)]},
        "median": {c: float(sample_df[c].median()) for c in features},
        "mad": {c: float((sample_df[c] - sample_df[c].median()).abs().median()) for c in features},
        "created_by": "Stage 10",
        "note": "For stage 11 track scoring; not used by stage 10 itself.",
    }


# =============================================================================
# The learning pass
# =============================================================================

class _FeatureAccumulator:
    """Exact counts + exact histogram + quantile reservoir for one feature."""

    def __init__(self, name: str, units: str, lo: float, hi: float, bins: int,
                 cap: int, seed: int):
        self.name, self.units, self.lo, self.hi = name, units, lo, hi
        self.edges = np.linspace(lo, hi, bins + 1)
        self.counts = np.zeros(bins, dtype=np.int64)
        self.reservoir = Reservoir(cap, seed)
        self.n_total = 0
        self.n_nonfinite = 0
        self.n_out_of_range = 0

    def add(self, values: np.ndarray) -> None:
        kept, drops = apply_feature_filters(values, self.lo, self.hi)
        self.n_total += int(np.isfinite(values).sum())
        self.n_nonfinite += drops["n_nonfinite"]
        self.n_out_of_range += drops["n_out_of_range"]
        if len(kept):
            self.counts += np.histogram(kept, bins=self.edges)[0]
            self.reservoir.add(kept)

    @property
    def n_used(self) -> int:
        return int(self.counts.sum())

    def to_prior(self) -> Dict:
        return fit_empirical_prior(
            self.reservoir.values(), self.counts, self.edges, self.name, self.units,
            self.n_total, self.n_used,
            {"lo": self.lo, "hi": self.hi, "n_nonfinite": self.n_nonfinite,
             "n_out_of_range": self.n_out_of_range,
             "quantiles_from_reservoir_cap": self.reservoir.cap})


def learn_motion_priors(cfg: MotionPriorConfig) -> Dict:
    """The full stage-10 pass. Returns accumulated state for reports/gate."""
    files = discover_stage04_files(cfg.input_dir)
    if not files:
        raise SystemExit(
            f"No stage-4 trajectory files found in {cfg.input_dir}.\n"
            f"Copy or symlink them in, e.g.:\n"
            f"  cp ../F01-Preprocessing/data/active/trajectories_10s/"
            f"states_*_trajectories_10s.csv {cfg.input_dir}/")

    if cfg.train_dates:
        files = [(d, p) for d, p in files if d in cfg.train_dates or
                 (cfg.holdout_dates and d in cfg.holdout_dates)]
    holdout = set(cfg.holdout_dates or [])
    train_files = [(d, p) for d, p in files if d not in holdout]
    holdout_files = [(d, p) for d, p in files if d in holdout]
    if not train_files:
        raise SystemExit("No training files left after date filtering")

    defs = feature_definitions(cfg)
    seed = RESERVOIR_SEED
    accums = {name: _FeatureAccumulator(name, d["units"], d["lo"], d["hi"],
                                        cfg.hist_bins, cfg.sample_per_feature, seed + i)
              for i, (name, d) in enumerate(defs.items())}
    holdout_accums = {name: _FeatureAccumulator(name, d["units"], d["lo"], d["hi"],
                                                cfg.hist_bins, DAY_RESERVOIR_CAP, seed + 100 + i)
                      for i, (name, d) in enumerate(defs.items())}
    signed_res = {name: Reservoir(DAY_RESERVOIR_CAP, seed + 200 + i)
                  for i, name in enumerate(SIGNED_REPORT_FEATURES)}
    day_res: Dict[Tuple[str, str], Reservoir] = {}
    day_stats: Dict[str, Dict] = {}
    joint = Reservoir(JOINT_RESERVOIR_CAP, seed + 300, ndim=len(defs))
    joint_features = list(defs.keys())
    available: Dict[str, bool] = {}

    for date, path in files:
        cols = load_stage04_columns(path)
        usecols = ["trajectory_id", "timestamp", cols["alt"], cols["speed"]]
        usecols += [c for c in (cols["accel"], cols["accel_vector"], cols["turn_rate"]) if c]
        is_holdout = date in holdout
        rows = 0
        traj_ids = set()
        print(f"[{date}] reading {os.path.basename(path)}"
              f"{' (holdout)' if is_holdout else ''} ...", flush=True)

        for chunk in iter_complete_trajectories(path, usecols, cfg.chunksize):
            rows += len(chunk)
            traj_ids.update(chunk["trajectory_id"].unique())
            feats = compute_motion_features_chunk(chunk, cols)

            for name in defs:
                if name not in feats.columns:
                    available.setdefault(name, False)
                    continue
                available[name] = True
                vals = feats[name].to_numpy()
                (holdout_accums if is_holdout else accums)[name].add(vals)
                if not is_holdout:
                    key = (date, name)
                    day_res.setdefault(key, Reservoir(
                        DAY_RESERVOIR_CAP, seed + 400 + hash(key) % 10_000))
                    kept, _ = apply_feature_filters(vals, defs[name]["lo"], defs[name]["hi"])
                    day_res[key].add(kept)
            if not is_holdout:
                for name in SIGNED_REPORT_FEATURES:
                    if name in feats.columns:
                        v = feats[name].to_numpy()
                        signed_res[name].add(v[np.isfinite(v)])
                present = [c for c in joint_features if c in feats.columns]
                if len(present) == len(joint_features):
                    block = feats[joint_features].to_numpy()
                    ok = np.isfinite(block).all(axis=1)
                    for j, name in enumerate(joint_features):
                        d = defs[name]
                        ok &= (block[:, j] >= d["lo"]) & (block[:, j] <= d["hi"])
                    joint.add(block[ok])

        day_stats[date] = {"rows_read": rows, "trajectories": len(traj_ids),
                           "holdout": is_holdout}

    # holdout days also appear in the day summary (flagged), using their accums
    for date, _ in holdout_files:
        for name in defs:
            day_res[(date, name)] = holdout_accums[name].reservoir

    return {
        "accums": accums, "holdout_accums": holdout_accums, "signed_res": signed_res,
        "day_res": day_res, "day_stats": day_stats, "joint": joint,
        "joint_features": joint_features, "available": available,
        "train_dates": [d for d, _ in train_files],
        "holdout_dates": [d for d, _ in holdout_files],
        "defs": defs,
    }


# =============================================================================
# Outputs: model JSONs, reports, plots
# =============================================================================

PRIOR_FILENAMES = {
    "speed_mps": "speed_prior.json",
    "accel_abs_mps2": "acceleration_prior.json",
    "accel_vector_mps2": "vector_acceleration_prior.json",
    "turn_rate_abs_deg_s": "turn_rate_prior.json",
    "vertical_speed_abs_mps": "vertical_speed_prior.json",
}
REQUIRED_PRIORS = ["speed_mps", "accel_abs_mps2", "turn_rate_abs_deg_s", "vertical_speed_abs_mps"]


def write_model_files(state: Dict, cfg: MotionPriorConfig) -> Dict[str, Dict]:
    os.makedirs(cfg.models_dir, exist_ok=True)
    priors = {}
    written = []
    for name, accum in state["accums"].items():
        if not state["available"].get(name, False):
            continue
        prior = accum.to_prior()
        path = os.path.join(cfg.models_dir, PRIOR_FILENAMES[name])
        write_prior_json(path, prior)
        priors[name] = prior
        written.append(PRIOR_FILENAMES[name])

    joint_vals = state["joint"].values()
    joint_df = pd.DataFrame(joint_vals, columns=state["joint_features"])
    joint_prior = build_joint_motion_prior(joint_df, cfg) if len(joint_df) else \
        {"features": state["joint_features"], "n_samples": 0, "created_by": "Stage 10"}
    write_prior_json(os.path.join(cfg.models_dir, "joint_motion_prior.json"), joint_prior)
    written.append("joint_motion_prior.json")

    config_json = {
        "created_by": "Stage 10",
        "input_dir": cfg.input_dir,
        "train_dates": state["train_dates"],
        "holdout_dates": state["holdout_dates"],
        "filters": {
            "min_speed_mps": cfg.min_speed_mps, "max_speed_mps": cfg.max_speed_mps,
            "max_abs_vertical_speed_mps": cfg.max_abs_vertical_speed_mps,
            "max_abs_turn_rate_deg_s": cfg.max_abs_turn_rate_deg_s,
            "max_accel_mps2": cfg.max_accel_mps2,
        },
        "hist_bins": cfg.hist_bins,
        "sample_per_feature": cfg.sample_per_feature,
        "notes": "Counts and histograms are exact; quantiles come from a seeded "
                 "streaming reservoir capped at sample_per_feature.",
    }
    write_prior_json(os.path.join(cfg.models_dir, "motion_prior_config.json"), config_json)
    written.append("motion_prior_config.json")

    manifest = {
        "created_by": "Stage 10",
        "files": sorted(written + ["motion_prior_manifest.json"]),
        "features_fitted": sorted(priors.keys()),
        "features_unavailable": sorted(n for n, ok in state["available"].items() if not ok),
        "n_samples_used": {n: p["n_samples_used"] for n, p in priors.items()},
        "train_dates": state["train_dates"],
        "holdout_dates": state["holdout_dates"],
    }
    write_prior_json(os.path.join(cfg.models_dir, "motion_prior_manifest.json"), manifest)
    return priors


def write_reports(state: Dict, priors: Dict[str, Dict], cfg: MotionPriorConfig) -> str:
    os.makedirs(cfg.report_dir, exist_ok=True)
    defs = state["defs"]

    # quantiles CSV
    qrows = []
    for name, prior in priors.items():
        row = {"feature": name, "units": prior["units"],
               "n_samples_total": prior["n_samples_total"],
               "n_samples_used": prior["n_samples_used"],
               "n_samples_dropped": prior["n_samples_dropped"]}
        row.update(prior["quantiles"])
        qrows.append(row)
    quantiles_df = pd.DataFrame(qrows)
    quantiles_df.to_csv(os.path.join(cfg.report_dir, "motion_prior_quantiles.csv"), index=False)

    # per-day summary
    drows = []
    for date in sorted(state["day_stats"]):
        st = state["day_stats"][date]
        row = {"date": date, "rows_read": st["rows_read"], "trajectories": st["trajectories"],
               "holdout": st["holdout"]}
        for name, prefix in [("speed_mps", "speed"), ("accel_abs_mps2", "accel_abs"),
                             ("turn_rate_abs_deg_s", "turn_abs"),
                             ("vertical_speed_abs_mps", "vertical_speed_abs")]:
            res = state["day_res"].get((date, name))
            vals = res.values() if res is not None else np.array([])
            for q in (50, 95, 99):
                row[f"{prefix}_p{q}"] = float(np.percentile(vals, q)) if len(vals) else np.nan
        drows.append(row)
    day_df = pd.DataFrame(drows)
    day_df.to_csv(os.path.join(cfg.report_dir, "motion_prior_day_summary.csv"), index=False)

    # per-feature summary (fitted + signed report-only features)
    frows = []
    for name, prior in priors.items():
        vals = state["accums"][name].reservoir.values()
        frows.append({
            "feature": name, "units": prior["units"], "fitted_prior": True,
            "n_samples_total": prior["n_samples_total"],
            "n_samples_used": prior["n_samples_used"],
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std()) if len(vals) else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "mad": float(np.median(np.abs(vals - np.median(vals)))) if len(vals) else np.nan,
            "p05": prior["quantiles"]["p05"], "p50": prior["quantiles"]["p50"],
            "p95": prior["quantiles"]["p95"], "p99": prior["quantiles"]["p99"],
            "filter_lo": defs[name]["lo"], "filter_hi": defs[name]["hi"],
        })
    for name, units in SIGNED_REPORT_FEATURES.items():
        vals = state["signed_res"][name].values()
        if not len(vals):
            continue
        frows.append({
            "feature": name, "units": units, "fitted_prior": False,
            "n_samples_total": state["signed_res"][name].seen, "n_samples_used": len(vals),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "median": float(np.median(vals)),
            "mad": float(np.median(np.abs(vals - np.median(vals)))),
            "p05": float(np.percentile(vals, 5)), "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99)),
            "filter_lo": np.nan, "filter_hi": np.nan,
        })
    pd.DataFrame(frows).to_csv(os.path.join(cfg.report_dir, "motion_prior_feature_summary.csv"),
                               index=False)

    # correlation CSV
    joint_df = pd.DataFrame(state["joint"].values(), columns=state["joint_features"])
    corr = joint_df.corr() if len(joint_df) else pd.DataFrame()
    corr.to_csv(os.path.join(cfg.report_dir, "motion_prior_correlation.csv"))

    # ---- Markdown report -------------------------------------------------
    stage9 = pd.DataFrame([
        {"stage 9 knee": k, "hand-tuned": v} for k, v in STAGE09_KNEES])
    emp = {
        "speed_mps p95": priors.get("speed_mps", {}).get("quantiles", {}).get("p95"),
        "accel p95": priors.get("accel_abs_mps2", {}).get("quantiles", {}).get("p95"),
        "turn rate p95": priors.get("turn_rate_abs_deg_s", {}).get("quantiles", {}).get("p95"),
        "vertical speed p95": priors.get("vertical_speed_abs_mps", {}).get("quantiles", {}).get("p95"),
    }
    stage9["empirical ADS-B p95"] = [f"{emp[k]:.2f}" if emp[k] is not None and np.isfinite(emp[k])
                                     else "n/a" for k in stage9["stage 9 knee"]]

    total_rows = sum(s["rows_read"] for s in state["day_stats"].values())
    total_traj = sum(s["trajectories"] for s in state["day_stats"].values())
    train_days = ", ".join(state["train_dates"]) or "none"
    holdout_days = ", ".join(state["holdout_dates"]) or "none"

    lines = [
        "# Stage 10 Empirical ADS-B Motion Priors",
        "",
        "## Purpose",
        "",
        "- Stage 10 learns **empirical motion priors** from the F01 stage-4",
        "  ADS-B trajectories: histogram densities and quantiles for speed,",
        "  |acceleration|, vector acceleration, |turn rate|, and |vertical",
        "  speed| of real fixed-wing GA flight.",
        "- These priors are **not yet applied to tracks** -- nothing in stages",
        "  7-9 changes.",
        "- **Stage 11** will use them to replace or augment the stage-9",
        "  hand-designed penalties with data-derived likelihoods. Nothing here",
        "  is a neural network: the priors are transparent empirical",
        "  histograms.",
        "",
        "## Input data",
        "",
        f"- Input directory: `{cfg.input_dir}`",
        f"- Training dates: {train_days}",
        f"- Holdout dates: {holdout_days}",
        f"- Rows read: {total_rows:,}; trajectories: {total_traj:,}",
        "- Fitting filters (disclosed, applied only for prior fitting): speed in",
        f"  [{cfg.min_speed_mps:g}, {cfg.max_speed_mps:g}] m/s, |accel| <= {cfg.max_accel_mps2:g} m/s^2,",
        f"  |turn rate| <= {cfg.max_abs_turn_rate_deg_s:g} deg/s, |vertical speed| <= "
        f"{cfg.max_abs_vertical_speed_mps:g} m/s.",
        "",
        "## Fitted features",
        "",
        "- **speed_mps** -- stage-4 ground speed on the 10 s grid.",
        "- **accel_abs_mps2** -- absolute longitudinal acceleration.",
        "- **accel_vector_mps2** -- vector (centripetal-inclusive) acceleration"
        + (" (available)." if state["available"].get("accel_vector_mps2") else
           " (NOT available in these inputs)."),
        "- **turn_rate_abs_deg_s** -- absolute turn rate.",
        "- **vertical_speed_abs_mps** -- |d(alt)/dt| per trajectory",
        "  (np.gradient over the trajectory's timestamps).",
        "",
        "## Quantile summary",
        "",
    ]
    lines += md_table(quantiles_df.round(4))
    lines += [
        "",
        "## Comparison to Stage 09 hand thresholds",
        "",
    ]
    lines += md_table(stage9)
    lines += [
        "",
        "Reading: where the empirical p95 sits well BELOW a stage-9 'good' knee,",
        "the hand threshold was conservative (it under-penalizes); where the",
        "empirical p95 approaches the knee, the hand threshold was aggressive.",
        "Stage 11 should score with the full histogram likelihoods instead of",
        "knees, making this comparison moot.",
        "",
        "## Day-to-day consistency",
        "",
    ]
    day_show = day_df[["date", "holdout", "rows_read", "trajectories",
                       "speed_p50", "speed_p95", "accel_abs_p95",
                       "turn_abs_p95", "vertical_speed_abs_p95"]].round(3)
    lines += md_table(day_show)
    lines += [
        "",
        "## Model files",
        "",
    ]
    manifest = load_prior_json(os.path.join(cfg.models_dir, "motion_prior_manifest.json"))
    lines += [f"- `{f}`" for f in manifest["files"]]
    lines += [
        "",
        "## Validation",
        "",
        "See the validation gate output in the run log: prior files present,",
        "monotonic quantiles, valid histogram shapes, plausibility anchors",
        "(speed p50 in 20-90 m/s), and the report-only stage-9 comparison.",
        "",
        "## Next stage",
        "",
        "Stage 11 will apply these empirical priors to Stage 08 tracks and",
        "compare against Stage 09 hand-designed physics scoring.",
        "",
    ]
    path = os.path.join(cfg.report_dir, "motion_prior_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


PLOT_SPECS = [
    ("speed_mps", "speed_hist.png", "Speed (m/s)"),
    ("accel_abs_mps2", "acceleration_hist.png", "|longitudinal acceleration| (m/s^2)"),
    ("accel_vector_mps2", "vector_acceleration_hist.png", "vector acceleration (m/s^2)"),
    ("turn_rate_abs_deg_s", "turn_rate_hist.png", "|turn rate| (deg/s)"),
    ("vertical_speed_abs_mps", "vertical_speed_hist.png", "|vertical speed| (m/s)"),
]


def make_plots(priors: Dict[str, Dict], plots_dir: str) -> List[str]:
    os.makedirs(plots_dir, exist_ok=True)
    written = []
    for name, fname, xlabel in PLOT_SPECS:
        if name not in priors:
            continue
        prior = priors[name]
        edges = np.asarray(prior["histogram"]["bin_edges"])
        density = np.asarray(prior["histogram"]["density"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.stairs(density, edges, fill=True, alpha=0.7)
        if (density > 0).any() and density.max() / max(density[density > 0].min(), 1e-300) > 1e3:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(f"Empirical ADS-B prior: {name} "
                     f"(n={prior['n_samples_used']:,})")
        ax.grid(True, linewidth=0.5)
        fig.tight_layout()
        p = os.path.join(plots_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


# =============================================================================
# Validation gate
# =============================================================================

def run_validation_gate(state: Dict, priors: Dict[str, Dict], cfg: MotionPriorConfig) -> None:
    def fail(message: str) -> None:
        raise ValueError(f"Stage 10 validation failed: {message}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    if not state["day_stats"]:
        fail("no stage-4 files were processed")
    print(f"  files processed: {len(state['day_stats'])} "
          f"({sum(s['rows_read'] for s in state['day_stats'].values()):,} rows)")

    for name in REQUIRED_PRIORS:
        path = os.path.join(cfg.models_dir, PRIOR_FILENAMES[name])
        if not os.path.exists(path):
            fail(f"required prior file missing: {path}")
        if name not in priors or priors[name]["n_samples_used"] <= 0:
            fail(f"required prior '{name}' has no samples")
    print("  required prior JSONs exist with n_samples_used > 0: OK")

    order = list(QUANTILE_KEYS.keys())
    for name, prior in priors.items():
        q = [prior["quantiles"][k] for k in order]
        if any(b < a - 1e-12 for a, b in zip(q, q[1:])):
            fail(f"{name}: quantiles not monotonic")
        h = prior["histogram"]
        if len(h["bin_edges"]) != len(h["density"]) + 1 or len(h["density"]) != len(h["counts"]):
            fail(f"{name}: histogram edge/density/count lengths inconsistent")
    print("  quantiles monotonic; histogram shapes valid: OK")

    for fn in ["motion_prior_quantiles.csv", "motion_prior_day_summary.csv",
               "motion_prior_feature_summary.csv", "motion_prior_correlation.csv"]:
        p = os.path.join(cfg.report_dir, fn)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            fail(f"report file missing or empty: {fn}")
    print("  report CSVs exist and are nonempty: OK")

    sp = priors["speed_mps"]["quantiles"]
    if not (20.0 <= sp["p50"] <= 90.0):
        fail(f"speed p50 {sp['p50']:.1f} m/s outside plausible 20-90 m/s")
    if sp["p99"] > cfg.max_speed_mps + 1e-9:
        fail("speed p99 above max_speed_mps")
    if priors["turn_rate_abs_deg_s"]["quantiles"]["p95"] > cfg.max_abs_turn_rate_deg_s + 1e-9:
        fail("turn-rate p95 above filter bound")
    if priors["vertical_speed_abs_mps"]["quantiles"]["p95"] > cfg.max_abs_vertical_speed_mps + 1e-9:
        fail("vertical-speed p95 above filter bound")
    print(f"  plausibility: speed p50 {sp['p50']:.1f} m/s in [20, 90]; "
          f"tails within filter bounds: OK")

    print("\n  stage-9 hand knees vs empirical p95 (report-only):")
    for label, knee in STAGE09_KNEES:
        key = {"speed_mps p95": "speed_mps", "accel p95": "accel_abs_mps2",
               "turn rate p95": "turn_rate_abs_deg_s",
               "vertical speed p95": "vertical_speed_abs_mps"}[label]
        if key in priors:
            print(f"    {label:<22} {knee:<32} empirical p95 = "
                  f"{priors[key]['quantiles']['p95']:.2f}")

    if state["holdout_dates"]:
        print("\n  holdout-vs-train quantile differences (report-only):")
        for name in REQUIRED_PRIORS:
            tr = priors[name]["quantiles"]
            hv = state["holdout_accums"][name].reservoir.values()
            if not len(hv):
                continue
            for k, q in [("p50", 50), ("p95", 95)]:
                h = float(np.percentile(hv, q))
                print(f"    {name:<24} {k}: train {tr[k]:.3f} vs holdout {h:.3f} "
                      f"(diff {h - tr[k]:+.3f})")


# =============================================================================
# Self-test (no real data required)
# =============================================================================

def self_test() -> None:
    """Two synthetic days x three trajectories of plausible GA motion."""
    import tempfile

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        os.makedirs(in_dir)
        for date in ("2022-01-01", "2022-01-02"):
            rows = []
            for k in range(3):
                n = 60
                t = 1_000.0 + 10.0 * np.arange(n)
                speed = 55.0 + 5 * np.sin(np.linspace(0, 3, n)) + rng.normal(0, 1, n)
                alt = 1200.0 + np.cumsum(rng.normal(0, 8, n))
                rows.append(pd.DataFrame({
                    "trajectory_id": f"{date}_traj{k}", "timestamp": t,
                    "alt_smooth": alt, "speed_mps": speed,
                    "accel_mps2": rng.normal(0, 0.3, n),
                    "accel_vector_mps2": np.abs(rng.normal(0.5, 0.3, n)),
                    "turn_rate_deg_s": rng.normal(0, 1.0, n),
                    "trajectory_duration_s": 10.0 * (n - 1), "n_samples": n,
                }))
            pd.concat(rows).to_csv(
                os.path.join(in_dir, f"states_{date}_conventionalGA_trajectories_10s.csv"),
                index=False)

        cfg = MotionPriorConfig(input_dir=in_dir,
                                models_dir=os.path.join(tmp, "models"),
                                report_dir=os.path.join(tmp, "reports"),
                                chunksize=100, sample_per_feature=100_000, hist_bins=50)
        state = learn_motion_priors(cfg)
        priors = write_model_files(state, cfg)
        report_path = write_reports(state, priors, cfg)
        make_plots(priors, os.path.join(cfg.report_dir, "plots"))
        run_validation_gate(state, priors, cfg)

        for name in REQUIRED_PRIORS + ["accel_vector_mps2"]:
            assert os.path.exists(os.path.join(cfg.models_dir, PRIOR_FILENAMES[name])), \
                f"missing prior {name}"
        assert os.path.exists(os.path.join(cfg.report_dir, "motion_prior_quantiles.csv"))
        assert os.path.exists(report_path)

        sp = priors["speed_mps"]["quantiles"]
        assert 45 <= sp["p50"] <= 65, f"speed median implausible: {sp['p50']}"
        order = list(QUANTILE_KEYS.keys())
        for prior in priors.values():
            q = [prior["quantiles"][k] for k in order]
            assert all(b >= a - 1e-12 for a, b in zip(q, q[1:])), "quantiles not monotonic"

        lp = evaluate_empirical_logpdf([55.0, 57.0, 59.0], priors["speed_mps"])
        assert np.isfinite(lp).all(), "logpdf non-finite for plausible speeds"
        lp_bad = evaluate_empirical_logpdf([500.0], priors["speed_mps"])[0]
        assert lp.min() > lp_bad, "plausible speed should out-score implausible speed"

        text = open(report_path).read()
        for needle in ["Stage 10 Empirical ADS-B Motion Priors",
                       "not yet applied to tracks", "Stage 11"]:
            assert needle in text, f"report missing expected text: {needle!r}"

    print("\nStage 10 self-test passed.")
