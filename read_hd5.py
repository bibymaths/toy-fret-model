'''
BSD 3-Clause License

Copyright (c) 2025, Abhinav Mishra
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

"""
HDF5 FRET Tracking Data Processor
=================================

This script processes single-molecule FRET tracking data stored in HDF5 format.
It is designed to inspect raw data, filter and export individual particle
trajectories, and combine them into a unified time-series matrix.

The workflow consists of three main stages:
1.  **Inspection**: Iterates through all track files, logging metadata,
    data dimensions, and basic statistics. Plots a representative trajectory
    for each file to aid in quick quality control.
2.  **Export**: Reads raw data again, applies validity filters (FRET range,
    minimum trajectory length), and saves each valid particle's trace as an
    individual CSV file in a designated output directory.
3.  **Combination**: Reads all exported individual CSVs, determines a global
    time vector, and interpolates all traces onto this uniform grid using
    cubic splines (or linear interpolation for short traces). The result is a
    single matrix where columns represent individual molecules and rows represent
    synchronized time points.
"""

import logging
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.interpolate import CubicSpline

# === Logger Setup ===
# Configured to mimic 'print' output cleanly while allowing for future redirection/levels
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# === Configuration ===
data_dir = Path("data/Hugel_2025")   # adjust path
key = "/tracks/Data"                 # default key
frame_interval = 0.1                # seconds per frame
fret_max = 2                         # max FRET efficiency to consider
fret_min = 0.16                         # min FRET efficiency to consider
USE_INTERPOLATION = False             # Set to False to disable cubic splines/filling

def interpolate_trace(time_grid: np.ndarray,
                      t_trace: np.ndarray,
                      E_trace: np.ndarray,
                      interpolate: bool = True) -> np.ndarray:
    """
    Interpolate a single FRET trace onto a common time grid using cubic splines.

    Handles edge cases such as missing data (NaNs), single-point traces,
    and short traces where only linear interpolation is feasible.

    Args:
        time_grid (np.ndarray): The uniform time vector to interpolate onto.
        t_trace (np.ndarray): Original time points of the trace.
        E_trace (np.ndarray): Original FRET efficiency values.

    Returns:
        np.ndarray: The FRET trace interpolated onto `time_grid`. Returns NaNs
        for time points outside the original observation range or if
        insufficient data exists.
    """
    # Ensure arrays
    t_trace = np.asarray(t_trace, float)
    E_trace = np.asarray(E_trace, float)

    # Mask out non-finite values
    mask = np.isfinite(t_trace) & np.isfinite(E_trace)
    t_clean = t_trace[mask]
    E_clean = E_trace[mask]

    # Not enough points to interpolate
    if t_clean.size == 0:
        return np.full_like(time_grid, np.nan, dtype=float)
    if t_clean.size == 1:
        # Single point: constant over its range, NaN elsewhere
        y = np.full_like(time_grid, np.nan, dtype=float)
        idx = np.argmin(np.abs(time_grid - t_clean[0]))
        y[idx] = E_clean[0]
        return y

    # Remove duplicate time stamps, if any
    t_unique, idx_unique = np.unique(t_clean, return_index=True)
    E_unique = E_clean[idx_unique]

    if not interpolate:
        y = np.full_like(time_grid, np.nan, dtype=float)
        if len(time_grid) > 1:
            dt = time_grid[1] - time_grid[0]
            # Find nearest grid index for each observation
            idx = np.rint(t_unique / dt).astype(np.int64)
            # Ensure indices are within bounds
            valid = (idx >= 0) & (idx < len(time_grid))
            y[idx[valid]] = E_unique[valid]
        elif len(time_grid) == 1 and t_unique.size > 0:
            # Edge case: 1-point grid, just take the first point if close enough
            if np.abs(t_unique[0] - time_grid[0]) < frame_interval / 2:
                y[0] = E_unique[0]
        return y

    if t_unique.size < 2:
        return np.full_like(time_grid, np.nan, dtype=float)

    # For 2 points, a spline is pointless; use linear interpolation
    if t_unique.size == 2:
        y = np.interp(time_grid, t_unique, E_unique,
                      left=np.nan, right=np.nan)
        return y

    # For >=3 points, try cubic spline
    try:
        cs = CubicSpline(t_unique, E_unique, extrapolate=False)
        y = cs(time_grid)
    except Exception:
        y = np.interp(time_grid, t_unique, E_unique,
                      left=np.nan, right=np.nan)

    outside = (time_grid < t_unique.min()) | (time_grid > t_unique.max())
    y[outside] = np.nan
    y[(y < fret_min) | (y > fret_max)] = np.nan
    return y

# === Inspect and plot data ===
for path in sorted(data_dir.glob("*.tracks*")):
    logger.info("=" * 80)

    # Extract metadata from filename
    fname = path.stem  # e.g. filtered-241107-Hsp90_409_601-v014.tracks
    parts = fname.split("-")
    if len(parts) >= 3:
        exp_id = parts[1]
        construct = parts[2].split(".")[0]
    else:
        exp_id = "unknown"
        construct = "unknown"

    logger.info(f"File: {path.name}")
    logger.info(f"Experiment: {construct}, Date/ID: {exp_id}")

    # Step 1 — list keys
    try:
        store = pd.HDFStore(path, mode="r")
        keys = store.keys()
        store.close()
        logger.info(f"Keys in file: {keys}")
    except Exception as e:
        logger.error(f"Could not open file: {e}")
        continue

    # Step 2 — read dataset
    k = key if key in keys else keys[0]
    try:
        df = pd.read_hdf(path, key=k)
    except Exception as e:
        logger.error(f"Error reading {k}: {e}")
        continue

    # Step 3 — flatten multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, col)).strip() for col in df.columns]

    # Step 4 — basic info
    logger.info(f"Loaded with shape: {df.shape}")
    logger.info(f"Columns ({len(df.columns)}):")
    logger.info("   " + ", ".join(df.columns[:10]) + ("..." if len(df.columns) > 10 else ""))
    logger.info("")

    # Step 5 — show sample rows
    logger.info(df.head(5))
    logger.info("")

    # Step 6 — numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[num_cols].describe().T[["mean", "std", "min", "max"]]
    logger.info("Numeric summary (mean ± std, min–max):")
    logger.info(summary.head(10))
    logger.info("")

    # Step 7 — add time column
    if "donor_frame" in df.columns:
        df["time_s"] = df["donor_frame"] * frame_interval
        logger.info(f"Added time_s (first 5): {df['time_s'].head().tolist()}")
    logger.info("")

    # Step 8 — detect FRET column
    fret_candidates = ["fret_eff", "fret_eff_app", "fret_efficiency"]
    fret_col = next((c for c in fret_candidates if c in df.columns), None)

    if fret_col is None:
        logger.warning(f"No FRET column found (checked: {', '.join(fret_candidates)}). Skipping plot.\n")
        continue

    # Ensure we have a time axis
    if "time_s" not in df.columns:
        if "donor_frame" in df.columns:
            df["time_s"] = df["donor_frame"] * frame_interval
        else:
            logger.warning("No donor_frame/time info found. Skipping plot.\n")
            continue

    # Choose a representative particle: longest trajectory
    part_col = "fret_particle" if "fret_particle" in df.columns else None

    if part_col is not None:
        counts = df.groupby(part_col).size()
        longest_pid = counts.sort_values(ascending=False).index[0]
        traj = df[df[part_col] == longest_pid].sort_values("time_s")
        label = f"{construct} {exp_id} – particle {int(longest_pid)}"
    else:
        traj = df.sort_values("time_s")
        label = f"{construct} {exp_id} – all data (no particle id)"

    # Plot representative trace
    plt.figure()
    plt.plot(traj["time_s"], traj[fret_col], marker="o", linestyle="-", markersize=3)
    plt.xlabel("time (s)")
    plt.ylabel(f"{fret_col}")
    plt.title(label)
    plt.tight_layout()
    plt.show()


# === Export per-particle time series ===
export_dir = Path("data/timeseries")
export_dir.mkdir(exist_ok=True)

for path in sorted(data_dir.glob("*.tracks*.h5")):
    fname = path.stem
    parts = fname.split("-")
    if len(parts) >= 3:
        exp_id = parts[1]
        construct = parts[2].split(".")[0]
    else:
        exp_id = "unknown"
        construct = "unknown"

    logger.info(f"\nExporting per-particle time series from: {path.name}")
    logger.info(f"Experiment: {construct}, Date/ID: {exp_id}")

    try:
        df = pd.read_hdf(path, key="/tracks/Data")
    except Exception as e:
        logger.error(f"Could not read {path.name}: {e}")
        continue

    # flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, c)).strip() for c in df.columns]

    # ensure required columns
    needed_cols = {"donor_frame", "fret_particle"}
    if not needed_cols.issubset(df.columns):
        logger.warning("Missing required columns, skipping.")
        continue

    df["time_s"] = df["donor_frame"] * frame_interval

    fret_candidates = ["fret_eff", "fret_eff_app"]
    fret_col = next((c for c in fret_candidates if c in df.columns), None)
    if fret_col is None:
        logger.warning("No FRET efficiency column found, skipping file.")
        continue

    if "filter_manual" in df.columns:
        df = df[df["filter_manual"] == 1]

    # keep only finite, physically reasonable FRET values
    df = df[np.isfinite(df[fret_col])]
    df = df[(df[fret_col] > fret_min) & (df[fret_col] < fret_max)]

    # drop short trajectories AFTER filtering
    lengths = df.groupby("fret_particle")["donor_frame"].nunique()
    keep = lengths[lengths >= 20].index
    df = df[df["fret_particle"].isin(keep)]

    count = 0
    for pid, traj in df.groupby("fret_particle"):
        traj = traj.sort_values("donor_frame")
        out = traj[["time_s", fret_col]].rename(columns={fret_col: "FRET"})
        out_name = f"{path.stem}_particle_{int(pid):05d}.csv"
        out.to_csv(export_dir / out_name, index=False)
        count += 1

    logger.info(f"Exported {count} per-particle traces to {export_dir}/")


# === Combine all exported trajectories into a single matrix ===
combined_out = export_dir / "combined_fret_matrix.csv"
logger.info("\nBuilding combined FRET matrix (uniform 0–max_t grid)...")

csv_files = sorted(export_dir.glob("*.csv"))
if not csv_files:
    logger.warning("No per-particle CSV files found, skipping matrix creation.")
else:
    max_t = 0.0

    # Find max time across all traces
    for f in csv_files:
        df_tmp = pd.read_csv(f)
        if "time_s" in df_tmp.columns and len(df_tmp) > 0:
            max_t = max(max_t, df_tmp["time_s"].max())

    time_grid = np.arange(0.0, max_t + frame_interval / 2, frame_interval)
    # Collect all interpolated traces first
    columns = {"time_s": time_grid}

    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        if len(df) == 0:
            continue

        t_trace = df["time_s"].values
        E_trace = df["FRET"].values
        interp = interpolate_trace(time_grid, t_trace, E_trace, interpolate=USE_INTERPOLATION)

        stem = f.stem
        parts = stem.split("-")
        if len(parts) >= 3:
            exp_id = parts[1]
            construct = parts[2].split(".")[0]
        else:
            exp_id = "unknown"
            construct = "unknown"

        pid = stem.split("particle_")[-1]
        col_name = f"{construct}_{exp_id}_p{pid}"

        columns[col_name] = interp

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} traces ...")

    # Build one DataFrame in a single allocation
    combined = pd.DataFrame(columns)
    combined.to_csv(combined_out, index=False)
    logger.info(f"Combined matrix saved → {combined_out}")
    logger.info(f"Time points: {len(time_grid)}, trajectories: {combined.shape[1] - 1}")