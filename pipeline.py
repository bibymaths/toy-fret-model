import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed

# ----------------------------------------------------------------------
# Extended Model: Three-state Hsp90 dynamics + bleaching
# States: Open (O) <-> Intermediate (I) <-> Closed (C) -> Bleached (B)
# ----------------------------------------------------------------------

@dataclass
class Hsp90Params3State:
    """
    Three-state conformational model with bleaching:
      O <-> I <-> C, each can irreversibly bleach to B with rate k_B.

    We explicitly track P_O, P_I, P_C; P_B = 1 - P_O - P_I - P_C.
    FRET is from O, I, C; bleached B is assumed dark (E_bleach = 0).
    """
    # Conformational rates
    k_OI: float  # Open -> Intermediate rate (1/s)
    k_IO: float  # Intermediate -> Open rate (1/s)
    k_IC: float  # Intermediate -> Closed rate (1/s)
    k_CI: float  # Closed -> Intermediate rate (1/s)

    # Bleaching rate (same from all fluorescent states)
    k_B: float   # O,I,C -> B (1/s)

    # FRET levels
    E_open: float   # FRET in Open state
    E_inter: float  # FRET in Intermediate state
    E_closed: float # FRET in Closed state

    # Initial probabilities (P_I0 = 1 - P_O0 - P_C0, P_B0 = 0)
    P_O0: float  # Initial Open probability
    P_C0: float  # Initial Closed probability


@dataclass
class Hsp90Fit3State:
    """
    Container for a full fit: kinetics + static subpopulation.
    """
    params: Hsp90Params3State
    f_dyn: float       # fraction of molecules following the kinetic model
    E_static: float    # FRET level of static subpopulation


def rhs_hsp90_3state(t: float, y: np.ndarray, p: Hsp90Params3State) -> np.ndarray:
    """
    ODE for P_O, P_I, P_C in the presence of bleaching.
    """
    P_O, P_I, P_C = y

    # dP_O/dt = -k_OI*P_O + k_IO*P_I - k_B*P_O
    dP_O = -p.k_OI * P_O + p.k_IO * P_I - p.k_B * P_O

    # dP_I/dt = k_OI*P_O - (k_IO + k_IC + k_B)*P_I + k_CI*P_C
    dP_I = p.k_OI * P_O - (p.k_IO + p.k_IC + p.k_B) * P_I + p.k_CI * P_C

    # dP_C/dt = k_IC*P_I - k_CI*P_C - k_B*P_C
    dP_C = p.k_IC * P_I - p.k_CI * P_C - p.k_B * P_C

    return np.array([dP_O, dP_I, dP_C], dtype=float)


@njit("float64[:](float64, float64[:], float64[:])", cache=True, fastmath=False, nogil=False)
def rhs_hsp90_numba(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack params array for speed (avoiding python object attribute access)
    k_OI, k_IO, k_IC, k_CI, k_B = params[0], params[1], params[2], params[3], params[4]

    P_O, P_I, P_C = y[0], y[1], y[2]

    dP_O = -k_OI * P_O + k_IO * P_I - k_B * P_O
    dP_I = k_OI * P_O - (k_IO + k_IC + k_B) * P_I + k_CI * P_C
    dP_C = k_IC * P_I - k_CI * P_C - k_B * P_C

    return np.array([dP_O, dP_I, dP_C], dtype=np.float64)

def model_fret_3state(t_eval: np.ndarray, p: Hsp90Params3State) -> np.ndarray:
    """
    Dynamic part only: E_dyn(t) = E_O*P_O + E_I*P_I + E_C*P_C.
    """
    P_O0 = p.P_O0
    P_C0 = p.P_C0
    P_I0 = 1.0 - P_O0 - P_C0
    # crude fix if guesses are pathological
    if P_I0 < 0.0:
        total = max(P_O0 + P_C0, 1e-9)
        P_O0 = P_O0 / total * 0.999
        P_C0 = P_C0 / total * 0.001
        P_I0 = 1.0 - P_O0 - P_C0

    # y0 = [P_O0, P_I0, P_C0]
    k_params = np.array([p.k_OI, p.k_IO, p.k_IC, p.k_CI, p.k_B], dtype=float)
    y0 = np.array([P_O0, P_I0, P_C0], dtype=float)

    sol = solve_ivp(
        fun=rhs_hsp90_numba,
        t_span=(t_eval.min(), t_eval.max()),
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        args=(k_params,),  # <--- CRITICAL FIX: Must be a tuple
        method='RK45'
    )

    if not sol.success:
        return np.full_like(t_eval, np.nan, dtype=float)

    P_O_t = sol.y[0]
    P_I_t = sol.y[1]
    P_C_t = sol.y[2]

    E_t = (
        p.E_open * P_O_t +
        p.E_inter * P_I_t +
        p.E_closed * P_C_t
    )
    return E_t


def model_total_fret(t_eval: np.ndarray, fit: Hsp90Fit3State) -> np.ndarray:
    """
    Full observation model: dynamic + static subpopulation.

        E_total(t) = f_dyn * E_dyn(t) + (1 - f_dyn) * E_static
    """
    E_dyn = model_fret_3state(t_eval, fit.params)
    return fit.f_dyn * E_dyn + (1.0 - fit.f_dyn) * fit.E_static


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_combined_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError("Expected a 'time_s' column in the combined matrix.")

    t = df["time_s"].values
    traj_cols = [c for c in df.columns if c != "time_s"]
    E_mat = df[traj_cols].to_numpy()

    row_valid = np.isfinite(E_mat).any(axis=1)
    t = t[row_valid]
    E_mat = E_mat[row_valid, :]

    return t, E_mat, traj_cols



def parse_column_metadata(col_names: List[str]) -> pd.DataFrame:
    """
    Parse combined_fret_matrix trajectory column names into a metadata DataFrame.

    Expected format (from your export code):
        <construct>_<exp_id>_p<particle>
    e.g. "Hsp90_409_601_241107_p00001"

    Returns
    -------
    meta : DataFrame with columns:
        - col: original column name
        - construct: e.g. "Hsp90_409_601" or "esDNA"
        - exp_id: e.g. "241107"
        - particle: string/ID after 'p'
        - condition: default grouping key "<construct>_<exp_id>"
    """
    records = []
    for c in col_names:
        if c == "time_s":
            continue

        # Split from the right: [..., construct, exp_id, pXXXXX]
        parts = c.split("_")
        if len(parts) < 3:
            # Fallback: treat whole name as "construct"
            construct = c
            exp_id = "unknown"
            particle = "unknown"
        else:
            particle = parts[-1]              # e.g. "p00001"
            exp_id = parts[-2]                # e.g. "241107"
            construct = "_".join(parts[:-2])  # e.g. "Hsp90_409_601" or "esDNA"

        condition = f"{construct}_{exp_id}"
        records.append((c, construct, exp_id, particle, condition))

    meta = pd.DataFrame(
        records,
        columns=["col", "construct", "exp_id", "particle", "condition"]
    )
    return meta

def subset_matrix_by_columns(
    t: np.ndarray,
    E_mat: np.ndarray,
    all_cols: List[str],
    cols_subset: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the full time grid and matrix, extract a sub-matrix restricted to
    a subset of trajectory columns, and remove rows where all subset entries are NaN.

    Parameters
    ----------
    t : (T,) array
        Full time grid.
    E_mat : (T, N) array
        FRET trajectories for all columns in all_cols.
    all_cols : list of str
        Names of all trajectory columns (corresponding to E_mat columns).
    cols_subset : list of str
        Names of columns to keep for this subset.

    Returns
    -------
    t_sub : (T_sub,) array
        Time grid for which at least one trajectory in the subset is finite.
    E_sub : (T_sub, N_sub) array
        Subset FRET matrix.
    """
    name_to_idx = {c: i for i, c in enumerate(all_cols)}
    idx = [name_to_idx[c] for c in cols_subset if c in name_to_idx]

    if not idx:
        raise ValueError("No matching columns found for subset.")

    E_sub_full = E_mat[:, idx]
    row_valid = np.isfinite(E_sub_full).any(axis=1)
    t_sub = t[row_valid]
    E_sub = E_sub_full[row_valid, :]
    return t_sub, E_sub


def compute_ensemble_metrics(
    t: np.ndarray,
    E_mat: np.ndarray,
    fit: Hsp90Fit3State
) -> dict:
    """
    Compute ensemble RMSE and R^2 for a given condition and fitted model.
    Uses ensemble mean vs model prediction (same logic as plot_ensemble_fit).
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    if t_plot.size == 0:
        return dict(rmse=np.nan, r2=np.nan, n_time=0, n_traj=0)

    E_mean = np.nanmean(E_plot, axis=1)
    E_model = model_total_fret(t_plot, fit)

    mask = np.isfinite(E_mean) & np.isfinite(E_model)
    E_obs = E_mean[mask]
    E_mod = E_model[mask]

    if E_obs.size == 0:
        return dict(rmse=np.nan, r2=np.nan, n_time=0, n_traj=E_plot.shape[1])

    residuals = E_obs - E_mod
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((E_obs - E_obs.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(
        rmse=float(rmse),
        r2=float(r2),
        n_time=int(len(E_obs)),
        n_traj=int(E_mat.shape[1]),
    )

def _fit_single_condition_worker(
        key: str,
        t: np.ndarray,
        E_mat: np.ndarray,
        col_names: List[str],
        meta: pd.DataFrame,
        group_by: str
) -> Optional[Tuple[str, Hsp90Fit3State, dict]]:
    """
    Internal worker for parallel fitting of a single condition.
    """
    cols_subset = meta.loc[meta[group_by] == key, "col"].tolist()
    if not cols_subset:
        return None

    try:
        # 1. Subset data for this specific condition
        t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols_subset)

        # 2. Perform the heavy computational fit
        fit = fit_global_3state(t_sub, E_sub)

        # 3. Compute metrics
        metrics = compute_ensemble_metrics(t_sub, E_sub, fit)

        # 4. Pack results into a dictionary record
        p = fit.params
        rec = dict(
            group_by=group_by,
            group_key=key,
            n_traj=metrics["n_traj"],
            n_time=metrics["n_time"],
            rmse=metrics["rmse"],
            r2=metrics["r2"],
            k_OI=p.k_OI, k_IO=p.k_IO, k_IC=p.k_IC, k_CI=p.k_CI, k_B=p.k_B,
            E_open=p.E_open, E_inter=p.E_inter, E_closed=p.E_closed,
            P_O0=p.P_O0, P_C0=p.P_C0,
            f_dyn=fit.f_dyn, E_static=fit.E_static,
        )
        return (key, fit, rec)

    except Exception as e:
        print(f"  Fit failed for group '{key}': {e}")
        return None

def fit_all_conditions(
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: List[str],
    group_by: str = "condition",
    do_plots: bool = False,
    max_overlay_traces: int = 100,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fit the 3-state+bleaching+static model separately for each condition.

    Parameters
    ----------
    t : (T,) array
        Global time grid.
    E_mat : (T, N) array
        Full combined FRET matrix.
    col_names : list of str
        Names of trajectory columns (same order as E_mat columns).
    group_by : {"condition", "construct", "exp_id"}
        How to group trajectories:
        - "condition": construct+exp_id (default, per coverslip/day)
        - "construct": pool all days for each construct
        - "exp_id": per date across constructs (usually less useful)
    do_plots : bool
        If True, make per-condition time plots using plot_hsp90_fit_time.
    max_overlay_traces : int
        Maximum number of individual trajectories to overlay per condition.

    Returns
    -------
    summary_df : DataFrame
        One row per condition with fitted parameters and metrics.
    fits : dict
        Mapping: condition_key -> Hsp90Fit3State
    """
    meta = parse_column_metadata(col_names)
    if group_by not in meta.columns:
        raise ValueError(f"group_by must be one of {', '.join(['condition', 'construct', 'exp_id'])}")

    group_keys = sorted(meta[group_by].unique())
    print(f"Starting parallel fit for {len(group_keys)} groups...")

    # --- PARALLEL FITTING ---
    # n_jobs=-1 uses all available CPU cores.
    # verbose=1 provides a progress update in the console.
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_fit_single_condition_worker)(
            key, t, E_mat, col_names, meta, group_by
        ) for key in group_keys
    )
    # ------------------------

    # Unpack results from parallel workers
    fits_list = []
    fit_dict = {}
    for res in results:
        if res is not None:
            key, fit, rec = res
            fit_dict[key] = fit
            fits_list.append(rec)

    # Create summary DataFrame
    if not fits_list:
        summary_df = pd.DataFrame(columns=[
            "group_by", "group_key", "n_traj", "n_time", "rmse", "r2",
            "k_OI", "k_IO", "k_IC", "k_CI", "k_B",
            "E_open", "E_inter", "E_closed", "P_O0", "P_C0", "f_dyn", "E_static"
        ])
    else:
        summary_df = pd.DataFrame(fits_list)

    # --- SEQUENTIAL PLOTTING (if requested) ---
    # Must be done serially because Matplotlib is not thread-safe.
    if do_plots and fit_dict:
        print("\nGenerating requested plots sequentially...")
        for key in sorted(fit_dict.keys()):
            print(f"  Plotting {key}...")
            # Re-subset data just for plotting
            cols = meta.loc[meta[group_by] == key, "col"].tolist()
            t_sub, E_sub = subset_matrix_by_columns(t, E_mat, col_names, cols)

            plot_hsp90_fit_time(
                t_sub, E_sub, fit_dict[key],
                n_traces_overlay=max_overlay_traces,
                random_seed=0,
            )

    return summary_df, fit_dict

# ----------------------------------------------------------------------
# Global fitting using curve_fit (ensemble mean)
# ----------------------------------------------------------------------

def fit_global_3state(
        t: np.ndarray,
        E_mat: np.ndarray,
        theta0: np.ndarray = None,
) -> Hsp90Fit3State:
    """
    Fit the 3-state + bleaching model plus static fraction to the ensemble mean.
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_fit = t[row_valid]
    E_mean = np.nanmean(E_mat[row_valid, :], axis=1)

    mask = np.isfinite(E_mean)
    t_fit = t_fit[mask]
    E_fit = E_mean[mask]

    if t_fit.size == 0:
        raise RuntimeError("No valid data for fitting.")

    # theta = [k_OI, k_IO, k_IC, k_CI, k_B, E_o, E_i, E_c, P_O0, P_C0, f_dyn, E_static]
    if theta0 is None:
        theta0 = np.array([
            0.01, 0.01, 0.01, 0.01,  # conformational rates
            0.001,                   # bleaching rate
            0.1, 0.3, 0.6,           # FRET levels
            0.7, 0.2,                # P_O0, P_C0 (P_I0 = 0.1)
            0.7, 0.18                # f_dyn, E_static
        ], dtype=float)

    lower = np.array([
        0.0, 0.0, 0.0, 0.0,   # rates >= 0
        0.0,                  # k_B >= 0
        0.0, 0.0, 0.0,        # FRET levels >= 0
        0.0, 0.0,             # P_O0, P_C0 >= 0
        0.0, 0.0              # f_dyn, E_static in [0,1]
    ], dtype=float)

    upper = np.array([
        10.0, 10.0, 10.0, 10.0,  # conformational rates
        10.0,                    # k_B
        1.0, 1.0, 1.0,           # FRET levels in [0,1]
        1.0, 1.0,                # P_O0, P_C0 in [0,1]
        1.0, 1.0                 # f_dyn, E_static in [0,1]
    ], dtype=float)

    def fret_wrapper_3s(t_in,
                        k_oi, k_io, k_ic, k_ci, k_b,
                        e_o, e_i, e_c,
                        p_o0, p_c0,
                        f_dyn, e_static):
        params = Hsp90Params3State(
            k_OI=k_oi, k_IO=k_io, k_IC=k_ic, k_CI=k_ci,
            k_B=k_b,
            E_open=e_o, E_inter=e_i, E_closed=e_c,
            P_O0=p_o0, P_C0=p_c0
        )
        E_dyn = model_fret_3state(t_in, params)
        return f_dyn * E_dyn + (1.0 - f_dyn) * e_static

    popt, pcov = curve_fit(
        fret_wrapper_3s,
        t_fit,
        E_fit,
        p0=theta0,
        bounds=(lower, upper),
        maxfev=20000,
    )

    # Unpack fitted parameters
    (k_oi, k_io, k_ic, k_ci, k_b,
     e_o, e_i, e_c,
     p_o0, p_c0,
     f_dyn, e_static) = popt

    params = Hsp90Params3State(
        k_OI=float(k_oi),
        k_IO=float(k_io),
        k_IC=float(k_ic),
        k_CI=float(k_ci),
        k_B=float(k_b),
        E_open=float(e_o),
        E_inter=float(e_i),
        E_closed=float(e_c),
        P_O0=float(p_o0),
        P_C0=float(p_c0),
    )
    return Hsp90Fit3State(params=params,
                          f_dyn=float(f_dyn),
                          E_static=float(e_static))


# ----------------------------------------------------------------------
# Diagnostics / plotting
# ----------------------------------------------------------------------

def plot_ensemble_fit(t: np.ndarray, E_mat: np.ndarray, fit: Hsp90Fit3State) -> None:
    """
    Goodness-of-fit plot based on ensemble-averaged FRET.
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    if t_plot.size == 0:
        raise RuntimeError("No valid time points for ensemble fit plot.")

    E_mean = np.nanmean(E_plot, axis=1)
    E_std  = np.nanstd(E_plot, axis=1)

    E_model = model_total_fret(t_plot, fit)

    mask = np.isfinite(E_mean) & np.isfinite(E_model)
    E_obs = E_mean[mask]
    E_mod = E_model[mask]

    if E_obs.size == 0:
        raise RuntimeError("No valid (mean, model) pairs for goodness-of-fit plot.")

    residuals = E_obs - E_mod
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((E_obs - E_obs.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"Ensemble RMSE (mean data - model): {rmse:.6f}")
    print(f"Ensemble R^2: {r2:.6f}")

    xmin = np.min(E_mod)
    xmax = np.max(E_mod)
    ymin = np.min(E_obs)
    ymax = np.max(E_obs)
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    plt.figure(figsize=(6, 6))
    plt.scatter(E_mod, E_obs, s=20, alpha=0.8, label="Time points (ensemble mean)")
    plt.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")

    plt.xlabel("Model FRET (ensemble)")
    plt.ylabel("Observed FRET (ensemble mean)")
    plt.title(
        "Goodness of fit (ensemble mean vs model)\n"
        f"RMSE = {rmse:.4f}, R^2 = {r2:.4f}"
    )
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_hsp90_fit_time(
    t: np.ndarray,
    E_mat: np.ndarray,
    fit: Hsp90Fit3State,
    n_traces_overlay: int = 200,
    random_seed: int = 0,
) -> None:
    """
    Plot Hsp90 3-state + bleaching + static fraction vs data across time.
    """
    row_valid = np.isfinite(E_mat).any(axis=1)
    t_plot = t[row_valid]
    E_plot = E_mat[row_valid, :]

    E_mean = np.nanmean(E_plot, axis=1)
    E_std = np.nanstd(E_plot, axis=1)

    E_model = model_total_fret(t_plot, fit)

    n_traj_total = E_plot.shape[1]
    if n_traces_overlay > 0 and n_traj_total > 0:
        n_traces_overlay = min(n_traces_overlay, n_traj_total)
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_traj_total, size=n_traces_overlay, replace=False)
        E_subset = E_plot[:, idx]
    else:
        E_subset = None

    fig, ax = plt.subplots(figsize=(8, 4))

    if E_subset is not None:
        for j in range(E_subset.shape[1]):
            ax.plot(t_plot, E_subset[:, j], color="gray", alpha=0.05, lw=0.5)

    ax.plot(t_plot, E_mean, color="tab:blue", lw=2, label="Data mean (all trajectories)")
    ax.fill_between(
        t_plot,
        E_mean - E_std,
        E_mean + E_std,
        color="tab:blue",
        alpha=0.2,
        label="Data ±1 SD",
    )

    ax.plot(t_plot, E_model, color="tab:orange", lw=2, label="Model")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("FRET")
    ax.set_title(
        f"Hsp90 global 3-state + bleaching + static fraction\n"
        f"{E_plot.shape[1]} trajectories, {len(t_plot)} time points"
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def _bootstrap_worker(
        t_sub: np.ndarray,
        E_sub: np.ndarray,
        seed: int
) -> Optional[dict]:
    """
    Internal worker for a single bootstrap resample and fit.
    """
    try:
        # 1. Create a local RNG for this worker
        rng = np.random.default_rng(seed)

        # 2. Resample trajectories with replacement
        idx_boot = rng.integers(0, E_sub.shape[1], size=E_sub.shape[1])
        E_boot = E_sub[:, idx_boot]

        # 3. Fit the resampled data
        fit_b = fit_global_3state(t_sub, E_boot)

        # 4. Pack results
        p = fit_b.params
        rec = dict(
            k_OI=p.k_OI,
            k_IO=p.k_IO,
            k_IC=p.k_IC,
            k_CI=p.k_CI,
            k_B=p.k_B,
            E_open=p.E_open,
            E_inter=p.E_inter,
            E_closed=p.E_closed,
            P_O0=p.P_O0,
            P_C0=p.P_C0,
            f_dyn=fit_b.f_dyn,
            E_static=fit_b.E_static,
        )
        return rec
    except Exception:
        # Fit failed for this replicate, return None
        return None

def bootstrap_condition_params(
    t: np.ndarray,
    E_mat: np.ndarray,
    col_names: list[str],
    meta: pd.DataFrame,
    group_key: str,
    group_by: str = "condition",
    n_boot: int = 100,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap parameter estimates for a single condition by resampling trajectories.

    Returns a DataFrame with one row per bootstrap replicate and columns:
    k_OI, k_IO, k_IC, k_CI, k_B, E_open, E_inter, E_closed, P_O0, P_C0, f_dyn, E_static
    """
    cols_subset = meta.loc[meta[group_by] == group_key, "col"].tolist()
    if not cols_subset:
        raise ValueError(f"No columns for {group_by}={group_key}")

    # Build submatrix for this condition (once)
    name_to_idx = {c: i for i, c in enumerate(col_names)}
    idx_all = [name_to_idx[c] for c in cols_subset if c in name_to_idx]
    E_full = E_mat[:, idx_all]

    row_valid = np.isfinite(E_full).any(axis=1)
    t_sub = t[row_valid]
    E_sub = E_full[row_valid, :]

    print(f"Starting {n_boot} parallel bootstrap fits for {group_key}...")

    # --- PARALLEL EXECUTION ---
    # We pass t_sub and E_sub (large arrays) once,
    # and iterate over the unique seeds.
    # verbose=5 will show a progress bar
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_bootstrap_worker)(t_sub, E_sub, random_seed + b)
        for b in range(n_boot)
    )
    # ------------------------

    # Unpack results, filtering out any failed (None) runs
    records = [res for res in results if res is not None]

    if not records:
        print(f"Warning: All {n_boot} bootstrap fits failed for {group_key}.")
        return pd.DataFrame()

    return pd.DataFrame(records)

def plot_param_vs_condition(summary_df, param):
    plt.figure()
    plt.plot(summary_df["group_key"], summary_df[param], "o-")
    plt.xticks(rotation=90)
    plt.ylabel(param)
    plt.tight_layout()
    plt.show()

def summarize_bootstrap(boot_df, name):
    vals = boot_df[name].values
    mean = np.nanmean(vals)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return mean, lo, hi


def plot_bootstrap_compare(boot_A, boot_B, param, label_A, label_B):
    A_vals = boot_A[param].values
    B_vals = boot_B[param].values

    plt.figure()
    plt.hist(A_vals, bins=30, alpha=0.5, density=True, label=label_A)
    plt.hist(B_vals, bins=30, alpha=0.5, density=True, label=label_B)
    plt.xlabel(param)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    combined_path = Path("data/timeseries/combined_fret_matrix.csv")
    t, E_mat, col_names = load_combined_matrix(combined_path)

    print(f"Loaded combined matrix: {E_mat.shape[0]} time points, {E_mat.shape[1]} trajectories")

    # Global fit (all trajectories pooled)
    fit_hat = fit_global_3state(t, E_mat)
    print("\nBest-fit parameters (global):")
    print(fit_hat)

    # Global diagnostics
    plot_ensemble_fit(t, E_mat, fit_hat)
    plot_hsp90_fit_time(t, E_mat, fit_hat, n_traces_overlay=200)

    # Now: per-condition fits
    # 1) Per coverslip/day (construct + exp_id)
    print("\n=== Per-condition fits (construct + exp_id) ===")
    summary_cond, fits_cond = fit_all_conditions(
        t,
        E_mat,
        col_names,
        group_by="condition",     # "<construct>_<exp_id>"
        do_plots=True,           # set True if you want per-condition time plots
        max_overlay_traces=100,
    )
    print("\nCondition-level summary:")
    print(summary_cond.sort_values(["group_key"]))

    # 2) Per construct (pooling all days), if you want
    print("\n=== Per-construct fits (pool all exp_id for each construct) ===")
    summary_constr, fits_constr = fit_all_conditions(
        t,
        E_mat,
        col_names,
        group_by="construct",
        do_plots=False,
        max_overlay_traces=100,
    )
    print("\nConstruct-level summary:")
    print(summary_constr.sort_values(["group_key"]))

    plot_param_vs_condition(summary_cond, "k_OI")
    plot_param_vs_condition(summary_cond, "f_dyn")
    plot_param_vs_condition(summary_cond, "E_closed")

    meta = parse_column_metadata(col_names)

    cond_A = "Hsp90_409_601_241107"
    cond_B = "Hsp90_409_601_241108"

    boot_A = bootstrap_condition_params(t, E_mat, col_names, meta, cond_A, n_boot=100)
    boot_B = bootstrap_condition_params(t, E_mat, col_names, meta, cond_B, n_boot=100)

    for param in ["k_OI", "k_IC", "f_dyn", "E_closed"]:
        mA, loA, hiA = summarize_bootstrap(boot_A, param)
        mB, loB, hiB = summarize_bootstrap(boot_B, param)
        print(param)
        print(f"  cond A: mean={mA:.4f}, 95% CI [{loA:.4f}, {hiA:.4f}]")
        print(f"  cond B: mean={mB:.4f}, 95% CI [{loB:.4f}, {hiB:.4f}]")
        print()

        plot_bootstrap_compare(boot_A, boot_B, param, cond_A, cond_B)

if __name__ == "__main__":
    main()
