"""Step 5: Patlak graphical analysis and compartment model kinetics.

Part A: Patlak plot (0 to T_END min) to estimate Ki (net influx rate).
Part B: Compartment model fitting:
         - 1TCM (K1, k2): single tissue compartment
         - 2TCM irreversible (K1, k2, k3; k4=0)
         - 2TCM reversible (K1, k2, k3, k4)
         Vb (blood volume fraction) is fixed at a configurable value (default 5%).
"""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.stats import linregress

from pipeline.cache import check_outputs_current, log_skip, write_provenance_header
from pipeline.config import PipelineConfig

log = logging.getLogger("s05_kinetics")


# ---------------------------------------------------------------------------
# Patlak helpers
# ---------------------------------------------------------------------------

def compute_patlak(Ct, Cp, t_s, t_star_s, t_end_s):
    """Compute Patlak coordinates and linear regression.

    Parameters
    ----------
    Ct : array, tissue concentration (Bq/mL)
    Cp : array, plasma concentration (Bq/mL)
    t_s : array, mid-frame times (seconds)
    t_star_s : float, start of linear phase (seconds)
    t_end_s : float, end of analysis window (seconds)

    Returns
    -------
    dict with keys: X_min, Y, valid, fit_mask, Ki, Ki_se, V0, V0_se, R2,
                    n_fit, integral_Cp
    """
    n = len(Ct)

    # Cumulative integral of Cp via trapezoidal rule
    integral_Cp = np.zeros(n)
    if n > 1:
        integral_Cp[1:] = cumulative_trapezoid(Cp, x=t_s)

    # Valid frames: Cp must be large enough to avoid division noise
    Cp_max = np.max(Cp)
    epsilon = max(Cp_max * 1e-4, 1.0)  # at least 1 Bq/mL
    valid = (Cp > epsilon) & (t_s <= t_end_s)

    # Patlak coordinates
    X_s = np.full(n, np.nan)
    Y = np.full(n, np.nan)
    X_s[valid] = integral_Cp[valid] / Cp[valid]
    Y[valid] = Ct[valid] / Cp[valid]
    X_min = X_s / 60.0  # stretched time in minutes

    # Fit mask: valid frames within [t*, t_end]
    fit_mask = valid & (t_s >= t_star_s) & (t_s <= t_end_s)
    n_fit = int(np.sum(fit_mask))

    Ki, Ki_se, V0, V0_se, R2 = np.nan, np.nan, np.nan, np.nan, np.nan
    if n_fit >= 3:
        result = linregress(X_min[fit_mask], Y[fit_mask])
        Ki = result.slope
        Ki_se = result.stderr
        V0 = result.intercept
        V0_se = result.intercept_stderr
        R2 = result.rvalue ** 2
    else:
        log.warning("Patlak: only %d points in fit window, need >= 3", n_fit)

    return {
        "X_min": X_min, "Y": Y, "valid": valid, "fit_mask": fit_mask,
        "Ki": Ki, "Ki_se": Ki_se, "V0": V0, "V0_se": V0_se, "R2": R2,
        "n_fit": n_fit, "integral_Cp": integral_Cp,
    }


# ---------------------------------------------------------------------------
# Logan helpers
# ---------------------------------------------------------------------------

def compute_logan(Ct, Cp, t_s, t_star_s, t_end_s):
    """Compute Logan graphical analysis for VT estimation.

    Logan plot (reversible tracer):
        Y = integral_Ct(t) / Ct(t)
        X = integral_Cp(t) / Ct(t)
        Slope = VT (total distribution volume)

    Parameters
    ----------
    Ct : array, tissue concentration (Bq/mL)
    Cp : array, plasma concentration (Bq/mL)
    t_s : array, mid-frame times (seconds)
    t_star_s : float, start of linear phase (seconds)
    t_end_s : float, end of analysis window (seconds)

    Returns
    -------
    dict with keys: X_min, Y_min, valid, fit_mask, VT, VT_se,
                    intercept, intercept_se, R2, n_fit,
                    integral_Cp, integral_Ct
    """
    n = len(Ct)

    # Cumulative integrals via trapezoidal rule
    integral_Cp = np.zeros(n)
    integral_Ct = np.zeros(n)
    if n > 1:
        integral_Cp[1:] = cumulative_trapezoid(Cp, x=t_s)
        integral_Ct[1:] = cumulative_trapezoid(Ct, x=t_s)

    # Valid frames: Ct must be large enough to avoid division noise
    Ct_max = np.max(Ct)
    epsilon = max(Ct_max * 1e-4, 1.0)
    valid = (Ct > epsilon) & (t_s <= t_end_s)

    # Logan coordinates (in seconds, then convert to minutes for display)
    X_s = np.full(n, np.nan)
    Y_s = np.full(n, np.nan)
    X_s[valid] = integral_Cp[valid] / Ct[valid]
    Y_s[valid] = integral_Ct[valid] / Ct[valid]
    X_min = X_s / 60.0
    Y_min = Y_s / 60.0

    # Fit mask: valid frames within [t*, t_end]
    fit_mask = valid & (t_s >= t_star_s) & (t_s <= t_end_s)
    n_fit = int(np.sum(fit_mask))

    VT, VT_se, intercept, intercept_se, R2 = (
        np.nan, np.nan, np.nan, np.nan, np.nan
    )
    if n_fit >= 3:
        result = linregress(X_min[fit_mask], Y_min[fit_mask])
        VT = result.slope
        VT_se = result.stderr
        intercept = result.intercept
        intercept_se = result.intercept_stderr
        R2 = result.rvalue ** 2
    else:
        log.warning("Logan: only %d points in fit window, need >= 3", n_fit)

    return {
        "X_min": X_min, "Y_min": Y_min, "valid": valid, "fit_mask": fit_mask,
        "VT": VT, "VT_se": VT_se,
        "intercept": intercept, "intercept_se": intercept_se, "R2": R2,
        "n_fit": n_fit, "integral_Cp": integral_Cp, "integral_Ct": integral_Ct,
    }


# ---------------------------------------------------------------------------
# 2TCM helpers
# ---------------------------------------------------------------------------

def _tcm_forward(params, Cp_fine, t_fine_min, dt_min, Vb, reversible):
    """Compute model tissue concentration for given rate constants.

    All rate constants are in per-minute units. Time grid is in MINUTES.

    Parameters
    ----------
    params : array [K1, k2, k3] or [K1, k2, k3, k4]
        K1 in mL/mL/min, k2/k3/k4 in 1/min
    Cp_fine : array, plasma input on fine time grid (Bq/mL)
    t_fine_min : array, fine time grid (MINUTES)
    dt_min : float, time step (MINUTES)
    Vb : float, blood volume fraction
    reversible : bool

    Returns
    -------
    Ct_model : array, same length as t_fine_min
    """
    K1, k2, k3 = params[0], params[1], params[2]
    k4 = params[3] if reversible else 0.0

    # Eigenvalues and coefficients of IRF (all in 1/min)
    s = k2 + k3 + k4
    disc = s * s - 4.0 * k2 * k4
    if disc < 0:
        disc = 0.0
    sqrt_disc = np.sqrt(disc)

    b1 = (s + sqrt_disc) / 2.0
    b2 = (s - sqrt_disc) / 2.0

    # Avoid division by zero when b1 == b2
    if abs(b1 - b2) < 1e-12:
        # Degenerate case: single exponential
        IRF = K1 * np.exp(-b1 * t_fine_min)
    else:
        a1 = (k3 + k4 - b1) / (b2 - b1)
        a2 = 1.0 - a1
        IRF = K1 * (a1 * np.exp(-b1 * t_fine_min) + a2 * np.exp(-b2 * t_fine_min))

    # Convolution: C_tissue(t) = integral Cp(tau) * IRF(t-tau) d(tau)
    # dt_min is in minutes, matching the 1/min units of the rate constants
    C_tissue = np.convolve(Cp_fine, IRF)[:len(t_fine_min)] * dt_min

    # Total measured signal with blood volume correction
    Ct_model = (1.0 - Vb) * C_tissue + Vb * Cp_fine
    return Ct_model


def _tcm_residuals(params, Cp_fine, t_fine_min, dt_min, Vb, reversible,
                   Ct_measured, frame_indices, weights=None):
    """Residual function for least_squares (optionally weighted)."""
    Ct_model_fine = _tcm_forward(params, Cp_fine, t_fine_min, dt_min, Vb, reversible)
    Ct_model_at_frames = Ct_model_fine[frame_indices]
    residuals = Ct_model_at_frames - Ct_measured
    if weights is not None:
        residuals = residuals * weights
    return residuals


def fit_2tcm(Ct, Cp, t_s, t_end_s, Vb, dt_s, reversible, durations_s=None):
    """Fit 2-tissue compartment model.

    Parameters
    ----------
    Ct : array, measured tissue concentration (Bq/mL) at frame mid-times
    Cp : array, plasma concentration (Bq/mL) at frame mid-times
    t_s : array, frame mid-times (seconds)
    t_end_s : float, end of fitting window (seconds)
    Vb : float, blood volume fraction (fixed)
    dt_s : float, fine grid time step in seconds (converted to minutes internally)
    reversible : bool, if True fit k4, else k4=0
    durations_s : array or None, frame durations for weighting (longer = more counts)

    Returns
    -------
    dict with fitted parameters, standard errors, AIC, BIC, RSS, etc.
    """
    # Select frames within fitting window
    sel = t_s <= t_end_s
    Ct_sel = Ct[sel]
    Cp_sel = Cp[sel]
    t_sel = t_s[sel]
    n_frames = int(np.sum(sel))

    # Frame-duration weights: sqrt(duration) as proxy for count statistics
    if durations_s is not None:
        dur_sel = durations_s[sel]
        weights = np.sqrt(dur_sel / np.min(dur_sel))
    else:
        weights = np.ones(n_frames)

    # Convert to minutes for convolution (rate constants are in 1/min)
    t_sel_min = t_sel / 60.0
    dt_min = dt_s / 60.0
    t_max_min = t_sel_min[-1]

    # Build fine time grid in MINUTES
    t_fine_min = np.arange(0, t_max_min + dt_min, dt_min)

    # Interpolate Cp onto fine grid (using minutes)
    Cp_fine = np.interp(t_fine_min, t_sel_min, Cp_sel, left=0.0, right=Cp_sel[-1])

    # Map frame mid-times to fine grid indices
    frame_indices = np.round(t_sel_min / dt_min).astype(int)
    frame_indices = np.clip(frame_indices, 0, len(t_fine_min) - 1)

    # Initial guesses and bounds
    if reversible:
        init_sets = [
            np.array([0.05, 0.10, 0.01, 0.005]),
            np.array([0.02, 0.05, 0.005, 0.002]),
            np.array([0.10, 0.20, 0.02, 0.01]),
        ]
        lower = np.array([0.001, 0.001, 0.0001, 0.0001])
        upper = np.array([2.0, 2.0, 1.0, 1.0])
        n_params = 4
        param_names = ["K1", "k2", "k3", "k4"]
    else:
        init_sets = [
            np.array([0.05, 0.10, 0.01]),
            np.array([0.02, 0.05, 0.005]),
            np.array([0.10, 0.20, 0.02]),
        ]
        lower = np.array([0.001, 0.001, 0.0001])
        upper = np.array([2.0, 2.0, 1.0])
        n_params = 3
        param_names = ["K1", "k2", "k3"]

    # Try multiple initial guesses, keep best
    best_result = None
    best_cost = np.inf
    for x0 in init_sets:
        try:
            result = least_squares(
                _tcm_residuals, x0,
                args=(Cp_fine, t_fine_min, dt_min, Vb, reversible,
                      Ct_sel, frame_indices, weights),
                bounds=(lower, upper),
                method="trf",
                max_nfev=10000,
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue

    result = best_result
    if result is None:
        log.error("2TCM: all initial guesses failed")
        nan_arr = np.full(n_frames, np.nan)
        return {
            "K1": np.nan, "k2": np.nan, "k3": np.nan, "k4": np.nan,
            "K1_se": np.nan, "k2_se": np.nan, "k3_se": np.nan, "k4_se": np.nan,
            "Ki_derived": np.nan, "VT": np.nan, "VS": np.nan,
            "RSS": np.nan, "AIC": np.nan, "BIC": np.nan, "R2": np.nan,
            "n_frames": n_frames, "n_params": n_params,
            "Ct_model_at_frames": nan_arr, "t_frames": t_sel,
            "Ct_measured": Ct_sel, "residuals": nan_arr,
            "success": False, "message": "All initial guesses failed",
            "param_names": param_names, "popt": np.full(n_params, np.nan),
            "se": np.full(n_params, np.nan),
        }

    popt = result.x

    # Compute unweighted residuals for RSS and R2 (weighted used only for fitting)
    Ct_model_fine = _tcm_forward(popt, Cp_fine, t_fine_min, dt_min, Vb, reversible)
    Ct_model_at_frames = Ct_model_fine[frame_indices]
    residuals = Ct_model_at_frames - Ct_sel
    RSS = float(np.sum(residuals ** 2))

    # Standard errors from Jacobian
    J = result.jac
    try:
        # Covariance = (J^T J)^-1 * s^2, where s^2 = RSS / (n - k)
        cov = np.linalg.inv(J.T @ J) * (RSS / max(n_frames - n_params, 1))
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)
        log.warning("2TCM (%s): could not compute standard errors",
                    "reversible" if reversible else "irreversible")

    # AIC and BIC
    if RSS > 0 and n_frames > 0:
        AIC = n_frames * np.log(RSS / n_frames) + 2 * n_params
        BIC = n_frames * np.log(RSS / n_frames) + n_params * np.log(n_frames)
    else:
        AIC, BIC = np.nan, np.nan

    # Extract parameters
    K1, k2, k3 = popt[0], popt[1], popt[2]
    k4 = popt[3] if reversible else 0.0
    K1_se, k2_se, k3_se = se[0], se[1], se[2]
    k4_se = se[3] if reversible else 0.0

    # Derived kinetic parameters
    if reversible and k4 > 1e-8:
        Ki_derived = K1 * k3 / (k2 + k3 + k4)
        VT = K1 / k2 * (1.0 + k3 / k4)
        VS = K1 * k3 / (k2 * k4)
    else:
        Ki_derived = K1 * k3 / (k2 + k3) if (k2 + k3) > 1e-8 else np.nan
        VT = np.nan
        VS = np.nan

    # R-squared (using unweighted residuals computed above)
    SS_tot = np.sum((Ct_sel - np.mean(Ct_sel)) ** 2)
    R2 = 1.0 - RSS / SS_tot if SS_tot > 0 else np.nan

    return {
        "K1": K1, "k2": k2, "k3": k3, "k4": k4,
        "K1_se": K1_se, "k2_se": k2_se, "k3_se": k3_se, "k4_se": k4_se,
        "Ki_derived": Ki_derived, "VT": VT, "VS": VS,
        "RSS": RSS, "AIC": AIC, "BIC": BIC, "R2": R2,
        "n_frames": n_frames, "n_params": n_params,
        "Ct_model_at_frames": Ct_model_at_frames,
        "t_frames": t_sel,
        "Ct_measured": Ct_sel,
        "residuals": residuals,
        "success": result.success,
        "message": result.message,
        "param_names": param_names,
        "popt": popt,
        "se": se,
    }


# ---------------------------------------------------------------------------
# 1TCM helpers
# ---------------------------------------------------------------------------

def _1tcm_forward(params, Cp_fine, t_fine_min, dt_min, Vb):
    """Compute 1TCM tissue concentration: IRF(t) = K1 * exp(-k2 * t).

    Parameters
    ----------
    params : array [K1, k2]
    Cp_fine, t_fine_min, dt_min, Vb : as in _tcm_forward
    """
    K1, k2 = params[0], params[1]
    IRF = K1 * np.exp(-k2 * t_fine_min)
    C_tissue = np.convolve(Cp_fine, IRF)[:len(t_fine_min)] * dt_min
    Ct_model = (1.0 - Vb) * C_tissue + Vb * Cp_fine
    return Ct_model


def _1tcm_residuals(params, Cp_fine, t_fine_min, dt_min, Vb,
                    Ct_measured, frame_indices, weights=None):
    """Residual function for 1TCM least_squares."""
    Ct_model_fine = _1tcm_forward(params, Cp_fine, t_fine_min, dt_min, Vb)
    Ct_model_at_frames = Ct_model_fine[frame_indices]
    residuals = Ct_model_at_frames - Ct_measured
    if weights is not None:
        residuals = residuals * weights
    return residuals


def fit_1tcm(Ct, Cp, t_s, t_end_s, Vb, dt_s, durations_s=None):
    """Fit 1-tissue compartment model (K1, k2).

    Parameters
    ----------
    Ct, Cp, t_s, t_end_s, Vb, dt_s, durations_s : same as fit_2tcm

    Returns
    -------
    dict with K1, k2, standard errors, VT, AIC, BIC, RSS, etc.
    """
    sel = t_s <= t_end_s
    Ct_sel = Ct[sel]
    Cp_sel = Cp[sel]
    t_sel = t_s[sel]
    n_frames = int(np.sum(sel))
    n_params = 2
    param_names = ["K1", "k2"]

    # Frame-duration weights
    if durations_s is not None:
        dur_sel = durations_s[sel]
        weights = np.sqrt(dur_sel / np.min(dur_sel))
    else:
        weights = np.ones(n_frames)

    # Convert to minutes
    t_sel_min = t_sel / 60.0
    dt_min = dt_s / 60.0
    t_max_min = t_sel_min[-1]

    t_fine_min = np.arange(0, t_max_min + dt_min, dt_min)
    Cp_fine = np.interp(t_fine_min, t_sel_min, Cp_sel, left=0.0, right=Cp_sel[-1])

    frame_indices = np.round(t_sel_min / dt_min).astype(int)
    frame_indices = np.clip(frame_indices, 0, len(t_fine_min) - 1)

    # Multiple initial guesses
    init_sets = [
        np.array([0.05, 0.10]),
        np.array([0.02, 0.05]),
        np.array([0.10, 0.20]),
        np.array([0.01, 0.03]),
    ]
    lower = np.array([0.001, 0.001])
    upper = np.array([2.0, 2.0])

    best_result = None
    best_cost = np.inf
    for x0 in init_sets:
        try:
            result = least_squares(
                _1tcm_residuals, x0,
                args=(Cp_fine, t_fine_min, dt_min, Vb,
                      Ct_sel, frame_indices, weights),
                bounds=(lower, upper),
                method="trf",
                max_nfev=10000,
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue

    result = best_result
    if result is None:
        log.error("1TCM: all initial guesses failed")
        nan_arr = np.full(n_frames, np.nan)
        return {
            "K1": np.nan, "k2": np.nan, "k3": np.nan, "k4": np.nan,
            "K1_se": np.nan, "k2_se": np.nan, "k3_se": np.nan, "k4_se": np.nan,
            "Ki_derived": np.nan, "VT": np.nan, "VS": np.nan,
            "RSS": np.nan, "AIC": np.nan, "BIC": np.nan, "R2": np.nan,
            "n_frames": n_frames, "n_params": n_params,
            "Ct_model_at_frames": nan_arr, "t_frames": t_sel,
            "Ct_measured": Ct_sel, "residuals": nan_arr,
            "success": False, "message": "All initial guesses failed",
            "param_names": param_names, "popt": np.full(n_params, np.nan),
            "se": np.full(n_params, np.nan),
        }

    popt = result.x

    # Unweighted residuals for RSS and R2
    Ct_model_fine = _1tcm_forward(popt, Cp_fine, t_fine_min, dt_min, Vb)
    Ct_model_at_frames = Ct_model_fine[frame_indices]
    residuals = Ct_model_at_frames - Ct_sel
    RSS = float(np.sum(residuals ** 2))

    # Standard errors
    J = result.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (RSS / max(n_frames - n_params, 1))
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)
        log.warning("1TCM: could not compute standard errors")

    # AIC and BIC
    if RSS > 0 and n_frames > 0:
        AIC = n_frames * np.log(RSS / n_frames) + 2 * n_params
        BIC = n_frames * np.log(RSS / n_frames) + n_params * np.log(n_frames)
    else:
        AIC, BIC = np.nan, np.nan

    K1, k2 = popt[0], popt[1]
    K1_se, k2_se = se[0], se[1]

    # Derived: VT = K1/k2
    VT = K1 / k2 if k2 > 1e-8 else np.nan

    # R-squared
    SS_tot = np.sum((Ct_sel - np.mean(Ct_sel)) ** 2)
    R2 = 1.0 - RSS / SS_tot if SS_tot > 0 else np.nan

    return {
        "K1": K1, "k2": k2, "k3": np.nan, "k4": np.nan,
        "K1_se": K1_se, "k2_se": k2_se, "k3_se": np.nan, "k4_se": np.nan,
        "Ki_derived": np.nan, "VT": VT, "VS": np.nan,
        "RSS": RSS, "AIC": AIC, "BIC": BIC, "R2": R2,
        "n_frames": n_frames, "n_params": n_params,
        "Ct_model_at_frames": Ct_model_at_frames,
        "t_frames": t_sel,
        "Ct_measured": Ct_sel,
        "residuals": residuals,
        "success": result.success,
        "message": result.message,
        "param_names": param_names,
        "popt": popt,
        "se": se,
    }


# ---------------------------------------------------------------------------
# Main step function
# ---------------------------------------------------------------------------

def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """Patlak graphical analysis and 2-tissue compartment model fitting."""
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s05: Kinetic modeling (Patlak + 2TCM)")
    log.info("Subject: %s", subject_id)

    sub = f"sub-{subject_id}"

    # --- Inputs ---
    tac_raw_path = cfg.outputs_dir() / f"{sub}_tac-raw.tsv"
    idif_path = cfg.outputs_dir() / f"{sub}_idif.tsv"

    # --- Outputs ---
    outputs = {
        "patlak_tsv": cfg.outputs_dir() / f"{sub}_patlak.tsv",
        "logan_tsv": cfg.outputs_dir() / f"{sub}_logan.tsv",
        "results_tsv": cfg.outputs_dir() / f"{sub}_kinetics-results.tsv",
        "fig_patlak": cfg.figures_dir() / f"{sub}_patlak.png",
        "fig_logan": cfg.figures_dir() / f"{sub}_logan.png",
        "fig_tcm_fit": cfg.figures_dir() / f"{sub}_tcm-fit.png",
        "fig_tcm_fit_log": cfg.figures_dir() / f"{sub}_tcm-fit-log.png",
        "fig_tcm_residuals": cfg.figures_dir() / f"{sub}_tcm-residuals.png",
    }

    # --- Cache check ---
    input_paths = [tac_raw_path, idif_path]
    output_paths = list(outputs.values())
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s05_kinetics", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    log.info("Patlak: t*=%.1f min, t_end=%.1f min",
             cfg.PATLAK_T_STAR_MIN, cfg.PATLAK_T_END_MIN)
    log.info("2TCM: t_end=%.1f min, Vb=%.3f, dt=%.1f s",
             cfg.TCM_T_END_MIN, cfg.TCM_VB, cfg.TCM_DT_S)

    # --- Load data ---
    brain_tac = pd.read_csv(tac_raw_path, sep="\t", comment="#")
    idif = pd.read_csv(idif_path, sep="\t", comment="#")
    log.info("Brain TAC: %d frames", len(brain_tac))
    log.info("IDIF: %d frames", len(idif))

    if len(brain_tac) != len(idif):
        raise ValueError(
            f"Frame count mismatch: brain TAC has {len(brain_tac)}, "
            f"IDIF has {len(idif)}"
        )
    if not np.allclose(
        brain_tac["mid_time_s"].values, idif["mid_time_s"].values, atol=0.5
    ):
        raise ValueError("Frame mid-times do not match between brain TAC and IDIF")

    Ct = brain_tac["mean_activity_Bq_per_mL"].values.astype(np.float64)
    Cp = idif["mean_activity_Bq_per_mL"].values.astype(np.float64)
    t_s = brain_tac["mid_time_s"].values.astype(np.float64)
    t_min = brain_tac["mid_time_min"].values.astype(np.float64)
    durations_s = (brain_tac["end_s"].values - brain_tac["start_s"].values).astype(np.float64)

    # =====================================================================
    # Part A: Patlak graphical analysis
    # =====================================================================
    log.info("-" * 40)
    log.info("Part A: Patlak graphical analysis")

    t_star_s = cfg.PATLAK_T_STAR_MIN * 60.0
    t_end_s_patlak = cfg.PATLAK_T_END_MIN * 60.0

    patlak = compute_patlak(Ct, Cp, t_s, t_star_s, t_end_s_patlak)

    log.info("Patlak: %d valid frames, %d in fit window",
             int(np.sum(patlak["valid"])), patlak["n_fit"])
    log.info("  Ki = %.6f +/- %.6f mL/mL/min", patlak["Ki"], patlak["Ki_se"])
    log.info("  V0 = %.4f +/- %.4f mL/mL", patlak["V0"], patlak["V0_se"])
    log.info("  R2 = %.6f", patlak["R2"])

    if patlak["R2"] < 0.9:
        log.warning("Patlak R2 = %.4f < 0.9: linearity may be poor", patlak["R2"])

    # Save Patlak per-frame TSV
    in_window = t_s <= t_end_s_patlak
    patlak_df = pd.DataFrame({
        "frame": brain_tac["frame"].values,
        "mid_time_min": t_min,
        "Ct_Bq_per_mL": Ct,
        "Cp_Bq_per_mL": Cp,
        "integral_Cp_Bq_s_per_mL": patlak["integral_Cp"],
        "X_stretched_time_min": patlak["X_min"],
        "Y_ratio_mL_per_mL": patlak["Y"],
        "in_window": in_window.astype(int),
        "used_in_fit": patlak["fit_mask"].astype(int),
    })

    outputs["patlak_tsv"].parent.mkdir(parents=True, exist_ok=True)
    with open(outputs["patlak_tsv"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s05_kinetics.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "t_star_min": cfg.PATLAK_T_STAR_MIN,
                "t_end_min": cfg.PATLAK_T_END_MIN,
                "n_frames_fit": patlak["n_fit"],
            },
            extra_lines=[
                f"Ki_mL_per_mL_per_min: {patlak['Ki']:.6f}",
                f"Ki_se: {patlak['Ki_se']:.6f}",
                f"V0_mL_per_mL: {patlak['V0']:.4f}",
                f"V0_se: {patlak['V0_se']:.4f}",
                f"R2: {patlak['R2']:.6f}",
            ],
        )
        patlak_df.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote: %s", outputs["patlak_tsv"])

    # --- Patlak figure ---
    plt.style.use(cfg.FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    # All valid points in window (pre-equilibrium, gray)
    pre_eq = patlak["valid"] & ~patlak["fit_mask"] & in_window
    if np.any(pre_eq):
        ax.plot(patlak["X_min"][pre_eq], patlak["Y"][pre_eq],
                "o", color="gray", alpha=0.5, markersize=6,
                label="Pre-equilibrium", zorder=2)

    # Annotate each pre-eq point with its time
    for i in np.where(pre_eq)[0]:
        ax.annotate(f"{t_min[i]:.1f}",
                    (patlak["X_min"][i], patlak["Y"][i]),
                    fontsize=7, color="gray", alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))

    # Fit points (steelblue)
    if np.any(patlak["fit_mask"]):
        ax.plot(patlak["X_min"][patlak["fit_mask"]],
                patlak["Y"][patlak["fit_mask"]],
                "o", color="steelblue", markersize=7,
                label=f"Fit points (t >= {cfg.PATLAK_T_STAR_MIN:.0f} min)",
                zorder=3)

        # Annotate fit points with time
        for i in np.where(patlak["fit_mask"])[0]:
            ax.annotate(f"{t_min[i]:.0f}",
                        (patlak["X_min"][i], patlak["Y"][i]),
                        fontsize=7, color="steelblue", alpha=0.8,
                        textcoords="offset points", xytext=(4, 4))

    # Regression line
    if not np.isnan(patlak["Ki"]):
        x_fit = patlak["X_min"][patlak["fit_mask"]]
        x_line = np.linspace(np.nanmin(x_fit), np.nanmax(x_fit), 100)
        y_line = patlak["Ki"] * x_line + patlak["V0"]
        ax.plot(x_line, y_line, "-", color="crimson", linewidth=2, zorder=4,
                label=(f"Ki = {patlak['Ki']:.5f} +/- {patlak['Ki_se']:.5f} mL/mL/min\n"
                       f"V0 = {patlak['V0']:.4f} +/- {patlak['V0_se']:.4f} mL/mL\n"
                       f"R\u00b2 = {patlak['R2']:.4f} (n = {patlak['n_fit']})"))

    ax.set_xlabel("Stretched Time (min)", fontsize=12)
    ax.set_ylabel("Ct(t) / Cp(t)  (mL/mL)", fontsize=12)
    ax.set_title(f"{subject_id}: Patlak Plot", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig.savefig(outputs["fig_patlak"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_patlak"])

    # =====================================================================
    # Part A-2: Logan graphical analysis
    # =====================================================================
    log.info("-" * 40)
    log.info("Part A-2: Logan graphical analysis")

    logan = compute_logan(Ct, Cp, t_s, t_star_s, t_end_s_patlak)

    log.info("Logan: %d valid frames, %d in fit window",
             int(np.sum(logan["valid"])), logan["n_fit"])
    log.info("  VT = %.4f +/- %.4f mL/mL", logan["VT"], logan["VT_se"])
    log.info("  intercept = %.4f +/- %.4f", logan["intercept"], logan["intercept_se"])
    log.info("  R2 = %.6f", logan["R2"])

    if logan["R2"] < 0.9:
        log.warning("Logan R2 = %.4f < 0.9: linearity may be poor", logan["R2"])

    # Save Logan per-frame TSV
    logan_df = pd.DataFrame({
        "frame": brain_tac["frame"].values,
        "mid_time_min": t_min,
        "Ct_Bq_per_mL": Ct,
        "Cp_Bq_per_mL": Cp,
        "integral_Cp_Bq_s_per_mL": logan["integral_Cp"],
        "integral_Ct_Bq_s_per_mL": logan["integral_Ct"],
        "X_int_Cp_over_Ct_min": logan["X_min"],
        "Y_int_Ct_over_Ct_min": logan["Y_min"],
        "in_window": in_window.astype(int),
        "used_in_fit": logan["fit_mask"].astype(int),
    })

    with open(outputs["logan_tsv"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s05_kinetics.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "t_star_min": cfg.PATLAK_T_STAR_MIN,
                "t_end_min": cfg.PATLAK_T_END_MIN,
                "n_frames_fit": logan["n_fit"],
            },
            extra_lines=[
                f"VT_mL_per_mL: {logan['VT']:.4f}",
                f"VT_se: {logan['VT_se']:.4f}",
                f"intercept: {logan['intercept']:.4f}",
                f"intercept_se: {logan['intercept_se']:.4f}",
                f"R2: {logan['R2']:.6f}",
            ],
        )
        logan_df.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote: %s", outputs["logan_tsv"])

    # --- Logan figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pre-equilibrium points (gray)
    pre_eq_logan = logan["valid"] & ~logan["fit_mask"] & in_window
    if np.any(pre_eq_logan):
        ax.plot(logan["X_min"][pre_eq_logan], logan["Y_min"][pre_eq_logan],
                "o", color="gray", alpha=0.5, markersize=6,
                label="Pre-equilibrium", zorder=2)
        for i in np.where(pre_eq_logan)[0]:
            ax.annotate(f"{t_min[i]:.1f}",
                        (logan["X_min"][i], logan["Y_min"][i]),
                        fontsize=7, color="gray", alpha=0.7,
                        textcoords="offset points", xytext=(4, 4))

    # Fit points (teal)
    if np.any(logan["fit_mask"]):
        ax.plot(logan["X_min"][logan["fit_mask"]],
                logan["Y_min"][logan["fit_mask"]],
                "o", color="teal", markersize=7,
                label=f"Fit points (t >= {cfg.PATLAK_T_STAR_MIN:.0f} min)",
                zorder=3)
        for i in np.where(logan["fit_mask"])[0]:
            ax.annotate(f"{t_min[i]:.0f}",
                        (logan["X_min"][i], logan["Y_min"][i]),
                        fontsize=7, color="teal", alpha=0.8,
                        textcoords="offset points", xytext=(4, 4))

    # Regression line
    if not np.isnan(logan["VT"]):
        x_fit_logan = logan["X_min"][logan["fit_mask"]]
        x_line_logan = np.linspace(np.nanmin(x_fit_logan), np.nanmax(x_fit_logan), 100)
        y_line_logan = logan["VT"] * x_line_logan + logan["intercept"]
        ax.plot(x_line_logan, y_line_logan, "-", color="crimson", linewidth=2, zorder=4,
                label=(f"VT = {logan['VT']:.4f} +/- {logan['VT_se']:.4f} mL/mL\n"
                       f"intercept = {logan['intercept']:.4f} +/- {logan['intercept_se']:.4f}\n"
                       f"R\u00b2 = {logan['R2']:.4f} (n = {logan['n_fit']})"))

    ax.set_xlabel(r"$\int C_p(\tau)\,d\tau \;/\; C_t(t)$  (min)", fontsize=12)
    ax.set_ylabel(r"$\int C_t(\tau)\,d\tau \;/\; C_t(t)$  (min)", fontsize=12)
    ax.set_title(f"{subject_id}: Logan Plot", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig.savefig(outputs["fig_logan"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_logan"])

    # =====================================================================
    # Part B: 2-Tissue Compartment Model
    # =====================================================================
    log.info("-" * 40)
    log.info("Part B: 2-Tissue Compartment Model fitting")

    t_end_s_tcm = cfg.TCM_T_END_MIN * 60.0

    # --- Irreversible 2TCM (K1, k2, k3; k4=0) ---
    log.info("Fitting irreversible 2TCM (3 params: K1, k2, k3)...")
    irrev = fit_2tcm(Ct, Cp, t_s, t_end_s_tcm, cfg.TCM_VB, cfg.TCM_DT_S,
                     reversible=False, durations_s=durations_s)
    log.info("  K1 = %.5f +/- %.5f mL/mL/min", irrev["K1"], irrev["K1_se"])
    log.info("  k2 = %.5f +/- %.5f 1/min", irrev["k2"], irrev["k2_se"])
    log.info("  k3 = %.5f +/- %.5f 1/min", irrev["k3"], irrev["k3_se"])
    log.info("  Ki (derived) = %.6f mL/mL/min", irrev["Ki_derived"])
    log.info("  R2 = %.6f, RSS = %.2f", irrev["R2"], irrev["RSS"])
    log.info("  AIC = %.2f, BIC = %.2f", irrev["AIC"], irrev["BIC"])
    log.info("  Converged: %s (%s)", irrev["success"], irrev["message"])

    # --- Reversible 2TCM (K1, k2, k3, k4) ---
    log.info("Fitting reversible 2TCM (4 params: K1, k2, k3, k4)...")
    rev = fit_2tcm(Ct, Cp, t_s, t_end_s_tcm, cfg.TCM_VB, cfg.TCM_DT_S,
                   reversible=True, durations_s=durations_s)
    log.info("  K1 = %.5f +/- %.5f mL/mL/min", rev["K1"], rev["K1_se"])
    log.info("  k2 = %.5f +/- %.5f 1/min", rev["k2"], rev["k2_se"])
    log.info("  k3 = %.5f +/- %.5f 1/min", rev["k3"], rev["k3_se"])
    log.info("  k4 = %.5f +/- %.5f 1/min", rev["k4"], rev["k4_se"])
    log.info("  Ki (derived) = %.6f mL/mL/min", rev["Ki_derived"])
    log.info("  VT = %.4f mL/mL", rev["VT"])
    log.info("  R2 = %.6f, RSS = %.2f", rev["R2"], rev["RSS"])
    log.info("  AIC = %.2f, BIC = %.2f", rev["AIC"], rev["BIC"])
    log.info("  Converged: %s (%s)", rev["success"], rev["message"])

    # --- 1TCM (K1, k2) ---
    log.info("Fitting 1TCM (2 params: K1, k2)...")
    onetcm = fit_1tcm(Ct, Cp, t_s, t_end_s_tcm, cfg.TCM_VB, cfg.TCM_DT_S,
                       durations_s=durations_s)
    log.info("  K1 = %.5f +/- %.5f mL/mL/min", onetcm["K1"], onetcm["K1_se"])
    log.info("  k2 = %.5f +/- %.5f 1/min", onetcm["k2"], onetcm["k2_se"])
    log.info("  VT = %.4f mL/mL", onetcm["VT"])
    log.info("  R2 = %.6f, RSS = %.2f", onetcm["R2"], onetcm["RSS"])
    log.info("  AIC = %.2f, BIC = %.2f", onetcm["AIC"], onetcm["BIC"])
    log.info("  Converged: %s (%s)", onetcm["success"], onetcm["message"])

    # Model comparison (all three)
    aic_models = {
        "1TCM": onetcm["AIC"],
        "irreversible_2TCM": irrev["AIC"],
        "reversible_2TCM": rev["AIC"],
    }
    preferred = min(aic_models, key=aic_models.get)
    log.info("Model comparison (AIC): 1TCM=%.2f, irrev_2TCM=%.2f, rev_2TCM=%.2f",
             onetcm["AIC"], irrev["AIC"], rev["AIC"])
    log.info("  Preferred: %s", preferred)

    # --- Results summary TSV ---
    results_df = pd.DataFrame([
        {
            "model": "1TCM",
            "K1": onetcm["K1"], "K1_se": onetcm["K1_se"],
            "k2": onetcm["k2"], "k2_se": onetcm["k2_se"],
            "k3": np.nan, "k3_se": np.nan,
            "k4": np.nan, "k4_se": np.nan,
            "Ki_derived": np.nan,
            "VT": onetcm["VT"], "VS": np.nan,
            "Vb_fixed": cfg.TCM_VB,
            "R2": onetcm["R2"], "RSS": onetcm["RSS"],
            "AIC": onetcm["AIC"], "BIC": onetcm["BIC"],
            "n_frames": onetcm["n_frames"], "n_params": onetcm["n_params"],
            "converged": onetcm["success"],
        },
        {
            "model": "irreversible_2TCM",
            "K1": irrev["K1"], "K1_se": irrev["K1_se"],
            "k2": irrev["k2"], "k2_se": irrev["k2_se"],
            "k3": irrev["k3"], "k3_se": irrev["k3_se"],
            "k4": 0.0, "k4_se": 0.0,
            "Ki_derived": irrev["Ki_derived"],
            "VT": np.nan, "VS": np.nan,
            "Vb_fixed": cfg.TCM_VB,
            "R2": irrev["R2"], "RSS": irrev["RSS"],
            "AIC": irrev["AIC"], "BIC": irrev["BIC"],
            "n_frames": irrev["n_frames"], "n_params": irrev["n_params"],
            "converged": irrev["success"],
        },
        {
            "model": "reversible_2TCM",
            "K1": rev["K1"], "K1_se": rev["K1_se"],
            "k2": rev["k2"], "k2_se": rev["k2_se"],
            "k3": rev["k3"], "k3_se": rev["k3_se"],
            "k4": rev["k4"], "k4_se": rev["k4_se"],
            "Ki_derived": rev["Ki_derived"],
            "VT": rev["VT"], "VS": rev["VS"],
            "Vb_fixed": cfg.TCM_VB,
            "R2": rev["R2"], "RSS": rev["RSS"],
            "AIC": rev["AIC"], "BIC": rev["BIC"],
            "n_frames": rev["n_frames"], "n_params": rev["n_params"],
            "converged": rev["success"],
        },
    ])

    # Add Patlak row
    patlak_row = {
        "model": "Patlak",
        "K1": np.nan, "K1_se": np.nan,
        "k2": np.nan, "k2_se": np.nan,
        "k3": np.nan, "k3_se": np.nan,
        "k4": np.nan, "k4_se": np.nan,
        "Ki_derived": patlak["Ki"],
        "VT": np.nan, "VS": np.nan,
        "Vb_fixed": np.nan,
        "R2": patlak["R2"], "RSS": np.nan,
        "AIC": np.nan, "BIC": np.nan,
        "n_frames": patlak["n_fit"], "n_params": 2,
        "converged": True,
    }
    results_df = pd.concat([results_df, pd.DataFrame([patlak_row])],
                           ignore_index=True)

    # Add Logan row
    logan_row = {
        "model": "Logan",
        "K1": np.nan, "K1_se": np.nan,
        "k2": np.nan, "k2_se": np.nan,
        "k3": np.nan, "k3_se": np.nan,
        "k4": np.nan, "k4_se": np.nan,
        "Ki_derived": np.nan,
        "VT": logan["VT"], "VS": np.nan,
        "Vb_fixed": np.nan,
        "R2": logan["R2"], "RSS": np.nan,
        "AIC": np.nan, "BIC": np.nan,
        "n_frames": logan["n_fit"], "n_params": 2,
        "converged": True,
    }
    results_df = pd.concat([results_df, pd.DataFrame([logan_row])],
                           ignore_index=True)

    with open(outputs["results_tsv"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s05_kinetics.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "patlak_t_star_min": cfg.PATLAK_T_STAR_MIN,
                "patlak_t_end_min": cfg.PATLAK_T_END_MIN,
                "tcm_t_end_min": cfg.TCM_T_END_MIN,
                "tcm_Vb": cfg.TCM_VB,
                "tcm_dt_s": cfg.TCM_DT_S,
            },
            extra_lines=[
                f"preferred_model: {preferred} (by AIC)",
                f"patlak_Ki: {patlak['Ki']:.6f}",
                f"logan_VT: {logan['VT']:.4f}",
                f"1tcm_K1: {onetcm['K1']:.5f}",
                f"1tcm_VT: {onetcm['VT']:.4f}",
                f"irrev_K1: {irrev['K1']:.5f}",
                f"irrev_Ki: {irrev['Ki_derived']:.6f}",
                f"rev_K1: {rev['K1']:.5f}",
                f"rev_Ki: {rev['Ki_derived']:.6f}",
            ],
        )
        results_df.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote: %s", outputs["results_tsv"])

    # =====================================================================
    # Figures
    # =====================================================================

    sel_tcm = t_s <= t_end_s_tcm

    # --- Helper to plot fit time-course (reused for linear and log) ---
    def _plot_fit_panel(ax, use_log=False):
        """Plot measured data, IDIF, and all model fits."""
        # IDIF at true scale
        ax.plot(t_min[sel_tcm], Cp[sel_tcm], "-", color="crimson",
                linewidth=1.5, alpha=0.5, label="IDIF (Cp)", zorder=1)

        # Measured brain TAC
        ax.plot(t_min[sel_tcm], Ct[sel_tcm], "o", color="steelblue",
                markersize=5, label="Measured Ct (brain)", zorder=3)

        # 1TCM
        ax.plot(onetcm["t_frames"] / 60.0, onetcm["Ct_model_at_frames"],
                "d-", color="mediumpurple", markersize=4, linewidth=1.5, alpha=0.8,
                label=(f"1TCM: K1={onetcm['K1']:.4f}, k2={onetcm['k2']:.4f}, "
                       f"VT={onetcm['VT']:.3f}"),
                zorder=4)

        # Irreversible 2TCM
        ax.plot(irrev["t_frames"] / 60.0, irrev["Ct_model_at_frames"],
                "s--", color="darkorange", markersize=4, linewidth=1.5, alpha=0.8,
                label=(f"Irrev 2TCM: K1={irrev['K1']:.4f}, "
                       f"k2={irrev['k2']:.4f}, k3={irrev['k3']:.4f}"),
                zorder=4)

        # Reversible 2TCM
        ax.plot(rev["t_frames"] / 60.0, rev["Ct_model_at_frames"],
                "^--", color="forestgreen", markersize=4, linewidth=1.5, alpha=0.8,
                label=(f"Rev 2TCM: K1={rev['K1']:.4f}, "
                       f"k2={rev['k2']:.4f}, k3={rev['k3']:.4f}, k4={rev['k4']:.4f}"),
                zorder=4)

        ax.set_ylabel("Activity (Bq/mL)", fontsize=12)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim(left=0)
        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=50)
        else:
            ax.set_ylim(bottom=0)

    # --- Figure 2: Fit time-course (linear scale) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _plot_fit_panel(axes[0], use_log=False)
    axes[0].set_title(
        f"{subject_id}: Compartment Model Fits (Vb={cfg.TCM_VB:.0%} fixed)",
        fontsize=14)

    # Bottom panel: residuals
    ax = axes[1]
    bw = 0.4
    ax.bar(onetcm["t_frames"] / 60.0 - bw, onetcm["residuals"],
           width=bw, color="mediumpurple", alpha=0.6, label="1TCM")
    ax.bar(irrev["t_frames"] / 60.0, irrev["residuals"],
           width=bw, color="darkorange", alpha=0.6, label="Irrev 2TCM")
    ax.bar(rev["t_frames"] / 60.0 + bw, rev["residuals"],
           width=bw, color="forestgreen", alpha=0.6, label="Rev 2TCM")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Residual (Bq/mL)", fontsize=12)
    ax.set_title("Residuals", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(outputs["fig_tcm_fit"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_tcm_fit"])

    # --- Figure 2b: Fit time-course (LOG scale) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _plot_fit_panel(axes[0], use_log=True)
    axes[0].set_title(
        f"{subject_id}: Compartment Model Fits, log scale (Vb={cfg.TCM_VB:.0%} fixed)",
        fontsize=14)

    # Bottom panel: residuals (same as linear)
    ax = axes[1]
    ax.bar(onetcm["t_frames"] / 60.0 - bw, onetcm["residuals"],
           width=bw, color="mediumpurple", alpha=0.6, label="1TCM")
    ax.bar(irrev["t_frames"] / 60.0, irrev["residuals"],
           width=bw, color="darkorange", alpha=0.6, label="Irrev 2TCM")
    ax.bar(rev["t_frames"] / 60.0 + bw, rev["residuals"],
           width=bw, color="forestgreen", alpha=0.6, label="Rev 2TCM")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Residual (Bq/mL)", fontsize=12)
    ax.set_title("Residuals", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(outputs["fig_tcm_fit_log"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_tcm_fit_log"])

    # --- Figure 3: Residuals detail (3 panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model, result, color in [
        (axes[0], "1TCM", onetcm, "mediumpurple"),
        (axes[1], "Irrev 2TCM", irrev, "darkorange"),
        (axes[2], "Rev 2TCM", rev, "forestgreen"),
    ]:
        ax.plot(result["t_frames"] / 60.0, result["residuals"],
                "o-", color=color, markersize=4)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("Residual (Bq/mL)", fontsize=11)
        ax.set_title(
            f"{model}: R\u00b2={result['R2']:.4f}, AIC={result['AIC']:.1f}",
            fontsize=11,
        )
        ax.set_xlim(left=0)

    fig.suptitle(f"{subject_id}: Model Residuals", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(outputs["fig_tcm_residuals"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_tcm_residuals"])

    # --- Final summary ---
    log.info("=" * 40)
    log.info("KINETICS SUMMARY")
    log.info("  Patlak Ki: %.6f mL/mL/min (R2=%.4f)", patlak["Ki"], patlak["R2"])
    log.info("  Logan  VT: %.4f mL/mL (R2=%.4f)", logan["VT"], logan["R2"])
    log.info("  1TCM:       K1=%.5f, k2=%.5f, VT=%.4f (R2=%.4f, AIC=%.1f)",
             onetcm["K1"], onetcm["k2"], onetcm["VT"], onetcm["R2"], onetcm["AIC"])
    log.info("  Irrev 2TCM: K1=%.5f, Ki=%.6f (R2=%.4f, AIC=%.1f)",
             irrev["K1"], irrev["Ki_derived"], irrev["R2"], irrev["AIC"])
    log.info("  Rev 2TCM:   K1=%.5f, Ki=%.6f, VT=%.4f (R2=%.4f, AIC=%.1f)",
             rev["K1"], rev["Ki_derived"], rev["VT"], rev["R2"], rev["AIC"])
    log.info("  Preferred model (AIC): %s", preferred)
    log.info("  VT cross-check: Logan=%.4f vs 1TCM=%.4f (ratio=%.3f)",
             logan["VT"], onetcm["VT"],
             logan["VT"] / onetcm["VT"] if onetcm["VT"] > 1e-8 else np.nan)

    elapsed = time.time() - t0
    log.info("DONE s05: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(
        description="Step 5: Patlak + 2TCM kinetic modeling"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
