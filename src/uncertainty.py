"""Uncertainty quantification for the CFRP damage chain.

Bootstrap idea: re-fit the model many times on randomly resampled versions of
the data. The spread of the resulting parameters shows how uncertain the fit is.
"""

import numpy as np
from scipy.optimize import curve_fit
from src.damage_models import (
    delamination_area, residual_strength, predict_strength_from_energy,
)

PARAM_NAMES = ["E_th", "c", "sigma0", "beta", "r"]


def fit_chain(E, A, S):
    """Fit both links of the chain. Returns (area_params, strength_params).

    Bounds keep the fits physically sensible (no negative thresholds, and the
    retained-strength fraction r must lie between 0 and 1).
    """
    pa, _ = curve_fit(
        delamination_area, E, A, p0=[4.0, 20.0],
        bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=10000,
    )
    ps, _ = curve_fit(
        residual_strength, A, S, p0=[400.0, 0.003, 0.5],
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]), maxfev=10000,
    )
    return pa, ps


def bootstrap_chain(E, A, S, n_boot=1000, seed=42):
    """Bootstrap the full fit. Returns array of shape (n_successful, 5) with
    columns E_th, c, sigma0, beta, r."""
    E, A, S = np.asarray(E), np.asarray(A), np.asarray(S)
    rng = np.random.default_rng(seed)
    n = len(E)
    results = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)   # pick n specimens, with replacement
        try:
            pa, ps = fit_chain(E[idx], A[idx], S[idx])
        except (RuntimeError, ValueError):
            continue                         # skip the rare failed fit
        results.append(np.concatenate([pa, ps]))
    return np.array(results)


def _curves(E_grid, params):
    """One predicted strength curve per parameter set."""
    E_grid = np.asarray(E_grid, dtype=float)
    return np.array([predict_strength_from_energy(E_grid, *p) for p in params])


def confidence_band(E_grid, params, level=95):
    """Band containing the true MEAN strength curve (parameter uncertainty only)."""
    curves = _curves(E_grid, params)
    tail = (100 - level) / 2
    return np.percentile(curves, tail, axis=0), np.percentile(curves, 100 - tail, axis=0)


def prediction_interval(E_grid, params, resid_sd, level=95, seed=0):
    """Band where a NEW specimen's strength would likely fall.

    Adds specimen-to-specimen scatter (resid_sd, in MPa) on top of the
    parameter uncertainty.
    """
    rng = np.random.default_rng(seed)
    curves = _curves(E_grid, params)
    curves = curves + rng.normal(0.0, resid_sd, size=curves.shape)
    tail = (100 - level) / 2
    return np.percentile(curves, tail, axis=0), np.percentile(curves, 100 - tail, axis=0)


def energy_for_strength_fraction(params, fraction):
    """Impact energy (J) at which strength falls to fraction * sigma0.

    Inverts the model analytically. Returns nan if the fraction is not
    between the plateau r and 1 (that strength level is never reached).
    """
    E_th, c, sigma0, beta, r = params
    if not (r < fraction < 1.0):
        return np.nan
    A = -np.log((fraction - r) / (1.0 - r)) / beta
    return E_th + A / c