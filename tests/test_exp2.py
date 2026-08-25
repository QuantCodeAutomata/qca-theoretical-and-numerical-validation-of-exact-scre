"""
Tests for Experiment 2: Two-Information-Window Screening Test.

Uses small n=10 assets and T=500 observations for speed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from data.simulate import simulate_ar1_drivers, simulate_asset_returns


PHI = 0.95
INNOV_VAR = 0.0975
LOADING_VAR = 0.5
N_ASSETS = 10
N_DRIVERS = 3
T_TOTAL = 500
SEED = 42


def _simulate_small():
    """Simulate a small panel and return data dict."""
    n_obs = T_TOTAL + 1
    Z, eta = simulate_ar1_drivers(
        n_drivers=N_DRIVERS, n_obs=n_obs, phi=PHI, innov_var=INNOV_VAR, seed=SEED
    )
    returns, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z, eta=eta, n_assets=N_ASSETS, loading_var=LOADING_VAR,
        idio_vol_range=(0.008, 0.02), seed=SEED,
    )
    Z_lag = Z[:-1]                   # (T_TOTAL, N_DRIVERS)
    eta_hat = Z[1:] - PHI * Z[:-1]  # (T_TOTAL, N_DRIVERS)
    return {
        "returns": returns,
        "Z_lag": Z_lag,
        "eta_hat": eta_hat,
    }


def _ols_residuals(returns, regressors):
    T = returns.shape[0]
    X = np.hstack([np.ones((T, 1)), regressors])
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)
    return returns - X @ coeffs


def _max_offdiag_corr(residuals):
    corr = np.corrcoef(residuals, rowvar=False)
    n = corr.shape[0]
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


def _r2(returns, regressors):
    """Average R^2 across assets."""
    T = returns.shape[0]
    X = np.hstack([np.ones((T, 1)), regressors])
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((returns - pred)**2, axis=0)
    ss_tot = np.sum((returns - returns.mean(0))**2, axis=0)
    r2_per_asset = 1.0 - ss_res / (ss_tot + 1e-15)
    return float(r2_per_asset.mean())


# ------------------------------------------------------------------ #
# Test 1: Simulation shape
# ------------------------------------------------------------------ #

def test_simulation_shape():
    """Check dimensions of simulated data."""
    data = _simulate_small()
    assert data["returns"].shape == (T_TOTAL, N_ASSETS)
    assert data["Z_lag"].shape == (T_TOTAL, N_DRIVERS)
    assert data["eta_hat"].shape == (T_TOTAL, N_DRIVERS)


# ------------------------------------------------------------------ #
# Test 2: Past-only residuals retain common risk
# ------------------------------------------------------------------ #

def test_past_only_residuals_correlated():
    """
    Past-only model (condition on Z_t only) should leave substantial
    residual correlation (common risk not fully removed).
    """
    data = _simulate_small()
    resid_past = _ols_residuals(data["returns"], data["Z_lag"])
    max_corr = _max_offdiag_corr(resid_past)
    assert max_corr > 0.3, (
        f"Past-only max off-diag corr={max_corr:.4f} should exceed 0.3 "
        "(common innovation risk retained)"
    )


# ------------------------------------------------------------------ #
# Test 3: Horizon-closed residuals near diagonal
# ------------------------------------------------------------------ #

def test_horizon_closed_residuals_near_diagonal():
    """
    Horizon-closed model (condition on Z_t and eta_{t+1}) should remove
    the common innovation component, leaving near-diagonal residuals.
    """
    data = _simulate_small()
    joint_regressors = np.hstack([data["Z_lag"], data["eta_hat"]])
    resid_hc = _ols_residuals(data["returns"], joint_regressors)
    max_corr = _max_offdiag_corr(resid_hc)
    # With loading_var=0.5 and idio vols in [0.008, 0.02], the residuals
    # should be much less correlated than the past-only case.
    assert max_corr < 0.5, (
        f"Horizon-closed max off-diag corr={max_corr:.4f} should be < 0.5 "
        "(innovation channel screened out)"
    )


# ------------------------------------------------------------------ #
# Test 4: Regression R2 >= 0
# ------------------------------------------------------------------ #

def test_model_fit():
    """Both regression models should achieve non-negative in-sample R2."""
    data = _simulate_small()

    r2_past = _r2(data["returns"], data["Z_lag"])
    assert r2_past >= 0.0, f"Past-only R2={r2_past:.4f} is negative"

    joint_regressors = np.hstack([data["Z_lag"], data["eta_hat"]])
    r2_hc = _r2(data["returns"], joint_regressors)
    assert r2_hc >= 0.0, f"Horizon-closed R2={r2_hc:.4f} is negative"

    # Horizon-closed should have at least as high R2 as past-only
    assert r2_hc >= r2_past - 1e-10, (
        f"Horizon-closed R2={r2_hc:.4f} should be >= past-only R2={r2_past:.4f}"
    )
