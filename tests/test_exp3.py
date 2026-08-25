"""
Tests for Experiment 3: Monte Carlo GMV Covariance Estimation.

Uses small n=10 to keep tests fast (< 30 s).

Design notes:
- structured/PCA estimators estimate the CONDITIONAL covariance (given Z_t)
- sample/LW estimate the UNCONDITIONAL covariance
- oracle for OOS portfolio comparisons uses the UNCONDITIONAL population covariance
- structured vs sample quality is measured by Frobenius error vs conditional population
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from data.simulate import (
    get_population_covariance,
    simulate_ar1_drivers,
    simulate_asset_returns,
)
from src.covariance import (
    compute_gmv_weights,
    evaluate_gmv_out_of_sample,
    fit_ledoit_wolf,
    fit_sample_covariance,
    fit_structured_covariance,
)
from src.utils import frob_norm, is_pd, is_psd, make_spd


N_ASSETS = 10
N_DRIVERS = 3
T_TRAIN = 120
T_TEST = 300   # large test set for stable OOS vol estimates
PHI = 0.95
INNOV_VAR = 0.0975
LOADING_VAR = 0.3
SEED = 7


def _simulate_panel(n: int = N_ASSETS, T_train: int = T_TRAIN, T_test: int = T_TEST,
                    seed: int = SEED):
    """Simulate a panel; return train/test splits and population covariances."""
    n_obs = T_train + T_test + 1
    Z, eta = simulate_ar1_drivers(n_drivers=N_DRIVERS, n_obs=n_obs, phi=PHI,
                                   innov_var=INNOV_VAR, seed=seed)
    returns, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z, eta=eta, n_assets=n, loading_var=LOADING_VAR,
        idio_vol_range=(0.008, 0.02), seed=seed,
    )
    innov_cov = INNOV_VAR * np.eye(N_DRIVERS)
    # Conditional population covariance (used to evaluate structured estimator quality)
    Q_cond = get_population_covariance(c_load, innov_cov, idio_vols)
    # Unconditional population covariance (correct oracle for OOS vol comparison)
    state_var = INNOV_VAR / (1.0 - PHI**2)
    Q_uncond = Q_cond + b_load @ (state_var * np.eye(N_DRIVERS)) @ b_load.T

    train_ret = returns[:T_train]
    test_ret = returns[T_train:]
    train_Z = Z[:T_train]
    return train_ret, test_ret, train_Z, Q_cond, Q_uncond, b_load, c_load


# ------------------------------------------------------------------ #
# Test 1: Structured covariance is better estimator of conditional Q
# ------------------------------------------------------------------ #

def test_structured_beats_sample_small_T():
    """
    At small T, structured estimator should have lower Frobenius error vs the
    conditional population covariance than the sample estimator.
    Tested over 5 seeds; majority should pass.

    Rationale: the structured estimator exploits the factor model structure,
    so it estimates the conditional covariance more accurately than sample,
    which also captures the large state-driven (b @ Z_t) component.
    """
    wins = 0
    n_trials = 5
    for seed in range(n_trials):
        train_ret, _, train_Z, Q_cond, _, _, _ = _simulate_panel(seed=seed + 100)
        Q_sample = fit_sample_covariance(train_ret)
        Q_struct = fit_structured_covariance(train_ret, train_Z)["Q_struct"]
        err_sample = frob_norm(Q_sample - Q_cond)
        err_struct = frob_norm(Q_struct - Q_cond)
        if err_struct < err_sample:
            wins += 1
    assert wins >= 3, (
        f"Structured estimator had lower Frobenius error (vs conditional Q) in only "
        f"{wins}/{n_trials} seeds. Expected majority wins."
    )


# ------------------------------------------------------------------ #
# Test 2: Oracle (unconditional) achieves low OOS vol vs sample/LW
# ------------------------------------------------------------------ #

def test_oracle_is_best():
    """
    Oracle using unconditional population covariance should achieve lower or
    comparable OOS vol vs sample and LW (which also estimate unconditional Q).
    Uses large T_test=300 for stable OOS vol estimates.
    """
    wins = 0
    n_trials = 5
    for seed in range(n_trials):
        train_ret, test_ret, train_Z, _, Q_uncond, _, _ = _simulate_panel(seed=seed + 200)
        res = evaluate_gmv_out_of_sample(train_ret, test_ret, train_Z, k_pca=3,
                                          oracle_Q=Q_uncond)
        if "oracle" not in res:
            continue
        v_oracle = res["oracle"]["realized_vol_ann"]
        # Compare only to unconditional estimators (sample and LW)
        uncond_vols = [
            res[m]["realized_vol_ann"]
            for m in ("sample", "ledoit_wolf")
            if m in res
        ]
        if all(v_oracle <= v * 1.1 for v in uncond_vols):  # within 10%
            wins += 1
    assert wins >= 3, (
        f"Oracle only beat sample/LW in {wins}/5 seeds (allowing 10% slack). "
        "Expected majority wins with large T_test."
    )


# ------------------------------------------------------------------ #
# Test 3: Calibration ratio finite
# ------------------------------------------------------------------ #

def test_calibration_ratio_finite():
    """All calibration ratios (realized_var / predicted_var) should be positive and finite."""
    train_ret, test_ret, train_Z, Q_cond, Q_uncond, _, _ = _simulate_panel()
    res = evaluate_gmv_out_of_sample(train_ret, test_ret, train_Z, k_pca=3,
                                      oracle_Q=Q_uncond)
    for method, vals in res.items():
        ratio = vals.get("ratio", None)
        if ratio is None or np.isnan(ratio):
            continue  # equal_weight ratio is NaN by design
        assert ratio > 0, f"Method {method}: ratio={ratio} is not positive"
        assert np.isfinite(ratio), f"Method {method}: ratio={ratio} is not finite"


# ------------------------------------------------------------------ #
# Test 4: Ledoit-Wolf returns PD
# ------------------------------------------------------------------ #

def test_ledoit_wolf_returns_spd():
    """Ledoit-Wolf covariance estimate should be positive definite."""
    train_ret, _, _, _, _, _, _ = _simulate_panel()
    Q_lw = fit_ledoit_wolf(train_ret)
    assert Q_lw.shape == (N_ASSETS, N_ASSETS), "LW covariance has wrong shape"
    assert is_pd(Q_lw), "Ledoit-Wolf covariance is not positive definite"
