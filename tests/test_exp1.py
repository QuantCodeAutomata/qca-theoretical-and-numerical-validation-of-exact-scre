"""
Tests for Experiment 1: Theoretical Validation.

All tests use small dimensions (n <= 10, T <= 200) for speed (< 30 s total).
No mocks are used; all tests exercise real code paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.screening import validate_screening_decomposition
from src.projection import (
    validate_projection_decomposition_affine,
    validate_projection_decomposition_nonlinear,
)
from src.markowitz import validate_markowitz_identities, compute_optimal_portfolio
from src.perturbation import verify_perturbation_identities, verify_first_order_perturbation
from src.diagnostics import verify_diagnostic_bound, compute_residual_diagnostic
from src.intervention import run_intervention_experiment
from src.utils import make_spd, is_psd, is_pd
from data.simulate import simulate_ar1_drivers, simulate_asset_returns, get_population_covariance


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _small_setup(n: int = 5, m: int = 2, seed: int = 7):
    """Build a tiny SPD covariance, mu, A, b for Markowitz tests."""
    Q = make_spd(n, min_eig=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.05, size=n)
    A = np.ones((1, n))
    b = np.array([1.0])
    return Q, mu, A, b


# ------------------------------------------------------------------ #
# Test 1: AR(1) stationary variance
# ------------------------------------------------------------------ #

def test_ar1_stationary():
    """
    Stationary variance of Z_t should equal innov_var / (1 - phi^2).
    Verified empirically from a long simulation.
    """
    phi = 0.90
    innov_var = 0.0975
    expected_var = innov_var / (1.0 - phi**2)
    n_obs = 50_000
    Z, _ = simulate_ar1_drivers(n_drivers=1, n_obs=n_obs, phi=phi, innov_var=innov_var, seed=0)
    empirical_var = float(np.var(Z[:, 0]))
    # Allow 5% relative tolerance (large sample)
    assert abs(empirical_var - expected_var) / expected_var < 0.05, (
        f"Empirical var {empirical_var:.4f} deviates from theory {expected_var:.4f}"
    )


# ------------------------------------------------------------------ #
# Test 2: Screening decomposition
# ------------------------------------------------------------------ #

def test_screening_decomposition():
    """Relative Frobenius error of Q = D + Sigma_sys should be < 1e-10."""
    res = validate_screening_decomposition(n_assets=8, n_drivers=2, n_mc=500, seed=1)
    assert res["relative_frob_error"] < 1e-10, (
        f"relative_frob_error={res['relative_frob_error']:.3e} exceeds 1e-10"
    )


# ------------------------------------------------------------------ #
# Test 3: Projection decomposition — affine
# ------------------------------------------------------------------ #

def test_projection_affine_omega_zero():
    """omega_max_eig < 1e-8 for affine (linear) projection."""
    res = validate_projection_decomposition_affine(n_assets=8, n_drivers=2, n_mc=500, seed=2)
    assert res["omega_max_eig"] < 1e-8, (
        f"omega_max_eig={res['omega_max_eig']:.3e} exceeds 1e-8 in affine case"
    )


# ------------------------------------------------------------------ #
# Test 4: Projection decomposition — nonlinear
# ------------------------------------------------------------------ #

def test_projection_nonlinear_omega_psd():
    """Omega is PSD and nonzero for the nonlinear projection case."""
    res = validate_projection_decomposition_nonlinear(n_assets=8, n_drivers=2, n_mc=500, seed=3)
    assert res["omega_is_psd"], "Omega should be PSD in the nonlinear case"
    assert res["omega_max_eig"] > 0, (
        f"omega_max_eig={res['omega_max_eig']:.3e} should be > 0 in nonlinear case"
    )


# ------------------------------------------------------------------ #
# Test 5: Markowitz constraint
# ------------------------------------------------------------------ #

def test_markowitz_constraint():
    """Optimal portfolio should satisfy the equality constraint to < 1e-8."""
    res = validate_markowitz_identities(n_assets=8, n_constraints=1, gamma=1.0, seed=4)
    assert res["constraint_residual"] < 1e-8, (
        f"constraint_residual={res['constraint_residual']:.3e} exceeds 1e-8"
    )


# ------------------------------------------------------------------ #
# Test 6: Markowitz vs CVXPY
# ------------------------------------------------------------------ #

def test_markowitz_cvxpy_match():
    """Closed-form Markowitz should match CVXPY within 1e-6 relative error."""
    res = validate_markowitz_identities(n_assets=8, n_constraints=1, gamma=1.0, seed=5)
    rel_err = res["relative_error_vs_cvxpy"]
    assert not np.isnan(rel_err), "CVXPY did not converge"
    assert rel_err < 1e-6, (
        f"relative_error_vs_cvxpy={rel_err:.3e} exceeds 1e-6"
    )


# ------------------------------------------------------------------ #
# Test 7: Perturbation identities
# ------------------------------------------------------------------ #

def test_perturbation_identities():
    """All three perturbation identities should hold to < 1e-10."""
    n = 6
    Q0, mu, A, b = _small_setup(n=n, seed=10)
    rng = np.random.default_rng(10)
    B = rng.normal(0.0, 1.0, size=(n, 2))
    R_raw = B @ B.T
    from src.utils import op_norm
    R = 0.05 * R_raw / op_norm(R_raw)
    res = verify_perturbation_identities(Q0, R, mu, A, b, gamma=1.0)
    for key in ("id1a_frob", "id1b_frob", "id2a_frob", "id2b_frob", "id3_abs"):
        val = res[key]
        assert val < 1e-10, f"{key}={val:.3e} exceeds 1e-10"


# ------------------------------------------------------------------ #
# Test 8: First-order perturbation accuracy
# ------------------------------------------------------------------ #

def test_first_order_accuracy():
    """
    At very small tau, first-order approximation error should be small relative
    to the exact change.  The relative error is O(tau * ||M R||_op), so at
    tau=1e-6 with Q0 min_eig=0.1 we expect < 1e-3 relative error.
    """
    n = 6
    Q0, mu, A, b = _small_setup(n=n, seed=11)
    rng = np.random.default_rng(11)
    B = rng.normal(0.0, 1.0, size=(n, 2))
    R_template = B @ B.T
    from src.utils import op_norm
    R_template /= op_norm(R_template)

    tau_values = np.array([0.0, 1e-6, 2e-6])
    res = verify_first_order_perturbation(Q0, R_template, mu, A, b, gamma=1.0, tau_values=tau_values)

    # At tau=1e-6 relative error is O(tau * ||M @ R||_op) ≈ 1e-5, well below 1e-3
    exact_dw = res["exact_weight_change"][1]   # tau=1e-6
    fo_dw = res["first_order_weight_change"][1]
    norm_ex = float(np.linalg.norm(exact_dw))
    if norm_ex > 1e-14:
        rel_err = float(np.linalg.norm(exact_dw - fo_dw)) / norm_ex
        assert rel_err < 1e-3, f"First-order relative error={rel_err:.3e} at tau=1e-6 exceeds 1e-3"


# ------------------------------------------------------------------ #
# Test 9: Diagnostic bound
# ------------------------------------------------------------------ #

def test_diagnostic_bound():
    """All diagnostic bound chain inequalities should hold for every trial."""
    res = verify_diagnostic_bound(n_assets=10, n_trials=20, seed=42)
    assert res["all_bounds_hold"], (
        "Diagnostic bound chain violated in at least one trial: "
        f"op_le_frob={res['op_le_frob_frac']:.3f}, "
        f"frob_le_bound={res['frob_le_bound_frac']:.3f}"
    )


# ------------------------------------------------------------------ #
# Test 10: Intervention pre-equivalence
# ------------------------------------------------------------------ #

def test_intervention_pre_equivalence():
    """
    Pre-intervention common-state and proxy diagnostics should be close
    (both condition on information correlated with the same underlying state).
    """
    res = run_intervention_experiment(n_assets=10, n_replications=20, seed=42)
    diff = abs(res["pre_common_mean"] - res["pre_proxy_mean"])
    assert diff < 0.5, (
        f"Pre-intervention common={res['pre_common_mean']:.4f} and "
        f"proxy={res['pre_proxy_mean']:.4f} differ by {diff:.4f} > 0.5"
    )


# ------------------------------------------------------------------ #
# Test 11: Edge cases
# ------------------------------------------------------------------ #

def test_edge_cases():
    """Single asset, single driver: functions should not crash."""
    # Single asset, single driver drivers
    Z, eta = simulate_ar1_drivers(n_drivers=1, n_obs=100, phi=0.8, innov_var=0.05, seed=0)
    returns, b, c, vols = simulate_asset_returns(Z=Z, eta=eta, n_assets=1, seed=0)
    assert returns.shape == (99, 1)

    # Single-asset GMV (trivially w=1)
    Q_single = np.array([[0.01]])
    A = np.ones((1, 1))
    b_vec = np.array([1.0])
    mu_single = np.array([0.05])
    port = compute_optimal_portfolio(Q_single, mu_single, A, b_vec, gamma=1.0)
    assert abs(port["w_star"][0] - 1.0) < 1e-10


# ------------------------------------------------------------------ #
# Test 12: SPD generation
# ------------------------------------------------------------------ #

def test_spd_generation():
    """make_spd should return a positive definite matrix."""
    for n in [2, 5, 10]:
        A = make_spd(n, min_eig=0.01, seed=n)
        assert is_pd(A), f"make_spd({n}) did not return a PD matrix"


# ------------------------------------------------------------------ #
# Test 13: Population covariance PSD
# ------------------------------------------------------------------ #

def test_population_covariance_psd():
    """get_population_covariance should return a PSD matrix."""
    rng = np.random.default_rng(42)
    n_assets, n_drivers = 8, 3
    c_load = rng.normal(0.0, 0.5, size=(n_assets, n_drivers))
    innov_cov = 0.0975 * np.eye(n_drivers)
    idio_vols = rng.uniform(0.008, 0.02, size=n_assets)
    Q = get_population_covariance(c_load, innov_cov, idio_vols)
    assert is_psd(Q), "Population covariance is not PSD"
    assert Q.shape == (n_assets, n_assets)


# ------------------------------------------------------------------ #
# Test 14: GMV weights sum to 1
# ------------------------------------------------------------------ #

def test_gmv_weights():
    """GMV portfolio weights should sum to 1."""
    from src.covariance import compute_gmv_weights
    for n in [5, 10, 20]:
        Q = make_spd(n, min_eig=0.05, seed=n)
        w = compute_gmv_weights(Q)
        assert abs(w.sum() - 1.0) < 1e-10, f"GMV weights sum to {w.sum():.6f} for n={n}"
