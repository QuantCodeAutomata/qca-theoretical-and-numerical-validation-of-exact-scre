"""
Tests for Experiment 4: Perturbation Study.

Uses small n=8, m=2 for speed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.markowitz import compute_optimal_portfolio
from src.perturbation import verify_perturbation_identities
from src.utils import is_pd, make_spd, op_norm


def _baseline(n: int = 8, m: int = 2, seed: int = 42):
    """Construct a small baseline Q0, mu, A, b."""
    rng = np.random.default_rng(seed)
    sigma2 = rng.uniform(0.008**2, 0.02**2, size=n)
    D = np.diag(sigma2)
    L = rng.normal(0.0, np.sqrt(0.3), size=(n, m))
    Lambda = 0.0975 * np.eye(m)
    Q0 = D + L @ Lambda @ L.T
    mu = rng.normal(0.0, 0.001, size=n)
    A = np.ones((1, n))
    b = np.array([1.0])
    underobar_sigma2 = float(sigma2.min())
    return Q0, mu, A, b, underobar_sigma2


def _omega_template(n: int, m: int, seed: int = 10):
    """Random positive semidefinite perturbation template."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 1.0, size=(n, m))
    Omega = B @ B.T
    return Omega / op_norm(Omega)


def _e_template(n: int, seed: int = 20):
    """Random symmetric zero-diagonal perturbation template."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0.0, 1.0, size=(n, n))
    E = (E + E.T) / 2.0
    np.fill_diagonal(E, 0.0)
    return E / op_norm(E)


# ------------------------------------------------------------------ #
# Test 1: Perturbation identities exact (rho=0.1)
# ------------------------------------------------------------------ #

def test_perturbation_identities_exact():
    """All identity residuals < 1e-9 at rho=0.1 (exact algebraic identities)."""
    n, m = 8, 2
    Q0, mu, A, b, underobar_sigma2 = _baseline(n=n, m=m)
    Omega = _omega_template(n, m)
    rho = 0.1
    R = rho * underobar_sigma2 * Omega
    res = verify_perturbation_identities(Q0, R, mu, A, b, gamma=1.0)
    for key in ("id1a_frob", "id1b_frob", "id2a_frob", "id2b_frob", "id3_abs"):
        assert res[key] < 1e-9, f"{key}={res[key]:.3e} exceeds 1e-9 at rho=0.1"


# ------------------------------------------------------------------ #
# Test 2: Displacement monotone in rho
# ------------------------------------------------------------------ #

def test_displacement_monotone():
    """Portfolio displacement should increase (approximately) with rho."""
    n, m = 8, 2
    Q0, mu, A, b, underobar_sigma2 = _baseline(n=n, m=m)
    Omega = _omega_template(n, m)

    res0 = compute_optimal_portfolio(Q0, mu, A, b, gamma=1.0)
    w_star = res0["w_star"]

    rhos = [0.05, 0.15, 0.30, 0.50]
    displacements = []
    for rho in rhos:
        R = rho * underobar_sigma2 * Omega
        Q_R = Q0 + R
        if not is_pd(Q_R):
            continue
        res_R = compute_optimal_portfolio(Q_R, mu, A, b, gamma=1.0)
        w_R = res_R["w_star"]
        norm_wstar = np.linalg.norm(w_star)
        disp = float(np.linalg.norm(w_R - w_star)) / (norm_wstar + 1e-15)
        displacements.append(disp)

    # Displacements should be broadly increasing
    assert len(displacements) >= 3, "Too few valid rho values"
    n_increasing = sum(
        displacements[i] <= displacements[i + 1] * 1.5
        for i in range(len(displacements) - 1)
    )
    assert n_increasing >= len(displacements) - 2, (
        f"Displacement not broadly increasing: {[f'{d:.4f}' for d in displacements]}"
    )


# ------------------------------------------------------------------ #
# Test 3: Omega and E give different displacements
# ------------------------------------------------------------------ #

def test_omega_e_different_effects():
    """At the same rho, Omega and E perturbations produce different displacements."""
    n, m = 8, 2
    Q0, mu, A, b, underobar_sigma2 = _baseline(n=n, m=m)
    Omega = _omega_template(n, m)
    E = _e_template(n)

    res0 = compute_optimal_portfolio(Q0, mu, A, b, gamma=1.0)
    w_star = res0["w_star"]

    rho = 0.3

    def disp(R):
        Q_R = Q0 + R
        if not is_pd(Q_R):
            return float("nan")
        res = compute_optimal_portfolio(Q_R, mu, A, b, gamma=1.0)
        return float(np.linalg.norm(res["w_star"] - w_star))

    R_omega = rho * underobar_sigma2 * Omega
    R_E = rho * underobar_sigma2 * E

    d_omega = disp(R_omega)
    d_E = disp(R_E)

    assert not np.isnan(d_omega) and not np.isnan(d_E), "Q0+R not PD for rho=0.3"
    # The two channels should produce clearly different displacement magnitudes
    assert abs(d_omega - d_E) > 1e-8, (
        f"Omega and E channels give nearly identical displacements: {d_omega:.6f} vs {d_E:.6f}"
    )


# ------------------------------------------------------------------ #
# Test 4: Q0 + R_rho is PD for small rho
# ------------------------------------------------------------------ #

def test_positive_definiteness():
    """Q0 + R_rho should remain PD for rho in [0, 0.5]."""
    n, m = 8, 2
    Q0, mu, A, b, underobar_sigma2 = _baseline(n=n, m=m)
    Omega = _omega_template(n, m)

    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        R = rho * underobar_sigma2 * Omega
        Q_R = Q0 + R
        assert is_pd(Q_R), f"Q0 + R is not PD at rho={rho}"
