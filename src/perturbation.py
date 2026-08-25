"""
Exact finite covariance-perturbation identities for constrained Markowitz.

Let Q_R = Q0 + R be a perturbed covariance and M, M_R the corresponding
Markowitz complement operators.  The following exact identities hold:

  (I)   M_R - M   = -M  @ R @ M_R  =  -M_R @ R @ M
  (II)  w_R* - w* = -M_R @ R @ w*  =  -M  @ R @ w_R*
  (III) Delta_R - Delta = -(M @ mu).T @ R @ (M_R @ mu)

These follow from the KKT perturbation theory for the constrained QP.

First-order approximations (tau -> 0 with R = tau * R_template):
  M_R  ≈  M  - tau * M  @ R_template @ M
  w_R* ≈  w* - tau * M  @ R_template @ w*
  Delta_R ≈ Delta - tau * (M @ mu).T @ R_template @ (M @ mu)
"""

import numpy as np
import scipy.linalg

from src.markowitz import compute_markowitz_operators, compute_optimal_portfolio
from src.utils import frob_norm, make_spd


def compute_perturbed_operators(
    Q0: np.ndarray,
    R: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    Compute baseline and perturbed Markowitz complement operators.

    Parameters
    ----------
    Q0 : np.ndarray, shape (n, n)
        Baseline SPD covariance.
    R : np.ndarray, shape (n, n)
        Perturbation matrix (symmetric; Q0 + R must remain SPD).
    A : np.ndarray, shape (m, n)
    b : np.ndarray, shape (m,)

    Returns
    -------
    dict with keys:
        M  : (n, n) baseline complement operator for Q0
        MR : (n, n) perturbed complement operator for Q0 + R
    """
    Q_R = Q0 + R

    ops0 = compute_markowitz_operators(Q0, A, b)
    opsR = compute_markowitz_operators(Q_R, A, b)

    return {"M": ops0["M"], "MR": opsR["M"]}


def verify_perturbation_identities(
    Q0: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    gamma: float = 1.0,
) -> dict:
    """
    Verify the three exact perturbation identities (I)–(III).

    Parameters
    ----------
    Q0 : np.ndarray, shape (n, n)
    R  : np.ndarray, shape (n, n)
        Symmetric perturbation; Q0 + R must be SPD.
    mu : np.ndarray, shape (n,)
    A  : np.ndarray, shape (m, n)
    b  : np.ndarray, shape (m,)
    gamma : float

    Returns
    -------
    dict with keys:
        M, MR           : complement operators
        w_star, w_R_star: optimal portfolios
        Delta, Delta_R  : quadratic forms
        # Residuals (Frobenius norm of identity violations)
        id1a_frob : ||M_R - M - (-M @ R @ M_R)||_F
        id1b_frob : ||M_R - M - (-M_R @ R @ M)||_F
        id2a_frob : ||w_R* - w* - (-M_R @ R @ w*)||_2
        id2b_frob : ||w_R* - w* - (-M @ R @ w_R*)||_2
        id3_abs   : |Delta_R - Delta - (-(M@mu).T @ R @ (M_R@mu))|
    """
    Q_R = Q0 + R

    ops0 = compute_markowitz_operators(Q0, A, b)
    opsR = compute_markowitz_operators(Q_R, A, b)

    M = ops0["M"]
    MR = opsR["M"]

    res0 = compute_optimal_portfolio(Q0, mu, A, b, gamma)
    resR = compute_optimal_portfolio(Q_R, mu, A, b, gamma)

    w_star = res0["w_star"]
    w_R_star = resR["w_star"]
    Delta = res0["Delta"]
    Delta_R = resR["Delta"]

    # Identity (I): M_R - M = -M @ R @ M_R = -M_R @ R @ M
    dM = MR - M
    id1a_frob = frob_norm(dM - (-M @ R @ MR))
    id1b_frob = frob_norm(dM - (-MR @ R @ M))

    # Identity (II): w_R* - w* = -M_R @ R @ w* = -M @ R @ w_R*
    dw = w_R_star - w_star
    id2a_frob = float(np.linalg.norm(dw - (-MR @ R @ w_star)))
    id2b_frob = float(np.linalg.norm(dw - (-M @ R @ w_R_star)))

    # Identity (III): Delta_R - Delta = -(M @ mu).T @ R @ (M_R @ mu)
    dDelta = Delta_R - Delta
    id3_abs = abs(dDelta - (-(M @ mu) @ R @ (MR @ mu)))

    return {
        "M": M,
        "MR": MR,
        "w_star": w_star,
        "w_R_star": w_R_star,
        "Delta": Delta,
        "Delta_R": Delta_R,
        "id1a_frob": id1a_frob,
        "id1b_frob": id1b_frob,
        "id2a_frob": id2a_frob,
        "id2b_frob": id2b_frob,
        "id3_abs": id3_abs,
    }


def verify_first_order_perturbation(
    Q0: np.ndarray,
    R_template: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    gamma: float = 1.0,
    tau_values: np.ndarray = None,
) -> dict:
    """
    Compare exact perturbation changes to first-order approximations as tau -> 0.

    Perturbed covariance: Q_tau = Q0 + tau * R_template.

    First-order approximations:
      Delta w   ≈ -tau * M @ R_template @ w*       (weight change)
      Delta_tau ≈ Delta - tau * (M@mu).T R_template (M@mu)  (quad-form change)

    Parameters
    ----------
    Q0 : np.ndarray, shape (n, n)
    R_template : np.ndarray, shape (n, n)
        Direction of perturbation (symmetric).
    mu : np.ndarray, shape (n,)
    A : np.ndarray, shape (m, n)
    b : np.ndarray, shape (m,)
    gamma : float
    tau_values : np.ndarray, optional
        Perturbation magnitudes.  Defaults to linspace(0, 0.1, 20).

    Returns
    -------
    dict with keys:
        tau_values             : (k,)
        exact_weight_change    : (k, n) w*_tau - w*
        first_order_weight_change : (k, n)  -tau * M @ R_template @ w*
        exact_delta_change     : (k,)  Delta_tau - Delta
        first_order_delta_change : (k,)  -tau * (M@mu).T R_template (M@mu)
    """
    if tau_values is None:
        tau_values = np.linspace(0.0, 0.1, 20)

    # Baseline operators and portfolio
    ops0 = compute_markowitz_operators(Q0, A, b)
    M = ops0["M"]
    res0 = compute_optimal_portfolio(Q0, mu, A, b, gamma)
    w_star = res0["w_star"]
    Delta = res0["Delta"]

    # Pre-compute first-order ingredients
    MRw = M @ R_template @ w_star   # direction for weight change
    MmuRMmu = (M @ mu) @ R_template @ (M @ mu)  # scalar for Delta change

    n = Q0.shape[0]
    k = len(tau_values)
    exact_dw = np.zeros((k, n))
    fo_dw = np.zeros((k, n))
    exact_dDelta = np.zeros(k)
    fo_dDelta = np.zeros(k)

    for i, tau in enumerate(tau_values):
        R = tau * R_template
        Q_tau = Q0 + R
        res_tau = compute_optimal_portfolio(Q_tau, mu, A, b, gamma)

        exact_dw[i] = res_tau["w_star"] - w_star
        fo_dw[i] = -tau * MRw

        exact_dDelta[i] = res_tau["Delta"] - Delta
        fo_dDelta[i] = -tau * MmuRMmu

    return {
        "tau_values": tau_values,
        "exact_weight_change": exact_dw,
        "first_order_weight_change": fo_dw,
        "exact_delta_change": exact_dDelta,
        "first_order_delta_change": fo_dDelta,
    }
