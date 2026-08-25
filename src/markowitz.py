"""
Constrained Markowitz portfolio operators.

Problem:  max_{w}  w.T @ mu - (1 / (2*gamma)) * w.T @ Q @ w
          s.t.     A @ w = b

KKT solution:  w* = w0 + gamma * M @ mu

where
  QinvA   = Q^{-1} A.T
  AQinvAT = A @ Q^{-1} @ A.T   (Gram matrix of constraints in Q^{-1} metric)
  w0      = QinvA @ (AQinvAT)^{-1} @ b    (minimum-norm feasible point)
  M       = Q^{-1} - QinvA @ (AQinvAT)^{-1} @ QinvA.T  (complement operator)
"""

import warnings

import cvxpy as cp
import numpy as np
import scipy.linalg

from src.utils import frob_norm, is_pd, make_spd, relative_error


def compute_markowitz_operators(
    Q: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    Compute constrained Markowitz operators w0 and M.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Positive definite covariance matrix.
    A : np.ndarray, shape (m, n)
        Constraint matrix  (m constraints, n assets).
    b : np.ndarray, shape (m,)
        Constraint RHS.

    Returns
    -------
    dict with keys:
        w0       : (n,)   minimum-norm feasible point
        M        : (n, n) Markowitz complement operator
        QinvA    : (n, m) Q^{-1} @ A.T
        AQinvAT  : (m, m) A @ Q^{-1} @ A.T

    Raises
    ------
    ValueError
        If Q is not positive definite or if AQinvAT is ill-conditioned.
    """
    n = Q.shape[0]
    m = A.shape[0]

    cond_Q = np.linalg.cond(Q)
    if cond_Q > 1e12:
        warnings.warn(f"Q is ill-conditioned (cond={cond_Q:.2e}).", stacklevel=2)

    # QinvA = Q^{-1} A.T  via solving Q @ X = A.T
    QinvA = scipy.linalg.solve(Q, A.T, assume_a="pos")  # (n, m)

    # Gram matrix of constraints in the Q^{-1} metric
    AQinvAT = A @ QinvA  # (m, m)

    cond_gram = np.linalg.cond(AQinvAT)
    if cond_gram > 1e12:
        raise ValueError(
            f"AQinvAT is ill-conditioned (cond={cond_gram:.2e}). "
            "Check that constraints are linearly independent."
        )

    # w0 = Q^{-1} A.T (A Q^{-1} A.T)^{-1} b
    lam = scipy.linalg.solve(AQinvAT, b)   # Lagrange multipliers: (m,)
    w0 = QinvA @ lam                        # (n,)

    # M = Q^{-1} - Q^{-1} A.T (A Q^{-1} A.T)^{-1} A Q^{-1}
    #   = Q^{-1} - QinvA @ inv(AQinvAT) @ QinvA.T
    Qinv = scipy.linalg.solve(Q, np.eye(n), assume_a="pos")  # (n, n)
    M = Qinv - QinvA @ scipy.linalg.solve(AQinvAT, QinvA.T)  # (n, n)

    return {
        "w0": w0,
        "M": M,
        "QinvA": QinvA,
        "AQinvAT": AQinvAT,
    }


def compute_optimal_portfolio(
    Q: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    gamma: float,
) -> dict:
    """
    Compute the constrained mean-variance optimal portfolio.

    w* = w0 + gamma * M @ mu

    Portfolio moments:
      mu_p    = w0.T @ mu + gamma * Delta      (expected return)
      sigma2_p = w0.T @ Q @ w0 + gamma^2 * Delta  (portfolio variance)

    where Delta = mu.T @ M @ mu.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
    mu : np.ndarray, shape (n,)
        Expected returns.
    A : np.ndarray, shape (m, n)
    b : np.ndarray, shape (m,)
    gamma : float
        Risk-tolerance parameter (larger = more aggressive).

    Returns
    -------
    dict with keys:
        w_star             : (n,)   optimal portfolio weights
        w0                 : (n,)   minimum-norm feasible point
        M                  : (n, n) Markowitz complement
        mu_p               : float  portfolio expected return
        sigma2_p           : float  portfolio variance
        Delta              : float  mu.T @ M @ mu
        constraint_residual: float  ||A @ w_star - b||_2
    """
    ops = compute_markowitz_operators(Q, A, b)
    w0 = ops["w0"]
    M = ops["M"]

    # Optimal portfolio
    w_star = w0 + gamma * (M @ mu)  # (n,)

    # Quadratic form: Delta = mu.T M mu
    Delta = float(mu @ M @ mu)

    # Expected return and variance
    # mu_p = (w0 + gamma*M@mu).T @ mu = w0.T@mu + gamma * mu.T@M@mu
    mu_p = float(w0 @ mu) + gamma * Delta

    # Variance: w*.T Q w* = w0.T Q w0 + gamma^2 * Delta
    # (cross term vanishes because Q @ w0 = A.T @ lam and A @ M = 0)
    sigma2_p = float(w0 @ Q @ w0) + gamma**2 * Delta

    constraint_residual = float(np.linalg.norm(A @ w_star - b))

    return {
        "w_star": w_star,
        "w0": w0,
        "M": M,
        "mu_p": mu_p,
        "sigma2_p": sigma2_p,
        "Delta": Delta,
        "constraint_residual": constraint_residual,
    }


def validate_markowitz_identities(
    n_assets: int = 30,
    n_constraints: int = 3,
    gamma: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Verify constrained Markowitz identities numerically, cross-checking
    the closed-form solution against a CVXPY reference.

    Parameters
    ----------
    n_assets : int
    n_constraints : int
        Total number of linear equality constraints (includes budget row).
    gamma : float
    seed : int

    Returns
    -------
    dict with keys:
        constraint_residual     : ||A @ w_star - b||_2  (should be ~0)
        relative_error_vs_cvxpy : ||w_star - w_cvxpy||_2 / ||w_cvxpy||_2
        w_star                  : (n,)
        w_cvxpy                 : (n,)
        mu_p                    : float
        sigma2_p                : float
        Delta                   : float
    """
    rng = np.random.default_rng(seed)

    # Random SPD covariance
    Q = make_spd(n_assets, min_eig=0.05, seed=int(rng.integers(1e6)))

    # Random expected returns
    mu = rng.normal(0.0, 0.05, size=n_assets)

    # Budget constraint: 1.T @ w = 1
    A_budget = np.ones((1, n_assets))
    b_budget = np.ones(1)

    # Additional random sensitivity constraints
    if n_constraints > 1:
        A_extra = rng.normal(0.0, 1.0, size=(n_constraints - 1, n_assets))
        A = np.vstack([A_budget, A_extra])
        b_extra = rng.normal(0.0, 0.1, size=n_constraints - 1)
        b = np.concatenate([b_budget, b_extra])
    else:
        A = A_budget
        b = b_budget

    # Closed-form solution
    result = compute_optimal_portfolio(Q, mu, A, b, gamma)
    w_star = result["w_star"]

    # CVXPY reference solution
    w_var = cp.Variable(n_assets)
    objective = cp.Maximize(mu @ w_var - (0.5 / gamma) * cp.quad_form(w_var, Q))
    constraints = [A @ w_var == b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(f"CVXPY did not converge: status={prob.status}", stacklevel=2)
        w_cvxpy = np.full(n_assets, np.nan)
        rel_err = float("nan")
    else:
        w_cvxpy = w_var.value
        rel_err = relative_error(w_star, w_cvxpy, norm="frob")

    return {
        "constraint_residual": result["constraint_residual"],
        "relative_error_vs_cvxpy": rel_err,
        "w_star": w_star,
        "w_cvxpy": w_cvxpy,
        "mu_p": result["mu_p"],
        "sigma2_p": result["sigma2_p"],
        "Delta": result["Delta"],
    }
