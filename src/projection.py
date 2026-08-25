"""
Validate the innovation projection decomposition.

The decomposition is:

    Sigma_sys = L @ Lambda @ L.T + Omega

where
  xi_tilde  in R^{n_drivers}  are the centered innovations
  Lambda    = Cov(xi_tilde)
  L         = Cov(m_sys, xi_tilde) @ inv(Lambda)   (regression of m_sys on xi)
  u         = m_sys - L @ xi_tilde                   (projection residual)
  Omega     = Cov(u)  >= 0

Affine case:  m_sys = B @ xi_tilde  => L = B, Omega = 0 (exact)
Nonlinear:    m_sys = B @ xi_tilde + c*(xi_tilde^2 - E[xi^2]) => Omega > 0
"""

import numpy as np

from src.utils import frob_norm, is_psd, op_norm, relative_error


def validate_projection_decomposition_affine(
    n_assets: int = 20,
    n_drivers: int = 3,
    n_mc: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Verify the projection decomposition in the affine (linear) case.

    Design: m_sys = B @ xi_tilde  where  xi_tilde ~ N(0, Lambda).
    Since the relationship is linear, the projection residual u = 0, so Omega = 0.

    Parameters
    ----------
    n_assets : int
    n_drivers : int
    n_mc : int
        Monte Carlo sample size.
    seed : int

    Returns
    -------
    dict with keys:
        Lambda              : (n_drivers, n_drivers) innovation covariance
        L                   : (n_assets, n_drivers) projection matrix
        Omega               : (n_assets, n_assets)  residual covariance (~0)
        Sigma_sys_reconstructed : L @ Lambda @ L.T + Omega
        Sigma_sys_true      : B @ Lambda @ B.T  (population)
        omega_max_eig       : maximum eigenvalue of Omega  (should be ~0)
        relative_error      : ||Sigma_sys_rec - Sigma_sys_true||_F / ||Sigma_sys_true||_F
    """
    rng = np.random.default_rng(seed)

    # Random innovation loadings  B: (n_assets, n_drivers)
    B = rng.normal(0.0, 1.0, size=(n_assets, n_drivers))

    # Random positive-definite Lambda (n_drivers x n_drivers)
    F = rng.normal(0.0, 1.0, size=(n_drivers, n_drivers))
    Lambda = (F.T @ F) / n_drivers + 0.1 * np.eye(n_drivers)

    # Population systematic covariance
    Sigma_sys_true = B @ Lambda @ B.T  # (n_assets, n_assets)

    # ------------------------------------------------------------------ #
    # Monte Carlo: xi ~ N(0, Lambda), m_sys = B @ xi
    # ------------------------------------------------------------------ #
    L_chol = np.linalg.cholesky(Lambda)  # Lambda = L_chol @ L_chol.T
    xi_raw = rng.normal(0.0, 1.0, size=(n_mc, n_drivers))
    xi = xi_raw @ L_chol.T  # (n_mc, n_drivers)  ~  N(0, Lambda)

    m_sys = xi @ B.T  # (n_mc, n_assets);  m_sys[k] = B @ xi[k]

    # Lambda_mc = sample covariance of xi
    Lambda_mc = np.cov(xi, rowvar=False)

    # L = Cov(m_sys, xi) @ inv(Lambda_mc)
    # Cov(m_sys, xi) has shape (n_assets, n_drivers)
    Cov_m_xi = (m_sys - m_sys.mean(0)).T @ (xi - xi.mean(0)) / (n_mc - 1)
    L = Cov_m_xi @ np.linalg.inv(Lambda_mc)  # (n_assets, n_drivers)

    # Projection residual: u = m_sys - xi @ L.T
    u = m_sys - xi @ L.T  # (n_mc, n_assets)

    Omega = np.cov(u, rowvar=False)  # should be ~0

    # Reconstructed systematic covariance
    Sigma_sys_rec = L @ Lambda_mc @ L.T + Omega

    omega_max_eig = float(np.linalg.eigvalsh(Omega).max())
    rel_err = relative_error(Sigma_sys_rec, Sigma_sys_true, norm="frob")

    return {
        "Lambda": Lambda_mc,
        "L": L,
        "Omega": Omega,
        "Sigma_sys_reconstructed": Sigma_sys_rec,
        "Sigma_sys_true": Sigma_sys_true,
        "omega_max_eig": omega_max_eig,
        "relative_error": rel_err,
    }


def validate_projection_decomposition_nonlinear(
    n_assets: int = 20,
    n_drivers: int = 3,
    n_mc: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Verify the projection decomposition in the nonlinear case.

    Design: m_sys = B @ xi_tilde + c_nl * (xi_tilde^2 - E[xi_tilde^2])
    where xi_tilde^2 denotes element-wise squaring.

    Because the mapping is nonlinear, the best linear projection of m_sys
    onto xi leaves a nonzero residual, so Omega > 0.

    Parameters
    ----------
    n_assets : int
    n_drivers : int
    n_mc : int
    seed : int

    Returns
    -------
    dict with keys:
        Lambda              : (n_drivers, n_drivers) innovation covariance
        L                   : (n_assets, n_drivers) linear projection matrix
        Omega               : (n_assets, n_assets)  residual covariance (> 0)
        Sigma_sys_reconstructed : L @ Lambda @ L.T + Omega
        Sigma_sys_true      : sample Cov(m_sys)
        omega_max_eig       : maximum eigenvalue of Omega  (should be > 0)
        omega_is_psd        : bool, whether Omega is PSD
        relative_error      : ||Sigma_sys_rec - Sigma_sys_true||_F / ||Sigma_sys_true||_F
    """
    rng = np.random.default_rng(seed)

    B = rng.normal(0.0, 1.0, size=(n_assets, n_drivers))
    c_nl = rng.normal(0.0, 0.5, size=(n_assets, n_drivers))  # nonlinear loadings

    # Random positive-definite Lambda
    F = rng.normal(0.0, 1.0, size=(n_drivers, n_drivers))
    Lambda = (F.T @ F) / n_drivers + 0.1 * np.eye(n_drivers)

    # ------------------------------------------------------------------ #
    # Monte Carlo: xi ~ N(0, Lambda)
    # ------------------------------------------------------------------ #
    L_chol = np.linalg.cholesky(Lambda)
    xi_raw = rng.normal(0.0, 1.0, size=(n_mc, n_drivers))
    xi = xi_raw @ L_chol.T  # (n_mc, n_drivers)  ~  N(0, Lambda)

    # E[xi^2] = diag(Lambda) (element-wise variance)
    E_xi2 = np.diag(Lambda)

    # Nonlinear systematic component
    xi2_centered = xi**2 - E_xi2[np.newaxis, :]  # (n_mc, n_drivers)
    m_sys = xi @ B.T + xi2_centered @ c_nl.T  # (n_mc, n_assets)

    # Sample covariance of xi and m_sys
    Lambda_mc = np.cov(xi, rowvar=False)

    Cov_m_xi = (m_sys - m_sys.mean(0)).T @ (xi - xi.mean(0)) / (n_mc - 1)
    L = Cov_m_xi @ np.linalg.inv(Lambda_mc)  # (n_assets, n_drivers)

    # Projection residual
    u = m_sys - xi @ L.T  # (n_mc, n_assets); nonzero because of quadratic terms

    Omega = np.cov(u, rowvar=False)  # should be > 0

    # Sigma_sys_true = sample covariance of m_sys
    Sigma_sys_true = np.cov(m_sys, rowvar=False)

    # Reconstructed
    Sigma_sys_rec = L @ Lambda_mc @ L.T + Omega

    omega_max_eig = float(np.linalg.eigvalsh(Omega).max())
    omega_is_psd = is_psd(Omega)
    rel_err = relative_error(Sigma_sys_rec, Sigma_sys_true, norm="frob")

    return {
        "Lambda": Lambda_mc,
        "L": L,
        "Omega": Omega,
        "Sigma_sys_reconstructed": Sigma_sys_rec,
        "Sigma_sys_true": Sigma_sys_true,
        "omega_max_eig": omega_max_eig,
        "omega_is_psd": omega_is_psd,
        "relative_error": rel_err,
    }
