"""
Validate the exact conditional risk decomposition (screening identity).

The decomposition is:

    Q_{t,h} = D_{t,h} + Sigma_sys_{t,h}

where
  Q_{t,h}       = Cov(r_{t+h} | G_t)          total conditional covariance
  D_{t,h}       = E[Cov(r_{t+h} | H_{t,h}) | G_t]  expected idiosyncratic block
  Sigma_sys_{t,h} = Cov(m_{t,h} | G_t)          systematic covariance

Given the return model  r_{i,t+1} = a_i + b_i @ Z_t + c_i @ eta_{t+1} + eps_{i,t+1}:
  - m_{t,h} = a + b @ Z_t + c @ eta_{t+1}   (systematic part)
  - Given Z_t, the remaining randomness is eta_{t+1} and eps_{t+1}
  - D_{t,h}     = diag(idio_vols^2)
  - Sigma_sys   = C @ Cov(eta) @ C.T
  - Q_{t,h}     = D_{t,h} + Sigma_sys  (population exact)
"""

import numpy as np

from data.simulate import (
    get_population_covariance,
    simulate_ar1_drivers,
    simulate_asset_returns,
)
from src.utils import frob_norm, op_norm, relative_error


def validate_screening_decomposition(
    n_assets: int = 20,
    n_drivers: int = 3,
    n_mc: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Verify the screening decomposition Q = D + Sigma_sys both analytically and
    via Monte Carlo estimation.

    Procedure
    ---------
    1. Simulate AR(1) drivers and asset returns with a fixed seed.
    2. Compute population quantities analytically.
    3. Estimate Q, D, Sigma_sys numerically via n_mc conditional simulations
       (fix Z_0 = Z[0] and re-draw eta and eps).
    4. Verify Q_pop - (D_pop + Sigma_sys_pop) ≈ 0  (should be machine-epsilon).
    5. Verify Q_mc  ≈ D_pop + Sigma_sys_pop         (Monte Carlo validation).

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_drivers : int
        Number of AR(1) drivers.
    n_mc : int
        Monte Carlo sample size for numerical verification.
    seed : int
        Master random seed.

    Returns
    -------
    dict with keys:
        Q              : population conditional covariance (n_assets, n_assets)
        D              : idiosyncratic block  diag(sigma_i^2)
        Sigma_sys      : systematic covariance  C @ Lambda @ C.T
        residual_frob  : ||Q - (D + Sigma_sys)||_F  (should be ~machine epsilon)
        residual_op    : ||Q - (D + Sigma_sys)||_2
        relative_frob_error : residual_frob / ||Q||_F
        Q_mc           : Monte Carlo estimate of Q (for reference)
        mc_frob_error  : ||Q_mc - Q||_F
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Simulate AR(1) drivers
    # ------------------------------------------------------------------ #
    n_obs = 200  # short series; only used to obtain model parameters
    Z, eta = simulate_ar1_drivers(
        n_drivers=n_drivers, n_obs=n_obs, seed=int(rng.integers(1e6))
    )
    _, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z, eta=eta, n_assets=n_assets, seed=int(rng.integers(1e6))
    )

    # ------------------------------------------------------------------ #
    # 2. Population quantities (analytical)
    # ------------------------------------------------------------------ #
    innov_var = 0.0975
    Lambda = innov_var * np.eye(n_drivers)          # Cov(eta) = innov_var * I

    D = np.diag(idio_vols**2)                       # idiosyncratic block
    Sigma_sys = c_load @ Lambda @ c_load.T          # systematic covariance
    Q = get_population_covariance(c_load, Lambda, idio_vols)  # D + Sigma_sys

    # ------------------------------------------------------------------ #
    # 3. Verify identity analytically (should be machine-epsilon exact)
    # ------------------------------------------------------------------ #
    residual = Q - (D + Sigma_sys)
    residual_frob = frob_norm(residual)
    residual_op = op_norm(residual)
    rel_frob = relative_error(D + Sigma_sys, Q, norm="frob")

    # ------------------------------------------------------------------ #
    # 4. Monte Carlo: fix Z_0 = Z[0], draw n_mc copies of (eta, eps)
    # ------------------------------------------------------------------ #
    Z0 = Z[0]  # fixed conditioning state
    innov_std = np.sqrt(innov_var)

    eta_mc = rng.normal(0.0, innov_std, size=(n_mc, n_drivers))    # eta_{t+1}
    eps_mc = rng.normal(0.0, 1.0, size=(n_mc, n_assets)) * idio_vols  # eps_{t+1}

    # Small intercept a is not needed for covariance; skip it
    # r = a + b @ Z0 + c @ eta + eps;  the constant a + b @ Z0 drops out
    r_mc = eta_mc @ c_load.T + eps_mc  # (n_mc, n_assets); centered

    Q_mc = np.cov(r_mc, rowvar=False)
    mc_frob_error = frob_norm(Q_mc - Q)

    return {
        "Q": Q,
        "D": D,
        "Sigma_sys": Sigma_sys,
        "residual_frob": residual_frob,
        "residual_op": residual_op,
        "relative_frob_error": rel_frob,
        "Q_mc": Q_mc,
        "mc_frob_error": mc_frob_error,
    }
