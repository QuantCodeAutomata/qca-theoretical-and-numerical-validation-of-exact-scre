"""
Simulate AR(1) factor drivers and synthetic asset return panels.

State equation  :  Z_{t+1} = phi * Z_t + eta_{t+1},   eta_t ~ N(0, innov_var * I)
Return equation :  r_{i,t+1} = a_i + b_i @ Z_t + c_i @ eta_{t+1} + eps_{i,t+1}
"""

import numpy as np


def simulate_ar1_drivers(
    n_drivers: int,
    n_obs: int,
    phi: float = 0.95,
    innov_var: float = 0.0975,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate AR(1) latent drivers with stationary initialization.

    Parameters
    ----------
    n_drivers : int
        Number of independent AR(1) processes.
    n_obs : int
        Number of time steps to simulate (including t=0).
    phi : float
        Autoregressive coefficient (|phi| < 1 required for stationarity).
    innov_var : float
        Innovation variance (scalar, same for all drivers).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Z : np.ndarray, shape (n_obs, n_drivers)
        Driver states.  Z[t] = Z_t.
    eta : np.ndarray, shape (n_obs, n_drivers)
        Innovations.  eta[t] = eta_t for t >= 1; eta[0] = 0 (unused).
    """
    if abs(phi) >= 1.0:
        raise ValueError(f"phi must satisfy |phi| < 1 for stationarity; got {phi}.")

    rng = np.random.default_rng(seed)
    innov_std = np.sqrt(innov_var)

    # Stationary marginal std: Var(Z_t) = innov_var / (1 - phi^2)
    stat_std = innov_std / np.sqrt(1.0 - phi**2)

    eta = np.zeros((n_obs, n_drivers))
    Z = np.zeros((n_obs, n_drivers))

    # Draw Z_0 from stationary distribution
    Z[0] = rng.normal(0.0, stat_std, size=n_drivers)

    # Draw innovations for t = 1, ..., n_obs-1
    eta[1:] = rng.normal(0.0, innov_std, size=(n_obs - 1, n_drivers))

    # Propagate state: Z_{t} = phi * Z_{t-1} + eta_{t}
    for t in range(1, n_obs):
        Z[t] = phi * Z[t - 1] + eta[t]

    return Z, eta


def simulate_asset_returns(
    Z: np.ndarray,
    eta: np.ndarray,
    n_assets: int,
    loading_var: float = 0.5,
    idio_vol_range: tuple = (0.008, 0.02),
    seed: int = 42,
    b_loading_var: float = None,
    c_loading_var: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate asset returns driven by AR(1) states and innovations.

    r_{i,t+1} = a_i + b_i @ Z_t + c_i @ eta_{t+1} + eps_{i,t+1}

    where eps_{i,t+1} ~ N(0, sigma_i^2) independently.

    Parameters
    ----------
    Z : np.ndarray, shape (n_obs, n_drivers)
        Driver states from ``simulate_ar1_drivers``.
    eta : np.ndarray, shape (n_obs, n_drivers)
        Innovations from ``simulate_ar1_drivers``.
    n_assets : int
        Number of assets.
    loading_var : float
        Variance of each element of b and c loadings (used when
        b_loading_var and c_loading_var are not specified).
    idio_vol_range : tuple of (float, float)
        Uniform range for idiosyncratic volatilities.
    seed : int
        Random seed.
    b_loading_var : float or None
        If provided, overrides ``loading_var`` for state (b) loadings.
        Set to 0.0 to eliminate state-predictive risk so that the
        conditional covariance equals the unconditional covariance.
    c_loading_var : float or None
        If provided, overrides ``loading_var`` for innovation (c) loadings.

    Returns
    -------
    returns : np.ndarray, shape (n_obs-1, n_assets)
        Simulated return panel.
    b_loadings : np.ndarray, shape (n_assets, n_drivers)
        State (predictive) loadings.
    c_loadings : np.ndarray, shape (n_assets, n_drivers)
        Innovation loadings.
    idio_vols : np.ndarray, shape (n_assets,)
        Per-asset idiosyncratic volatilities.
    """
    rng = np.random.default_rng(seed)
    n_obs, n_drivers = Z.shape

    b_var = loading_var if b_loading_var is None else b_loading_var
    c_var = loading_var if c_loading_var is None else c_loading_var

    b_std = np.sqrt(b_var) if b_var > 0 else 0.0
    c_std = np.sqrt(c_var)

    # Random intercepts (small), state loadings, innovation loadings
    a = rng.normal(0.0, 0.001, size=n_assets)
    b = (rng.normal(0.0, b_std, size=(n_assets, n_drivers))
         if b_std > 0 else np.zeros((n_assets, n_drivers)))
    c = rng.normal(0.0, c_std, size=(n_assets, n_drivers))  # (n_assets, n_drivers)

    # Idiosyncratic vols drawn uniformly from range
    idio_vols = rng.uniform(idio_vol_range[0], idio_vol_range[1], size=n_assets)

    # T = n_obs - 1 return observations
    T = n_obs - 1
    eps = rng.normal(0.0, 1.0, size=(T, n_assets)) * idio_vols[np.newaxis, :]  # (T, n_assets)

    # r_{t+1}: uses Z_t = Z[t] and eta_{t+1} = eta[t+1]
    # Z[:-1] has rows Z_0, ..., Z_{T-1}; eta[1:] has rows eta_1, ..., eta_T
    returns = (
        a[np.newaxis, :]
        + Z[:-1] @ b.T      # (T, n_assets)  predictive component
        + eta[1:] @ c.T     # (T, n_assets)  contemporaneous innovation component
        + eps               # (T, n_assets)  idiosyncratic noise
    )

    return returns, b, c, idio_vols


def get_population_covariance(
    c_loadings: np.ndarray,
    innov_cov: np.ndarray,
    idio_vols: np.ndarray,
) -> np.ndarray:
    """
    Compute population conditional covariance.

    Q = D + C @ innov_cov @ C.T

    where D = diag(idio_vols^2).  This is the covariance of r_{t+1} conditional
    on Z_t, because the only residual randomness is eta_{t+1} and eps_{t+1}.

    Parameters
    ----------
    c_loadings : np.ndarray, shape (n_assets, n_drivers)
        Innovation loadings C.
    innov_cov : np.ndarray, shape (n_drivers, n_drivers)
        Covariance matrix of the innovations eta.
    idio_vols : np.ndarray, shape (n_assets,)
        Idiosyncratic volatilities.

    Returns
    -------
    Q : np.ndarray, shape (n_assets, n_assets)
        Population conditional covariance matrix.
    """
    D = np.diag(idio_vols**2)
    Q = D + c_loadings @ innov_cov @ c_loadings.T  # rank-n_drivers update
    return Q
