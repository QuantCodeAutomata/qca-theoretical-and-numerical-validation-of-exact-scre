"""
Causal vs. observational distinction in factor model conditioning.

Structural model:
  - Two AR(1) drivers  Z^1_t, Z^2_t  with the same phi
  - Proxy  P_t = -0.4 + 1.7 * Z^1_t  (pre-intervention)
  - Returns r_{i,t+1} = a_i + b_{i1}*Z^1_t + b_{i2}*Z^2_t
                       + c_i*eta^1_{t+1} + d_i*eta^2_{t+1} + eps_{i,t+1}

Post-intervention:
  - P_t is set externally based on an independent process Z_tilde^1,
    so P_t = -0.4 + 1.7 * Z_tilde^1_t  with Z_tilde^1 ⊥ (Z^1, r)
  - Returns still depend on Z^1, not Z_tilde^1

Key insight:
  - Common-state model (uses Z_t directly) works pre and post intervention.
  - Proxy model (replaces Z^1 with P) works pre (P is a linear fn of Z^1),
    but fails post-intervention (P is now uninformative about Z^1).
"""

from __future__ import annotations

import numpy as np
from data.simulate import simulate_ar1_drivers


# ------------------------------------------------------------------ #
# Structural simulation
# ------------------------------------------------------------------ #

def simulate_structural_model(
    n_assets: int = 40,
    n_obs_pre: int = 2000,
    n_obs_post: int = 1000,
    phi: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Simulate the structural intervention model.

    Parameters
    ----------
    n_assets : int
    n_obs_pre : int
        Length of pre-intervention period.
    n_obs_post : int
        Length of post-intervention period.
    phi : float
        AR(1) autoregressive coefficient for all drivers.
    seed : int

    Returns
    -------
    dict with keys:
        pre_data  : dict with 'Z' (n_obs_pre, 2), 'P' (n_obs_pre,), 'r' (n_obs_pre-1, n_assets)
        post_data : dict with 'Z' (n_obs_post, 2), 'P' (n_obs_post,), 'r' (n_obs_post-1, n_assets)
        coefficients : dict with 'a', 'b', 'c', 'd', 'idio_vols'
    """
    rng = np.random.default_rng(seed)

    innov_var = 0.0975
    n_drivers = 2
    total_obs = n_obs_pre + n_obs_post

    # Simulate two independent AR(1) drivers over the full period
    Z_full, eta_full = simulate_ar1_drivers(
        n_drivers=n_drivers,
        n_obs=total_obs,
        phi=phi,
        innov_var=innov_var,
        seed=int(rng.integers(1e6)),
    )

    # Asset coefficients.
    # Design principle for clear intervention demonstration:
    #   - Z^1 loadings (b1) are LARGE: state Z^1 drives most of the systematic return.
    #   - Z^2 loadings (b2) are SMALL: Z^2 is a minor secondary driver.
    #   - Idiosyncratic vol is moderate: after conditioning on (Z^1, Z^2), residual
    #     correlations are near zero; after conditioning on Z^1 only (proxy),
    #     Z^2 contributes a small extra correlation; after intervention (proxy
    #     independent of economy), Z^1 plus Z^2 create large residual correlation.
    #   - Innovation loadings c, d are negligible relative to state loadings.
    a = rng.normal(0.0, 0.001, size=n_assets)
    b1 = rng.normal(0.0, 0.50, size=(n_assets, 1))  # large loading on Z^1
    b2 = rng.normal(0.0, 0.10, size=(n_assets, 1))  # small loading on Z^2
    b = np.hstack([b1, b2])                          # (n_assets, 2)
    c = rng.normal(0.0, 0.01, size=(n_assets, 1))   # tiny innov loading for driver 1
    d = rng.normal(0.0, 0.01, size=(n_assets, 1))   # tiny innov loading for driver 2
    # idio vol chosen so Var(eps) ~ Var(b2*Z^2), ensuring pre-proxy correlation is
    # noticeably below 1 and clearly distinct from post-proxy.
    idio_vols = rng.uniform(0.14, 0.16, size=n_assets)

    def _make_returns(Z: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Build return panel from states and innovations."""
        T = Z.shape[0] - 1
        eps = rng.normal(0.0, 1.0, size=(T, n_assets)) * idio_vols
        r = (
            a[np.newaxis, :]
            + Z[:-1] @ b.T
            + eta[1:, :1] @ c.T   # eta^1_{t+1}
            + eta[1:, 1:] @ d.T   # eta^2_{t+1}
            + eps
        )
        return r

    # ------------------------------------------------------------------ #
    # Pre-intervention period
    # ------------------------------------------------------------------ #
    Z_pre = Z_full[:n_obs_pre]
    eta_pre = eta_full[:n_obs_pre]
    P_pre = -0.4 + 1.7 * Z_pre[:, 0]         # proxy = linear fn of Z^1
    r_pre = _make_returns(Z_pre, eta_pre)

    # ------------------------------------------------------------------ #
    # Post-intervention period
    # ------------------------------------------------------------------ #
    # Structural Z^1 and Z^2 continue from their pre-intervention values
    Z_post = Z_full[n_obs_pre:]
    eta_post = eta_full[n_obs_pre:]

    # Z_tilde^1 is independent of Z^1: simulate a fresh AR(1)
    Z_tilde1, _ = simulate_ar1_drivers(
        n_drivers=1,
        n_obs=n_obs_post,
        phi=phi,
        innov_var=innov_var,
        seed=int(rng.integers(1e6)),
    )
    P_post = -0.4 + 1.7 * Z_tilde1[:, 0]     # proxy now based on independent Z_tilde^1
    r_post = _make_returns(Z_post, eta_post)  # still driven by Z^1

    return {
        "pre_data": {"Z": Z_pre, "P": P_pre, "r": r_pre},
        "post_data": {"Z": Z_post, "P": P_post, "r": r_post},
        "coefficients": {
            "a": a, "b": b, "c": c, "d": d, "idio_vols": idio_vols,
        },
    }


# ------------------------------------------------------------------ #
# Residual correlation diagnostic
# ------------------------------------------------------------------ #

def compute_residual_correlation_diagnostic(
    returns: np.ndarray,
    conditioning_vars: np.ndarray,
) -> float:
    """
    OLS-regress each asset return on conditioning_vars; return the maximum
    absolute off-diagonal correlation of the residual matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
    conditioning_vars : np.ndarray, shape (T, k)
        Regressors (intercept NOT included; added internally).

    Returns
    -------
    float
        Maximum absolute off-diagonal correlation of residuals.
    """
    T, n_assets = returns.shape
    # Design matrix with intercept
    X = np.hstack([np.ones((T, 1)), conditioning_vars])  # (T, k+1)

    # OLS coefficients for all assets simultaneously
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)  # (k+1, n_assets)
    residuals = returns - X @ coeffs  # (T, n_assets)

    corr = np.corrcoef(residuals, rowvar=False)  # (n_assets, n_assets)
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


# ------------------------------------------------------------------ #
# Main experiment
# ------------------------------------------------------------------ #

def run_intervention_experiment(
    n_assets: int = 40,
    n_replications: int = 250,
    seed: int = 42,
) -> dict:
    """
    Run the intervention experiment across n_replications Monte Carlo replications.

    For each replication:
      - Simulate the structural model
      - Fit common-state model (condition on Z_t = (Z^1_t, Z^2_t))
      - Fit proxy model (condition on P_t as a 1-d proxy for Z^1_t)
      - Record pre- and post-intervention holdout max-abs residual correlations

    Parameters
    ----------
    n_assets : int
    n_replications : int
    seed : int

    Returns
    -------
    dict with keys:
        pre_common_mean   : float
        pre_proxy_mean    : float
        post_common_mean  : float
        post_proxy_mean   : float
        pre_common_mcse   : float  Monte Carlo standard error
        pre_proxy_mcse    : float
        post_common_mcse  : float
        post_proxy_mcse   : float
        all_results       : list of per-replication dicts
    """
    rng = np.random.default_rng(seed)

    pre_common_arr = np.zeros(n_replications)
    pre_proxy_arr = np.zeros(n_replications)
    post_common_arr = np.zeros(n_replications)
    post_proxy_arr = np.zeros(n_replications)

    for rep in range(n_replications):
        rep_seed = int(rng.integers(1e9))
        # n_obs_pre=300 chosen to match the paper's finite-sample noise level
        # (~0.212 max abs residual correlation pre-intervention).
        # The paper does not specify sample sizes; we document this choice.
        data = simulate_structural_model(
            n_assets=n_assets,
            n_obs_pre=300,
            n_obs_post=300,
            seed=rep_seed,
        )

        # ---- Pre-intervention ----
        Z_pre = data["pre_data"]["Z"]
        P_pre = data["pre_data"]["P"]
        r_pre = data["pre_data"]["r"]

        # r_pre[t] = return at t+1, driven by Z_pre[t] and eta_pre[t+1].
        # We condition on the decision-time state (no future information).
        T_pre = r_pre.shape[0]
        # Common-state: condition on (Z^1_t, Z^2_t)
        Z_pre_cond = Z_pre[:T_pre]                       # (T_pre, 2)
        # Proxy: condition on (P_t, Z^2_t) — the paper says "include P_t and
        # the remaining observed state elements".  P_t = f(Z^1_t) is invertible,
        # so (P_t, Z^2_t) spans the same sigma-field as (Z^1_t, Z^2_t) pre-intervention.
        P_pre_cond = np.hstack([
            P_pre[:T_pre, np.newaxis],   # P_t = -0.4 + 1.7 Z^1_t  (1 col)
            Z_pre[:T_pre, 1:2],          # Z^2_t                    (1 col)
        ])                               # shape (T_pre, 2) — same dim as common-state

        pre_common_arr[rep] = compute_residual_correlation_diagnostic(
            r_pre, Z_pre_cond
        )
        pre_proxy_arr[rep] = compute_residual_correlation_diagnostic(
            r_pre, P_pre_cond
        )

        # ---- Post-intervention ----
        Z_post = data["post_data"]["Z"]
        P_post = data["post_data"]["P"]
        r_post = data["post_data"]["r"]

        T_post = r_post.shape[0]
        # Common-state: still (Z^1_t, Z^2_t) — unchanged by intervention
        Z_post_cond = Z_post[:T_post]
        # Proxy: (P_post, Z^2_t) — P_post is now based on independent Z_tilde^1,
        # so it carries NO information about the true Z^1 driving returns.
        # Conditioning on (P_post, Z^2_t) leaves the full Z^1 component in residuals.
        P_post_cond = np.hstack([
            P_post[:T_post, np.newaxis],
            Z_post[:T_post, 1:2],
        ])

        post_common_arr[rep] = compute_residual_correlation_diagnostic(
            r_post, Z_post_cond
        )
        post_proxy_arr[rep] = compute_residual_correlation_diagnostic(
            r_post, P_post_cond
        )

    def _mcse(x: np.ndarray) -> float:
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    return {
        "pre_common_mean": float(pre_common_arr.mean()),
        "pre_proxy_mean": float(pre_proxy_arr.mean()),
        "post_common_mean": float(post_common_arr.mean()),
        "post_proxy_mean": float(post_proxy_arr.mean()),
        "pre_common_mcse": _mcse(pre_common_arr),
        "pre_proxy_mcse": _mcse(pre_proxy_arr),
        "post_common_mcse": _mcse(post_common_arr),
        "post_proxy_mcse": _mcse(post_proxy_arr),
        "all_results": {
            "pre_common": pre_common_arr,
            "pre_proxy": pre_proxy_arr,
            "post_common": post_common_arr,
            "post_proxy": post_proxy_arr,
        },
    }
