"""
Covariance estimators for the Global Minimum Variance (GMV) portfolio study.

Estimators implemented:
  - Sample covariance
  - Ledoit-Wolf shrinkage
  - PCA factor covariance (rank-k approximation + diagonal residual)
  - Structured covariance (AR(1) factor model fit by OLS)

The structured estimator exploits the return DGP:
  r_{i,t+1} = a_i + b_i @ Z_t + c_i @ eta_{t+1} + eps_{i,t+1}

by first extracting AR(1) innovations from driver states, then regressing
asset returns on (intercept, Z_t, eta_{t+1}) to recover c_i.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
from sklearn.covariance import LedoitWolf

from src.utils import frob_norm, project_spd


# ------------------------------------------------------------------ #
# Structured covariance estimator
# ------------------------------------------------------------------ #

def fit_structured_covariance(
    returns: np.ndarray,
    driver_states: np.ndarray,
    phi_known: float = None,
) -> dict:
    """
    Fit a structured covariance model by exploiting the factor structure.

    Steps
    -----
    1. For each driver, fit AR(1) with intercept  x_t = a + phi * x_{t-1} + innov_t
       and extract innovations eta_hat.
    2. Regress each asset return on (1, Z_{t-1}, eta_hat_t):
         r_{i,t} = alpha_i + b_i @ Z_{t-1} + c_i @ eta_hat_t + eps_{i,t}
       Residuals eps_hat give idiosyncratic variances.
    3. Form the structured covariance:
         Q_struct = diag(D_hat) + L_hat @ Lambda_hat @ L_hat.T
       where Lambda_hat = sample covariance of eta_hat.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
        Asset return panel.  Rows are time, columns are assets.
    driver_states : np.ndarray, shape (T, n_drivers)
        Observed AR(1) driver states aligned with returns.
        driver_states[t] = Z_t  (used to predict r_{t+1} = returns[t]).
    phi_known : float or None
        If provided, use this as the AR(1) coefficient for all drivers;
        otherwise estimate phi for each driver individually.

    Returns
    -------
    dict with keys:
        Q_struct      : (n_assets, n_assets)  structured covariance estimate
        D_hat         : (n_assets,)  idiosyncratic variances
        L_hat         : (n_assets, n_drivers)  innovation loadings (c_i estimates)
        Lambda_hat    : (n_drivers, n_drivers)  innovation covariance
        phi_estimates : (n_drivers,)  AR(1) coefficients
    """
    T, n_assets = returns.shape
    _, n_drivers = driver_states.shape

    # ------------------------------------------------------------------ #
    # Step 1: Extract AR(1) innovations for each driver
    # ------------------------------------------------------------------ #
    # Align: driver_states[t] = Z_t; we model Z_t = a + phi * Z_{t-1} + eta_t
    # Use t = 1, ..., T-1 for the AR(1) regression (need lagged value)
    # driver_states has the same length as returns rows, so states run 0..T-1
    # eta_hat[t] corresponds to innovations at time t (used in r_t = r[t-1])

    n_ar = T - 1  # number of AR(1) residual observations (t=1,...,T-1)
    eta_hat = np.zeros((n_ar, n_drivers))
    phi_estimates = np.zeros(n_drivers)

    Z_lag = driver_states[:-1]  # Z_{t-1} for t=1,...,T-1  shape (n_ar, n_drivers)
    Z_cur = driver_states[1:]   # Z_t    for t=1,...,T-1  shape (n_ar, n_drivers)

    for k in range(n_drivers):
        y_k = Z_cur[:, k]    # Z_t^k
        x_k = Z_lag[:, k]    # Z_{t-1}^k

        if phi_known is not None:
            phi_k = phi_known
            alpha_k = np.mean(y_k - phi_k * x_k)
        else:
            # OLS: [1, Z_{t-1}^k] -> Z_t^k
            X_ar = np.column_stack([np.ones(n_ar), x_k])
            coeffs_k, _, _, _ = np.linalg.lstsq(X_ar, y_k, rcond=None)
            alpha_k, phi_k = coeffs_k

        phi_estimates[k] = phi_k
        eta_hat[:, k] = Z_cur[:, k] - alpha_k - phi_k * Z_lag[:, k]

    # ------------------------------------------------------------------ #
    # Step 2: Regress asset returns on (1, Z_{t-1}, eta_hat_t)
    # ------------------------------------------------------------------ #
    # Alignment:
    #   returns[t] = r_{t+1}  (return at period t+1, driven by Z_t and eta_{t+1})
    #   driver_states[t] = Z_t
    #   eta_hat[t] = Z_{t+1} - alpha - phi*Z_t = eta_{t+1}
    #
    # For t = 0, ..., T-2  (n_ar = T-1 observations):
    #   r_reg[t]   = returns[t]       = r_{t+1}      <- return to predict
    #   Z_cond2[t] = driver_states[t] = Z_t          <- decision-time state
    #   eta_hat[t]                    = eta_{t+1}    <- contemporaneous innovation
    #
    # This matches the DGP: r_{t+1} = a + b @ Z_t + c @ eta_{t+1} + eps_{t+1}
    r_reg = returns[:n_ar]            # (n_ar, n_assets)  r_{t+1} for t=0,...,T-2
    Z_cond2 = driver_states[:n_ar]    # (n_ar, n_drivers)  Z_t = driver_states[0..T-2]

    X_ret = np.hstack([
        np.ones((n_ar, 1)),
        Z_cond2,          # b_i loadings (state-predictive, not needed for Q_struct)
        eta_hat,          # c_i loadings (innovation loadings => Q_struct)
    ])  # (n_ar, 1 + 2*n_drivers)

    # OLS for all assets simultaneously
    coeffs_ret, _, _, _ = np.linalg.lstsq(X_ret, r_reg, rcond=None)
    # coeffs_ret: (1 + 2*n_drivers, n_assets)
    L_hat = coeffs_ret[1 + n_drivers:].T  # (n_assets, n_drivers)  c_i estimates

    residuals = r_reg - X_ret @ coeffs_ret   # (n_ar, n_assets)
    D_hat = np.var(residuals, axis=0, ddof=1 + 2 * n_drivers)  # bias-corrected

    # ------------------------------------------------------------------ #
    # Step 3: Structured covariance
    # ------------------------------------------------------------------ #
    Lambda_hat = np.cov(eta_hat, rowvar=False)  # (n_drivers, n_drivers)

    Q_struct = np.diag(D_hat) + L_hat @ Lambda_hat @ L_hat.T
    Q_struct = project_spd(Q_struct)  # ensure PSD after floating-point errors

    return {
        "Q_struct": Q_struct,
        "D_hat": D_hat,
        "L_hat": L_hat,
        "Lambda_hat": Lambda_hat,
        "phi_estimates": phi_estimates,
    }


# ------------------------------------------------------------------ #
# Other estimators
# ------------------------------------------------------------------ #

def fit_sample_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Standard sample covariance matrix (unbiased, ddof=1).

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)

    Returns
    -------
    S : np.ndarray, shape (n_assets, n_assets)
    """
    return np.cov(returns, rowvar=False)


def fit_ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Uses sklearn.covariance.LedoitWolf with analytic shrinkage.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)

    Returns
    -------
    Q_lw : np.ndarray, shape (n_assets, n_assets)
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns)
    return lw.covariance_


def fit_pca_covariance(returns: np.ndarray, k: int) -> np.ndarray:
    """
    PCA factor covariance estimator.

    S_k = U_k @ diag(lam_k) @ U_k.T + diag(diag(S - U_k @ diag(lam_k) @ U_k.T))

    Top-k factors capture systematic risk; residual diagonal captures idio variance.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
    k : int
        Number of PCA factors.

    Returns
    -------
    Q_pca : np.ndarray, shape (n_assets, n_assets)
    """
    n_assets = returns.shape[1]
    k = int(np.clip(k, 1, n_assets))

    S = fit_sample_covariance(returns)

    eigvals, eigvecs = np.linalg.eigh(S)      # ascending order
    # Take top-k eigenvectors/values
    idx = np.argsort(eigvals)[::-1]           # descending
    lam_k = eigvals[idx[:k]]
    U_k = eigvecs[:, idx[:k]]                 # (n_assets, k)

    # Low-rank factor part
    factor_part = U_k @ np.diag(lam_k) @ U_k.T   # (n_assets, n_assets)

    # Idiosyncratic diagonal from residual S - factor_part
    idio_diag = np.maximum(np.diag(S - factor_part), 0.0)

    Q_pca = factor_part + np.diag(idio_diag)
    return project_spd(Q_pca)  # ensure PSD


# ------------------------------------------------------------------ #
# GMV weights and evaluation
# ------------------------------------------------------------------ #

def compute_gmv_weights(Q: np.ndarray) -> np.ndarray:
    """
    Compute Global Minimum Variance (GMV) portfolio weights.

    Solves  Q @ x = 1_n  and returns  w = x / (1_n.T @ x).

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Positive definite covariance matrix.

    Returns
    -------
    w : np.ndarray, shape (n,)
        Portfolio weights summing to 1.
    """
    n = Q.shape[0]
    ones = np.ones(n)
    try:
        x = scipy.linalg.solve(Q, ones, assume_a="pos")
    except np.linalg.LinAlgError:
        warnings.warn("Q appears singular; using pseudoinverse for GMV.", stacklevel=2)
        x = np.linalg.pinv(Q) @ ones

    total = ones @ x
    if abs(total) < 1e-12:
        raise ValueError("GMV denominator is zero; covariance may be degenerate.")
    return x / total


def evaluate_gmv_out_of_sample(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    train_driver_states: np.ndarray,
    k_pca: int = 3,
    oracle_Q: np.ndarray = None,
) -> dict:
    """
    Estimate covariances on the training set, form GMV portfolios, and
    evaluate realized variance on the test set.

    Methods evaluated: sample, LW, PCA(k-2), PCA(k), PCA(k+2),
    structured, oracle (if provided), 1/N equal-weight.

    Parameters
    ----------
    train_returns : np.ndarray, shape (T_train, n_assets)
    test_returns  : np.ndarray, shape (T_test,  n_assets)
    train_driver_states : np.ndarray, shape (T_train, n_drivers)
        Driver states aligned with train_returns.
    k_pca : int
        Central number of PCA factors; k-2 and k+2 variants are also fitted.
    oracle_Q : np.ndarray or None
        Population covariance (used as "oracle" method).

    Returns
    -------
    dict mapping method name -> sub-dict with:
        weights         : (n_assets,)
        realized_var    : float  (test-period sample variance of portfolio return)
        realized_vol_ann: float  annualised realised vol (assuming 252 trading days)
        predicted_var   : float  w.T Q_hat w
        ratio           : float  realized_var / predicted_var
    """
    T_train, n_assets = train_returns.shape
    T_test = test_returns.shape[0]
    n_drivers = train_driver_states.shape[1]

    # ---- Fit all covariance estimators ----
    estimates: dict[str, np.ndarray] = {}
    estimates["sample"] = fit_sample_covariance(train_returns)
    estimates["ledoit_wolf"] = fit_ledoit_wolf(train_returns)

    for k_offset in (-2, 0, 2):
        k_use = int(np.clip(k_pca + k_offset, 1, n_assets - 1))
        label = f"pca_{k_use}"
        estimates[label] = fit_pca_covariance(train_returns, k_use)

    struct = fit_structured_covariance(train_returns, train_driver_states)
    estimates["structured"] = struct["Q_struct"]

    if oracle_Q is not None:
        estimates["oracle"] = oracle_Q

    # Equal-weight (no covariance needed, but compute predicted variance)
    w_ew = np.full(n_assets, 1.0 / n_assets)

    results: dict[str, dict] = {}

    for method, Q_hat in estimates.items():
        try:
            w = compute_gmv_weights(project_spd(Q_hat))
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"GMV failed for '{method}': {exc}", stacklevel=2)
            continue

        port_ret = test_returns @ w            # (T_test,) portfolio returns
        realized_var = float(np.var(port_ret, ddof=1))
        realized_vol_ann = float(np.sqrt(realized_var * 252))
        predicted_var = float(w @ Q_hat @ w)
        ratio = realized_var / predicted_var if predicted_var > 0 else float("nan")

        results[method] = {
            "weights": w,
            "realized_var": realized_var,
            "realized_vol_ann": realized_vol_ann,
            "predicted_var": predicted_var,
            "ratio": ratio,
        }

    # Equal-weight
    port_ret_ew = test_returns @ w_ew
    realized_var_ew = float(np.var(port_ret_ew, ddof=1))
    results["equal_weight"] = {
        "weights": w_ew,
        "realized_var": realized_var_ew,
        "realized_vol_ann": float(np.sqrt(realized_var_ew * 252)),
        "predicted_var": float(w_ew @ estimates["sample"] @ w_ew),
        "ratio": float("nan"),
    }

    return results
