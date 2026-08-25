"""
Experiment 2: Two-Information-Window Screening Test

Demonstrates that conditioning on the lagged driver state (Z_t) alone leaves
substantial residual correlation (past-only model), while additionally
conditioning on the contemporaneous innovation eta_{t+1} = Z_{t+1} - phi*Z_t
removes it (horizon-closed model).

Settings: n=30 assets, m=3 AR(1) drivers, phi=0.95, innov_var=0.0975, T=5000.
Loading variance: loading_var=0.5 (both b_i and c_i drawn N(0, loading_var)).
Sigma_i in [0.008, 0.02].  Seed=42.

Target results (approximately):
  Past-only:       max|off-diag corr| ~ 0.83
  Horizon-closed:  max|off-diag corr| ~ 0.025
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.simulate import simulate_ar1_drivers, simulate_asset_returns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Simulation parameters
N_ASSETS = 30
N_DRIVERS = 3
PHI = 0.95
INNOV_VAR = 0.0975
# loading_var=0.5: b_i and c_i loadings drawn from N(0, sqrt(0.5)) so that
# Var(b_i @ Z_t) and Var(c_i @ eta_{t+1}) are both substantial relative to
# idiosyncratic variance, creating strong residual correlation before screening.
LOADING_VAR = 0.5
IDIO_VOL_RANGE = (0.008, 0.02)
T_TOTAL = 5000  # total observations (driver states)
SEED = 42
TRAIN_FRAC = 0.80


def simulate_data(seed: int = SEED) -> dict:
    """Simulate AR(1) drivers and asset returns."""
    n_obs = T_TOTAL + 1  # +1 because simulate_ar1_drivers returns Z[0..n_obs-1]
    Z, eta = simulate_ar1_drivers(
        n_drivers=N_DRIVERS,
        n_obs=n_obs,
        phi=PHI,
        innov_var=INNOV_VAR,
        seed=seed,
    )
    returns, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z,
        eta=eta,
        n_assets=N_ASSETS,
        loading_var=LOADING_VAR,
        idio_vol_range=IDIO_VOL_RANGE,
        seed=seed,
    )
    # returns shape: (T_TOTAL, N_ASSETS); Z shape: (T_TOTAL+1, N_DRIVERS)
    return {
        "Z": Z,
        "eta": eta,
        "returns": returns,
        "b_load": b_load,
        "c_load": c_load,
        "idio_vols": idio_vols,
    }


def build_regressors(Z: np.ndarray, eta: np.ndarray) -> dict:
    """
    Build regressor matrices for both models.

    returns[t] = r_{t+1} uses Z[t] (lagged state) and eta[t+1] (contemporaneous innovation).
    Array alignment (for t=0,...,T-1):
      Z_lag  = Z[:-1]   shape (T, n_drivers)
      eta_next = eta[1:] shape (T, n_drivers)

    Horizon-closed innovation (reconstructed from data):
      eta_hat_{t+1} = Z_{t+1} - phi * Z_t  =  Z[t+1] - PHI * Z[t]
    """
    T = Z.shape[0] - 1  # number of return observations

    Z_lag = Z[:-1]          # (T, n_drivers)  = Z_t
    Z_next = Z[1:]          # (T, n_drivers)  = Z_{t+1}
    eta_hat = Z_next - PHI * Z_lag   # (T, n_drivers)  ~ eta_{t+1}

    return {
        "Z_lag": Z_lag,       # for past-only model:  [1, Z_t]
        "eta_hat": eta_hat,   # for horizon-closed:   [1, Z_t, eta_hat_{t+1}]
        "T": T,
    }


def ols_residuals(
    returns: np.ndarray, regressors: np.ndarray
) -> np.ndarray:
    """
    OLS regression of returns on [1 | regressors]; return residuals.

    Parameters
    ----------
    returns    : (T, n_assets)
    regressors : (T, k)

    Returns
    -------
    residuals : (T, n_assets)
    """
    T = returns.shape[0]
    X = np.hstack([np.ones((T, 1)), regressors])  # (T, 1+k)
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)
    return returns - X @ coeffs


def residual_metrics(residuals: np.ndarray) -> dict:
    """Compute covariance, correlation, and off-diagonal statistics."""
    cov = np.cov(residuals, rowvar=False)
    corr = np.corrcoef(residuals, rowvar=False)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag_abs = np.abs(corr[mask])
    return {
        "cov": cov,
        "corr": corr,
        "max_abs_offdiag_corr": float(off_diag_abs.max()),
        "mean_abs_offdiag_corr": float(off_diag_abs.mean()),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Simulate data
    # ------------------------------------------------------------------ #
    print("=== Experiment 2: Two-Information-Window Screening Test ===\n")
    print(f"  n_assets={N_ASSETS}, n_drivers={N_DRIVERS}, T={T_TOTAL}, seed={SEED}")
    print(f"  phi={PHI}, innov_var={INNOV_VAR}, loading_var={LOADING_VAR}")
    print(f"  idio_vol_range={IDIO_VOL_RANGE}, train_frac={TRAIN_FRAC}\n")

    data = simulate_data(SEED)
    Z = data["Z"]
    eta = data["eta"]
    returns = data["returns"]

    # ------------------------------------------------------------------ #
    # 2. Train/eval split
    # ------------------------------------------------------------------ #
    T_total = returns.shape[0]
    T_train = int(T_total * TRAIN_FRAC)
    T_eval = T_total - T_train

    reg = build_regressors(Z, eta)

    Z_lag_tr = reg["Z_lag"][:T_train]
    eta_hat_tr = reg["eta_hat"][:T_train]
    returns_tr = returns[:T_train]

    Z_lag_ev = reg["Z_lag"][T_train:]
    eta_hat_ev = reg["eta_hat"][T_train:]
    returns_ev = returns[T_train:]

    print(f"  Training observations : {T_train}")
    print(f"  Evaluation observations: {T_eval}\n")

    # ------------------------------------------------------------------ #
    # 3. Fit models on training data
    # ------------------------------------------------------------------ #
    # Model A: Past-only  — regress r_{t+1} on [1, Z_t]
    coeffs_past, _, _, _ = np.linalg.lstsq(
        np.hstack([np.ones((T_train, 1)), Z_lag_tr]), returns_tr, rcond=None
    )
    # Model B: Horizon-closed — regress r_{t+1} on [1, Z_t, eta_{t+1}]
    coeffs_hc, _, _, _ = np.linalg.lstsq(
        np.hstack([np.ones((T_train, 1)), Z_lag_tr, eta_hat_tr]), returns_tr, rcond=None
    )

    # ------------------------------------------------------------------ #
    # 4. Compute residuals on evaluation data
    # ------------------------------------------------------------------ #
    X_past_ev = np.hstack([np.ones((T_eval, 1)), Z_lag_ev])
    X_hc_ev = np.hstack([np.ones((T_eval, 1)), Z_lag_ev, eta_hat_ev])

    resid_past = returns_ev - X_past_ev @ coeffs_past
    resid_hc = returns_ev - X_hc_ev @ coeffs_hc

    # ------------------------------------------------------------------ #
    # 5. Compute metrics
    # ------------------------------------------------------------------ #
    m_past = residual_metrics(resid_past)
    m_hc = residual_metrics(resid_hc)

    print("  +--------------------------+--------------------+--------------------+")
    print("  | Metric                   | Past-only          | Horizon-closed     |")
    print("  +--------------------------+--------------------+--------------------+")
    print(f"  | Max abs off-diag corr    | {m_past['max_abs_offdiag_corr']:18.4f} | {m_hc['max_abs_offdiag_corr']:18.4f} |")
    print(f"  | Mean abs off-diag corr   | {m_past['mean_abs_offdiag_corr']:18.4f} | {m_hc['mean_abs_offdiag_corr']:18.4f} |")
    print("  +--------------------------+--------------------+--------------------+")
    print("\n  Target: past-only max~0.83, horizon-closed max~0.025")

    # ------------------------------------------------------------------ #
    # 6. Save metrics
    # ------------------------------------------------------------------ #
    metrics = {
        "past_only": {
            "max_abs_offdiag_corr": m_past["max_abs_offdiag_corr"],
            "mean_abs_offdiag_corr": m_past["mean_abs_offdiag_corr"],
        },
        "horizon_closed": {
            "max_abs_offdiag_corr": m_hc["max_abs_offdiag_corr"],
            "mean_abs_offdiag_corr": m_hc["mean_abs_offdiag_corr"],
        },
        "simulation": {
            "n_assets": N_ASSETS,
            "n_drivers": N_DRIVERS,
            "T_total": T_TOTAL,
            "T_train": T_train,
            "T_eval": T_eval,
            "phi": PHI,
            "innov_var": INNOV_VAR,
            "loading_var": LOADING_VAR,
            "seed": SEED,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "exp2_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics JSON saved -> {json_path}")

    csv_path = os.path.join(RESULTS_DIR, "exp2_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "max_abs_offdiag_corr", "mean_abs_offdiag_corr"])
        writer.writerow(["past_only", m_past["max_abs_offdiag_corr"], m_past["mean_abs_offdiag_corr"]])
        writer.writerow(["horizon_closed", m_hc["max_abs_offdiag_corr"], m_hc["mean_abs_offdiag_corr"]])
    print(f"  Metrics CSV saved -> {csv_path}")

    # ------------------------------------------------------------------ #
    # 7. Heatmaps
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax = max(np.abs(m_past["corr"]).max(), np.abs(m_hc["corr"]).max())
    for ax, corr, title in zip(
        axes,
        [m_past["corr"], m_hc["corr"]],
        ["Past-only (Z_t)", "Horizon-closed (Z_t + eta_{t+1})"],
    ):
        im = ax.imshow(corr, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Asset")
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        "Residual Correlation Matrices\n"
        f"n={N_ASSETS} assets, T_eval={T_eval}",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp2_corr_heatmaps.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Heatmap saved -> {plot_path}")


if __name__ == "__main__":
    main()
