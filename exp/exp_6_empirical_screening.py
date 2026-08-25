"""
Experiment 6: Empirical Screening (Synthetic Fallback)

# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.

Bloomberg data is unavailable in this environment.  This script implements a
synthetic version using the same algorithmic pipeline as the paper:
  - n=12 assets, m=123 candidate drivers (synthetic AR(1) processes)
  - 5653 daily observations (matching paper length)
  - Local-search driver selector with kappa=0.01, max_card=8
  - Rolling 39 windows of 756 obs with 126-step stride
  - Comparison: selected drivers vs PCA-matched vs no-conditioning

The reported max abs residual correlation serves as a synthetic proxy for the
paper's empirical screening diagnostic.
"""

import csv
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.simulate import simulate_ar1_drivers, simulate_asset_returns
from src.utils import op_norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.
print("# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.")

# ------------------------------------------------------------------ #
# Parameters matching the paper's empirical setup (sizes only)
# ------------------------------------------------------------------ #
N_ASSETS = 12
N_CANDIDATE_DRIVERS = 123   # total driver candidates
N_OBS = 5653                # daily observations (matching paper)
PHI = 0.97
INNOV_VAR = 0.02
LOADING_VAR = 0.3
IDIO_VOL_RANGE = (0.005, 0.015)
KAPPA = 0.01                # regularization / cardinality penalty
MAX_CARD = 8                # maximum number of selected drivers
WINDOW_SIZE = 756           # rolling window length (rows)
STRIDE = 126                # step size between windows
SEED = 42


def simulate_synthetic_panel(seed: int = SEED) -> dict:
    """Simulate a synthetic panel of n assets and m candidate drivers."""
    # True drivers: only the first TRUE_M drive returns
    TRUE_M = 5
    rng = np.random.default_rng(seed)

    # Simulate candidate drivers (all as AR(1))
    Z_all = np.zeros((N_OBS + 1, N_CANDIDATE_DRIVERS))
    eta_all = np.zeros((N_OBS + 1, N_CANDIDATE_DRIVERS))
    for k in range(N_CANDIDATE_DRIVERS):
        phi_k = rng.uniform(0.90, 0.99)
        Z_k, eta_k = simulate_ar1_drivers(
            n_drivers=1, n_obs=N_OBS + 1, phi=phi_k, innov_var=INNOV_VAR,
            seed=int(rng.integers(1e8))
        )
        Z_all[:, k] = Z_k[:, 0]
        eta_all[:, k] = eta_k[:, 0]

    # Only the first TRUE_M drivers actually drive asset returns
    Z_true = Z_all[:, :TRUE_M]
    eta_true = eta_all[:, :TRUE_M]

    returns, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z_true, eta=eta_true, n_assets=N_ASSETS,
        loading_var=LOADING_VAR, idio_vol_range=IDIO_VOL_RANGE,
        seed=seed,
    )
    return {
        "returns": returns,         # (N_OBS, N_ASSETS)
        "Z_all": Z_all[:-1],        # (N_OBS, N_CANDIDATE_DRIVERS) - aligned with returns
        "eta_all": eta_all[1:],     # (N_OBS, N_CANDIDATE_DRIVERS)
        "true_driver_indices": list(range(TRUE_M)),
    }


def compute_max_offdiag_corr(residuals: np.ndarray) -> float:
    """Max absolute off-diagonal correlation of the residual matrix."""
    n = residuals.shape[1]
    corr = np.corrcoef(residuals, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    return float(np.abs(corr).max())


def ols_residuals(returns: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """OLS regression with intercept; return residuals."""
    T = returns.shape[0]
    X = np.hstack([np.ones((T, 1)), regressors])
    coeffs, _, _, _ = np.linalg.lstsq(X, returns, rcond=None)
    return returns - X @ coeffs


def local_search_driver_selector(
    returns: np.ndarray,
    Z_candidates: np.ndarray,
    kappa: float = KAPPA,
    max_card: int = MAX_CARD,
) -> list:
    """
    Greedy local-search driver selector.

    Iteratively adds the candidate driver that most reduces the maximum
    absolute off-diagonal residual correlation, penalized by kappa per
    added driver, until no further improvement or max_card reached.

    Parameters
    ----------
    returns      : (T, n_assets)
    Z_candidates : (T, n_candidates)
    kappa        : penalty per additional driver
    max_card     : maximum number of selected drivers

    Returns
    -------
    selected : list of int
        Indices of selected drivers (1-indexed into Z_candidates).
    """
    n_cand = Z_candidates.shape[1]
    selected = []
    remaining = list(range(n_cand))

    # Baseline: intercept-only residuals
    T = returns.shape[0]
    baseline_corr = compute_max_offdiag_corr(
        returns - returns.mean(axis=0)
    )
    current_score = baseline_corr   # lower is better

    for _ in range(max_card):
        best_improvement = kappa  # must improve by at least kappa to add
        best_driver = None

        for cand in remaining:
            trial_drivers = selected + [cand]
            trial_Z = Z_candidates[:, trial_drivers]
            trial_resid = ols_residuals(returns, trial_Z)
            trial_corr = compute_max_offdiag_corr(trial_resid)
            improvement = current_score - trial_corr
            if improvement > best_improvement:
                best_improvement = improvement
                best_driver = cand

        if best_driver is None:
            break  # no more improvement
        selected.append(best_driver)
        remaining.remove(best_driver)
        sel_Z = Z_candidates[:, selected]
        sel_resid = ols_residuals(returns, sel_Z)
        current_score = compute_max_offdiag_corr(sel_resid)

    return selected


def pca_regressors(returns: np.ndarray, k: int) -> np.ndarray:
    """Return top-k PCA factors of the returns as regressors."""
    S = np.cov(returns, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(S)
    top_k = eigvecs[:, np.argsort(eigvals)[::-1][:k]]  # (n_assets, k)
    return returns @ top_k   # (T, k)  factor scores


def run_rolling_windows(data: dict) -> list:
    """Run the local-search selection over rolling windows."""
    returns = data["returns"]
    Z_all = data["Z_all"]
    T = returns.shape[0]

    window_records = []
    window_starts = list(range(0, T - WINDOW_SIZE, STRIDE))[:39]  # at most 39 windows
    n_win = len(window_starts)

    print(f"\n  Running {n_win} rolling windows (window={WINDOW_SIZE}, stride={STRIDE}) ...")

    for i, t0 in enumerate(window_starts):
        t1 = t0 + WINDOW_SIZE
        ret_win = returns[t0:t1]
        Z_win = Z_all[t0:t1]

        # 1. Local-search selected drivers
        try:
            selected = local_search_driver_selector(ret_win, Z_win, KAPPA, MAX_CARD)
        except Exception as exc:
            warnings.warn(f"Window {i}: local search failed: {exc}")
            selected = []

        if selected:
            Z_sel = Z_win[:, selected]
            resid_sel = ols_residuals(ret_win, Z_sel)
            corr_sel = compute_max_offdiag_corr(resid_sel)
        else:
            resid_sel = ret_win - ret_win.mean(axis=0)
            corr_sel = compute_max_offdiag_corr(resid_sel)

        # 2. PCA-matched (same cardinality)
        k_pca = max(1, len(selected))
        Z_pca = pca_regressors(ret_win, k_pca)
        resid_pca = ols_residuals(ret_win, Z_pca)
        corr_pca = compute_max_offdiag_corr(resid_pca)

        # 3. No conditioning (intercept only)
        resid_none = ret_win - ret_win.mean(axis=0)
        corr_none = compute_max_offdiag_corr(resid_none)

        window_records.append({
            "window": i,
            "t0": t0,
            "t1": t1,
            "n_selected": len(selected),
            "corr_selected": corr_sel,
            "corr_pca": corr_pca,
            "corr_none": corr_none,
        })

        if (i + 1) % 10 == 0 or i == n_win - 1:
            print(f"    Window {i+1}/{n_win}: n_sel={len(selected)}, "
                  f"corr_sel={corr_sel:.3f}, corr_pca={corr_pca:.3f}, corr_none={corr_none:.3f}")

    return window_records


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Experiment 6: Empirical Screening (Synthetic Fallback)")
    print("  NOTE: This is a synthetic approximation.")
    print("        Bloomberg data required for exact replication.")
    print("=" * 70)
    print(f"\n  n_assets={N_ASSETS}, n_candidate_drivers={N_CANDIDATE_DRIVERS}")
    print(f"  n_obs={N_OBS}, kappa={KAPPA}, max_card={MAX_CARD}, seed={SEED}")

    data = simulate_synthetic_panel(SEED)
    window_records = run_rolling_windows(data)

    # Aggregate
    corr_sel_vals = [r["corr_selected"] for r in window_records]
    corr_pca_vals = [r["corr_pca"] for r in window_records]
    corr_none_vals = [r["corr_none"] for r in window_records]

    print(f"\n  Median max abs residual correlation:")
    print(f"    Selected drivers: {np.median(corr_sel_vals):.4f}")
    print(f"    PCA-matched     : {np.median(corr_pca_vals):.4f}")
    print(f"    No conditioning : {np.median(corr_none_vals):.4f}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp6_synthetic_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_records[0].keys()))
        writer.writeheader()
        writer.writerows(window_records)
    print(f"\n  Metrics CSV saved -> {csv_path}")

    # Plot
    windows = [r["window"] for r in window_records]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(windows, corr_none_vals, "k-o", markersize=4, label="No conditioning", alpha=0.7)
    ax.plot(windows, corr_pca_vals, "b-s", markersize=4, label="PCA-matched", alpha=0.7)
    ax.plot(windows, corr_sel_vals, "r-^", markersize=4, label="Selected drivers", alpha=0.7)
    ax.set_xlabel("Rolling window index")
    ax.set_ylabel("Max abs off-diagonal residual correlation")
    ax.set_title(
        "Exp 6 (Synthetic): Rolling Screening Diagnostics\n"
        "NOTE: Synthetic approximation — Bloomberg data required for exact replication"
    )
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp6_rolling_diagnostics.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Rolling diagnostics plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
