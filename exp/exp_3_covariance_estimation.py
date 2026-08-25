"""
Experiment 3: Monte Carlo GMV Covariance Estimation Study

For each (n, T) cell in {50,100,200} x {126,252,504}, run 250 Monte Carlo
replications and compare annualized out-of-sample GMV volatility across:
  sample, LW, PCA-1, PCA-3, PCA-5, structured, oracle, equal_weight.

DGP design: b_loading_var=0.0 (no state-predictive component), so the
conditional covariance equals the unconditional covariance:
    Q = D + C @ Lambda @ C.T    (population target for all estimators)
This is the correct setting for exp_3 "correctly specified affine structure":
all methods estimate the SAME population covariance, and the structured
estimator's advantage lies in correctly imposing the low-rank + diagonal form.

c_loading_var=0.3: innovation loadings (c_i) with per-element std 0.3.
Annualized oracle GMV vol ~ sigma_idio / sqrt(n - m) ~ 3% at n=50.

Parallel processing via joblib if available; otherwise sequential with progress.
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

from data.simulate import (
    get_population_covariance,
    simulate_ar1_drivers,
    simulate_asset_returns,
)
from src.covariance import evaluate_gmv_out_of_sample

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Experiment parameters
N_ASSETS_LIST = [50, 100, 200]
T_TRAIN_LIST = [126, 252, 504]
N_REPS = 250
T_TEST = 252
N_DRIVERS = 3
PHI = 0.95
INNOV_VAR = 0.0975
# b_loading_var=0.0: state (b) loadings are zero, so conditional cov = unconditional cov.
# This is required so all estimators target the same quantity Q = D + C*Lambda*C.T.
B_LOADING_VAR = 0.0
# c_loading_var=0.3: innovation loadings create a non-trivial rank-3 factor structure.
# With phi=0.95, innov_var=0.0975: stationary state variance = 0.0975/(1-0.95^2) = 1.0.
C_LOADING_VAR = 0.3
IDIO_VOL_RANGE = (0.008, 0.02)
BURN_IN = 500
K_PCA = 3  # central PCA factor count; k-2=1, k=3, k+2=5 are all evaluated

METHODS = ["sample", "ledoit_wolf", "pca_1", "pca_3", "pca_5", "structured", "oracle", "equal_weight"]
METHOD_LABELS = {
    "sample": "Sample",
    "ledoit_wolf": "LW",
    "pca_1": "PCA-1",
    "pca_3": "PCA-3",
    "pca_5": "PCA-5",
    "structured": "Structured",
    "oracle": "Oracle",
    "equal_weight": "1/N",
}


def run_one_replication(n: int, T_train: int, seed: int) -> dict:
    """Run a single Monte Carlo replication for given (n, T_train) and return vols+ratios."""
    n_obs = BURN_IN + T_train + T_TEST + 1
    try:
        Z, eta = simulate_ar1_drivers(
            n_drivers=N_DRIVERS, n_obs=n_obs, phi=PHI, innov_var=INNOV_VAR, seed=seed
        )
        # b_loading_var=0: no state-predictive component, so Q_cond == Q_uncond = D + C*Lambda*C.T.
        # All estimators (sample, LW, PCA, structured) therefore target the same quantity.
        returns, b_load, c_load, idio_vols = simulate_asset_returns(
            Z=Z, eta=eta, n_assets=n,
            b_loading_var=B_LOADING_VAR, c_loading_var=C_LOADING_VAR,
            idio_vol_range=IDIO_VOL_RANGE, seed=seed + 1,
        )
        # Discard burn-in
        returns = returns[BURN_IN:]          # (T_train + T_TEST, n)
        Z_states = Z[BURN_IN:-1]            # aligned with returns: (T_train + T_TEST, n_drivers)

        train_ret = returns[:T_train]
        test_ret = returns[T_train:]
        train_Z = Z_states[:T_train]

        innov_cov = INNOV_VAR * np.eye(N_DRIVERS)
        # Oracle = true conditional covariance = D + C @ Lambda @ C.T.
        # Since b=0, this equals the unconditional covariance.
        oracle_Q = get_population_covariance(c_load, innov_cov, idio_vols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = evaluate_gmv_out_of_sample(
                train_ret, test_ret, train_Z, k_pca=K_PCA, oracle_Q=oracle_Q
            )
        return {method: res[method] for method in METHODS if method in res}
    except Exception as exc:
        warnings.warn(f"Replication failed (n={n}, T={T_train}, seed={seed}): {exc}")
        return {}


def _run_cell(n: int, T_train: int) -> tuple:
    """Run all replications for one (n, T_train) cell. Returns (vols_dict, ratios_dict)."""
    rng = np.random.default_rng(seed=n * 10000 + T_train)
    seeds = rng.integers(0, 10**8, size=N_REPS).tolist()

    vols = {m: [] for m in METHODS}
    ratios = {m: [] for m in METHODS}

    for rep, seed in enumerate(seeds):
        if (rep + 1) % 50 == 0:
            print(f"    n={n}, T={T_train}: rep {rep+1}/{N_REPS}")
        rep_res = run_one_replication(n, T_train, int(seed))
        for method in METHODS:
            if method in rep_res:
                vols[method].append(rep_res[method].get("realized_vol_ann", np.nan) * 100)
                ratio = rep_res[method].get("ratio", np.nan)
                ratios[method].append(ratio if np.isfinite(ratio) else np.nan)

    mean_vols = {m: float(np.nanmean(vols[m])) if vols[m] else float("nan") for m in METHODS}
    mean_ratios = {m: float(np.nanmean(ratios[m])) if ratios[m] else float("nan") for m in METHODS}
    return mean_vols, mean_ratios


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Experiment 3: Monte Carlo GMV Covariance Estimation Study ===")
    print(f"  n_assets: {N_ASSETS_LIST}")
    print(f"  T_train:  {T_TRAIN_LIST}")
    print(f"  n_reps:   {N_REPS},  T_test: {T_TEST}")
    print(f"  b_loading_var={B_LOADING_VAR} (zero state loadings: cond cov = uncond cov)")
    print(f"  c_loading_var={C_LOADING_VAR} (innovation loadings create rank-3 factor)\n")

    # Try parallel with joblib
    try:
        from joblib import Parallel, delayed
        USE_PARALLEL = True
        print("  Using joblib parallel processing (n_jobs=-1).")
    except ImportError:
        USE_PARALLEL = False
        print("  joblib not available; running sequential loops with progress reporting.")

    vol_table = {}    # (n, T_train) -> {method: mean_vol_%}
    ratio_table = {}  # (n, T_train) -> {method: mean_ratio}

    for n in N_ASSETS_LIST:
        for T_train in T_TRAIN_LIST:
            print(f"\n  Cell: n={n}, T_train={T_train}")

            if USE_PARALLEL:
                rng = np.random.default_rng(seed=n * 10000 + T_train)
                seeds = rng.integers(0, 10**8, size=N_REPS).tolist()

                def _rep(seed):
                    return run_one_replication(n, T_train, int(seed))

                all_res = Parallel(n_jobs=-1, verbose=0)(
                    delayed(_rep)(int(s)) for s in seeds
                )
                vols = {m: [] for m in METHODS}
                ratios = {m: [] for m in METHODS}
                for rep_res in all_res:
                    for method in METHODS:
                        if method in rep_res:
                            vols[method].append(rep_res[method].get("realized_vol_ann", np.nan) * 100)
                            ratio = rep_res[method].get("ratio", np.nan)
                            ratios[method].append(ratio if np.isfinite(ratio) else np.nan)
                mean_vols = {m: float(np.nanmean(vols[m])) if vols[m] else float("nan") for m in METHODS}
                mean_ratios = {m: float(np.nanmean(ratios[m])) if ratios[m] else float("nan") for m in METHODS}
            else:
                mean_vols, mean_ratios = _run_cell(n, T_train)

            vol_table[(n, T_train)] = mean_vols
            ratio_table[(n, T_train)] = mean_ratios

            # Print row
            print(f"    Mean annualized GMV vol (%):")
            for m in METHODS:
                v = mean_vols.get(m, float("nan"))
                print(f"      {METHOD_LABELS.get(m, m):<12}: {v:.3f}%")

    # ------------------------------------------------------------------ #
    # Save CSV tables
    # ------------------------------------------------------------------ #
    # Vol table
    vol_csv_path = os.path.join(RESULTS_DIR, "exp3_gmv_table.csv")
    with open(vol_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["n_assets", "T_train"] + [METHOD_LABELS.get(m, m) for m in METHODS]
        writer.writerow(header)
        for n in N_ASSETS_LIST:
            for T_train in T_TRAIN_LIST:
                row = [n, T_train] + [vol_table[(n, T_train)].get(m, float("nan")) for m in METHODS]
                writer.writerow(row)
    print(f"\n  GMV vol table saved -> {vol_csv_path}")

    # Calibration ratio table
    ratio_csv_path = os.path.join(RESULTS_DIR, "exp3_calibration_table.csv")
    with open(ratio_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["n_assets", "T_train"] + [METHOD_LABELS.get(m, m) for m in METHODS]
        writer.writerow(header)
        for n in N_ASSETS_LIST:
            for T_train in T_TRAIN_LIST:
                row = [n, T_train] + [ratio_table[(n, T_train)].get(m, float("nan")) for m in METHODS]
                writer.writerow(row)
    print(f"  Calibration table saved -> {ratio_csv_path}")

    # ------------------------------------------------------------------ #
    # Summary table print
    # ------------------------------------------------------------------ #
    print("\n  === Summary: Mean Annualized GMV Vol (%) ===")
    header_str = f"  {'n':>6} {'T':>6}" + "".join(f"  {METHOD_LABELS.get(m, m):>10}" for m in METHODS)
    print(header_str)
    print("  " + "-" * (len(header_str) - 2))
    for n in N_ASSETS_LIST:
        for T_train in T_TRAIN_LIST:
            vals = [vol_table[(n, T_train)].get(m, float("nan")) for m in METHODS]
            row_str = f"  {n:>6} {T_train:>6}" + "".join(f"  {v:>10.3f}" for v in vals)
            print(row_str)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    # Bar chart: for each (n, T) cell, compare methods
    plot_methods = [m for m in METHODS if m != "equal_weight"]
    n_methods = len(plot_methods)
    n_cells = len(N_ASSETS_LIST) * len(T_TRAIN_LIST)

    fig, axes = plt.subplots(len(N_ASSETS_LIST), len(T_TRAIN_LIST), figsize=(14, 9), sharey=False)
    axes = np.atleast_2d(axes)

    for i, n in enumerate(N_ASSETS_LIST):
        for j, T_train in enumerate(T_TRAIN_LIST):
            ax = axes[i, j]
            vals = [vol_table[(n, T_train)].get(m, float("nan")) for m in plot_methods]
            bars = ax.bar(range(n_methods), vals, color=plt.cm.tab10(np.linspace(0, 0.8, n_methods)))
            ax.set_xticks(range(n_methods))
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in plot_methods], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"n={n}, T={T_train}", fontsize=9)
            ax.set_ylabel("Ann. vol (%)" if j == 0 else "")

    fig.suptitle("Mean Annualized GMV Out-of-Sample Volatility (%)", fontsize=12)
    plt.tight_layout()
    vol_plot_path = os.path.join(RESULTS_DIR, "exp3_volatility_plot.png")
    plt.savefig(vol_plot_path, dpi=120)
    plt.close()
    print(f"\n  Volatility plot saved -> {vol_plot_path}")

    # Calibration ratio plot (exclude NaN methods)
    ratio_methods = [m for m in METHODS if m not in ("equal_weight", "oracle")]
    n_rm = len(ratio_methods)
    fig2, axes2 = plt.subplots(len(N_ASSETS_LIST), len(T_TRAIN_LIST), figsize=(14, 9), sharey=False)
    axes2 = np.atleast_2d(axes2)

    for i, n in enumerate(N_ASSETS_LIST):
        for j, T_train in enumerate(T_TRAIN_LIST):
            ax = axes2[i, j]
            vals = [ratio_table[(n, T_train)].get(m, float("nan")) for m in ratio_methods]
            ax.bar(range(n_rm), vals, color=plt.cm.tab10(np.linspace(0, 0.7, n_rm)))
            ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
            ax.set_xticks(range(n_rm))
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in ratio_methods], rotation=45, ha="right", fontsize=7)
            ax.set_title(f"n={n}, T={T_train}", fontsize=9)
            ax.set_ylabel("Ratio" if j == 0 else "")

    fig2.suptitle("Mean Calibration Ratio (Realized Var / Predicted Var)", fontsize=12)
    plt.tight_layout()
    cal_plot_path = os.path.join(RESULTS_DIR, "exp3_calibration_plot.png")
    plt.savefig(cal_plot_path, dpi=120)
    plt.close()
    print(f"  Calibration plot saved -> {cal_plot_path}")


if __name__ == "__main__":
    main()
