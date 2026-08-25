"""
Experiment 7: Empirical Covariance (Synthetic Fallback)

# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.

Bloomberg data is unavailable in this environment.  This script implements a
synthetic version using n=150 assets and m=3 drivers:
  - 3267 daily observations (matching paper length)
  - T=252 and T=504 fold studies with 126-day test blocks
  - Comparison: structured, LW, PCA-k (k=1,3,5), sample, 1/N

Results are reported as mean annualized GMV volatility per fold and method.
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

# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.
print("# NOTE: This is a synthetic approximation. Bloomberg data required for exact replication.")

# ------------------------------------------------------------------ #
# Synthetic setup matching paper length/size
# ------------------------------------------------------------------ #
N_ASSETS = 150
N_DRIVERS = 3
N_OBS = 3267        # daily obs (paper length)
PHI = 0.95
INNOV_VAR = 0.0975
# b_loading_var=0.0: the state term b_i@Z_t contributes only to the conditional
# mean (not covariance). Setting it to zero makes population cov = D + C*Λ*C',
# so oracle_Q == structured target.  This matches exp_3's DGP specification.
B_LOADING_VAR = 0.0
C_LOADING_VAR = 0.3
IDIO_VOL_RANGE = (0.008, 0.02)
T_TEST_BLOCK = 126  # 126-day out-of-sample block
T_TRAIN_LIST = [252, 504]
SEED = 42

METHODS = ["sample", "ledoit_wolf", "pca_1", "pca_3", "pca_5", "structured", "equal_weight"]
METHOD_LABELS = {
    "sample": "Sample",
    "ledoit_wolf": "LW",
    "pca_1": "PCA-1",
    "pca_3": "PCA-3",
    "pca_5": "PCA-5",
    "structured": "Structured",
    "equal_weight": "1/N",
}


def simulate_full_panel(seed: int = SEED) -> dict:
    """Simulate N_OBS daily observations for n=150 assets."""
    Z, eta = simulate_ar1_drivers(
        n_drivers=N_DRIVERS, n_obs=N_OBS + 1, phi=PHI, innov_var=INNOV_VAR, seed=seed
    )
    returns, b_load, c_load, idio_vols = simulate_asset_returns(
        Z=Z, eta=eta, n_assets=N_ASSETS,
        b_loading_var=B_LOADING_VAR, c_loading_var=C_LOADING_VAR,
        idio_vol_range=IDIO_VOL_RANGE, seed=seed,
    )
    innov_cov = INNOV_VAR * np.eye(N_DRIVERS)
    oracle_Q = get_population_covariance(c_load, innov_cov, idio_vols)
    return {
        "returns": returns,    # (N_OBS, N_ASSETS)
        "Z": Z[:-1],           # (N_OBS, N_DRIVERS) aligned with returns
        "oracle_Q": oracle_Q,
    }


def run_fold_study(data: dict, T_train: int) -> dict:
    """
    Rolling fold study with T_train-day training windows and T_TEST_BLOCK-day test blocks.
    """
    returns = data["returns"]
    Z = data["Z"]
    oracle_Q = data["oracle_Q"]
    T_total = returns.shape[0]

    fold_vols = {m: [] for m in METHODS}
    fold_start = 0
    fold_idx = 0

    while fold_start + T_train + T_TEST_BLOCK <= T_total:
        train_ret = returns[fold_start: fold_start + T_train]
        test_ret = returns[fold_start + T_train: fold_start + T_train + T_TEST_BLOCK]
        train_Z = Z[fold_start: fold_start + T_train]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = evaluate_gmv_out_of_sample(
                train_ret, test_ret, train_Z, k_pca=3, oracle_Q=oracle_Q
            )

        for m in METHODS:
            if m in res:
                fold_vols[m].append(res[m]["realized_vol_ann"] * 100)

        fold_idx += 1
        fold_start += T_TEST_BLOCK  # non-overlapping test blocks

    mean_vols = {
        m: float(np.mean(fold_vols[m])) if fold_vols[m] else float("nan")
        for m in METHODS
    }
    n_folds = fold_idx
    return {"mean_vols": mean_vols, "n_folds": n_folds, "fold_vols": fold_vols}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Experiment 7: Empirical Covariance (Synthetic Fallback)")
    print("  NOTE: This is a synthetic approximation.")
    print("        Bloomberg data required for exact replication.")
    print("=" * 70)
    print(f"\n  n_assets={N_ASSETS}, n_drivers={N_DRIVERS}")
    print(f"  n_obs={N_OBS}, T_train in {T_TRAIN_LIST}, T_test_block={T_TEST_BLOCK}")
    print(f"  b_loading_var={B_LOADING_VAR} (zero => oracle=structured target)")
    print(f"  c_loading_var={C_LOADING_VAR}")

    data = simulate_full_panel(SEED)
    T_total = data["returns"].shape[0]
    print(f"  Simulated returns shape: {data['returns'].shape}")

    results = {}
    for T_train in T_TRAIN_LIST:
        print(f"\n  Running T_train={T_train} fold study ...")
        fold_res = run_fold_study(data, T_train)
        results[T_train] = fold_res
        n_folds = fold_res["n_folds"]
        print(f"    n_folds={n_folds}")
        print(f"    Mean annualized GMV vol (%):")
        for m in METHODS:
            v = fold_res["mean_vols"].get(m, float("nan"))
            print(f"      {METHOD_LABELS.get(m, m):<12}: {v:.3f}%")

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(RESULTS_DIR, "exp7_synthetic_gmv_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T_train", "n_folds"] + [METHOD_LABELS.get(m, m) for m in METHODS])
        for T_train, fold_res in results.items():
            row = [T_train, fold_res["n_folds"]] + [
                fold_res["mean_vols"].get(m, float("nan")) for m in METHODS
            ]
            writer.writerow(row)
    print(f"\n  GMV table saved -> {csv_path}")

    # ------------------------------------------------------------------ #
    # Plot: bar chart comparing methods for each T_train
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, len(T_TRAIN_LIST), figsize=(12, 5), sharey=False)
    if len(T_TRAIN_LIST) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(METHODS)))

    for ax, T_train in zip(axes, T_TRAIN_LIST):
        fold_res = results[T_train]
        vals = [fold_res["mean_vols"].get(m, float("nan")) for m in METHODS]
        bars = ax.bar(range(len(METHODS)), vals, color=colors)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in METHODS], rotation=45, ha="right")
        ax.set_title(f"T_train={T_train} ({fold_res['n_folds']} folds)")
        ax.set_ylabel("Mean Ann. GMV Vol (%)")

    fig.suptitle(
        f"Exp 7 (Synthetic): GMV Out-of-Sample Volatility Comparison\n"
        f"n={N_ASSETS} assets, {N_OBS} obs — SYNTHETIC APPROXIMATION",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp7_gmv_comparison.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  GMV comparison plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
