"""
Experiment 4: Perturbation Study

n=60 assets, m=3 drivers.  Varies rho in [0, 0.05, ..., 0.80] and measures:
  A. Response-error channel  (Omega only, E=0)
  B. Screening-error channel (E only, Omega=0)
  C. Sign-reversed screening channel (-E, Omega=0)

At rho=0.5 for the response-error channel the target is:
  displacement ~ 13.4%,  frontier_change ~ -0.39%
(Discrepancies from exact numbers are documented below the results table.)
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

from src.markowitz import compute_optimal_portfolio
from src.perturbation import verify_perturbation_identities
from src.utils import op_norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ------------------------------------------------------------------ #
# Baseline parameters
# ------------------------------------------------------------------ #
N_ASSETS = 60
N_DRIVERS = 3
INNOV_VAR = 0.0975
LOADING_VAR = 0.3
IDIO_VOL_RANGE = (0.008, 0.02)
GAMMA = 1.0
RHO_GRID = np.arange(0.0, 0.85, 0.05)


def build_baseline(seed: int = 42) -> dict:
    """Construct baseline covariance Q0, expected returns mu, constraints."""
    rng = np.random.default_rng(seed)

    # D: diagonal idiosyncratic variances
    sigma2_vals = rng.uniform(IDIO_VOL_RANGE[0]**2, IDIO_VOL_RANGE[1]**2, size=N_ASSETS)
    D = np.diag(sigma2_vals)

    # L: n x m innovation loadings (centered Gaussian)
    L = rng.normal(0.0, np.sqrt(LOADING_VAR), size=(N_ASSETS, N_DRIVERS))

    # Lambda: diagonal innovation covariance
    Lambda = INNOV_VAR * np.eye(N_DRIVERS)

    # Q0 = D + L @ Lambda @ L.T
    Q0 = D + L @ Lambda @ L.T

    # mu: centered Gaussian expected returns
    mu = rng.normal(0.0, 0.001, size=N_ASSETS)

    # Budget-only constraint: 1.T @ w = 1
    A = np.ones((1, N_ASSETS))
    b = np.array([1.0])

    # Underobar_sigma2 = min(diag(D))
    underobar_sigma2 = float(sigma2_vals.min())

    return {
        "Q0": Q0,
        "D": D,
        "L": L,
        "Lambda": Lambda,
        "mu": mu,
        "A": A,
        "b": b,
        "sigma2_vals": sigma2_vals,
        "underobar_sigma2": underobar_sigma2,
    }


def compute_perturbed_metrics(
    Q0: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    w_star: np.ndarray,
    Delta: float,
) -> dict:
    """
    Compute exact perturbed portfolio and first-order approximations.

    Returns displacement, frontier_change, and first-order errors.
    """
    from src.markowitz import compute_markowitz_operators

    Q_R = Q0 + R
    ops0 = compute_markowitz_operators(Q0, A, b)
    M = ops0["M"]

    res_R = compute_optimal_portfolio(Q_R, mu, A, b, GAMMA)
    w_R_star = res_R["w_star"]
    Delta_R = res_R["Delta"]

    # First-order approximations
    dw_fo = -M @ R @ w_star
    dDelta_fo = -(M @ mu) @ R @ (M @ mu)

    # Displacement (relative)
    norm_wstar = float(np.linalg.norm(w_star))
    displacement = float(np.linalg.norm(w_R_star - w_star)) / (norm_wstar if norm_wstar > 1e-12 else 1.0)

    # Frontier change (relative)
    frontier_change = (Delta_R - Delta) / abs(Delta) if abs(Delta) > 1e-12 else float("nan")

    # First-order relative errors
    dw_exact = w_R_star - w_star
    dw_err = float(np.linalg.norm(dw_exact - dw_fo)) / (float(np.linalg.norm(dw_exact)) + 1e-15)
    dDelta_exact = Delta_R - Delta
    dD_err = abs(dDelta_exact - dDelta_fo) / (abs(dDelta_exact) + 1e-15)

    return {
        "displacement": displacement,
        "frontier_change": frontier_change,
        "dw_fo_rel_err": dw_err,
        "dDelta_fo_rel_err": dD_err,
        "w_R_star": w_R_star,
        "Delta_R": Delta_R,
    }


def make_omega_perturbation(rng, underobar_sigma2: float, seed_offset: int = 10) -> np.ndarray:
    """Response-error perturbation matrix (Omega channel)."""
    rng2 = np.random.default_rng(seed_offset)
    B = rng2.normal(0.0, 1.0, size=(N_ASSETS, N_DRIVERS))
    Omega_raw = B @ B.T
    return Omega_raw / op_norm(Omega_raw)   # normalised template


def make_E_perturbation(seed_offset: int = 20) -> np.ndarray:
    """Screening-error perturbation matrix (E channel): symmetric, zero diagonal."""
    rng = np.random.default_rng(seed_offset)
    E_raw = rng.normal(0.0, 1.0, size=(N_ASSETS, N_ASSETS))
    E_raw = (E_raw + E_raw.T) / 2.0
    np.fill_diagonal(E_raw, 0.0)
    E_raw = E_raw / op_norm(E_raw)
    return E_raw


def verify_identities_at_rho(
    Q0: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> dict:
    """Verify all three perturbation identities and return residual norms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = verify_perturbation_identities(Q0, R, mu, A, b, GAMMA)
    return {
        "id1a_frob": res["id1a_frob"],
        "id1b_frob": res["id1b_frob"],
        "id2a_frob": res["id2a_frob"],
        "id2b_frob": res["id2b_frob"],
        "id3_abs": res["id3_abs"],
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Experiment 4: Perturbation Study ===")
    print(f"  n_assets={N_ASSETS}, n_drivers={N_DRIVERS}, gamma={GAMMA}")
    print(f"  rho grid: {RHO_GRID.tolist()}\n")

    # ------------------------------------------------------------------ #
    # Build baseline
    # ------------------------------------------------------------------ #
    baseline = build_baseline(seed=42)
    Q0 = baseline["Q0"]
    mu = baseline["mu"]
    A = baseline["A"]
    b = baseline["b"]
    underobar_sigma2 = baseline["underobar_sigma2"]

    # Baseline portfolio
    res0 = compute_optimal_portfolio(Q0, mu, A, b, GAMMA)
    w_star = res0["w_star"]
    Delta = res0["Delta"]

    print(f"  Baseline Delta (mu.T M mu)         : {Delta:.6f}")
    print(f"  underobar_sigma2 (min diag D)      : {underobar_sigma2:.6e}\n")

    # ------------------------------------------------------------------ #
    # Build perturbation templates
    # ------------------------------------------------------------------ #
    Omega_template = make_omega_perturbation(None, underobar_sigma2, seed_offset=10)
    E_template = make_E_perturbation(seed_offset=20)

    channels = {
        "omega": Omega_template,
        "E": E_template,
        "neg_E": -E_template,
    }
    channel_labels = {
        "omega": "Response error (Omega)",
        "E": "Screening error (+E)",
        "neg_E": "Screening error (-E)",
    }

    # ------------------------------------------------------------------ #
    # Main rho sweep
    # ------------------------------------------------------------------ #
    records = []

    for channel_key, R_template in channels.items():
        print(f"  Channel: {channel_labels[channel_key]}")

        for rho in RHO_GRID:
            # Scale perturbation: R_rho = rho * underobar_sigma2 * R_template
            R_rho = rho * underobar_sigma2 * R_template

            # Check that Q0 + R_rho is PD (necessary for portfolio computation)
            Q_try = Q0 + R_rho
            eigvals = np.linalg.eigvalsh(Q_try)
            if eigvals.min() <= 1e-12:
                print(f"    rho={rho:.2f}: Q0+R is not PD; skipping.")
                continue

            metrics = compute_perturbed_metrics(Q0, R_rho, mu, A, b, w_star, Delta)
            id_res = verify_identities_at_rho(Q0, R_rho, mu, A, b)

            record = {
                "channel": channel_key,
                "rho": float(rho),
                "displacement": metrics["displacement"],
                "frontier_change": metrics["frontier_change"],
                "dw_fo_rel_err": metrics["dw_fo_rel_err"],
                "dDelta_fo_rel_err": metrics["dDelta_fo_rel_err"],
                "id1a_frob": id_res["id1a_frob"],
                "id1b_frob": id_res["id1b_frob"],
                "id2a_frob": id_res["id2a_frob"],
                "id2b_frob": id_res["id2b_frob"],
                "id3_abs": id_res["id3_abs"],
            }
            records.append(record)

            if abs(rho - 0.5) < 1e-9 and channel_key == "omega":
                print(f"    rho=0.50: displacement={metrics['displacement']:.4f} "
                      f"(target~13.4%), frontier_change={metrics['frontier_change']*100:.4f}% "
                      f"(target~-0.39%)")

        print()

    # ------------------------------------------------------------------ #
    # Print summary at selected rho
    # ------------------------------------------------------------------ #
    rho_target = 0.5
    print(f"  === Metrics at rho={rho_target:.2f} ===")
    print(f"  {'Channel':<35} {'Displacement':>14} {'Frontier Chg %':>16}")
    print(f"  {'-'*35} {'-'*14} {'-'*16}")
    for rec in records:
        if abs(rec["rho"] - rho_target) < 1e-9:
            label = channel_labels.get(rec["channel"], rec["channel"])
            fc = rec["frontier_change"] * 100 if rec["frontier_change"] is not None else float("nan")
            print(f"  {label:<35} {rec['displacement']:>14.4f} {fc:>16.4f}")

    print("\n  Note: Exact displacement/frontier_change at rho=0.5 depends on the")
    print("  specific random seed and loading_var. Target values (13.4%, -0.39%)")
    print("  are paper-specific and may differ with this synthetic baseline.")

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(RESULTS_DIR, "exp4_perturbation_metrics.csv")
    fieldnames = list(records[0].keys()) if records else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  Metrics CSV saved -> {csv_path}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    # 1. Displacement vs rho for 3 channels
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"omega": "b", "E": "r", "neg_E": "g"}
    for ch in ["omega", "E", "neg_E"]:
        recs = [r for r in records if r["channel"] == ch]
        rho_vals = [r["rho"] for r in recs]
        disp_vals = [r["displacement"] * 100 for r in recs]
        ax.plot(rho_vals, disp_vals, "o-", color=colors[ch],
                label=channel_labels[ch], markersize=4)
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("Displacement (% of $\\|w^*\\|$)")
    ax.set_title("Portfolio Displacement vs Perturbation Magnitude")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_displacement_plot.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Displacement plot saved -> {path}")

    # 2. Frontier change vs rho
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for ch in ["omega", "E", "neg_E"]:
        recs = [r for r in records if r["channel"] == ch]
        rho_vals = [r["rho"] for r in recs]
        fc_vals = [r["frontier_change"] * 100 if r["frontier_change"] is not None else float("nan")
                   for r in recs]
        ax2.plot(rho_vals, fc_vals, "o-", color=colors[ch],
                 label=channel_labels[ch], markersize=4)
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("$\\rho$")
    ax2.set_ylabel("Frontier change (% of $|\\Delta|$)")
    ax2.set_title("Efficient Frontier Change vs Perturbation Magnitude")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "exp4_frontier_change_plot.png")
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"  Frontier change plot saved -> {path2}")

    # 3. Exact vs first-order at rho=0.5 (omega channel)
    rho_plot = 0.5
    recs_05 = [r for r in records if r["channel"] == "omega" and abs(r["rho"] - rho_plot) < 1e-9]
    if recs_05:
        rec_05 = recs_05[0]
        R_rho05 = rho_plot * underobar_sigma2 * Omega_template
        from src.markowitz import compute_markowitz_operators
        ops0 = compute_markowitz_operators(Q0, A, b)
        M = ops0["M"]
        res_R05 = compute_optimal_portfolio(Q0 + R_rho05, mu, A, b, GAMMA)
        w_R05 = res_R05["w_star"]
        dw_exact_05 = w_R05 - w_star
        dw_fo_05 = -M @ R_rho05 @ w_star
        asset_idx = np.arange(N_ASSETS)
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        ax3.bar(asset_idx - 0.2, dw_exact_05, width=0.4, label="Exact $\\Delta w$", alpha=0.7)
        ax3.bar(asset_idx + 0.2, dw_fo_05, width=0.4, label="First-order approx.", alpha=0.7)
        ax3.set_xlabel("Asset index")
        ax3.set_ylabel("Weight change")
        ax3.set_title(f"Exact vs First-order $\\Delta w$ at $\\rho$={rho_plot} (Omega channel)")
        ax3.legend()
        plt.tight_layout()
        path3 = os.path.join(RESULTS_DIR, "exp4_firstorder_comparison.png")
        plt.savefig(path3, dpi=120)
        plt.close()
        print(f"  First-order comparison plot saved -> {path3}")


if __name__ == "__main__":
    main()
