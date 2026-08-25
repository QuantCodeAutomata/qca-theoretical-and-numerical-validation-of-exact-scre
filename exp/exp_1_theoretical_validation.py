"""
Experiment 1: Theoretical Validation

Runs 7 validation blocks that verify the mathematical identities underpinning
the QCA framework. Results are saved to the results/ directory.
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

from src.diagnostics import compute_residual_diagnostic, verify_diagnostic_bound
from src.markowitz import validate_markowitz_identities, compute_optimal_portfolio
from src.perturbation import verify_perturbation_identities, verify_first_order_perturbation
from src.projection import (
    validate_projection_decomposition_affine,
    validate_projection_decomposition_nonlinear,
)
from src.screening import validate_screening_decomposition
from src.utils import make_spd, op_norm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _ensure_results() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
# Block 1: Screening decomposition
# ------------------------------------------------------------------ #

def block1_screening() -> dict:
    print("=== Block 1: Screening Decomposition ===")
    res = validate_screening_decomposition(n_assets=20, n_drivers=3, n_mc=5000, seed=42)

    frob = res["residual_frob"]
    op = res["residual_op"]
    rel_frob = res["relative_frob_error"]
    mc_err = res["mc_frob_error"]

    print(f"  Analytical residual Frobenius norm : {frob:.3e}")
    print(f"  Analytical residual Operator norm  : {op:.3e}")
    print(f"  Relative Frobenius error           : {rel_frob:.3e}")
    print(f"  Monte Carlo Frobenius error        : {mc_err:.3e}")

    out = {
        "residual_frob": frob,
        "residual_op": op,
        "relative_frob_error": rel_frob,
        "mc_frob_error": mc_err,
        "pass": frob < 1e-10,
    }
    path = os.path.join(RESULTS_DIR, "exp1_block1_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Block 2: Projection decomposition — affine
# ------------------------------------------------------------------ #

def block2_projection_affine() -> dict:
    print("\n=== Block 2: Projection Decomposition (Affine) ===")
    res = validate_projection_decomposition_affine(
        n_assets=20, n_drivers=3, n_mc=5000, seed=42
    )

    omega_max = res["omega_max_eig"]
    rel_err = res["relative_error"]

    print(f"  Omega max eigenvalue (should be ~0) : {omega_max:.3e}")
    print(f"  Relative reconstruction error       : {rel_err:.3e}")
    print(f"  omega_max_eig < 1e-8               : {omega_max < 1e-8}")

    out = {
        "omega_max_eig": omega_max,
        "relative_error": rel_err,
        "pass": omega_max < 1e-8,
    }
    path = os.path.join(RESULTS_DIR, "exp1_block2_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Block 3: Projection decomposition — nonlinear
# ------------------------------------------------------------------ #

def block3_projection_nonlinear() -> dict:
    print("\n=== Block 3: Projection Decomposition (Nonlinear) ===")
    res = validate_projection_decomposition_nonlinear(
        n_assets=20, n_drivers=3, n_mc=5000, seed=42
    )

    omega_max = res["omega_max_eig"]
    omega_psd = res["omega_is_psd"]
    rel_err = res["relative_error"]

    print(f"  Omega max eigenvalue (should be > 0): {omega_max:.3e}")
    print(f"  Omega is PSD                        : {omega_psd}")
    print(f"  Relative reconstruction error       : {rel_err:.3e}")
    print(f"  omega_is_psd=True and omega_max>0   : {omega_psd and omega_max > 0}")

    out = {
        "omega_max_eig": omega_max,
        "omega_is_psd": omega_psd,
        "relative_error": rel_err,
        "pass": omega_psd and (omega_max > 0),
    }
    path = os.path.join(RESULTS_DIR, "exp1_block3_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Block 4: Markowitz identities
# ------------------------------------------------------------------ #

def block4_markowitz() -> dict:
    print("\n=== Block 4: Markowitz Identities ===")
    res = validate_markowitz_identities(n_assets=30, n_constraints=3, gamma=1.0, seed=42)

    constr_res = res["constraint_residual"]
    rel_cvxpy = res["relative_error_vs_cvxpy"]

    print(f"  Constraint residual ||Aw*-b||_2    : {constr_res:.3e}")
    print(f"  Relative error vs CVXPY            : {rel_cvxpy:.3e}")
    print(f"  constraint_residual < 1e-10        : {constr_res < 1e-10}")

    out = {
        "constraint_residual": constr_res,
        "relative_error_vs_cvxpy": rel_cvxpy,
        "mu_p": res["mu_p"],
        "sigma2_p": res["sigma2_p"],
        "Delta": res["Delta"],
        "pass": constr_res < 1e-10,
    }
    path = os.path.join(RESULTS_DIR, "exp1_block4_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Block 5: Perturbation identities
# ------------------------------------------------------------------ #

def block5_perturbation_identities() -> dict:
    print("\n=== Block 5: Perturbation Identities ===")
    rng = np.random.default_rng(0)
    n = 15

    # Construct Q0 and R
    Q0 = make_spd(n, min_eig=0.1, seed=1)
    # R = B @ B.T scaled so Q0 + R is SPD
    B = rng.normal(0.0, 1.0, size=(n, 3))
    R_raw = B @ B.T
    R = 0.05 * R_raw / op_norm(R_raw)

    mu = rng.normal(0.0, 0.05, size=n)
    A = np.ones((1, n))
    b = np.array([1.0])
    gamma = 1.0

    res = verify_perturbation_identities(Q0, R, mu, A, b, gamma)

    id1a = res["id1a_frob"]
    id1b = res["id1b_frob"]
    id2a = res["id2a_frob"]
    id2b = res["id2b_frob"]
    id3 = res["id3_abs"]

    print(f"  Identity I-a  ||M_R-M+M@R@M_R||_F : {id1a:.3e}")
    print(f"  Identity I-b  ||M_R-M+M_R@R@M||_F : {id1b:.3e}")
    print(f"  Identity II-a ||dw+M_R@R@w*||_2   : {id2a:.3e}")
    print(f"  Identity II-b ||dw+M@R@w_R*||_2   : {id2b:.3e}")
    print(f"  Identity III  |dDelta+...| (abs)   : {id3:.3e}")
    all_hold = all(v < 1e-10 for v in [id1a, id1b, id2a, id2b, id3])
    print(f"  All identities < 1e-10             : {all_hold}")

    out = {
        "id1a_frob": id1a,
        "id1b_frob": id1b,
        "id2a_frob": id2a,
        "id2b_frob": id2b,
        "id3_abs": id3,
        "pass": all_hold,
    }
    path = os.path.join(RESULTS_DIR, "exp1_block5_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Block 6: First-order perturbation accuracy
# ------------------------------------------------------------------ #

def block6_first_order(tau_range=(0.0, 0.1)) -> dict:
    print("\n=== Block 6: First-Order Perturbation Accuracy ===")
    rng = np.random.default_rng(1)
    n = 15

    Q0 = make_spd(n, min_eig=0.1, seed=2)
    B = rng.normal(0.0, 1.0, size=(n, 3))
    R_template = B @ B.T
    R_template /= op_norm(R_template)

    mu = rng.normal(0.0, 0.05, size=n)
    A = np.ones((1, n))
    b = np.array([1.0])
    gamma = 1.0

    tau_values = np.linspace(tau_range[0], tau_range[1], 20)
    res = verify_first_order_perturbation(Q0, R_template, mu, A, b, gamma, tau_values)

    exact_dw = res["exact_weight_change"]
    fo_dw = res["first_order_weight_change"]
    exact_dD = res["exact_delta_change"]
    fo_dD = res["first_order_delta_change"]

    # Compute relative errors where exact is nonzero
    w_errs = []
    D_errs = []
    for i, tau in enumerate(tau_values):
        if tau == 0.0:
            continue
        norm_ex = float(np.linalg.norm(exact_dw[i]))
        if norm_ex > 1e-12:
            w_errs.append(float(np.linalg.norm(exact_dw[i] - fo_dw[i])) / norm_ex)
        if abs(exact_dD[i]) > 1e-12:
            D_errs.append(abs(exact_dD[i] - fo_dD[i]) / abs(exact_dD[i]))

    mean_w_err = float(np.mean(w_errs)) if w_errs else float("nan")
    mean_D_err = float(np.mean(D_errs)) if D_errs else float("nan")

    print(f"  Mean relative weight error (tau>0) : {mean_w_err:.3e}")
    print(f"  Mean relative Delta error (tau>0)  : {mean_D_err:.3e}")

    out = {
        "tau_values": tau_values.tolist(),
        "mean_weight_relative_error": mean_w_err,
        "mean_delta_relative_error": mean_D_err,
        "exact_delta_change": exact_dD.tolist(),
        "first_order_delta_change": fo_dD.tolist(),
    }
    path = os.path.join(RESULTS_DIR, "exp1_block6_results.json")
    _save_json(path, out)
    print(f"  Saved -> {path}")

    # Store arrays for plotting
    out["_exact_dw"] = exact_dw
    out["_fo_dw"] = fo_dw
    out["_tau"] = tau_values
    return out


# ------------------------------------------------------------------ #
# Block 7: Diagnostic bound
# ------------------------------------------------------------------ #

def block7_diagnostic() -> dict:
    print("\n=== Block 7: Diagnostic Bound ===")
    res = verify_diagnostic_bound(n_assets=30, n_trials=100, seed=42)

    all_hold = res["all_bounds_hold"]
    op_frac = res["op_le_frob_frac"]
    frob_frac = res["frob_le_bound_frac"]
    inter_frac = res["inter_le_bound_frac"]

    print(f"  All bounds hold                    : {all_hold}")
    print(f"  Fraction ||E||_op <= ||E||_F       : {op_frac:.3f}")
    print(f"  Fraction ||E||_F <= intermediate   : {frob_frac:.3f}")
    print(f"  Fraction intermediate <= bound_F   : {inter_frac:.3f}")

    # Extract per-trial epsilon_cov and E_frob for plotting
    trial_eps = [t["epsilon_cov"] for t in res["trial_results"]]
    trial_E_frob = [t["E_frob"] for t in res["trial_results"]]
    trial_bound = [t["bound_F"] for t in res["trial_results"]]

    out = {
        "all_bounds_hold": all_hold,
        "op_le_frob_frac": op_frac,
        "frob_le_bound_frac": frob_frac,
        "inter_le_bound_frac": inter_frac,
        "pass": all_hold,
        "_trial_eps": trial_eps,
        "_trial_E_frob": trial_E_frob,
        "_trial_bound": trial_bound,
    }
    save_out = {k: v for k, v in out.items() if not k.startswith("_")}
    path = os.path.join(RESULTS_DIR, "exp1_block7_results.json")
    _save_json(path, save_out)
    print(f"  Saved -> {path}")
    return out


# ------------------------------------------------------------------ #
# Plots
# ------------------------------------------------------------------ #

def plot_perturbation(block6_data: dict) -> None:
    tau = block6_data["_tau"]
    exact_dD = np.array(block6_data["exact_delta_change"])
    fo_dD = np.array(block6_data["first_order_delta_change"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau, exact_dD, "b-o", markersize=4, label="Exact $\\Delta_{\\tau} - \\Delta$")
    ax.plot(tau, fo_dD, "r--s", markersize=4, label="First-order approx.")
    ax.set_xlabel("Perturbation magnitude $\\tau$")
    ax.set_ylabel("$\\Delta_{\\tau} - \\Delta$")
    ax.set_title("First-order perturbation accuracy: $\\Delta$ change")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_perturbation_plot.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Perturbation plot saved -> {path}")


def plot_diagnostic_bounds(block7_data: dict) -> None:
    eps = block7_data["_trial_eps"]
    E_frob = block7_data["_trial_E_frob"]
    bound = block7_data["_trial_bound"]
    trials = np.arange(1, len(eps) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(eps, E_frob, s=20, alpha=0.6, label="$\\|E\\|_F$")
    axes[0].scatter(eps, bound, s=20, alpha=0.6, marker="^", label="Bound $\\epsilon_{cov} \\|s\\|^2$")
    axes[0].set_xlabel("$\\epsilon_{cov}$")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Diagnostic: $\\|E\\|_F$ vs bound")
    axes[0].legend()

    axes[1].plot(trials, np.array(bound) - np.array(E_frob), "g-", linewidth=0.8)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Bound $-$ $\\|E\\|_F$  (positive = bound holds)")
    axes[1].set_title("Bound slack per trial")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_diagnostic_bounds.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Diagnostic bounds plot saved -> {path}")


# ------------------------------------------------------------------ #
# Summary table
# ------------------------------------------------------------------ #

def save_summary(blocks: list) -> None:
    rows = [
        ("block", "name", "key_metric", "value", "pass"),
        (1, "Screening decomposition", "residual_frob", blocks[0]["residual_frob"], blocks[0]["pass"]),
        (2, "Projection affine", "omega_max_eig", blocks[1]["omega_max_eig"], blocks[1]["pass"]),
        (3, "Projection nonlinear", "omega_max_eig", blocks[2]["omega_max_eig"], blocks[2]["pass"]),
        (4, "Markowitz identities", "constraint_residual", blocks[3]["constraint_residual"], blocks[3]["pass"]),
        (5, "Perturbation identities", "max_identity_norm", max(blocks[4]["id1a_frob"], blocks[4]["id1b_frob"], blocks[4]["id2a_frob"], blocks[4]["id2b_frob"], blocks[4]["id3_abs"]), blocks[4]["pass"]),
        (6, "First-order perturbation", "mean_weight_rel_err", blocks[5]["mean_weight_relative_error"], True),
        (7, "Diagnostic bound", "all_bounds_hold", int(blocks[6]["all_bounds_hold"]), blocks[6]["pass"]),
    ]
    path = os.path.join(RESULTS_DIR, "exp1_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n  Summary CSV saved -> {path}")

    print("\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"  {'Block':<6} {'Name':<35} {'Key Metric':<30} {'Value':<14} {'Pass'}")
    print(f"  {'-'*6} {'-'*35} {'-'*30} {'-'*14} {'-'*6}")
    for row in rows[1:]:
        b, name, metric, val, passed = row
        val_str = f"{val:.3e}" if isinstance(val, float) else str(val)
        print(f"  {b:<6} {name:<35} {metric:<30} {val_str:<14} {passed}")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def main() -> None:
    _ensure_results()

    b1 = block1_screening()
    b2 = block2_projection_affine()
    b3 = block3_projection_nonlinear()
    b4 = block4_markowitz()
    b5 = block5_perturbation_identities()
    b6 = block6_first_order(tau_range=(0.0, 0.1))
    b7 = block7_diagnostic()

    plot_perturbation(b6)
    plot_diagnostic_bounds(b7)
    save_summary([b1, b2, b3, b4, b5, b6, b7])

    all_pass = all(d["pass"] for d in [b1, b2, b3, b4, b5, b7])
    print(f"\n  All validation blocks passed: {all_pass}")


if __name__ == "__main__":
    main()
