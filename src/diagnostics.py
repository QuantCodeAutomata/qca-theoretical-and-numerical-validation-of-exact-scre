"""
Diagnostic-to-operator bound for the off-diagonal residual of a covariance matrix.

Given a positive (semi)definite matrix V, decompose it as:

    V = D + E

where D = diag(V) and E = V - diag(diag(V)) is the off-diagonal part.

Define the normalised maximum off-diagonal correlation:

    epsilon_cov = max_{i < j} |E_{ij}| / (s_i * s_j),    s_i = sqrt(V_{ii})

The following chain of inequalities holds:

    ||E||_2  <=  ||E||_F
             <=  epsilon_cov * sqrt( (sum_i s_i^2)^2 - sum_i s_i^4 )
             <=  epsilon_cov * (sum_i s_i^2)
              =  epsilon_cov * ||s||_2^2
"""

import numpy as np

from src.utils import frob_norm, make_spd, op_norm


def compute_residual_diagnostic(V: np.ndarray) -> dict:
    """
    Decompose V = diag(V) + E and compute norm/bound quantities.

    Parameters
    ----------
    V : np.ndarray, shape (n, n)
        Symmetric positive (semi)definite matrix.

    Returns
    -------
    dict with keys:
        epsilon_cov : float  max_{i<j} |E_{ij}| / (s_i * s_j)
        E_frob      : float  ||E||_F
        E_op        : float  ||E||_2
        bound_F     : float  epsilon_cov * ||s||_2^2  (upper bound on ||E||_F)
        bound_op    : float  same as bound_F           (upper bound on ||E||_2)
        s           : (n,)  vector of standard deviations  sqrt(diag(V))
        E           : (n, n) off-diagonal matrix
    """
    n = V.shape[0]
    diag_V = np.diag(V)                         # s_i^2
    s = np.sqrt(np.maximum(diag_V, 0.0))        # (n,)

    D = np.diag(diag_V)                         # diagonal part
    E = V - D                                   # off-diagonal part

    # epsilon_cov = max_{i < j} |E_{ij}| / (s_i * s_j)
    # Avoid division by zero (if any variance is zero)
    outer_s = np.outer(s, s)                    # s_i * s_j  (n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_abs = np.where(outer_s > 0.0, np.abs(E) / outer_s, 0.0)
    # Zero out diagonal (not relevant)
    np.fill_diagonal(corr_abs, 0.0)
    # Upper-triangular pairs only
    i_up, j_up = np.triu_indices(n, k=1)
    epsilon_cov = float(corr_abs[i_up, j_up].max()) if len(i_up) > 0 else 0.0

    E_frob = frob_norm(E)
    E_op = op_norm(E)

    # Tighter intermediate bound: epsilon_cov * sqrt(||s||_2^4 - sum(s^4))
    s2 = s**2
    sum_s2 = float(s2.sum())                    # sum_i s_i^2 = ||s||_2^2
    sum_s4 = float((s2**2).sum())               # sum_i s_i^4
    intermediate = epsilon_cov * float(np.sqrt(max(sum_s2**2 - sum_s4, 0.0)))

    # Final (looser) bound: epsilon_cov * sum_i s_i^2 = epsilon_cov * ||s||_2^2
    bound_F = epsilon_cov * sum_s2
    bound_op = bound_F  # ||E||_2 <= ||E||_F <= bound_F

    return {
        "epsilon_cov": epsilon_cov,
        "E_frob": E_frob,
        "E_op": E_op,
        "intermediate_bound": intermediate,
        "bound_F": bound_F,
        "bound_op": bound_op,
        "s": s,
        "E": E,
    }


def verify_diagnostic_bound(
    n_assets: int = 30,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """
    Verify the diagnostic bound chain across n_trials random PD matrices.

    For each trial:
      (a) ||E||_2 <= ||E||_F
      (b) ||E||_F <= epsilon_cov * sqrt((sum s_i^2)^2 - sum s_i^4)
      (c) epsilon_cov * sqrt(...) <= epsilon_cov * ||s||_2^2

    Parameters
    ----------
    n_assets : int
    n_trials : int
    seed : int

    Returns
    -------
    dict with keys:
        trial_results       : list of per-trial diagnostic dicts
        all_bounds_hold     : bool  True if (a)+(b)+(c) all hold for every trial
        frob_le_bound_frac  : float  fraction of trials where (b) holds
        op_le_frob_frac     : float  fraction of trials where (a) holds
        inter_le_bound_frac : float  fraction of trials where (c) holds
    """
    rng = np.random.default_rng(seed)
    trial_results = []

    a_holds = []  # ||E||_2 <= ||E||_F
    b_holds = []  # ||E||_F <= intermediate_bound
    c_holds = []  # intermediate_bound <= bound_F

    for t in range(n_trials):
        V = make_spd(n_assets, min_eig=0.05, seed=int(rng.integers(1e9)))
        diag = compute_residual_diagnostic(V)
        trial_results.append(diag)

        tol = 1e-10  # floating-point tolerance
        a_holds.append(diag["E_op"] <= diag["E_frob"] + tol)
        b_holds.append(diag["E_frob"] <= diag["intermediate_bound"] + tol)
        c_holds.append(diag["intermediate_bound"] <= diag["bound_F"] + tol)

    all_bounds_hold = all(a_holds) and all(b_holds) and all(c_holds)

    return {
        "trial_results": trial_results,
        "all_bounds_hold": all_bounds_hold,
        "op_le_frob_frac": float(np.mean(a_holds)),
        "frob_le_bound_frac": float(np.mean(b_holds)),
        "inter_le_bound_frac": float(np.mean(c_holds)),
    }
