"""
Matrix utility functions for the QCA library.

Provides helpers for generating random SPD matrices, computing norms,
checking definiteness, and projecting matrices onto the PSD cone.
"""

import numpy as np
import scipy.linalg


def make_spd(n: int, min_eig: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Generate a random symmetric positive definite (SPD) matrix of size n.

    Constructs A = F.T @ F + min_eig * I, where F is a random n×n matrix,
    ensuring all eigenvalues are at least min_eig.

    Parameters
    ----------
    n : int
        Matrix dimension.
    min_eig : float
        Minimum eigenvalue floor added to ensure strict positivity.
    seed : int
        Random seed.

    Returns
    -------
    A : np.ndarray, shape (n, n)
        Random SPD matrix.
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, n))
    A = F.T @ F / n + min_eig * np.eye(n)  # scale by 1/n to keep entries O(1)
    return A


def frob_norm(A: np.ndarray) -> float:
    """
    Frobenius norm of matrix (or Euclidean norm of vector) A.

    ||A||_F = sqrt(sum_{i,j} A_{ij}^2)

    Parameters
    ----------
    A : np.ndarray
        1-D vector or 2-D matrix.

    Returns
    -------
    float
    """
    # np.linalg.norm with no ord is Frobenius for matrices, Euclidean for vectors
    return float(np.linalg.norm(A))


def op_norm(A: np.ndarray) -> float:
    """
    Operator (spectral) norm of matrix A, or Euclidean norm of vector A.

    For a matrix: ||A||_2 = sigma_max(A)  (largest singular value).
    For a vector: ||A||_2 (Euclidean norm, consistent with frob_norm).

    Uses scipy.linalg.svdvals for numerical stability on matrices.

    Parameters
    ----------
    A : np.ndarray
        1-D vector or 2-D matrix.

    Returns
    -------
    float
    """
    if A.ndim == 1:
        return float(np.linalg.norm(A))
    return float(scipy.linalg.svdvals(A)[0])


def relative_error(A: np.ndarray, B: np.ndarray, norm: str = "frob") -> float:
    """
    Relative error ||A - B|| / ||B||.

    Parameters
    ----------
    A : np.ndarray
        Approximation or perturbed matrix.
    B : np.ndarray
        Reference matrix.
    norm : str
        'frob' for Frobenius norm, 'op' for operator (spectral) norm.

    Returns
    -------
    float
        Relative error.  Returns inf when ||B|| == 0.

    Raises
    ------
    ValueError
        If norm is not 'frob' or 'op'.
    """
    if norm == "frob":
        norm_fn = frob_norm
    elif norm == "op":
        norm_fn = op_norm
    else:
        raise ValueError(f"norm must be 'frob' or 'op'; got '{norm}'.")

    denom = norm_fn(B)
    if denom == 0.0:
        return float("inf")
    return norm_fn(A - B) / denom


def is_psd(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check whether A is positive semidefinite via its minimum eigenvalue.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric matrix.
    tol : float
        Eigenvalues >= -tol are considered non-negative.

    Returns
    -------
    bool
    """
    # Force symmetry before eigendecomposition
    A_sym = (A + A.T) / 2.0
    min_ev = float(np.linalg.eigvalsh(A_sym).min())
    return min_ev >= -tol


def is_pd(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check whether A is positive definite via its minimum eigenvalue.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric matrix.
    tol : float
        Eigenvalues > tol are considered strictly positive.

    Returns
    -------
    bool
    """
    A_sym = (A + A.T) / 2.0
    min_ev = float(np.linalg.eigvalsh(A_sym).min())
    return min_ev > tol


def project_spd(A: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone by flooring negative eigenvalues.

    Computes A_+ = V @ diag(max(lambda, floor)) @ V.T.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric (or nearly symmetric) matrix.
    floor : float
        Minimum eigenvalue to enforce (>= 0).

    Returns
    -------
    A_psd : np.ndarray, shape (n, n)
        PSD projection of A.
    """
    A_sym = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_floored = np.maximum(eigvals, floor)
    A_psd = eigvecs @ np.diag(eigvals_floored) @ eigvecs.T
    # Symmetrise to remove floating-point asymmetry
    return (A_psd + A_psd.T) / 2.0
