from __future__ import annotations

import numpy as np


def mass_matrix_from_quadrature(
    V: np.ndarray,
    weights: np.ndarray,
    area: float = 1.0,
) -> np.ndarray:
    """
    Build the quadrature-based mass matrix

        M = area * V^T W V

    where W = diag(weights).

    Parameters
    ----------
    V : np.ndarray
        Vandermonde-like matrix of shape (n_points, n_modes).
    weights : np.ndarray
        Quadrature weights of shape (n_points,).
    area : float
        Geometric area factor.

    Returns
    -------
    np.ndarray
        Mass matrix of shape (n_modes, n_modes).
    """
    V = np.asarray(V, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if V.ndim != 2:
        raise ValueError("V must be a 2D array.")
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if V.shape[0] != weights.size:
        raise ValueError("V.shape[0] must match len(weights).")

    WV = weights[:, None] * V
    M = area * (V.T @ WV)
    return M
