from __future__ import annotations

import numpy as np

from operators.mass import mass_matrix_from_quadrature


def differentiation_matrices_square(
    V: np.ndarray,
    Vr: np.ndarray,
    Vs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build square nodal differentiation matrices:

        Dr = Vr V^{-1}
        Ds = Vs V^{-1}

    Parameters
    ----------
    V : np.ndarray
        Square Vandermonde matrix at nodal points.
    Vr, Vs : np.ndarray
        Gradient Vandermonde matrices at the same nodal points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Dr, Ds), both of shape (n_points, n_points).
    """
    V = np.asarray(V, dtype=float)
    Vr = np.asarray(Vr, dtype=float)
    Vs = np.asarray(Vs, dtype=float)

    if V.ndim != 2 or Vr.ndim != 2 or Vs.ndim != 2:
        raise ValueError("V, Vr, and Vs must be 2D arrays.")
    if V.shape[0] != V.shape[1]:
        raise ValueError("Square differentiation requires V to be square.")
    if Vr.shape != V.shape or Vs.shape != V.shape:
        raise ValueError("Vr and Vs must have the same shape as V.")

    Vinv = np.linalg.inv(V)
    Dr = Vr @ Vinv
    Ds = Vs @ Vinv
    return Dr, Ds


def differentiation_matrices_weighted(
    V: np.ndarray,
    Vr: np.ndarray,
    Vs: np.ndarray,
    weights: np.ndarray,
    area: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build weighted projection differentiation matrices on the sampling grid:

        Dr = Vr (A V^T W V)^{-1} (A V^T W)
        Ds = Vs (A V^T W V)^{-1} (A V^T W)

    where A = area.

    Parameters
    ----------
    V : np.ndarray
        Vandermonde matrix of shape (n_points, n_modes).
    Vr, Vs : np.ndarray
        Gradient Vandermonde matrices of shape (n_points, n_modes).
    weights : np.ndarray
        Quadrature weights of shape (n_points,).
    area : float
        Geometric area factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Dr, Ds), both of shape (n_points, n_points).
    """
    V = np.asarray(V, dtype=float)
    Vr = np.asarray(Vr, dtype=float)
    Vs = np.asarray(Vs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if V.ndim != 2 or Vr.ndim != 2 or Vs.ndim != 2:
        raise ValueError("V, Vr, and Vs must be 2D arrays.")
    if Vr.shape != V.shape or Vs.shape != V.shape:
        raise ValueError("Vr and Vs must have the same shape as V.")
    if V.shape[0] != weights.size:
        raise ValueError("weights size must match the number of rows in V.")

    M = mass_matrix_from_quadrature(V, weights, area=area)

    # rhs = A * V^T * W
    rhs = area * (V.T * weights[None, :])

    # proj = (A V^T W V)^(-1) (A V^T W)
    proj = np.linalg.solve(M, rhs)

    # IMPORTANT:
    # Do NOT multiply by area again outside.
    Dr = Vr @ proj
    Ds = Vs @ proj
    return Dr, Ds
