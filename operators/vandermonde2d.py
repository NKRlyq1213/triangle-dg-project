from __future__ import annotations

import numpy as np

from basis.indexing import mode_indices_2d, num_modes_2d
from basis.simplex2d import simplex2d_mode, grad_simplex2d_mode


def vandermonde2d(N: int, r, s) -> np.ndarray:
    """
    Build the 2D Vandermonde matrix on the triangle.

    Parameters
    ----------
    N : int
        Maximum total polynomial degree.
    r, s : array_like
        Evaluation points on the reference triangle.

    Returns
    -------
    np.ndarray
        Vandermonde matrix of shape (num_points, num_modes).
    """
    if N < 0:
        raise ValueError("N must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    r = r.reshape(-1)
    s = s.reshape(-1)

    nmodes = num_modes_2d(N)
    V = np.zeros((r.size, nmodes), dtype=float)

    for k, (i, j) in enumerate(mode_indices_2d(N)):
        V[:, k] = simplex2d_mode(i, j, r, s)
    print(V.shape)
    return V


def grad_vandermonde2d(N: int, r, s) -> tuple[np.ndarray, np.ndarray]:
    """
    Build gradient Vandermonde matrices on the triangle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Vr, Vs), both of shape (num_points, num_modes)
    """
    if N < 0:
        raise ValueError("N must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    r = r.reshape(-1)
    s = s.reshape(-1)

    nmodes = num_modes_2d(N)
    Vr = np.zeros((r.size, nmodes), dtype=float)
    Vs = np.zeros((r.size, nmodes), dtype=float)

    for k, (i, j) in enumerate(mode_indices_2d(N)):
        dr, ds = grad_simplex2d_mode(i, j, r, s)
        Vr[:, k] = dr
        Vs[:, k] = ds

    return Vr, Vs
