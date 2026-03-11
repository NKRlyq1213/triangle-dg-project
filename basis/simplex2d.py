from __future__ import annotations

import numpy as np

from .jacobi import jacobi_orthonormal, grad_jacobi_orthonormal


def rstoab(r, s):
    """
    Map (r, s) on the reference triangle to (a, b) coordinates.

    Reference triangle in (r, s):
        -1 <= r <= 1
        -1 <= s <= 1
        r + s <= 0

    Mapping:
        a = 2*(1+r)/(1-s) - 1,  if s != 1
        b = s

    At s = 1, define a = -1 by continuity / convention.
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    if r.shape != s.shape:
        raise ValueError("r and s must have the same shape.")

    a = np.empty_like(r, dtype=float)
    mask = np.abs(1.0 - s) > 1e-14
    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0

    b = s.copy()
    return a, b


def simplex2d_mode(i: int, j: int, r, s) -> np.ndarray:
    """
    Evaluate the 2D orthonormal simplex basis mode on the triangle.

    Basis:
        psi_{ij}(r,s) = sqrt(2)
                       * P_i^{(0,0)}(a)
                       * P_j^{(2i+1,0)}(b)
                       * (1-b)^i
    where P_n^{(alpha,beta)} is orthonormal on [-1,1].
    """
    if i < 0 or j < 0:
        raise ValueError("i and j must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    a, b = rstoab(r, s)

    fa = jacobi_orthonormal(i, 0.0, 0.0, a)
    gb = jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)

    return np.sqrt(2.0) * fa * gb * (1.0 - b) ** i


def grad_simplex2d_mode(i: int, j: int, r, s) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate gradients of the 2D simplex basis mode with respect to r and s.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dpsi/dr, dpsi/ds)
    """
    if i < 0 or j < 0:
        raise ValueError("i and j must be >= 0")

    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    a, b = rstoab(r, s)

    fa = jacobi_orthonormal(i, 0.0, 0.0, a)
    dfa = grad_jacobi_orthonormal(i, 0.0, 0.0, a)

    gb = jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)
    dgb = grad_jacobi_orthonormal(j, 2.0 * i + 1.0, 0.0, b)

    one_minus_b = 1.0 - b

    # Handle the (1-b)^(i-1) term safely when i = 0
    if i == 0:
        h = np.ones_like(b, dtype=float)
        dh_db = np.zeros_like(b, dtype=float)
    else:
        h = one_minus_b ** i
        dh_db = -i * one_minus_b ** (i - 1)

    # a = 2*(1+r)/(1-s) - 1,  b = s
    # da/dr = 2/(1-s) = 2/(1-b)
    # da/ds = (1+a)/(1-b)
    # db/dr = 0
    # db/ds = 1
    eps = 1e-14
    denom = np.where(np.abs(one_minus_b) > eps, one_minus_b, eps)

    da_dr = 2.0 / denom
    da_ds = (1.0 + a) / denom
    db_ds = np.ones_like(b, dtype=float)

    pref = np.sqrt(2.0)

    dpsi_da = pref * dfa * gb * h
    dpsi_db = pref * fa * (dgb * h + gb * dh_db)

    dpsi_dr = dpsi_da * da_dr
    dpsi_ds = dpsi_da * da_ds + dpsi_db * db_ds

    return dpsi_dr, dpsi_ds
