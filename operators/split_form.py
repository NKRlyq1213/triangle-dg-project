from __future__ import annotations

import numpy as np

from geometry.metrics import physical_derivatives_2d


def split_advective_operator_2d(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
) -> np.ndarray:
    """
    2D split advective operator:

        L(v) =
        1/2 [ d_x(a v) + a d_x(v) - d_x(a) v ]
      + 1/2 [ d_y(b v) + b d_y(v) - d_y(b) v ]

    This approximates the advective form:

        a * v_x + b * v_y

    Parameters
    ----------
    v, a, b : np.ndarray
        Nodal values on physical elements, shape (K, Np) or (Np,)
    Dr, Ds : np.ndarray
        Reference differentiation matrices, shape (Np, Np)
    rx, sx, ry, sy : np.ndarray
        Geometric factors, same shape as v

    Returns
    -------
    np.ndarray
        Split-form operator applied to v, same shape as v
    """
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if not (v.shape == a.shape == b.shape == rx.shape == sx.shape == ry.shape == sy.shape):
        raise ValueError("v, a, b, rx, sx, ry, sy must all have the same shape.")

    # v_x, v_y
    vx, vy = physical_derivatives_2d(v, Dr, Ds, rx, sx, ry, sy)

    # a_x, b_y
    ax, _ = physical_derivatives_2d(a, Dr, Ds, rx, sx, ry, sy)
    _, by = physical_derivatives_2d(b, Dr, Ds, rx, sx, ry, sy)

    # d_x(a v), d_y(b v)
    av = a * v
    bv = b * v

    dav_dx, _ = physical_derivatives_2d(av, Dr, Ds, rx, sx, ry, sy)
    _, dbv_dy = physical_derivatives_2d(bv, Dr, Ds, rx, sx, ry, sy)

    return 0.5 * (dav_dx + a * vx - ax * v) + 0.5 * (dbv_dy + b * vy - by * v)