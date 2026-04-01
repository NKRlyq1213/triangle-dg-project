from __future__ import annotations

import numpy as np


def _apply_reference_operator(D: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Apply a reference differentiation matrix D to nodal data u.

    Supported shapes
    ----------------
    - u.shape == (Np,)
    - u.shape == (K, Np)
    """
    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D array.")

    Np = D.shape[0]

    if u.ndim == 1:
        if u.shape[0] != Np:
            raise ValueError("For 1D input, u.shape[0] must equal D.shape[0].")
        return D @ u

    if u.ndim == 2:
        if u.shape[1] != Np:
            raise ValueError("For 2D input, u.shape[1] must equal D.shape[0].")
        return u @ D.T

    raise ValueError("u must have shape (Np,) or (K, Np).")


def mapped_divergence_split_2d(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    xr: np.ndarray,
    xs: np.ndarray,
    yr: np.ndarray,
    ys: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    r"""
    Split-form mapped conservative divergence for

        F = (a v, b v)

    on the reference triangle:

        div(F)
        = 1/J [ D_r (J ∇r · F) + D_s (J ∇s · F) ]

    where
        J ∇r = (ys, -xs),
        J ∇s = (-yr, xr).

    Let
        alpha = ys * a - xs * b
        beta  = -yr * a + xr * b

    then the split form is

        div_h^split(F)
        =
        1/J * [ 1/2 ( D_r(alpha v) + alpha D_r(v) + v D_r(alpha) )
              + 1/2 ( D_s(beta  v) + beta  D_s(v) + v D_s(beta ) ) ]

    Parameters
    ----------
    v, a, b, xr, xs, yr, ys, J : np.ndarray
        Shape (Np,) or (K, Np)
    Dr, Ds : np.ndarray
        Reference differentiation matrices, shape (Np, Np)

    Returns
    -------
    np.ndarray
        Split-form divergence values, same shape as v
    """
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    xr = np.asarray(xr, dtype=float)
    xs = np.asarray(xs, dtype=float)
    yr = np.asarray(yr, dtype=float)
    ys = np.asarray(ys, dtype=float)
    J = np.asarray(J, dtype=float)

    if not (v.shape == a.shape == b.shape == xr.shape == xs.shape == yr.shape == ys.shape == J.shape):
        raise ValueError("v, a, b, xr, xs, yr, ys, J must all have the same shape.")

    alpha = ys * a - xs * b
    beta = -yr * a + xr * b

    vr = _apply_reference_operator(Dr, v)
    vs = _apply_reference_operator(Ds, v)

    ar = _apply_reference_operator(Dr, alpha)
    bs = _apply_reference_operator(Ds, beta)

    alpha_v = alpha * v
    beta_v = beta * v

    Dr_alpha_v = _apply_reference_operator(Dr, alpha_v)
    Ds_beta_v = _apply_reference_operator(Ds, beta_v)

    split_r = 0.5 * (Dr_alpha_v + alpha * vr + v * ar)
    split_s = 0.5 * (Ds_beta_v + beta * vs + v * bs)

    return (split_r + split_s) / J


def mapped_divergence_conservative_2d(
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    xr: np.ndarray,
    xs: np.ndarray,
    yr: np.ndarray,
    ys: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    r"""
    Conservative mapped divergence for

        F = (a v, b v)

    defined by

        div(F)
        = 1/J [ D_r(alpha v) + D_s(beta v) ]

    where
        alpha = ys * a - xs * b
        beta  = -yr * a + xr * b
    """
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    xr = np.asarray(xr, dtype=float)
    xs = np.asarray(xs, dtype=float)
    yr = np.asarray(yr, dtype=float)
    ys = np.asarray(ys, dtype=float)
    J = np.asarray(J, dtype=float)

    if not (v.shape == a.shape == b.shape == xr.shape == xs.shape == yr.shape == ys.shape == J.shape):
        raise ValueError("v, a, b, xr, xs, yr, ys, J must all have the same shape.")

    alpha = ys * a - xs * b
    beta = -yr * a + xr * b

    Dr_alpha_v = _apply_reference_operator(Dr, alpha * v)
    Ds_beta_v = _apply_reference_operator(Ds, beta * v)

    return (Dr_alpha_v + Ds_beta_v) / J