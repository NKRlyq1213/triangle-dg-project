from __future__ import annotations

import numpy as np

from geometry.metrics import divergence_2d


def sdg_conservative_volume_rhs(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
    *,
    J_area: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Volume-only conservative RHS for equal-area SDG sphere advection.

    Continuous conservative form:

        d_t(J q) + d_x(Fx_area q) + d_y(Fy_area q) = 0

    where:

        Fx_area = J u^x
        Fy_area = J u^y

    For equal-area mapping:

        J = pi R^2 = constant

    Therefore:

        q_t = -1/J * [d_x(Fx_area q) + d_y(Fy_area q)]

    Parameters
    ----------
    q:
        Scalar state, shape (K, Np).
    Fx_area, Fy_area:
        Area-weighted conservative flux components, shape (K, Np).
    Dr, Ds:
        Reference differentiation matrices.
    rx, sx, ry, sy:
        Physical derivative metric factors on flattened mesh.
    J_area:
        Constant equal-area surface Jacobian.
    mask:
        Optional boolean mask. True means excluded / invalid. Masked RHS is set to NaN.

    Returns
    -------
    rhs:
        q_t volume-only RHS, shape (K, Np).
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if not (q.shape == Fx_area.shape == Fy_area.shape):
        raise ValueError("q, Fx_area, and Fy_area must have the same shape.")

    if J_area == 0.0:
        raise ValueError("J_area must be nonzero.")

    flux_x = Fx_area * q
    flux_y = Fy_area * q

    div_flux = divergence_2d(
        flux_x,
        flux_y,
        Dr,
        Ds,
        rx,
        sx,
        ry,
        sy,
    )

    rhs = -(1.0 / float(J_area)) * div_flux

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rhs.shape:
            raise ValueError("mask must have the same shape as rhs.")
        rhs = np.where(mask, np.nan, rhs)

    return rhs


def sdg_conservative_volume_divergence(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Return only:

        div(F q) = d_x(Fx_area q) + d_y(Fy_area q)

    This is useful for diagnostics before dividing by J.
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if not (q.shape == Fx_area.shape == Fy_area.shape):
        raise ValueError("q, Fx_area, and Fy_area must have the same shape.")

    div_flux = divergence_2d(
        Fx_area * q,
        Fy_area * q,
        Dr,
        Ds,
        rx,
        sx,
        ry,
        sy,
    )

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != div_flux.shape:
            raise ValueError("mask must have the same shape as div_flux.")
        div_flux = np.where(mask, np.nan, div_flux)

    return div_flux
