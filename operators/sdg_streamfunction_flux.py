from __future__ import annotations

import math
import numpy as np

from geometry.metrics import physical_derivatives_2d, divergence_2d


def solid_body_omega(
    *,
    alpha0: float,
    u0: float = 1.0,
) -> np.ndarray:
    r"""
    Solid-body rotation vector.

    Omega = u0 * (-sin(alpha0), 0, cos(alpha0))

    This matches the sphere-advection convention already used in the project.
    """
    return np.array(
        [
            -math.sin(float(alpha0)) * float(u0),
            0.0,
            math.cos(float(alpha0)) * float(u0),
        ],
        dtype=float,
    )


def solid_body_streamfunction(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    alpha0: float,
    u0: float = 1.0,
    R: float = 1.0,
) -> np.ndarray:
    r"""
    Stream function for solid-body rotation on the sphere.

    Correct sign convention verified by velocity reconstruction diagnostic:

        psi = - R * Omega dot X

    where:

        Omega = u0 * (-sin(alpha0), 0, cos(alpha0))

    Parameters
    ----------
    X, Y, Z:
        Sphere coordinates at volume nodes, shape (K, Np).
    alpha0:
        Rotation-axis tilt parameter.
    u0:
        Velocity scale.
    R:
        Sphere radius.

    Returns
    -------
    psi:
        Stream function values, shape (K, Np).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, Z must have the same shape.")

    omega = solid_body_omega(alpha0=alpha0, u0=u0)

    return -float(R) * (
        omega[0] * X
        + omega[1] * Y
        + omega[2] * Z
    )


def streamfunction_area_flux(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
    *,
    alpha0: float,
    u0: float = 1.0,
    R: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Construct area-weighted conservative flux from the stream function.

    Given:

        psi = - R * Omega dot X

    define:

        Fx_area = - d_y psi
        Fy_area =   d_x psi

    These are area-weighted contravariant fluxes:

        Fx_area = J u^x
        Fy_area = J u^y

    where J = sqrt(G) is the surface area Jacobian.

    For equal-area SDG mapping, J is constant:

        J = pi R^2

    Returns
    -------
    Fx_area, Fy_area, psi:
        Each shape (K, Np).
    """
    psi = solid_body_streamfunction(
        X,
        Y,
        Z,
        alpha0=alpha0,
        u0=u0,
        R=R,
    )

    psi_x, psi_y = physical_derivatives_2d(
        psi,
        Dr,
        Ds,
        rx,
        sx,
        ry,
        sy,
    )

    Fx_area = -psi_y
    Fy_area = psi_x

    return Fx_area, Fy_area, psi


def streamfunction_area_divergence(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
    *,
    alpha0: float,
    u0: float = 1.0,
    R: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convenience diagnostic.

    Computes:

        Fx_area = -psi_y
        Fy_area =  psi_x
        div_area = d_x Fx_area + d_y Fy_area

    For q = 1, div_area should be near roundoff.
    """
    Fx_area, Fy_area, psi = streamfunction_area_flux(
        X,
        Y,
        Z,
        Dr,
        Ds,
        rx,
        sx,
        ry,
        sy,
        alpha0=alpha0,
        u0=u0,
        R=R,
    )

    div_area = divergence_2d(
        Fx_area,
        Fy_area,
        Dr,
        Ds,
        rx,
        sx,
        ry,
        sy,
    )

    return Fx_area, Fy_area, psi, div_area


def reconstruct_physical_velocity_from_area_flux(
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    rx: np.ndarray,
    sx: np.ndarray,
    ry: np.ndarray,
    sy: np.ndarray,
    *,
    J_area: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Reconstruct physical 3D velocity from area-weighted flux.

    Since:

        Fx_area = J u^x
        Fy_area = J u^y

    we compute:

        u^x = Fx_area / J
        u^y = Fy_area / J

    and reconstruct:

        U = u^x X_x + u^y X_y.

    This diagnostic is mainly for verification, not for the volume RHS.
    """
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if Fx_area.shape != Fy_area.shape:
        raise ValueError("Fx_area and Fy_area must have the same shape.")

    Xx, Xy = physical_derivatives_2d(X, Dr, Ds, rx, sx, ry, sy)
    Yx, Yy = physical_derivatives_2d(Y, Dr, Ds, rx, sx, ry, sy)
    Zx, Zy = physical_derivatives_2d(Z, Dr, Ds, rx, sx, ry, sy)

    ux = Fx_area / J_area
    uy = Fy_area / J_area

    Ux = ux * Xx + uy * Xy
    Uy = ux * Yx + uy * Yy
    Uz = ux * Zx + uy * Zy

    return Ux, Uy, Uz


def exact_solid_body_velocity_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    alpha0: float,
    u0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Exact physical velocity:

        U = Omega x X.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, Z must have the same shape.")

    omega = solid_body_omega(alpha0=alpha0, u0=u0)
    ox, oy, oz = omega

    Ux = oy * Z - oz * Y
    Uy = oz * X - ox * Z
    Uz = ox * Y - oy * X

    return Ux, Uy, Uz


def equal_area_jacobian(R: float = 1.0) -> float:
    r"""
    Equal-area SDG surface Jacobian.

        J = sqrt(G) = pi R^2
    """
    return math.pi * float(R) * float(R)
