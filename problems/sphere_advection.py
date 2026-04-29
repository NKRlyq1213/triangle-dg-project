from __future__ import annotations

import numpy as np

from geometry.sphere_flat_metrics import SphereFlatGeometryCache


def spherical_velocity_lambda_theta(
    lambda_: np.ndarray,
    theta: np.ndarray,
    u0: float = 1.0,
    alpha0: float = -np.pi / 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    SDG-style spherical velocity field in (lambda, theta) components.

    u = u0 * (cos(alpha0)*cos(theta) + sin(alpha0)*cos(lambda)*sin(theta))
    v = -u0 * sin(alpha0)*sin(lambda)

    Returns
    -------
    u, v : np.ndarray
        Components associated with the spherical orthonormal coordinates used by A.
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)

    u = u0 * (
        np.cos(alpha0) * np.cos(theta)
        + np.sin(alpha0) * np.cos(lambda_) * np.sin(theta)
    )
    v = -u0 * np.sin(alpha0) * np.sin(lambda_)
    return u, v


def flattened_velocity_from_cache(
    cache: SphereFlatGeometryCache,
    u0: float = 1.0,
    alpha0: float = -np.pi / 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical velocity (u,v) into flattened Cartesian velocity (u1,u2):

        [u1,u2]^T = Ainv [u,v]^T

    Pole limit values are not applied.
    """
    u, v = spherical_velocity_lambda_theta(
        lambda_=cache.lambda_,
        theta=cache.theta,
        u0=u0,
        alpha0=alpha0,
    )

    uv = np.stack([u, v], axis=-1)
    flat = np.einsum("...ij,...j->...i", cache.Ainv, uv)

    u1 = flat[..., 0]
    u2 = flat[..., 1]

    u1 = np.where(cache.pole_mask, np.nan, u1)
    u2 = np.where(cache.pole_mask, np.nan, u2)

    return u1, u2, u, v


def sphere_tangent_xyz_velocity(
    lambda_: np.ndarray,
    theta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical velocity components into 3D tangent vector.

    Basis:
        e_lambda = (-sin lambda, cos lambda, 0)
        e_theta  = (-cos lambda sin theta, -sin lambda sin theta, cos theta)

    tangent = u * e_lambda + v * e_theta
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    ex_l = -np.sin(lambda_)
    ey_l =  np.cos(lambda_)
    ez_l =  np.zeros_like(lambda_)

    ex_t = -np.cos(lambda_) * np.sin(theta)
    ey_t = -np.sin(lambda_) * np.sin(theta)
    ez_t =  np.cos(theta)

    VX = u * ex_l + v * ex_t
    VY = u * ey_l + v * ey_t
    VZ = u * ez_l + v * ez_t

    return VX, VY, VZ


def solid_body_velocity_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    u0: float = 1.0,
    alpha0: float = -np.pi / 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    3D solid-body rotation velocity on the sphere.

    This is the 3D equivalent of the SDG spherical velocity field:

        Omega = u0 * (-sin(alpha0), 0, cos(alpha0))
        V = Omega x r

    The resulting velocity is tangent to the sphere for every point r=(X,Y,Z).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, Z must have the same shape.")

    omega_x = -np.sin(alpha0) * u0
    omega_y = 0.0
    omega_z = np.cos(alpha0) * u0

    U = omega_y * Z - omega_z * Y
    V = omega_z * X - omega_x * Z
    W = omega_x * Y - omega_y * X

    return U, V, W


def _solid_body_omega(u0: float, alpha0: float) -> np.ndarray:
    return np.array(
        [-np.sin(alpha0) * u0, 0.0, np.cos(alpha0) * u0],
        dtype=float,
    )


def _rotate_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    omega: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=float).reshape(3)
    speed = float(np.linalg.norm(omega))
    if speed == 0.0 or t == 0.0:
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)

    axis = omega / speed
    angle = speed * float(t)
    c = np.cos(angle)
    s = np.sin(angle)

    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    z = np.asarray(Z, dtype=float)

    ax, ay, az = axis
    dot = ax * x + ay * y + az * z
    cx = ay * z - az * y
    cy = az * x - ax * z
    cz = ax * y - ay * x

    xr = x * c + cx * s + ax * dot * (1.0 - c)
    yr = y * c + cy * s + ay * dot * (1.0 - c)
    zr = z * c + cz * s + az * dot * (1.0 - c)
    return xr, yr, zr


def constant_field_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    value: float = 1.0,
) -> np.ndarray:
    del Y, Z
    return np.full_like(np.asarray(X, dtype=float), float(value), dtype=float)


def gaussian_bell_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    R: float = 1.0,
    center_xyz: tuple[float, float, float] = (0.0, 1.0, 0.0),
    width: float = 1.0 / np.sqrt(10.0),
) -> np.ndarray:
    """
    Smooth Gaussian bell defined by geodesic distance on the sphere.
    """
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if width <= 0.0:
        raise ValueError("width must be positive.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, Z must have the same shape.")

    center = np.asarray(center_xyz, dtype=float).reshape(3)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 0.0:
        raise ValueError("center_xyz must not be the zero vector.")
    center = R * center / center_norm

    dot = (X * center[0] + Y * center[1] + Z * center[2]) / (R * R)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    dist = R * angle
    return np.exp(-((dist / float(width)) ** 2))


def exact_gaussian_bell_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    t: float,
    u0: float = 1.0,
    R: float = 1.0,
    alpha0: float = -np.pi / 4.0,
    center_xyz: tuple[float, float, float] = (0.0, 1.0, 0.0),
    width: float = 1.0 / np.sqrt(10.0),
) -> np.ndarray:
    """
    Exact solid-body advection of a Gaussian bell via inverse Rodrigues rotation.
    """
    omega = _solid_body_omega(u0=u0, alpha0=alpha0)
    x0, y0, z0 = _rotate_xyz(X, Y, Z, omega=omega, t=-float(t))
    return gaussian_bell_xyz(
        x0,
        y0,
        z0,
        R=R,
        center_xyz=center_xyz,
        width=width,
    )
