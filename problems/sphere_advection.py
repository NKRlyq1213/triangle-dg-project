from __future__ import annotations

import numpy as np

from geometry.sphere_flat_metrics import SphereFlatGeometryCache


def spherical_velocity_lambda_theta(
    lambda_: np.ndarray,
    theta: np.ndarray,
    u0: float = 1.0,
    alpha0: float = 0.0,
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
    alpha0: float = 0.0,
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
