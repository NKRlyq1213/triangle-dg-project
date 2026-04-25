from __future__ import annotations

import numpy as np

from geometry.sphere_mapping import local_xy_to_patch_angles
from geometry.sphere_metrics import contravariant_velocity


def spherical_advection_velocity(
    lambda_,
    theta,
    *,
    u0: float = 1.0,
    alpha0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Velocity field from SDG4PDEOnSphere notes.

    V = (u, v) = u e_lambda + v e_theta

    u = u0 (cos(alpha0) cos(theta)
             + sin(alpha0) cos(lambda) sin(theta))
    v = -u0 sin(alpha0) sin(lambda)
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)

    u = u0 * (
        np.cos(alpha0) * np.cos(theta)
        + np.sin(alpha0) * np.cos(lambda_) * np.sin(theta)
    )
    v = -u0 * np.sin(alpha0) * np.sin(lambda_)
    return u, v


def local_contravariant_advection_velocity(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    u0: float = 1.0,
    alpha0: float = 0.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose:
        local (x,y)
        -> (lambda, theta)
        -> spherical velocity (u,v)
        -> local contravariant velocity (u1,u2)

    This is the object that should be passed into the planar triangle RHS:
        q_t + d_x(u1 q) + d_y(u2 q) = 0
    """
    lambda_, theta = local_xy_to_patch_angles(x, y, patch_id, tol=tol)
    u, v = spherical_advection_velocity(lambda_, theta, u0=u0, alpha0=alpha0)

    return contravariant_velocity(
        u, v, x, y, patch_id, radius=radius, tol=tol
    )


def smooth_sphere_initial_condition(lambda_, theta) -> np.ndarray:
    """
    Simple smooth scalar field for visualization / smoke tests.

    可換成你論文中的 exact solution。
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)

    return np.sin(lambda_) * np.cos(theta)