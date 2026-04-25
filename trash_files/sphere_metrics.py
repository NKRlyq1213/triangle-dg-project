from __future__ import annotations

import numpy as np

from geometry.sphere_patches import assert_patch_local_xy
from geometry.sphere_mapping import local_xy_to_patch_angles


def local_angle_derivatives(
    x,
    y,
    patch_id: int,
    *,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lambda_x, lambda_y, theta_x, theta_y
    for patch-local coordinates.

    Local variables:
        rho = x + y
        lambda = sector_base + (pi/2) y/rho

    Therefore:
        lambda_x = -pi*y/(2*rho^2)
        lambda_y =  pi*x/(2*rho^2)

    Upper patch:
        sin(theta) = 1 - rho^2
        theta_x = theta_y = -2*rho/cos(theta)

    Lower patch:
        sin(theta) = rho^2 - 1
        theta_x = theta_y =  2*rho/cos(theta)

    At rho = 0, longitude derivatives are singular.
    This corresponds to the pole. For numerical DG nodes, avoid placing
    a volume node exactly at the pole, or handle it by a limiting convention.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    assert_patch_local_xy(x, y, tol=tol)

    lambda_, theta = local_xy_to_patch_angles(x, y, patch_id, tol=tol)
    rho = x + y
    cos_theta = np.cos(theta)

    if np.any(rho <= tol):
        raise ValueError(
            "lambda derivatives are singular at rho=x+y=0. "
            "Avoid pole nodes for metric tests, or use a one-sided/limit convention."
        )

    lambda_x = -np.pi * y / (2.0 * rho**2)
    lambda_y =  np.pi * x / (2.0 * rho**2)

    # upper: odd patches, lower: even patches
    upper = patch_id in (1, 3, 5, 7)
    sign = -1.0 if upper else +1.0

    theta_x = sign * 2.0 * rho / cos_theta
    theta_y = sign * 2.0 * rho / cos_theta

    return lambda_x, lambda_y, theta_x, theta_y


def transformation_matrix_A(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Build A matrix at each point.

    A has shape (..., 2, 2), with
        [u, v]^T = A [u1, u2]^T

    where (u,v) are velocity components in the orthonormal
    spherical basis (e_lambda, e_theta), and (u1,u2) are
    contravariant components in local patch coordinates.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lambda_, theta = local_xy_to_patch_angles(x, y, patch_id, tol=tol)
    cos_theta = np.cos(theta)

    lambda_x, lambda_y, theta_x, theta_y = local_angle_derivatives(
        x, y, patch_id, tol=tol
    )

    A = np.empty(x.shape + (2, 2), dtype=float)
    A[..., 0, 0] = radius * cos_theta * lambda_x
    A[..., 0, 1] = radius * cos_theta * lambda_y
    A[..., 1, 0] = radius * theta_x
    A[..., 1, 1] = radius * theta_y
    return A


def metric_tensor_G(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Compute G = A^T A.

    Returns
    -------
    G:
        Shape (..., 2, 2).
    """
    A = transformation_matrix_A(x, y, patch_id, radius=radius, tol=tol)
    return np.einsum("...ki,...kj->...ij", A, A)


def sqrtG(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Compute sqrt(det(G)) = abs(det(A)).

    For the current equal-area mapping, this should be constant:
        sqrtG = pi * radius**2
    """
    A = transformation_matrix_A(x, y, patch_id, radius=radius, tol=tol)
    return np.abs(np.linalg.det(A))


def expected_sqrtG(*, radius: float = 1.0) -> float:
    """
    Expected constant area Jacobian for the current equal-area map.
    """
    return np.pi * radius**2


def contravariant_velocity(
    u,
    v,
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical velocity components (u,v) to local contravariant components (u1,u2).

    Mathematical relation:
        [u, v]^T = A [u1, u2]^T
        [u1, u2]^T = A^{-1} [u, v]^T

    Parameters
    ----------
    u, v:
        Velocity components in spherical orthonormal basis (e_lambda, e_theta).
    x, y:
        Patch-local coordinates.
    patch_id:
        1,...,8.

    Returns
    -------
    u1, u2:
        Contravariant components for local conservative form.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    A = transformation_matrix_A(x, y, patch_id, radius=radius, tol=tol)
    rhs = np.stack([u, v], axis=-1)

    sol = np.linalg.solve(A, rhs[..., None])[..., 0]
    return sol[..., 0], sol[..., 1]