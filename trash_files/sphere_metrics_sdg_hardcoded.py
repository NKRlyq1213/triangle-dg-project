from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SDGPatchFormula:
    """
    Hard-coded SDG formula descriptor.

    We write each patch formula in the form

        rho = a0 + ax*x + ay*y
        eta = b0 + bx*x + by*y
        lambda = lambda0 + sigma * (pi/2) * eta/rho

    where (x,y) are GLOBAL square coordinates.

    For upper patches:
        sin(theta) = 1 - rho^2

    For lower patches:
        sin(theta) = rho^2 - 1
    """

    patch_id: int
    rho_coeff: tuple[float, float, float]      # a0, ax, ay
    eta_coeff: tuple[float, float, float]      # b0, bx, by
    lambda0: float
    sigma: float
    upper: bool


# ---------------------------------------------------------------------
# SDG global-coordinate patch formulas.
#
# Global square:
#     (x,y) in [-1,1]^2
#
# Patch layout:
#     T1,T3,T5,T7 : upper hemisphere
#     T2,T4,T6,T8 : lower hemisphere
#
# For T7,T8 we use the SDG-style negative longitude sector:
#     T7,T8 use lambda in [-pi/2, 0]
# rather than [3pi/2, 2pi].
# This is equivalent for sin/cos velocity evaluation.
# ---------------------------------------------------------------------

_SDG_PATCH_FORMULAS: dict[int, SDGPatchFormula] = {
    # T1: x>=0, y>=0, x+y<=1
    # lambda = (pi/2) y/(x+y), sin(theta)=1-(x+y)^2
    1: SDGPatchFormula(
        patch_id=1,
        rho_coeff=(0.0, +1.0, +1.0),
        eta_coeff=(0.0,  0.0, +1.0),
        lambda0=0.0,
        sigma=+1.0,
        upper=True,
    ),

    # T2: first quadrant outer triangle
    # lambda = pi(1-x)/(2(2-x-y)), sin(theta)=(2-x-y)^2-1
    2: SDGPatchFormula(
        patch_id=2,
        rho_coeff=(2.0, -1.0, -1.0),
        eta_coeff=(1.0, -1.0,  0.0),
        lambda0=0.0,
        sigma=+1.0,
        upper=False,
    ),

    # T3: second quadrant inner triangle
    # lambda = pi/2 + (pi/2)(-x)/(-x+y)
    3: SDGPatchFormula(
        patch_id=3,
        rho_coeff=(0.0, -1.0, +1.0),
        eta_coeff=(0.0, -1.0,  0.0),
        lambda0=0.5 * np.pi,
        sigma=+1.0,
        upper=True,
    ),

    # T4: second quadrant outer triangle
    # lambda = pi - pi(1+x)/(2(2+x-y))
    4: SDGPatchFormula(
        patch_id=4,
        rho_coeff=(2.0, +1.0, -1.0),
        eta_coeff=(1.0, +1.0,  0.0),
        lambda0=np.pi,
        sigma=-1.0,
        upper=False,
    ),

    # T5: third quadrant inner triangle
    # lambda = pi + (pi/2)(-y)/(-x-y)
    5: SDGPatchFormula(
        patch_id=5,
        rho_coeff=(0.0, -1.0, -1.0),
        eta_coeff=(0.0,  0.0, -1.0),
        lambda0=np.pi,
        sigma=+1.0,
        upper=True,
    ),

    # T6: third quadrant outer triangle
    # lambda = pi + pi(1+x)/(2(2+x+y))
    6: SDGPatchFormula(
        patch_id=6,
        rho_coeff=(2.0, +1.0, +1.0),
        eta_coeff=(1.0, +1.0,  0.0),
        lambda0=np.pi,
        sigma=+1.0,
        upper=False,
    ),

    # T7: fourth quadrant inner triangle
    # SDG-style negative sector:
    # lambda = -pi/2 + (pi/2) x/(x-y)
    7: SDGPatchFormula(
        patch_id=7,
        rho_coeff=(0.0, +1.0, -1.0),
        eta_coeff=(0.0, +1.0,  0.0),
        lambda0=-0.5 * np.pi,
        sigma=+1.0,
        upper=True,
    ),

    # T8: fourth quadrant outer triangle
    # lambda = -pi(1-x)/(2(2-x+y)), sin(theta)=(2-x+y)^2-1
    8: SDGPatchFormula(
        patch_id=8,
        rho_coeff=(2.0, -1.0, +1.0),
        eta_coeff=(1.0, -1.0,  0.0),
        lambda0=0.0,
        sigma=-1.0,
        upper=False,
    ),
}


# ---------------------------------------------------------------------
# Local-to-global affine maps.
#
# The local triangle coordinate is:
#     xi >= 0, eta >= 0, xi+eta <= 1
#
# and physical/global patch coordinates are:
#     [xg, yg]^T = v0 + B [xi, eta]^T
#
# B = d(xg,yg)/d(xi,eta)
# ---------------------------------------------------------------------

_UNIT_PATCH_VERTICES: dict[int, np.ndarray] = {
    1: np.array([[0.0,  0.0], [1.0,  0.0], [0.0,  1.0]], dtype=float),
    2: np.array([[1.0,  1.0], [1.0,  0.0], [0.0,  1.0]], dtype=float),
    3: np.array([[0.0,  0.0], [0.0,  1.0], [-1.0, 0.0]], dtype=float),
    4: np.array([[-1.0, 1.0], [0.0,  1.0], [-1.0, 0.0]], dtype=float),
    5: np.array([[0.0,  0.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float),
    6: np.array([[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float),
    7: np.array([[0.0,  0.0], [0.0, -1.0], [1.0,  0.0]], dtype=float),
    8: np.array([[1.0, -1.0], [0.0, -1.0], [1.0,  0.0]], dtype=float),
}


def _get_sdg_patch_formula(patch_id: int) -> SDGPatchFormula:
    if patch_id not in _SDG_PATCH_FORMULAS:
        raise ValueError("patch_id must be one of 1,...,8.")
    return _SDG_PATCH_FORMULAS[patch_id]


def _local_to_global_xy(
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert local patch coordinates to global square coordinates.

    Input:
        xi, eta in local triangle.

    Output:
        xg, yg in global square [-R,R]^2.
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if xi.shape != eta.shape:
        raise ValueError("xi and eta must have the same shape.")

    verts = radius * _UNIT_PATCH_VERTICES[patch_id]
    v0 = verts[0]
    b1 = verts[1] - verts[0]
    b2 = verts[2] - verts[0]

    xg = v0[0] + b1[0] * xi + b2[0] * eta
    yg = v0[1] + b1[1] * xi + b2[1] * eta

    return xg, yg


def _local_to_global_B(
    patch_id: int,
    *,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Return B = d(xg,yg)/d(xi,eta), shape (2,2).

    Columns:
        column 0 = d(xg,yg)/d xi
        column 1 = d(xg,yg)/d eta
    """
    verts = radius * _UNIT_PATCH_VERTICES[patch_id]
    b1 = verts[1] - verts[0]
    b2 = verts[2] - verts[0]
    return np.column_stack([b1, b2])


def _safe_sqrt(value, *, tol: float = 1e-14):
    value = np.asarray(value, dtype=float)
    return np.sqrt(np.maximum(value, 0.0))


def _safe_cos_theta_from_sin(sin_theta: np.ndarray) -> np.ndarray:
    """
    cos(theta) for theta in [-pi/2, pi/2].
    """
    return _safe_sqrt(1.0 - sin_theta**2)


def sdg_lambda_theta_global(
    xg,
    yg,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard-coded SDG lambda/theta from global square coordinates.

    The formula is written using normalized coordinates X=xg/R, Y=yg/R.
    """
    f = _get_sdg_patch_formula(patch_id)

    X = np.asarray(xg, dtype=float) / radius
    Y = np.asarray(yg, dtype=float) / radius

    a0, ax, ay = f.rho_coeff
    b0, bx, by = f.eta_coeff

    rho = a0 + ax * X + ay * Y
    eta = b0 + bx * X + by * Y

    if np.any(rho <= tol):
        raise ValueError(
            "rho is too close to zero. This is the pole singularity of the SDG coordinates."
        )

    lam = f.lambda0 + f.sigma * 0.5 * np.pi * eta / rho

    if f.upper:
        sin_theta = 1.0 - rho**2
    else:
        sin_theta = rho**2 - 1.0

    theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    return lam, theta


def transformation_matrix_A_sdg_global(
    xg,
    yg,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Hard-coded SDG A matrix with respect to GLOBAL square coordinates.

    A_global satisfies:

        [u, v]^T = A_global [u_global^1, u_global^2]^T

    where u_global^1,u_global^2 are contravariant components with
    respect to global square coordinates (xg,yg).
    """
    f = _get_sdg_patch_formula(patch_id)

    X = np.asarray(xg, dtype=float) / radius
    Y = np.asarray(yg, dtype=float) / radius

    if X.shape != Y.shape:
        raise ValueError("xg and yg must have the same shape.")

    a0, ax, ay = f.rho_coeff
    b0, bx, by = f.eta_coeff

    rho = a0 + ax * X + ay * Y
    eta = b0 + bx * X + by * Y

    if np.any(rho <= tol):
        raise ValueError(
            "rho is too close to zero. A is singular at the pole in these coordinates."
        )

    # lambda = lambda0 + sigma*pi/2 * eta/rho
    # Derivatives below are with respect to normalized X,Y.
    coeff = f.sigma * 0.5 * np.pi
    lam_X = coeff * (bx * rho - eta * ax) / (rho**2)
    lam_Y = coeff * (by * rho - eta * ay) / (rho**2)

    if f.upper:
        sin_theta = 1.0 - rho**2
        # dS/dX = -2 rho rho_X
        theta_X = (-2.0 * rho * ax) / _safe_cos_theta_from_sin(sin_theta)
        theta_Y = (-2.0 * rho * ay) / _safe_cos_theta_from_sin(sin_theta)
    else:
        sin_theta = rho**2 - 1.0
        # dS/dX = +2 rho rho_X
        theta_X = (2.0 * rho * ax) / _safe_cos_theta_from_sin(sin_theta)
        theta_Y = (2.0 * rho * ay) / _safe_cos_theta_from_sin(sin_theta)

    cos_theta = _safe_cos_theta_from_sin(sin_theta)

    # Chain rule:
    # X = xg/R, Y = yg/R
    # lambda_xg = lambda_X / R
    # theta_xg  = theta_X  / R
    #
    # A entries:
    # R cos(theta) lambda_xg = cos(theta) lambda_X
    # R theta_xg             = theta_X
    #
    # Hence A_global is independent of R in entries, but determinant is
    # dimensionless in normalized coordinates. To match SDG's physical
    # coordinate convention with sphere radius R, use physical xg,yg but
    # retain [u,v] in physical tangent velocity. The formula above gives
    # the correct velocity conversion for xg,yg coordinates.
    A = np.empty(X.shape + (2, 2), dtype=float)

    A[..., 0, 0] = cos_theta * lam_X
    A[..., 0, 1] = cos_theta * lam_Y
    A[..., 1, 0] = theta_X
    A[..., 1, 1] = theta_Y

    return A


def transformation_matrix_A_sdg_local(
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Hard-coded SDG A matrix converted to LOCAL patch coordinates.

    This is the matrix your current solver should use.

    Chain rule:
        [xg, yg]^T = B [xi, eta]^T + v0

    Therefore:
        A_local = A_global @ B

    and

        [u, v]^T = A_local [u_local^1, u_local^2]^T.
    """
    xg, yg = _local_to_global_xy(
        xi,
        eta,
        patch_id,
        radius=radius,
    )

    A_global = transformation_matrix_A_sdg_global(
        xg,
        yg,
        patch_id,
        radius=radius,
        tol=tol,
    )

    B = _local_to_global_B(
        patch_id,
        radius=radius,
    )

    # A_local[..., i, j] = sum_k A_global[..., i, k] B[k, j]
    A_local = np.einsum("...ik,kj->...ij", A_global, B)
    return A_local


def metric_tensor_G_sdg_local(
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    G = A^T A using hard-coded SDG A in local coordinates.
    """
    A = transformation_matrix_A_sdg_local(
        xi,
        eta,
        patch_id,
        radius=radius,
        tol=tol,
    )
    return np.einsum("...ki,...kj->...ij", A, A)


def sqrtG_sdg_local(
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    sqrt(det(G)) = abs(det(A_local)).

    Expected:
        sqrtG = pi * R^2
    up to orientation and coordinate scaling.
    """
    A = transformation_matrix_A_sdg_local(
        xi,
        eta,
        patch_id,
        radius=radius,
        tol=tol,
    )
    return np.abs(np.linalg.det(A))


def expected_sqrtG_sdg(*, radius: float = 1.0) -> float:
    return np.pi * radius**2


def contravariant_velocity_sdg_local(
    u_sph,
    v_sph,
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical velocity components to local contravariant components:

        [u1,u2]^T = A_local^{-1} [u,v]^T
    """
    A = transformation_matrix_A_sdg_local(
        xi,
        eta,
        patch_id,
        radius=radius,
        tol=tol,
    )

    rhs = np.stack(
        [
            np.asarray(u_sph, dtype=float),
            np.asarray(v_sph, dtype=float),
        ],
        axis=-1,
    )

    sol = np.linalg.solve(A, rhs[..., None])[..., 0]
    return sol[..., 0], sol[..., 1]


def reconstruct_spherical_velocity_sdg_local(
    u1,
    u2,
    xi,
    eta,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check:
        [u,v]^T = A_local [u1,u2]^T
    """
    A = transformation_matrix_A_sdg_local(
        xi,
        eta,
        patch_id,
        radius=radius,
        tol=tol,
    )

    uv_local = np.stack(
        [
            np.asarray(u1, dtype=float),
            np.asarray(u2, dtype=float),
        ],
        axis=-1,
    )

    uv = np.einsum("...ij,...j->...i", A, uv_local)
    return uv[..., 0], uv[..., 1]