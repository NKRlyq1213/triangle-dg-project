from __future__ import annotations

import numpy as np

from geometry.sphere_patches import get_patch, assert_patch_local_xy


def _safe_arcsin(value: np.ndarray, *, tol: float = 1e-14) -> np.ndarray:
    """
    Numerically safe arcsin.

    中文：
    因為浮點誤差可能讓 value 稍微超出 [-1,1]，
    例如 1 + 1e-16，所以先 clip。
    """
    value = np.asarray(value, dtype=float)
    return np.arcsin(np.clip(value, -1.0 - tol, 1.0 + tol).clip(-1.0, 1.0))


def local_xy_to_patch_angles(
    x,
    y,
    patch_id: int,
    *,
    pole_lambda: float | None = None,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map patch-local triangle coordinates to spherical angles.

    Local coordinate convention:
        x >= 0, y >= 0, x + y <= 1

    The upper patch uses:
        sin(theta) = 1 - (x + y)^2

    The lower patch uses:
        sin(theta) = (x + y)^2 - 1

    Longitude sector:
        patch 1,2: [0, pi/2]
        patch 3,4: [pi/2, pi]
        patch 5,6: [pi, 3pi/2]
        patch 7,8: [3pi/2, 2pi]

    Parameters
    ----------
    x, y:
        Patch-local coordinates, normalized by R.
    patch_id:
        Integer in {1,...,8}.
    pole_lambda:
        Longitude assigned when x + y == 0.
        If None, use the sector center.
    tol:
        Numerical tolerance.

    Returns
    -------
    lambda_, theta:
        Spherical longitude and latitude.
    """
    p = get_patch(patch_id)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    assert_patch_local_xy(x, y, tol=tol)

    rho = x + y

    # sector base: 0, pi/2, pi, 3pi/2
    base = p.sector * (np.pi / 2.0)

    # In local coordinates, the within-sector fraction is y/(x+y).
    # At the pole rho=0, longitude is not unique; choose sector center.
    if pole_lambda is None:
        pole_lambda = base + np.pi / 4.0

    frac = np.empty_like(rho)
    nonpole = rho > tol
    frac[nonpole] = y[nonpole] / rho[nonpole]
    frac[~nonpole] = 0.5

    lambda_ = base + (np.pi / 2.0) * frac

    if p.upper:
        sin_theta = 1.0 - rho**2
    else:
        sin_theta = rho**2 - 1.0

    theta = _safe_arcsin(sin_theta)

    # Explicitly set pole longitude.
    lambda_ = np.where(nonpole, lambda_, pole_lambda)

    # Keep longitude inside [0, 2pi).  This avoids plotting discontinuities later.
    lambda_ = np.mod(lambda_, 2.0 * np.pi)

    return lambda_, theta


def patch_angles_to_sphere_xyz(
    lambda_,
    theta,
    *,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical angles to 3D Cartesian coordinates.

    X = R cos(lambda) cos(theta)
    Y = R sin(lambda) cos(theta)
    Z = R sin(theta)
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if lambda_.shape != theta.shape:
        raise ValueError("lambda_ and theta must have the same shape.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    cos_theta = np.cos(theta)
    X = radius * np.cos(lambda_) * cos_theta
    Y = radius * np.sin(lambda_) * cos_theta
    Z = radius * np.sin(theta)
    return X, Y, Z


def local_xy_to_sphere_xyz(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct map:
        local triangle (x,y) -> sphere Cartesian coordinates.

    This is the main function used by visualization and tests.
    """
    lambda_, theta = local_xy_to_patch_angles(x, y, patch_id, tol=tol)
    return patch_angles_to_sphere_xyz(lambda_, theta, radius=radius)


def local_xy_to_octahedron_xyz(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map patch-local coordinates to the corresponding octahedron face.

    Local convention:
        x,y are normalized coordinates in [0,1].
        Physical octahedron coordinates are scaled by radius.

    For upper patches:
        z_local = 1 - x - y

    For lower patches:
        z sign is handled by patch metadata.
    """
    p = get_patch(patch_id)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert_patch_local_xy(x, y, tol=tol)

    z = 1.0 - x - y

    X = radius * p.sx * x
    Y = radius * p.sy * y
    Z = radius * p.sz * z
    return X, Y, Z

from geometry.sphere_patches import assert_patch_local_xy


def local_xy_to_square_xy(
    x,
    y,
    patch_id: int,
    *,
    radius: float = 1.0,
    tol: float = 1e-14,
):
    """
    Map normalized patch-local triangle coordinates (x,y) to the 2D square [-R,R]^2.

    Local triangle:
        x >= 0, y >= 0, x + y <= 1

    Returns
    -------
    X2, Y2:
        2D square coordinates of the corresponding Ti region.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    assert_patch_local_xy(x, y, tol=tol)

    if patch_id == 1:
        X2 = radius * x
        Y2 = radius * y
    elif patch_id == 2:
        X2 = radius * (1.0 - x)
        Y2 = radius * (1.0 - y)
    elif patch_id == 3:
        X2 = -radius * x
        Y2 =  radius * y
    elif patch_id == 4:
        X2 = -radius * (1.0 - x)
        Y2 =  radius * (1.0 - y)
    elif patch_id == 5:
        X2 = -radius * x
        Y2 = -radius * y
    elif patch_id == 6:
        X2 = -radius * (1.0 - x)
        Y2 = -radius * (1.0 - y)
    elif patch_id == 7:
        X2 =  radius * x
        Y2 = -radius * y
    elif patch_id == 8:
        X2 =  radius * (1.0 - x)
        Y2 = -radius * (1.0 - y)
    else:
        raise ValueError("patch_id must be one of 1,...,8.")

    return X2, Y2