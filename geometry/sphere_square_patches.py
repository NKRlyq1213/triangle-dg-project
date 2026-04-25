from __future__ import annotations

import numpy as np


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def patch_id_from_xy(
    x,
    y,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> np.ndarray:
    """
    Classify flattened-square points into Mapping.pdf patches T1,...,T8.

    Patch convention
    ----------------
    F1: X>=0, Y>=0, Z>=0
    F2: X>=0, Y>=0, Z<=0
    F3: X<=0, Y>=0, Z>=0
    F4: X<=0, Y>=0, Z<=0
    F5: X<=0, Y<=0, Z>=0
    F6: X<=0, Y<=0, Z<=0
    F7: X>=0, Y<=0, Z>=0
    F8: X>=0, Y<=0, Z<=0

    Notes
    -----
    - This classifier is for geometric diagnostics.
    - In element-based computations, prefer elem_patch_id from the mesh generator.
      Do not reclassify nodes lying exactly on seams.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    ax = np.abs(x)
    ay = np.abs(y)
    top = (ax + ay) <= (R + tol)

    pid = np.zeros_like(x, dtype=int)

    q1 = (x >= -tol) & (y >= -tol)
    q2 = (x <=  tol) & (y >= -tol)
    q3 = (x <=  tol) & (y <=  tol)
    q4 = (x >= -tol) & (y <=  tol)

    pid[q1 & top] = 1
    pid[q1 & ~top] = 2

    pid[q2 & top] = 3
    pid[q2 & ~top] = 4

    pid[q3 & top] = 5
    pid[q3 & ~top] = 6

    pid[q4 & top] = 7
    pid[q4 & ~top] = 8

    if np.any(pid == 0):
        raise ValueError("Some points could not be classified into patches.")

    return pid


def _patch_parameters(patch_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        quadrant_id in {1,2,3,4}
        sx = d|x|/dx
        sy = d|y|/dy
        is_top
    """
    pid = np.asarray(patch_id, dtype=int)

    quadrant = np.zeros_like(pid, dtype=int)
    sx = np.zeros_like(pid, dtype=float)
    sy = np.zeros_like(pid, dtype=float)
    is_top = (pid % 2) == 1

    q1 = np.isin(pid, [1, 2])
    q2 = np.isin(pid, [3, 4])
    q3 = np.isin(pid, [5, 6])
    q4 = np.isin(pid, [7, 8])

    quadrant[q1] = 1
    quadrant[q2] = 2
    quadrant[q3] = 3
    quadrant[q4] = 4

    sx[q1 | q4] = 1.0
    sx[q2 | q3] = -1.0

    sy[q1 | q2] = 1.0
    sy[q3 | q4] = -1.0

    if np.any(quadrant == 0):
        raise ValueError("patch_id must contain only values 1,...,8.")

    return quadrant, sx, sy, is_top


def lambda_theta_from_xy_patch(
    x,
    y,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map flattened square coordinates to spherical longitude/latitude.

    Parameters
    ----------
    x, y : array_like
        Global flattened Cartesian coordinates on [-R,R]^2.
    patch_id : array_like
        Patch ids in {1,...,8}. Must be element-based for seam nodes.

    Returns
    -------
    lambda_ : np.ndarray
        Longitude. Values in [0, 2*pi].
    theta : np.ndarray
        Latitude.
    pole_mask : np.ndarray
        True where the raw formula is singular or near singular.

    Notes
    -----
    Pole values are not regularized here.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)
    pid = np.asarray(patch_id, dtype=int)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if pid.shape != x.shape:
        if pid.size == 1:
            pid = np.full_like(x, int(pid), dtype=int)
        else:
            raise ValueError("patch_id must be scalar or have the same shape as x.")

    quadrant, _, _, is_top = _patch_parameters(pid)

    p = np.abs(x)
    q = np.abs(y)
    denom = p + q
    rho = denom

    lambda_ = np.full_like(x, np.nan, dtype=float)
    c = 0.5 * np.pi

    good = denom > tol

    # Q1: [0, pi/2], fraction q/(p+q)
    mask = good & (quadrant == 1)
    lambda_[mask] = 0.0 + c * q[mask] / denom[mask]

    # Q2: [pi/2, pi], fraction p/(p+q)
    mask = good & (quadrant == 2)
    lambda_[mask] = 0.5 * np.pi + c * p[mask] / denom[mask]

    # Q3: [pi, 3pi/2], fraction q/(p+q)
    mask = good & (quadrant == 3)
    lambda_[mask] = np.pi + c * q[mask] / denom[mask]

    # Q4: [3pi/2, 2pi], fraction p/(p+q)
    mask = good & (quadrant == 4)
    lambda_[mask] = 1.5 * np.pi + c * p[mask] / denom[mask]

    sin_theta = np.full_like(x, np.nan, dtype=float)
    top_mask = is_top
    bot_mask = ~is_top

    # Top octants: sin(theta) = 1 - rho^2.
    sin_theta[top_mask] = 1.0 - rho[top_mask] ** 2

    # Bottom octants: sin(theta) = (2-rho)^2 - 1.
    sin_theta[bot_mask] = (2.0 - rho[bot_mask]) ** 2 - 1.0

    sin_theta_clip = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta_clip)

    cos_theta = np.cos(theta)
    pole_mask = (~np.isfinite(lambda_)) | (~np.isfinite(theta)) | (np.abs(cos_theta) <= 10.0 * tol)

    return lambda_, theta, pole_mask


def sphere_xyz_from_lambda_theta(
    lambda_,
    theta,
    R: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spherical coordinates to 3D Cartesian coordinates.
    """
    lambda_ = _as_float_array(lambda_)
    theta = _as_float_array(theta)

    X = R * np.cos(lambda_) * np.cos(theta)
    Y = R * np.sin(lambda_) * np.cos(theta)
    Z = R * np.sin(theta)
    return X, Y, Z


def sphere_xyz_from_xy_patch(
    x,
    y,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper:
        (x,y,patch_id) -> (lambda,theta,X,Y,Z,pole_mask)
    """
    lambda_, theta, pole_mask = lambda_theta_from_xy_patch(
        x=x,
        y=y,
        patch_id=patch_id,
        R=R,
        tol=tol,
    )
    X, Y, Z = sphere_xyz_from_lambda_theta(lambda_, theta, R=R)
    return lambda_, theta, X, Y, Z, pole_mask


def A_matrix_from_xy_patch(
    x,
    y,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute SDG-style A matrix:

        A = [[cos(theta) * lambda_x, cos(theta) * lambda_y],
             [theta_x,               theta_y]]

    where derivatives are with respect to global flattened Cartesian
    coordinates (x,y).

    This uses the same analytic mapping formulas used in Mapping.pdf / SDG,
    with chain rule through local quadrant coordinates p=|x|, q=|y|.

    Returns
    -------
    A : np.ndarray
        Shape x.shape + (2,2)
    bad_mask : np.ndarray
        True where the raw formula is singular / undefined.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)
    pid = np.asarray(patch_id, dtype=int)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if pid.shape != x.shape:
        if pid.size == 1:
            pid = np.full_like(x, int(pid), dtype=int)
        else:
            raise ValueError("patch_id must be scalar or have the same shape as x.")

    lambda_, theta, pole_mask = lambda_theta_from_xy_patch(
        x=x,
        y=y,
        patch_id=pid,
        R=R,
        tol=tol,
    )

    quadrant, sx, sy, is_top = _patch_parameters(pid)

    p = np.abs(x)
    q = np.abs(y)
    denom = p + q
    rho = denom

    c = 0.5 * np.pi

    dl_dp = np.full_like(x, np.nan, dtype=float)
    dl_dq = np.full_like(x, np.nan, dtype=float)

    good = denom > tol

    # Q1/Q3: lambda = base + c*q/(p+q)
    mask = good & ((quadrant == 1) | (quadrant == 3))
    dl_dp[mask] = -c * q[mask] / denom[mask] ** 2
    dl_dq[mask] =  c * p[mask] / denom[mask] ** 2

    # Q2/Q4: lambda = base + c*p/(p+q)
    mask = good & ((quadrant == 2) | (quadrant == 4))
    dl_dp[mask] =  c * q[mask] / denom[mask] ** 2
    dl_dq[mask] = -c * p[mask] / denom[mask] ** 2

    dl_dx = dl_dp * sx
    dl_dy = dl_dq * sy

    cos_theta = np.cos(theta)

    dS_drho = np.full_like(x, np.nan, dtype=float)
    dS_drho[is_top] = -2.0 * rho[is_top]
    dS_drho[~is_top] = -2.0 * (2.0 - rho[~is_top])

    with np.errstate(divide="ignore", invalid="ignore"):
        dtheta_drho = dS_drho / cos_theta

    dtheta_dx = dtheta_drho * sx
    dtheta_dy = dtheta_drho * sy

    A = np.empty(x.shape + (2, 2), dtype=float)
    A[..., 0, 0] = R * cos_theta * dl_dx
    A[..., 0, 1] = R * cos_theta * dl_dy
    A[..., 1, 0] = R * dtheta_dx
    A[..., 1, 1] = R * dtheta_dy

    bad_mask = pole_mask | (~np.all(np.isfinite(A), axis=(-1, -2)))

    return A, bad_mask


def Ainv_from_xy_patch(
    x,
    y,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw A^{-1} from SDG-style A.

    Pole limit values are intentionally not applied here.
    """
    A, bad_mask = A_matrix_from_xy_patch(
        x=x,
        y=y,
        patch_id=patch_id,
        R=R,
        tol=tol,
    )

    Ainv = np.full_like(A, np.nan, dtype=float)
    flat_A = A.reshape((-1, 2, 2))
    flat_bad = bad_mask.reshape(-1)
    flat_Ainv = Ainv.reshape((-1, 2, 2))

    for i in range(flat_A.shape[0]):
        if flat_bad[i]:
            continue
        try:
            flat_Ainv[i] = np.linalg.inv(flat_A[i])
        except np.linalg.LinAlgError:
            flat_bad[i] = True

    bad_mask = flat_bad.reshape(bad_mask.shape)
    Ainv = flat_Ainv.reshape(A.shape)
    return Ainv, bad_mask


def metric_sqrtG_from_A(A: np.ndarray) -> np.ndarray:
    """
    Compute sqrt(det(A^T A)).
    """
    A = np.asarray(A, dtype=float)
    if A.shape[-2:] != (2, 2):
        raise ValueError("A must have shape (...,2,2).")

    G = np.einsum("...ki,...kj->...ij", A, A)
    detG = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
    with np.errstate(invalid="ignore"):
        return np.sqrt(detG)


def A_Ainv_error(A: np.ndarray, Ainv: np.ndarray) -> np.ndarray:
    """
    Infinity norm of A @ Ainv - I at each node.
    """
    A = np.asarray(A, dtype=float)
    Ainv = np.asarray(Ainv, dtype=float)

    I = np.eye(2)
    prod = np.einsum("...ij,...jk->...ik", A, Ainv)
    err = np.max(np.abs(prod - I), axis=(-1, -2))
    return err

