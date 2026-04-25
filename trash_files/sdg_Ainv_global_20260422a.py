from __future__ import annotations

import numpy as np


def _check_face_id(face_id: int) -> None:
    if int(face_id) not in range(1, 9):
        raise ValueError("face_id must be one of 1,...,8.")


def _as_arrays(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    return x, y


def sdg_lambda_theta_20260422a(
    x,
    y,
    face_id: int,
    *,
    pole_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SDG4PDEOnSphere20260422a global flattened mapping.

    Input:
        x, y:
            global flattened coordinates on [-1,1]^2.
        face_id:
            T1,...,T8.

    Output:
        lambda, theta.

    Pole treatment:
        intentionally not handled.
        If the formula denominator is too close to zero, this raises ValueError.

    Notes:
        T7 and T8 use the negative longitude convention from the PDF:
            lambda in [-pi/2, 0].
    """
    _check_face_id(face_id)
    x, y = _as_arrays(x, y)

    pi = np.pi

    if face_id == 1:
        rho = x + y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T1 pole singularity: x+y=0.")
        lam = 0.5 * pi * y / rho
        sin_theta = 1.0 - rho**2

    elif face_id == 2:
        rho = 2.0 - x - y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T2 pole singularity: 2-x-y=0.")
        lam = pi * (1.0 - x) / (2.0 * rho)
        sin_theta = rho**2 - 1.0

    elif face_id == 3:
        rho = -x + y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T3 pole singularity: -x+y=0.")
        lam = pi - 0.5 * pi * y / rho
        sin_theta = 1.0 - rho**2

    elif face_id == 4:
        rho = 2.0 + x - y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T4 pole singularity: 2+x-y=0.")
        lam = pi - pi * (1.0 + x) / (2.0 * rho)
        sin_theta = rho**2 - 1.0

    elif face_id == 5:
        rho = x + y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T5 pole singularity: x+y=0.")
        lam = pi + pi * y / (2.0 * rho)
        sin_theta = 1.0 - rho**2

    elif face_id == 6:
        rho = 2.0 + x + y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T6 pole singularity: 2+x+y=0.")
        lam = pi + pi * (1.0 + x) / (2.0 * rho)
        sin_theta = rho**2 - 1.0

    elif face_id == 7:
        rho = x - y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T7 pole singularity: x-y=0.")
        lam = -0.5 * pi * (-y / rho)
        sin_theta = 1.0 - rho**2

    elif face_id == 8:
        rho = 2.0 - x + y
        if np.any(np.abs(rho) <= pole_tol):
            raise ValueError("T8 pole singularity: 2-x+y=0.")
        lam = -pi * (1.0 - x) / (2.0 * rho)
        sin_theta = rho**2 - 1.0

    else:
        raise RuntimeError("unreachable")

    theta = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    return lam, theta


def _scale_factor(theta: np.ndarray, sign: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Common factor for the PDF formulas.

    For upper patches:
        D = 1 - sin(theta)

    For lower patches:
        D = 1 + sin(theta)

    A = R / (2 cos(theta) sqrt(D)) * M
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    if sign == -1:
        D = 1.0 - sin_t
    elif sign == +1:
        D = 1.0 + sin_t
    else:
        raise ValueError("sign must be -1 for 1-sin(theta), +1 for 1+sin(theta).")

    factor = radius / (2.0 * cos_t * np.sqrt(D))
    return factor, D, cos_t


def sdg_A_global_20260422a(
    x,
    y,
    face_id: int,
    *,
    radius: float = 1.0,
    pole_tol: float = 1e-14,
) -> np.ndarray:
    """
    Hard-coded A matrix from SDG4PDEOnSphere20260422a.pdf.

    A satisfies:
        [u, v]^T = A [u1, u2]^T

    where:
        (u, v) are spherical components in (e_lambda, e_theta),
        (u1, u2) are global flattened Cartesian components on [-1,1]^2.

    Shape:
        if x,y have shape S, return shape S + (2,2).
    """
    _check_face_id(face_id)
    x, y = _as_arrays(x, y)

    lam, theta = sdg_lambda_theta_20260422a(
        x,
        y,
        face_id,
        pole_tol=pole_tol,
    )

    pi = np.pi
    c2 = np.cos(theta) ** 2

    # Upper patches use 1 - sin(theta), lower patches use 1 + sin(theta).
    if face_id in (1, 3, 5, 7):
        factor, D, _ = _scale_factor(theta, sign=-1, radius=radius)
    else:
        factor, D, _ = _scale_factor(theta, sign=+1, radius=radius)

    M = np.empty(np.shape(lam) + (2, 2), dtype=float)

    if face_id == 1:
        M[..., 0, 0] = -2.0 * lam * c2
        M[..., 0, 1] = (pi - 2.0 * lam) * c2
        M[..., 1, 0] = -4.0 * D
        M[..., 1, 1] = -4.0 * D

    elif face_id == 2:
        M[..., 0, 0] = -(pi - 2.0 * lam) * c2
        M[..., 0, 1] = 2.0 * lam * c2
        M[..., 1, 0] = -4.0 * D
        M[..., 1, 1] = -4.0 * D

    elif face_id == 3:
        M[..., 0, 0] = -(2.0 * pi - 2.0 * lam) * c2
        M[..., 0, 1] = (pi - 2.0 * lam) * c2
        M[..., 1, 0] = 4.0 * D
        M[..., 1, 1] = -4.0 * D

    elif face_id == 4:
        M[..., 0, 0] = (pi - 2.0 * lam) * c2
        M[..., 0, 1] = -(2.0 * pi - 2.0 * lam) * c2
        M[..., 1, 0] = 4.0 * D
        M[..., 1, 1] = -4.0 * D

    elif face_id == 5:
        M[..., 0, 0] = -(2.0 * lam - 2.0 * pi) * c2
        M[..., 0, 1] = (3.0 * pi - 2.0 * lam) * c2
        M[..., 1, 0] = -4.0 * D
        M[..., 1, 1] = -4.0 * D

    elif face_id == 6:
        M[..., 0, 0] = (3.0 * pi - 2.0 * lam) * c2
        M[..., 0, 1] = -(2.0 * lam - 2.0 * pi) * c2
        M[..., 1, 0] = 4.0 * D
        M[..., 1, 1] = 4.0 * D

    elif face_id == 7:
        M[..., 0, 0] = -2.0 * lam * c2
        M[..., 0, 1] = (pi + 2.0 * lam) * c2
        M[..., 1, 0] = -4.0 * D
        M[..., 1, 1] = 4.0 * D

    elif face_id == 8:
        M[..., 0, 0] = (pi + 2.0 * lam) * c2
        M[..., 0, 1] = -2.0 * lam * c2
        M[..., 1, 0] = -4.0 * D
        M[..., 1, 1] = 4.0 * D

    else:
        raise RuntimeError("unreachable")

    return factor[..., None, None] * M


def sdg_Ainv_global_20260422a(
    x,
    y,
    face_id: int,
    *,
    radius: float = 1.0,
    pole_tol: float = 1e-14,
) -> np.ndarray:
    """
    Hard-coded A^{-1} based on the PDF result det(A)=pi R^2.

    No numerical inversion is used.

    If:
        A = [[a, b],
             [c, d]]

    then:
        A^{-1} = 1/(pi R^2) [[ d, -b],
                              [-c,  a]]
    """
    A = sdg_A_global_20260422a(
        x,
        y,
        face_id,
        radius=radius,
        pole_tol=pole_tol,
    )

    detA = np.pi * radius**2

    Ainv = np.empty_like(A)
    Ainv[..., 0, 0] = A[..., 1, 1] / detA
    Ainv[..., 0, 1] = -A[..., 0, 1] / detA
    Ainv[..., 1, 0] = -A[..., 1, 0] / detA
    Ainv[..., 1, 1] = A[..., 0, 0] / detA

    return Ainv


def sdg_sqrtG_global_20260422a(
    x,
    y,
    face_id: int,
    *,
    radius: float = 1.0,
    pole_tol: float = 1e-14,
) -> np.ndarray:
    """
    sqrt(G)=pi R^2 for all T1,...,T8 according to the PDF.

    This function still evaluates A once to catch pole singularities.
    """
    x, y = _as_arrays(x, y)

    _ = sdg_A_global_20260422a(
        x,
        y,
        face_id,
        radius=radius,
        pole_tol=pole_tol,
    )

    return np.full_like(x, np.pi * radius**2, dtype=float)


def sdg_A_Ainv_batch_20260422a(
    X: np.ndarray,
    Y: np.ndarray,
    face_ids: np.ndarray,
    *,
    radius: float = 1.0,
    pole_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch construction for element-node arrays.

    Parameters
    ----------
    X, Y:
        shape (K, Np), global flattened coordinates.
    face_ids:
        shape (K,), values 1,...,8.

    Returns
    -------
    A, Ainv:
        shape (K, Np, 2, 2)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    face_ids = np.asarray(face_ids, dtype=int)

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    if X.ndim != 2:
        raise ValueError("X and Y must have shape (K,Np).")
    if face_ids.shape != (X.shape[0],):
        raise ValueError("face_ids must have shape (K,).")

    K, Np = X.shape

    A = np.zeros((K, Np, 2, 2), dtype=float)
    Ainv = np.zeros((K, Np, 2, 2), dtype=float)

    for fid in range(1, 9):
        mask = face_ids == fid
        if not np.any(mask):
            continue

        A[mask] = sdg_A_global_20260422a(
            X[mask],
            Y[mask],
            fid,
            radius=radius,
            pole_tol=pole_tol,
        )

        Ainv[mask] = sdg_Ainv_global_20260422a(
            X[mask],
            Y[mask],
            fid,
            radius=radius,
            pole_tol=pole_tol,
        )

    return A, Ainv