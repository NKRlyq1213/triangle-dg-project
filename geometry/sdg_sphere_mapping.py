from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SDGMappingResult:
    lambda_: np.ndarray
    theta: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    Ainv: np.ndarray
    detA: np.ndarray
    sqrtG: np.ndarray
    pole_mask: np.ndarray
    bad_mask: np.ndarray
    A_Ainv_err: np.ndarray
    Ainv_numpy_err: np.ndarray


def sdg_detA_expected(R: float = 1.0) -> float:
    return float(np.pi * R * R)


def sdg_sqrtG_expected(R: float = 1.0) -> float:
    return float(np.pi * R * R)


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def sdg_lambda_theta_from_xy_patch(
    x,
    y,
    patch_id,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SDG explicit patch mapping:
        (x,y,patch_id) -> (lambda,theta)

    This function follows SDG4PDEOnSphere20260425a.pdf patch formulas T1--T8.
    It does not normalize lambda into [0, 2*pi]; T7/T8 keep the negative
    longitude convention used by SDG.

    Flat domain is fixed as [-1,1] x [-1,1].
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

    lambda_ = np.full_like(x, np.nan, dtype=float)
    sin_theta = np.full_like(x, np.nan, dtype=float)
    bad = np.zeros_like(x, dtype=bool)

    pi = np.pi

    # T1
    m = pid == 1
    d = x + y
    use = m & (np.abs(d) > tol)
    lambda_[use] = 0.5 * pi * y[use] / d[use]
    sin_theta[use] = 1.0 - d[use] ** 2
    bad[m & ~use] = True

    # T2
    m = pid == 2
    d = 2.0 - x - y
    use = m & (np.abs(d) > tol)
    lambda_[use] = pi * (1.0 - x[use]) / (2.0 * d[use])
    sin_theta[use] = d[use] ** 2 - 1.0
    bad[m & ~use] = True

    # T3
    m = pid == 3
    d = -x + y
    use = m & (np.abs(d) > tol)
    lambda_[use] = pi - 0.5 * pi * y[use] / d[use]
    sin_theta[use] = 1.0 - d[use] ** 2
    bad[m & ~use] = True

    # T4
    m = pid == 4
    d = 2.0 + x - y
    use = m & (np.abs(d) > tol)
    lambda_[use] = pi - pi * (1.0 + x[use]) / (2.0 * d[use])
    sin_theta[use] = d[use] ** 2 - 1.0
    bad[m & ~use] = True

    # T5
    m = pid == 5
    d = x + y
    use = m & (np.abs(d) > tol)
    lambda_[use] = pi + 0.5 * pi * y[use] / d[use]
    sin_theta[use] = 1.0 - d[use] ** 2
    bad[m & ~use] = True

    # T6
    m = pid == 6
    d = 2.0 + x + y
    use = m & (np.abs(d) > tol)
    lambda_[use] = pi + pi * (1.0 + x[use]) / (2.0 * d[use])
    sin_theta[use] = d[use] ** 2 - 1.0
    bad[m & ~use] = True

    # T7
    m = pid == 7
    d = x - y
    use = m & (np.abs(d) > tol)
    lambda_[use] = -0.5 * pi * ((-y[use]) / d[use])
    sin_theta[use] = 1.0 - d[use] ** 2
    bad[m & ~use] = True

    # T8
    m = pid == 8
    d = 2.0 - x + y
    use = m & (np.abs(d) > tol)
    lambda_[use] = -pi * (1.0 - x[use]) / (2.0 * d[use])
    sin_theta[use] = d[use] ** 2 - 1.0
    bad[m & ~use] = True

    invalid_patch = (pid < 1) | (pid > 8)
    if np.any(invalid_patch):
        raise ValueError("patch_id must contain only values 1,...,8.")

    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    bad |= ~np.isfinite(lambda_)
    bad |= ~np.isfinite(theta)

    return lambda_, theta, bad


def sdg_sphere_xyz_from_lambda_theta(lambda_, theta, R: float = 1.0):
    lambda_ = _as_float_array(lambda_)
    theta = _as_float_array(theta)

    X = R * np.cos(lambda_) * np.cos(theta)
    Y = R * np.sin(lambda_) * np.cos(theta)
    Z = R * np.sin(theta)
    return X, Y, Z


def _sdg_H(theta: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """
    H = 1 - sin(theta) for T1,T3,T5,T7
    H = 1 + sin(theta) for T2,T4,T6,T8
    """
    s = np.sin(theta)
    top = np.isin(pid, [1, 3, 5, 7])
    return np.where(top, 1.0 - s, 1.0 + s)


def sdg_A_from_lambda_theta_patch(
    lambda_,
    theta,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SDG explicit A matrix for T1--T8.

    A = R / (2 cos(theta) sqrt(H)) * M,

    where M is exactly the patch-wise matrix listed in SDG:
        first row entries contain cos(theta)^2;
        second row entries contain +/- 4H.
    """
    lam = _as_float_array(lambda_)
    theta = _as_float_array(theta)
    pid = np.asarray(patch_id, dtype=int)

    if lam.shape != theta.shape:
        raise ValueError("lambda_ and theta must have the same shape.")

    if pid.shape != lam.shape:
        if pid.size == 1:
            pid = np.full_like(lam, int(pid), dtype=int)
        else:
            raise ValueError("patch_id must be scalar or have the same shape as lambda_.")

    pi = np.pi
    c = np.cos(theta)
    c2 = c * c
    H = _sdg_H(theta, pid)

    A = np.full(lam.shape + (2, 2), np.nan, dtype=float)

    bad = np.zeros_like(lam, dtype=bool)
    bad |= (np.abs(c) <= tol)
    bad |= (H <= tol)
    bad |= ~np.isfinite(lam)
    bad |= ~np.isfinite(theta)

    with np.errstate(divide="ignore", invalid="ignore"):
        fac = R / (2.0 * c * np.sqrt(H))

    # T1
    m = pid == 1
    A[..., 0, 0] = np.where(m, fac * (-2.0 * lam) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (pi - 2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (-4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (-4.0 * H), A[..., 1, 1])

    # T2
    m = pid == 2
    A[..., 0, 0] = np.where(m, fac * (-(pi - 2.0 * lam)) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (-4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (-4.0 * H), A[..., 1, 1])

    # T3
    m = pid == 3
    A[..., 0, 0] = np.where(m, fac * (-(2.0 * pi - 2.0 * lam)) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (pi - 2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (-4.0 * H), A[..., 1, 1])

    # T4
    m = pid == 4
    A[..., 0, 0] = np.where(m, fac * (pi - 2.0 * lam) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (-(2.0 * pi - 2.0 * lam)) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (-4.0 * H), A[..., 1, 1])

    # T5
    m = pid == 5
    A[..., 0, 0] = np.where(m, fac * (-(2.0 * lam - 2.0 * pi)) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (3.0 * pi - 2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (-4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (-4.0 * H), A[..., 1, 1])

    # T6
    m = pid == 6
    A[..., 0, 0] = np.where(m, fac * (3.0 * pi - 2.0 * lam) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (-(2.0 * lam - 2.0 * pi)) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (4.0 * H), A[..., 1, 1])

    # T7
    m = pid == 7
    A[..., 0, 0] = np.where(m, fac * (-2.0 * lam) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (pi + 2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (-4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (4.0 * H), A[..., 1, 1])

    # T8
    m = pid == 8
    A[..., 0, 0] = np.where(m, fac * (pi + 2.0 * lam) * c2, A[..., 0, 0])
    A[..., 0, 1] = np.where(m, fac * (-2.0 * lam) * c2, A[..., 0, 1])
    A[..., 1, 0] = np.where(m, fac * (-4.0 * H), A[..., 1, 0])
    A[..., 1, 1] = np.where(m, fac * (4.0 * H), A[..., 1, 1])

    bad |= ~np.all(np.isfinite(A), axis=(-1, -2))

    return A, bad



def sdg_Ainv_T1_stable(lambda_, theta, R: float = 1.0) -> np.ndarray:
    r"""
    SDG pole-stable A^{-1} formula for T1.

    For T1:
        h = 1 - sin(theta)

    The original inverse contains:
        sqrt(h)/cos(theta),  cos(theta)/sqrt(h)

    Near the north pole this is 0/0. Using

        cos(theta)^2 = (1-sin(theta))(1+sin(theta))

    and theta in [0, pi/2), we obtain the stable equivalent:

        sqrt(1-sin(theta))/cos(theta) = 1/sqrt(1+sin(theta))
        cos(theta)/sqrt(1-sin(theta)) = sqrt(1+sin(theta))

    Therefore,

        Ainv_T1 =
        1/(pi R) *
        [[ -2/sqrt(1+sin(theta)),
           -(pi/2-lambda)*sqrt(1+sin(theta)) ],
         [  2/sqrt(1+sin(theta)),
           -lambda*sqrt(1+sin(theta)) ]]

    Important:
    ----------
    At the exact pole, lambda is directional data. This function assumes
    lambda is already supplied. It does not infer lambda from x=y=0.
    """
    lam = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if lam.shape != theta.shape:
        raise ValueError("lambda_ and theta must have the same shape.")

    s = np.sin(theta)
    root = np.sqrt(1.0 + s)

    Ainv = np.empty(lam.shape + (2, 2), dtype=float)
    fac = 1.0 / (np.pi * R)

    Ainv[..., 0, 0] = fac * (-2.0 / root)
    Ainv[..., 0, 1] = fac * (-(0.5 * np.pi - lam) * root)
    Ainv[..., 1, 0] = fac * ( 2.0 / root)
    Ainv[..., 1, 1] = fac * (-lam * root)

    return Ainv



def sdg_Ainv_stable_from_lambda_theta_patch(
    lambda_,
    theta,
    patch_id,
    R: float = 1.0,
) -> np.ndarray:
    r"""
    SDG pole-stable A^{-1} formulas for all eight patches T1--T8.

    This function is derived algebraically from the SDG explicit A matrices
    and det(A)=pi R^2.

    It does not use np.linalg.inv(A).

    Top / north-pole patches:
        T1, T3, T5, T7 use P = sqrt(1 + sin(theta)).

    Bottom / south-pole patches:
        T2, T4, T6, T8 use M = sqrt(1 - sin(theta)).

    Important
    ---------
    At exact poles, lambda is directional information. This function assumes
    lambda is already supplied. It does not infer lambda from a collapsed pole
    point such as x=y=0.
    """
    lam = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)
    pid = np.asarray(patch_id, dtype=int)

    if lam.shape != theta.shape:
        raise ValueError("lambda_ and theta must have the same shape.")

    if pid.shape != lam.shape:
        if pid.size == 1:
            pid = np.full_like(lam, int(pid), dtype=int)
        else:
            raise ValueError("patch_id must be scalar or have the same shape as lambda_.")

    if np.any((pid < 1) | (pid > 8)):
        raise ValueError("patch_id must contain only values 1,...,8.")

    Ainv = np.full(lam.shape + (2, 2), np.nan, dtype=float)
    fac = 1.0 / (np.pi * R)
    s = np.sin(theta)

    # ---------------------------
    # Top patches: T1, T3, T5, T7
    # P = sqrt(1 + sin(theta))
    # ---------------------------
    top = np.isin(pid, [1, 3, 5, 7])
    P = np.empty_like(lam, dtype=float)
    P[:] = np.nan
    P[top] = np.sqrt(1.0 + s[top])

    # T1
    m = pid == 1
    if np.any(m):
        Ainv[m, 0, 0] = fac * (-2.0 / P[m])
        Ainv[m, 0, 1] = fac * (-(0.5 * np.pi - lam[m]) * P[m])
        Ainv[m, 1, 0] = fac * ( 2.0 / P[m])
        Ainv[m, 1, 1] = fac * (-lam[m] * P[m])

    # T3
    m = pid == 3
    if np.any(m):
        Ainv[m, 0, 0] = fac * (-2.0 / P[m])
        Ainv[m, 0, 1] = fac * (-(0.5 * np.pi - lam[m]) * P[m])
        Ainv[m, 1, 0] = fac * (-2.0 / P[m])
        Ainv[m, 1, 1] = fac * (-(np.pi - lam[m]) * P[m])

    # T5
    m = pid == 5
    if np.any(m):
        Ainv[m, 0, 0] = fac * (-2.0 / P[m])
        Ainv[m, 0, 1] = fac * (-(1.5 * np.pi - lam[m]) * P[m])
        Ainv[m, 1, 0] = fac * ( 2.0 / P[m])
        Ainv[m, 1, 1] = fac * (-(lam[m] - np.pi) * P[m])

    # T7
    m = pid == 7
    if np.any(m):
        Ainv[m, 0, 0] = fac * ( 2.0 / P[m])
        Ainv[m, 0, 1] = fac * (-(0.5 * np.pi + lam[m]) * P[m])
        Ainv[m, 1, 0] = fac * ( 2.0 / P[m])
        Ainv[m, 1, 1] = fac * (-lam[m] * P[m])

    # ------------------------------
    # Bottom patches: T2, T4, T6, T8
    # M = sqrt(1 - sin(theta))
    # ------------------------------
    bottom = np.isin(pid, [2, 4, 6, 8])
    M = np.empty_like(lam, dtype=float)
    M[:] = np.nan
    M[bottom] = np.sqrt(1.0 - s[bottom])

    # T2
    m = pid == 2
    if np.any(m):
        Ainv[m, 0, 0] = fac * (-2.0 / M[m])
        Ainv[m, 0, 1] = fac * (-lam[m] * M[m])
        Ainv[m, 1, 0] = fac * ( 2.0 / M[m])
        Ainv[m, 1, 1] = fac * (-(0.5 * np.pi - lam[m]) * M[m])

    # T4
    m = pid == 4
    if np.any(m):
        Ainv[m, 0, 0] = fac * (-2.0 / M[m])
        Ainv[m, 0, 1] = fac * ((np.pi - lam[m]) * M[m])
        Ainv[m, 1, 0] = fac * (-2.0 / M[m])
        Ainv[m, 1, 1] = fac * ((0.5 * np.pi - lam[m]) * M[m])

    # T6
    m = pid == 6
    if np.any(m):
        Ainv[m, 0, 0] = fac * ( 2.0 / M[m])
        Ainv[m, 0, 1] = fac * ((lam[m] - np.pi) * M[m])
        Ainv[m, 1, 0] = fac * (-2.0 / M[m])
        Ainv[m, 1, 1] = fac * ((1.5 * np.pi - lam[m]) * M[m])

    # T8
    m = pid == 8
    if np.any(m):
        Ainv[m, 0, 0] = fac * ( 2.0 / M[m])
        Ainv[m, 0, 1] = fac * (lam[m] * M[m])
        Ainv[m, 1, 0] = fac * ( 2.0 / M[m])
        Ainv[m, 1, 1] = fac * ((0.5 * np.pi + lam[m]) * M[m])

    return Ainv


def sdg_Ainv_with_T1_stable_patch(
    A: np.ndarray,
    lambda_: np.ndarray,
    theta: np.ndarray,
    patch_id: np.ndarray,
    R: float = 1.0,
) -> np.ndarray:
    """
    Backward-compatible wrapper.

    Earlier this function only replaced T1. It now delegates to the
    all-patch SDG stable A^{-1} formula.
    """
    return sdg_Ainv_stable_from_lambda_theta_patch(
        lambda_=lambda_,
        theta=theta,
        patch_id=patch_id,
        R=R,
    )

def sdg_Ainv_from_A_explicit(A: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    SDG explicit A^{-1} using det(A)=pi R^2:

        A^{-1} = 1/(pi R^2) [[A22, -A12], [-A21, A11]]

    This is not np.linalg.inv. np.linalg.inv is only used separately for diagnostics.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[-2:] != (2, 2):
        raise ValueError("A must have shape (...,2,2).")

    det_expected = sdg_detA_expected(R)
    Ainv = np.empty_like(A, dtype=float)

    Ainv[..., 0, 0] = A[..., 1, 1] / det_expected
    Ainv[..., 0, 1] = -A[..., 0, 1] / det_expected
    Ainv[..., 1, 0] = -A[..., 1, 0] / det_expected
    Ainv[..., 1, 1] = A[..., 0, 0] / det_expected

    return Ainv


def sdg_sqrtG_from_A(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    G = np.einsum("...ki,...kj->...ij", A, A)
    detG = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
    with np.errstate(invalid="ignore"):
        return np.sqrt(detG)


def sdg_A_Ainv_error(A: np.ndarray, Ainv: np.ndarray) -> np.ndarray:
    I = np.eye(2)
    prod = np.einsum("...ij,...jk->...ik", A, Ainv)
    return np.max(np.abs(prod - I), axis=(-1, -2))


def sdg_Ainv_numpy_error(A: np.ndarray, Ainv_sdg: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    """
    Diagnostic only. This does not define Ainv.
    """
    A = np.asarray(A, dtype=float)
    Ainv_sdg = np.asarray(Ainv_sdg, dtype=float)
    bad_mask = np.asarray(bad_mask, dtype=bool)

    out = np.full(A.shape[:-2], np.nan, dtype=float)

    flat_A = A.reshape((-1, 2, 2))
    flat_B = Ainv_sdg.reshape((-1, 2, 2))
    flat_bad = bad_mask.reshape(-1)
    flat_out = out.reshape(-1)

    for i in range(flat_A.shape[0]):
        if flat_bad[i]:
            continue
        try:
            inv_np = np.linalg.inv(flat_A[i])
            flat_out[i] = np.max(np.abs(flat_B[i] - inv_np))
        except np.linalg.LinAlgError:
            flat_out[i] = np.nan

    return out


def sdg_mapping_from_xy_patch(
    x,
    y,
    patch_id,
    R: float = 1.0,
    tol: float = 1.0e-12,
) -> SDGMappingResult:
    """
    Full SDG mapping bundle:
        (x,y,patch_id) -> lambda, theta, sphere xyz, A, Ainv, diagnostics.
    """
    x = _as_float_array(x)
    y = _as_float_array(y)
    pid = np.asarray(patch_id, dtype=int)

    if pid.shape != x.shape:
        if pid.size == 1:
            pid = np.full_like(x, int(pid), dtype=int)
        else:
            raise ValueError("patch_id must be scalar or have the same shape as x.")

    lambda_, theta, bad_lt = sdg_lambda_theta_from_xy_patch(
        x=x,
        y=y,
        patch_id=pid,
        tol=tol,
    )

    X, Y, Z = sdg_sphere_xyz_from_lambda_theta(lambda_, theta, R=R)

    A, bad_A = sdg_A_from_lambda_theta_patch(
        lambda_=lambda_,
        theta=theta,
        patch_id=pid,
        R=R,
        tol=tol,
    )

    Ainv = sdg_Ainv_stable_from_lambda_theta_patch(lambda_, theta, pid, R=R)

    detA = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    sqrtG = sdg_sqrtG_from_A(A)

    c = np.cos(theta)
    pole_mask = np.abs(c) <= 10.0 * tol
    bad_mask = bad_lt | bad_A | pole_mask

    A_Ainv_err = sdg_A_Ainv_error(A, Ainv)
    Ainv_numpy_err = sdg_Ainv_numpy_error(A, Ainv, bad_mask=bad_mask)

    return SDGMappingResult(
        lambda_=lambda_,
        theta=theta,
        X=X,
        Y=Y,
        Z=Z,
        A=A,
        Ainv=Ainv,
        detA=detA,
        sqrtG=sqrtG,
        pole_mask=pole_mask,
        bad_mask=bad_mask,
        A_Ainv_err=A_Ainv_err,
        Ainv_numpy_err=Ainv_numpy_err,
    )
