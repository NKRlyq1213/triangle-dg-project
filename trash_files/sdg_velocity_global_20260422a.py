from __future__ import annotations

import numpy as np

from geometry.sdg_Ainv_global_20260422a import (
    sdg_lambda_theta_20260422a,
    sdg_A_global_20260422a,
    sdg_Ainv_global_20260422a,
    sdg_A_Ainv_batch_20260422a,
)


def sdg_spherical_velocity_20260422a(
    lam: np.ndarray,
    theta: np.ndarray,
    *,
    u0: float = 1.0,
    alpha0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherical velocity from SDG4PDEOnSphere20260422a.pdf.

    V = (u,v) = u e_lambda + v e_theta

        u = u0(cos(alpha0) cos(theta)
               + sin(alpha0) cos(lambda) sin(theta))

        v = -u0 sin(alpha0) sin(lambda)
    """
    lam = np.asarray(lam, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if lam.shape != theta.shape:
        raise ValueError("lam and theta must have the same shape.")

    u = u0 * (
        np.cos(alpha0) * np.cos(theta)
        + np.sin(alpha0) * np.cos(lam) * np.sin(theta)
    )
    v = -u0 * np.sin(alpha0) * np.sin(lam)

    return u, v


def sdg_global_contravariant_velocity_20260422a(
    x,
    y,
    face_id: int,
    *,
    radius: float = 1.0,
    u0: float = 1.0,
    alpha0: float = 0.0,
    pole_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute global flattened Cartesian velocity components (u1,u2).

    Input:
        x,y:
            global flattened square coordinates.
        face_id:
            T1,...,T8.

    Output:
        u1,u2:
            global Cartesian contravariant velocity components used in

                d_x(u1 q) + d_y(u2 q)

        u_sph,v_sph:
            spherical velocity components, for diagnostics.

    Relation:
        [u1,u2]^T = A^{-1} [u,v]^T.
    """
    lam, theta = sdg_lambda_theta_20260422a(
        x,
        y,
        face_id,
        pole_tol=pole_tol,
    )

    u_sph, v_sph = sdg_spherical_velocity_20260422a(
        lam,
        theta,
        u0=u0,
        alpha0=alpha0,
    )

    Ainv = sdg_Ainv_global_20260422a(
        x,
        y,
        face_id,
        radius=radius,
        pole_tol=pole_tol,
    )

    uv = np.stack([u_sph, v_sph], axis=-1)
    out = np.einsum("...ij,...j->...i", Ainv, uv)

    u1 = out[..., 0]
    u2 = out[..., 1]

    return u1, u2, u_sph, v_sph


def sdg_global_contravariant_velocity_on_mesh_20260422a(
    X: np.ndarray,
    Y: np.ndarray,
    face_ids: np.ndarray,
    *,
    radius: float = 1.0,
    u0: float = 1.0,
    alpha0: float = 0.0,
    pole_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch version for element-node arrays.

    Parameters
    ----------
    X,Y:
        Shape (K,Np), global flattened coordinates.
    face_ids:
        Shape (K,), values 1,...,8.

    Returns
    -------
    u1,u2,u_sph,v_sph:
        Each shape (K,Np).
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

    u1 = np.zeros_like(X)
    u2 = np.zeros_like(X)
    u_sph = np.zeros_like(X)
    v_sph = np.zeros_like(X)

    for fid in range(1, 9):
        mask = face_ids == fid
        if not np.any(mask):
            continue

        u1_i, u2_i, u_i, v_i = sdg_global_contravariant_velocity_20260422a(
            X[mask, :],
            Y[mask, :],
            fid,
            radius=radius,
            u0=u0,
            alpha0=alpha0,
            pole_tol=pole_tol,
        )

        u1[mask, :] = u1_i
        u2[mask, :] = u2_i
        u_sph[mask, :] = u_i
        v_sph[mask, :] = v_i

    return u1, u2, u_sph, v_sph


def sdg_velocity_roundtrip_error_20260422a(
    X: np.ndarray,
    Y: np.ndarray,
    face_ids: np.ndarray,
    *,
    radius: float = 1.0,
    u0: float = 1.0,
    alpha0: float = 0.0,
    pole_tol: float = 1e-14,
) -> float:
    """
    Check:

        A [u1,u2]^T = [u,v]^T

    using the SDG4PDEOnSphere20260422a A and Ainv.
    """
    u1, u2, u_sph, v_sph = sdg_global_contravariant_velocity_on_mesh_20260422a(
        X,
        Y,
        face_ids,
        radius=radius,
        u0=u0,
        alpha0=alpha0,
        pole_tol=pole_tol,
    )

    A, _ = sdg_A_Ainv_batch_20260422a(
        X,
        Y,
        face_ids,
        radius=radius,
        pole_tol=pole_tol,
    )

    uv_flat = np.stack([u1, u2], axis=-1)
    uv_rec = np.einsum("...ij,...j->...i", A, uv_flat)

    err_u = np.max(np.abs(uv_rec[..., 0] - u_sph))
    err_v = np.max(np.abs(uv_rec[..., 1] - v_sph))

    return float(max(err_u, err_v))