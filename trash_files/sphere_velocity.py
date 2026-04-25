from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from geometry.sphere_metrics_sdg_hardcoded import (
    transformation_matrix_A_sdg_local,
    contravariant_velocity_sdg_local,
    reconstruct_spherical_velocity_sdg_local,
)


@dataclass(frozen=True)
class SphereVelocityField:
    """
    Velocity data on a SpherePatchMesh.

    中文：
    - u_sph, v_sph 是球面正交基底 (e_lambda, e_theta) 下的速度。
    - u1, u2 是 local triangle coordinates 下的 contravariant velocity。
    """

    u_sph: np.ndarray      # shape (K, Nq)
    v_sph: np.ndarray      # shape (K, Nq)
    u1: np.ndarray         # shape (K, Nq)
    u2: np.ndarray         # shape (K, Nq)


def spherical_advection_velocity(
    lambda_: np.ndarray,
    theta: np.ndarray,
    *,
    u0: float = 1.0,
    alpha0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Velocity field on the sphere.

    Formula:
        u = u0 (cos(alpha0) cos(theta)
                + sin(alpha0) cos(lambda) sin(theta))
        v = -u0 sin(alpha0) sin(lambda)
    u = u0 * (np.cos(alpha0) * np.cos(lat) + np.sin(alpha0) * np.cos(lam) * np.sin(lat))
    v = -u0 * np.sin(alpha0) * np.sin(lam)
    Parameters
    ----------
    lambda_, theta:
        Spherical longitude and latitude.
    u0:
        Velocity scale.
    alpha0:
        Rotation angle parameter.

    Returns
    -------
    u, v:
        Spherical velocity components in (e_lambda, e_theta).
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if lambda_.shape != theta.shape:
        raise ValueError("lambda_ and theta must have the same shape.")

    u = u0 * (
        np.cos(alpha0) * np.cos(theta)
        + np.sin(alpha0) * np.cos(lambda_) * np.sin(theta)
    )

    v = -u0 * np.sin(alpha0) * np.sin(lambda_)

    return u, v


def contravariant_velocity_from_spherical(
    u_sph: np.ndarray,
    v_sph: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    patch_id: int,
    *,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard-coded SDG version:
        [u1,u2]^T = A_SDG_local^{-1} [u,v]^T
    """
    return contravariant_velocity_sdg_local(
        u_sph,
        v_sph,
        x_local,
        y_local,
        patch_id,
        radius=radius,
    )


def reconstruct_spherical_velocity_from_contravariant(
    u1: np.ndarray,
    u2: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    patch_id: int,
    *,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard-coded SDG version:
        [u,v]^T = A_SDG_local [u1,u2]^T
    """
    return reconstruct_spherical_velocity_sdg_local(
        u1,
        u2,
        x_local,
        y_local,
        patch_id,
        radius=radius,
    )


def build_velocity_on_sphere_mesh(
    mesh,
    *,
    u0: float = 1.0,
    alpha0: float = 0.0,
) -> SphereVelocityField:
    """
    Build velocity field on all volume nodes of a SpherePatchMesh.

    This function uses precomputed mesh.lambda/theta/local nodes.
    It does NOT recompute geometry.
    """
    u_sph, v_sph = spherical_advection_velocity(
        mesh.volume_lambda,
        mesh.volume_theta,
        u0=u0,
        alpha0=alpha0,
    )

    K, Nq = u_sph.shape
    u1 = np.zeros((K, Nq), dtype=float)
    u2 = np.zeros((K, Nq), dtype=float)

    for patch_id in range(1, 9):
        mask = mesh.elem_to_patch == patch_id

        x = mesh.volume_nodes_local[mask, :, 0]
        y = mesh.volume_nodes_local[mask, :, 1]

        u1_patch, u2_patch = contravariant_velocity_from_spherical(
            u_sph[mask, :],
            v_sph[mask, :],
            x,
            y,
            patch_id,
            radius=mesh.radius,
        )

        u1[mask, :] = u1_patch
        u2[mask, :] = u2_patch

    return SphereVelocityField(
        u_sph=u_sph,
        v_sph=v_sph,
        u1=u1,
        u2=u2,
    )


def velocity_roundtrip_error(
    mesh,
    velocity: SphereVelocityField,
) -> dict[int, float]:
    """
    Check:
        A [u1, u2]^T = [u, v]^T

    Returns
    -------
    errors:
        dict patch_id -> max roundtrip error.
    """
    errors: dict[int, float] = {}

    for patch_id in range(1, 9):
        mask = mesh.elem_to_patch == patch_id

        x = mesh.volume_nodes_local[mask, :, 0]
        y = mesh.volume_nodes_local[mask, :, 1]

        u_rec, v_rec = reconstruct_spherical_velocity_from_contravariant(
            velocity.u1[mask, :],
            velocity.u2[mask, :],
            x,
            y,
            patch_id,
            radius=mesh.radius,
        )

        err_u = np.max(np.abs(u_rec - velocity.u_sph[mask, :]))
        err_v = np.max(np.abs(v_rec - velocity.v_sph[mask, :]))

        errors[patch_id] = float(max(err_u, err_v))

    return errors