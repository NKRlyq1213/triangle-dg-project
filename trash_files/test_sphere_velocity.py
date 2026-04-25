from __future__ import annotations

import numpy as np

from geometry.sphere_mesh import build_sphere_patch_mesh
from geometry.sphere_velocity import (
    spherical_advection_velocity,
    build_velocity_on_sphere_mesh,
    velocity_roundtrip_error,
)


def test_spherical_velocity_shapes():
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table2",
        quad_order=4,
    )

    velocity = build_velocity_on_sphere_mesh(
        mesh,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    assert velocity.u_sph.shape == mesh.volume_lambda.shape
    assert velocity.v_sph.shape == mesh.volume_lambda.shape
    assert velocity.u1.shape == mesh.volume_lambda.shape
    assert velocity.u2.shape == mesh.volume_lambda.shape


def test_spherical_velocity_alpha0_zero():
    """
    If alpha0 = 0:
        u = u0 cos(theta)
        v = 0
    """
    lambda_ = np.array([0.0, np.pi / 2.0, np.pi])
    theta = np.array([0.0, np.pi / 6.0, -np.pi / 4.0])

    u0 = 2.5
    u, v = spherical_advection_velocity(
        lambda_,
        theta,
        u0=u0,
        alpha0=0.0,
    )

    assert np.max(np.abs(u - u0 * np.cos(theta))) < 1e-14
    assert np.max(np.abs(v)) < 1e-14


def test_velocity_roundtrip_on_sphere_mesh():
    """
    Verify:
        A [u1, u2]^T = [u, v]^T
    """
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table2",
        quad_order=4,
    )

    velocity = build_velocity_on_sphere_mesh(
        mesh,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    errors = velocity_roundtrip_error(mesh, velocity)

    for patch_id, err in errors.items():
        assert err < 1e-11


def test_velocity_is_finite():
    """
    Guard against pole / singular derivative issues.

    Current volume quadrature nodes should not sit exactly at the pole.
    """
    mesh = build_sphere_patch_mesh(
        nsub=5,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
    )

    velocity = build_velocity_on_sphere_mesh(
        mesh,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    assert np.all(np.isfinite(velocity.u_sph))
    assert np.all(np.isfinite(velocity.v_sph))
    assert np.all(np.isfinite(velocity.u1))
    assert np.all(np.isfinite(velocity.u2))