from __future__ import annotations

import numpy as np

from data.rule_registry import load_rule
from geometry.sphere_patches import all_patch_ids, local_xy_from_reference_rs, patch_signs
from geometry.sphere_mapping import local_xy_to_sphere_xyz
from geometry.sphere_metrics import (
    transformation_matrix_A,
    sqrtG,
    expected_sqrtG,
    contravariant_velocity,
)
from problems.sphere_advection import spherical_advection_velocity
from geometry.sphere_mapping import local_xy_to_patch_angles


def _interior_quadrature_nodes(order: int = 4):
    """
    Use existing repo quadrature data instead of defining nodes again.

    Table2 nodes are inside the reference triangle.
    """
    rule = load_rule("table2", order)
    rs = rule["rs"]
    x, y = local_xy_from_reference_rs(rs)

    # Remove possible pole-like points if future rules contain boundary nodes.
    # Current Table2 rules are interior for order 1..4.
    mask = (x + y) > 1e-12
    return x[mask], y[mask]


def test_sphere_radius_identity():
    radius = 2.0
    x, y = _interior_quadrature_nodes(order=4)

    for patch_id in all_patch_ids():
        X, Y, Z = local_xy_to_sphere_xyz(x, y, patch_id, radius=radius)
        err = np.max(np.abs(X**2 + Y**2 + Z**2 - radius**2))
        assert err < 1e-12


def test_patch_signs_match_faces():
    radius = 1.0
    x, y = _interior_quadrature_nodes(order=4)

    for patch_id in all_patch_ids():
        sx, sy, sz = patch_signs(patch_id)
        X, Y, Z = local_xy_to_sphere_xyz(x, y, patch_id, radius=radius)

        assert np.all(sx * X >= -1e-12)
        assert np.all(sy * Y >= -1e-12)
        assert np.all(sz * Z >= -1e-12)


def test_sqrtG_is_constant():
    radius = 1.0
    x, y = _interior_quadrature_nodes(order=4)

    for patch_id in all_patch_ids():
        J = sqrtG(x, y, patch_id, radius=radius)
        assert np.max(np.abs(J - expected_sqrtG(radius=radius))) < 1e-11


def test_contravariant_velocity_roundtrip():
    radius = 1.0
    alpha0 = np.pi / 4.0
    x, y = _interior_quadrature_nodes(order=4)

    for patch_id in all_patch_ids():
        lam, theta = local_xy_to_patch_angles(x, y, patch_id)
        u, v = spherical_advection_velocity(lam, theta, u0=1.0, alpha0=alpha0)

        u1, u2 = contravariant_velocity(u, v, x, y, patch_id, radius=radius)
        A = transformation_matrix_A(x, y, patch_id, radius=radius)

        uv_reconstructed = np.einsum(
            "...ij,...j->...i",
            A,
            np.stack([u1, u2], axis=-1),
        )

        assert np.max(np.abs(uv_reconstructed[:, 0] - u)) < 1e-11
        assert np.max(np.abs(uv_reconstructed[:, 1] - v)) < 1e-11