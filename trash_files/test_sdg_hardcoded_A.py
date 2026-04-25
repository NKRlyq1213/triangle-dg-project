from __future__ import annotations

import numpy as np

from data.rule_registry import load_rule
from geometry.sphere_patches import all_patch_ids, local_xy_from_reference_rs
from geometry.sphere_mapping import local_xy_to_patch_angles
from geometry.sphere_velocity import spherical_advection_velocity
from geometry.sphere_metrics_sdg_hardcoded import (
    sqrtG_sdg_local,
    expected_sqrtG_sdg,
    contravariant_velocity_sdg_local,
    reconstruct_spherical_velocity_sdg_local,
)


def _test_nodes():
    rule = load_rule("table1", 4)
    rs = rule["rs"]
    x, y = local_xy_from_reference_rs(rs)

    # avoid exact pole if table ever contains vertex node
    mask = (x + y) > 1e-12
    return x[mask], y[mask]


def test_sdg_hardcoded_sqrtG_constant():
    x, y = _test_nodes()
    radius = 1.0

    for patch_id in all_patch_ids():
        J = sqrtG_sdg_local(
            x,
            y,
            patch_id,
            radius=radius,
        )

        err = np.max(np.abs(J - expected_sqrtG_sdg(radius=radius)))
        assert err < 1e-11


def test_sdg_hardcoded_velocity_roundtrip():
    x, y = _test_nodes()
    radius = 1.0
    alpha0 = np.pi / 4.0

    for patch_id in all_patch_ids():
        lam, theta = local_xy_to_patch_angles(
            x,
            y,
            patch_id,
        )

        u, v = spherical_advection_velocity(
            lam,
            theta,
            u0=1.0,
            alpha0=alpha0,
        )

        u1, u2 = contravariant_velocity_sdg_local(
            u,
            v,
            x,
            y,
            patch_id,
            radius=radius,
        )

        u_rec, v_rec = reconstruct_spherical_velocity_sdg_local(
            u1,
            u2,
            x,
            y,
            patch_id,
            radius=radius,
        )

        err = max(
            float(np.max(np.abs(u_rec - u))),
            float(np.max(np.abs(v_rec - v))),
        )

        assert err < 1e-11