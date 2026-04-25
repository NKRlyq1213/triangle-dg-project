from __future__ import annotations

import numpy as np

from operators.sdg_cartesian_divergence_20260422a import (
    build_sdg_cartesian_divergence_setup_20260422a,
    sdg_cartesian_divergence_of_constant_state_20260422a,
    integrate_global_divergence_diagnostics_20260422a,
)
from geometry.sdg_velocity_global_20260422a import (
    sdg_velocity_roundtrip_error_20260422a,
)


def test_velocity_roundtrip_20260422a():
    setup = build_sdg_cartesian_divergence_setup_20260422a(
        nsub=4,
        order=4,
        N=4,
        radius=1.0,
    )

    err = sdg_velocity_roundtrip_error_20260422a(
        setup.X,
        setup.Y,
        setup.mesh.face_ids,
        radius=1.0,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    assert err < 1e-11


def test_constant_state_cartesian_divergence_alpha0_zero_20260422a():
    setup = build_sdg_cartesian_divergence_setup_20260422a(
        nsub=4,
        order=4,
        N=4,
        radius=1.0,
    )

    div, diag = sdg_cartesian_divergence_of_constant_state_20260422a(
        setup,
        u0=1.0,
        alpha0=0.0,
    )

    stats = integrate_global_divergence_diagnostics_20260422a(
        div,
        setup,
        surface_weighted=True,
    )

    assert np.all(np.isfinite(div))
    assert abs(stats["integral"]) < 1e-10
    assert stats["linf"] < 1e-8


def test_constant_state_cartesian_divergence_alpha0_pi_over_4_finite_20260422a():
    setup = build_sdg_cartesian_divergence_setup_20260422a(
        nsub=4,
        order=4,
        N=4,
        radius=1.0,
    )

    div, diag = sdg_cartesian_divergence_of_constant_state_20260422a(
        setup,
        u0=1.0,
        alpha0=np.pi / 4.0,
    )

    stats = integrate_global_divergence_diagnostics_20260422a(
        div,
        setup,
        surface_weighted=True,
    )

    assert np.all(np.isfinite(div))
    assert abs(stats["integral"]) < 1e-8

def test_T1_velocity_pole_corrected_finite_20260425a():
    from geometry.sdg_velocity_global_20260422a import (
        sdg_global_contravariant_velocity_20260422a,
    )

    x = np.array([0.0, 1e-15, 1e-12, 1e-8])
    y = np.array([0.0, 2e-15, 3e-12, 2e-8])

    u1, u2, u, v = sdg_global_contravariant_velocity_20260422a(
        x,
        y,
        face_id=1,
        radius=1.0,
        u0=1.0,
        alpha0=np.pi / 4.0,
        pole_tol=1e-14,
    )

    assert np.all(np.isfinite(u1))
    assert np.all(np.isfinite(u2))
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))