from __future__ import annotations

import numpy as np

from problems.sphere_single_patch_rhs import (
    compute_constant_state_single_patch_rhs,
)


def test_single_patch_rhs_constant_state_patch1():
    """
    For q = 1 and divergence-free local contravariant velocity,
    RHS should be approximately zero on one patch.
    """
    rhs, diag, setup = compute_constant_state_single_patch_rhs(
        patch_id=1,
        nsub=5,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
        N=4,
        alpha0=0.0,
        tau=0.0,
    )

    assert rhs.shape == setup.X.shape

    total_err = np.max(np.abs(rhs))
    volume_err = np.max(np.abs(diag["volume_rhs"]))
    surface_err = np.max(np.abs(diag["surface_rhs"]))

    assert surface_err < 1e-12
    assert volume_err < 1e-10
    assert total_err < 1e-10


def test_single_patch_rhs_constant_state_all_patches():
    """
    Same constant-state RHS check for all 8 patches.
    """
    for patch_id in range(1, 9):
        rhs, diag, setup = compute_constant_state_single_patch_rhs(
            patch_id=patch_id,
            nsub=5,
            radius=1.0,
            quad_table="table1",
            quad_order=4,
            N=4,
            alpha0=0.0,
            tau=0.0,
        )

        assert rhs.shape == setup.X.shape

        total_err = np.max(np.abs(rhs))
        surface_err = np.max(np.abs(diag["surface_rhs"]))

        assert surface_err < 1e-12
        assert total_err < 1e-10


def test_single_patch_rhs_uses_table1_by_default():
    """
    Safety check: default quadrature table should be table1.
    """
    rhs, diag, setup = compute_constant_state_single_patch_rhs(
        patch_id=1,
        nsub=2,
        radius=1.0,
        quad_order=4,
        N=4,
    )

    assert setup.quad_table == "table1"
    assert rhs.shape == setup.X.shape