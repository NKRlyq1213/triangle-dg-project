from __future__ import annotations

import numpy as np
import pytest

from experiments.lsrk_h_convergence import (
    LSRKHConvergenceConfig,
    build_sinx_rk_stage_boundary_correction,
    run_lsrk_h_convergence,
    run_lsrk_h_convergence_compare_qb_correction,
)


def test_sinx_rk_stage_boundary_correction_callback_shape() -> None:
    corr = build_sinx_rk_stage_boundary_correction(dt=1.0e-2)

    x_face = np.linspace(-1.0, 1.0, 6, dtype=float).reshape(1, 3, 2)
    y_face = np.zeros_like(x_face)
    qM = np.zeros_like(x_face)
    ndotV = -np.ones_like(x_face)
    is_boundary = np.ones((1, 3), dtype=bool)

    qB_exact_0 = np.sin(np.pi * (x_face - 0.0))
    delta0 = corr(x_face, y_face, 0.0, qM, ndotV, is_boundary, qB_exact_0)

    assert delta0.shape == qB_exact_0.shape
    assert np.all(np.isfinite(delta0))

    qB_exact_1 = np.sin(np.pi * (x_face - 2.0e-3))
    delta1 = corr(x_face, y_face, 2.0e-3, qM, ndotV, is_boundary, qB_exact_1)

    assert delta1.shape == qB_exact_1.shape
    assert np.all(np.isfinite(delta1))


def test_compare_qb_correction_runs_for_small_case() -> None:
    cfg = LSRKHConvergenceConfig(
        mesh_levels=(1,),
        tf_values=(0.1,),
        use_numba=False,
        verbose=False,
    )

    compared = run_lsrk_h_convergence_compare_qb_correction(cfg)

    assert set(compared.keys()) == {"baseline", "rk_stage_correction"}

    tf_key = next(iter(compared["baseline"].keys()))
    baseline_row = compared["baseline"][tf_key][0]
    corrected_row = compared["rk_stage_correction"][tf_key][0]

    assert baseline_row["q_boundary_correction_kind"] == "none"
    assert corrected_row["q_boundary_correction_kind"] == "sinx_rk_stage"


def test_config_rejects_mixed_qb_correction_sources() -> None:
    def custom_corr(*args):
        return 0.0

    cfg = LSRKHConvergenceConfig(
        mesh_levels=(1,),
        tf_values=(0.1,),
        use_sinx_rk_stage_boundary_correction=True,
        q_boundary_correction=custom_corr,
        verbose=False,
    )

    with pytest.raises(ValueError, match="either"):
        run_lsrk_h_convergence(cfg)
