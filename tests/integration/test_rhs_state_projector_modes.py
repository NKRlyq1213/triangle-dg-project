from __future__ import annotations

import numpy as np
import pytest

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import (
    build_polynomial_l2_projector_from_rule,
    build_reference_diff_operators_from_rule,
)
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.rhs_split_conservative_exchange import (
    build_surface_exchange_cache,
    build_volume_split_cache,
    rhs_split_conservative_exchange,
    surface_term_from_exchange,
)
from operators.trace_policy import build_trace_policy


def q_boundary_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.sin(x - t)


def velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def q_boundary_correction_zero(
    x_face: np.ndarray,
    y_face: np.ndarray,
    t: float,
    qM: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    return np.zeros_like(q_boundary_exact)


def q_boundary_correction_const(
    x_face: np.ndarray,
    y_face: np.ndarray,
    t: float,
    qM: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    return np.full_like(q_boundary_exact, 0.25)


def _build_rhs_fixture() -> dict:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, 4)
    projector = build_polynomial_l2_projector_from_rule(rule, 4)

    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)
    surface_cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
    )

    q0 = np.sin(X) @ projector.T
    u_elem, v_elem = velocity_one_one(X, Y, t=0.0)

    return {
        "q0": q0,
        "u_elem": u_elem,
        "v_elem": v_elem,
        "Dr": Dr,
        "Ds": Ds,
        "geom": geom,
        "rule": rule,
        "trace": trace,
        "conn": conn,
        "face_geom": face_geom,
        "projector": projector,
        "surface_cache": surface_cache,
    }


def test_state_projector_post_matches_both_for_projected_state() -> None:
    fx = _build_rhs_fixture()

    rhs_both, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        state_projector=fx["projector"],
        state_projector_mode="both",
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    rhs_post, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        state_projector=fx["projector"],
        state_projector_mode="post",
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    assert np.allclose(rhs_both, rhs_post, atol=1e-12, rtol=1e-12)


def test_state_projector_mode_validation() -> None:
    fx = _build_rhs_fixture()

    with pytest.raises(ValueError, match="state_projector_mode"):
        rhs_split_conservative_exchange(
            q_elem=fx["q0"],
            u_elem=fx["u_elem"],
            v_elem=fx["v_elem"],
            Dr=fx["Dr"],
            Ds=fx["Ds"],
            geom=fx["geom"],
            rule=fx["rule"],
            trace=fx["trace"],
            conn=fx["conn"],
            face_geom=fx["face_geom"],
            q_boundary=q_boundary_sinx,
            velocity=velocity_one_one,
            t=0.125,
            tau=0.0,
            compute_mismatches=False,
            return_diagnostics=False,
            use_numba=False,
            state_projector=fx["projector"],
            state_projector_mode="invalid",
            surface_backend="face-major",
            surface_cache=fx["surface_cache"],
        )


def test_state_projector_T_matches_state_projector() -> None:
    fx = _build_rhs_fixture()

    rhs_ref, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        state_projector=fx["projector"],
        state_projector_mode="post",
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    rhs_fast, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        state_projector_T=fx["projector"].T,
        state_projector_mode="post",
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    assert np.allclose(rhs_ref, rhs_fast, atol=1e-12, rtol=1e-12)


def test_ndotv_precomputed_matches_velocity_callback() -> None:
    fx = _build_rhs_fixture()
    ndotV_precomputed = np.asarray(fx["face_geom"]["nx"], dtype=float) + np.asarray(
        fx["face_geom"]["ny"],
        dtype=float,
    )

    rhs_ref, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    rhs_fast, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        ndotV_precomputed=ndotV_precomputed,
    )

    assert np.allclose(rhs_ref, rhs_fast, atol=1e-12, rtol=1e-12)


def test_ndotv_flat_precomputed_matches_ndotv_precomputed() -> None:
    fx = _build_rhs_fixture()
    ndotV_precomputed = np.asarray(fx["face_geom"]["nx"], dtype=float) + np.asarray(
        fx["face_geom"]["ny"],
        dtype=float,
    )
    ndotV_flat_precomputed = ndotV_precomputed.reshape(-1, int(fx["trace"]["nfp"]))

    rhs_ref, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        ndotV_precomputed=ndotV_precomputed,
    )

    rhs_fast, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        ndotV_precomputed=ndotV_precomputed,
        ndotV_flat_precomputed=ndotV_flat_precomputed,
    )

    assert np.allclose(rhs_ref, rhs_fast, atol=1e-12, rtol=1e-12)


def test_q_boundary_correction_zero_matches_baseline() -> None:
    fx = _build_rhs_fixture()

    rhs_ref, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    rhs_corr, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        q_boundary_correction=q_boundary_correction_zero,
        q_boundary_correction_mode="inflow",
    )

    assert np.allclose(rhs_ref, rhs_corr, atol=1e-12, rtol=1e-12)


def test_q_boundary_correction_inflow_and_all_modes() -> None:
    fx = _build_rhs_fixture()
    t_eval = 0.125

    _, diag_inflow = surface_term_from_exchange(
        q_elem=fx["q0"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=t_eval,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        q_boundary_correction=q_boundary_correction_const,
        q_boundary_correction_mode="inflow",
    )

    qB_exact = q_boundary_sinx(diag_inflow["x_face"], diag_inflow["y_face"], t=t_eval)
    bd = np.asarray(fx["conn"]["is_boundary"], dtype=bool)[:, :, None]
    inflow = bd & (diag_inflow["ndotV"] < 0.0)
    outflow = bd & (diag_inflow["ndotV"] >= 0.0)

    expected_inflow = np.array(qB_exact, copy=True)
    expected_inflow[inflow] += 0.25
    assert np.allclose(diag_inflow["qB"], expected_inflow, atol=1e-12, rtol=1e-12)
    assert np.allclose(diag_inflow["qB"][outflow], qB_exact[outflow], atol=1e-12, rtol=1e-12)

    _, diag_all = surface_term_from_exchange(
        q_elem=fx["q0"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=t_eval,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        q_boundary_correction=q_boundary_correction_const,
        q_boundary_correction_mode="all",
    )

    expected_all = np.array(qB_exact, copy=True)
    expected_all[np.broadcast_to(bd, qB_exact.shape)] += 0.25
    assert np.allclose(diag_all["qB"], expected_all, atol=1e-12, rtol=1e-12)


def test_q_boundary_correction_mode_validation() -> None:
    fx = _build_rhs_fixture()

    with pytest.raises(ValueError, match="q_boundary_correction_mode"):
        rhs_split_conservative_exchange(
            q_elem=fx["q0"],
            u_elem=fx["u_elem"],
            v_elem=fx["v_elem"],
            Dr=fx["Dr"],
            Ds=fx["Ds"],
            geom=fx["geom"],
            rule=fx["rule"],
            trace=fx["trace"],
            conn=fx["conn"],
            face_geom=fx["face_geom"],
            q_boundary=q_boundary_sinx,
            velocity=velocity_one_one,
            t=0.125,
            tau=0.0,
            compute_mismatches=False,
            return_diagnostics=False,
            use_numba=False,
            surface_backend="face-major",
            surface_cache=fx["surface_cache"],
            q_boundary_correction=q_boundary_correction_zero,
            q_boundary_correction_mode="invalid",
        )


def test_volume_split_cache_matches_direct_volume_path() -> None:
    fx = _build_rhs_fixture()
    vol_cache = build_volume_split_cache(
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
    )

    rhs_ref, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=True,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    rhs_fast, _ = rhs_split_conservative_exchange(
        q_elem=fx["q0"],
        u_elem=fx["u_elem"],
        v_elem=fx["v_elem"],
        Dr=fx["Dr"],
        Ds=fx["Ds"],
        geom=fx["geom"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=True,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        volume_split_cache=vol_cache,
    )

    assert np.allclose(rhs_ref, rhs_fast, atol=1e-11, rtol=1e-11)
