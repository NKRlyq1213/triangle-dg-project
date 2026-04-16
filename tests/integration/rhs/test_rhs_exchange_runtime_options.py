from __future__ import annotations

import numpy as np
import pytest

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import build_reference_diff_operators_from_rule
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
    del y
    return np.sin(x - t)


def velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    del t
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
    del x_face, y_face, t, qM, ndotV, is_boundary
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
    del x_face, y_face, t, qM, ndotV, is_boundary
    return np.full_like(q_boundary_exact, 0.25)


def _build_rhs_fixture() -> dict:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, 4)

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

    q0 = np.sin(X)
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
        "surface_cache": surface_cache,
    }


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
    assert np.allclose(diag_inflow["qB_exact"], qB_exact, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        diag_inflow["qP_boundary"][np.broadcast_to(bd, qB_exact.shape)],
        expected_inflow[np.broadcast_to(bd, qB_exact.shape)],
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(diag_inflow["qP"][outflow], qB_exact[outflow], atol=1e-12, rtol=1e-12)

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
    assert np.allclose(diag_all["qB_exact"], qB_exact, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        diag_all["qP_boundary"][np.broadcast_to(bd, qB_exact.shape)],
        expected_all[np.broadcast_to(bd, qB_exact.shape)],
        atol=1e-12,
        rtol=1e-12,
    )


def test_tau_nonzero_uses_full_flux_formula_on_outflow_boundary() -> None:
    fx = _build_rhs_fixture()
    t_eval = 0.125

    _, diag_tau0 = surface_term_from_exchange(
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
    )

    tau = 0.4
    _, diag_tau = surface_term_from_exchange(
        q_elem=fx["q0"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=t_eval,
        tau=tau,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
    )

    ndotV = np.asarray(diag_tau["ndotV"], dtype=float)
    qM = np.asarray(diag_tau["qM"], dtype=float)
    qP = np.asarray(diag_tau["qP"], dtype=float)
    expected_p = 0.5 * (ndotV - (1.0 - tau) * np.abs(ndotV)) * (qM - qP)

    assert np.allclose(diag_tau["p"], expected_p, atol=1e-12, rtol=1e-12)

    bd = np.asarray(fx["conn"]["is_boundary"], dtype=bool)[:, :, None]
    outflow = bd & (ndotV > 0.0)

    assert np.allclose(diag_tau0["p"][outflow], 0.0, atol=1e-12, rtol=1e-12)
    assert np.max(np.abs(diag_tau["p"][outflow])) > 1e-8


def test_split_tau_uses_tau_qb_only_on_exact_qb_boundary() -> None:
    fx = _build_rhs_fixture()
    t_eval = 0.125

    tau_interior = 0.15
    tau_qb = 0.7
    _, diag = surface_term_from_exchange(
        q_elem=fx["q0"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=t_eval,
        tau=0.0,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        physical_boundary_mode="exact_qb",
    )

    is_boundary = np.asarray(fx["conn"]["is_boundary"], dtype=bool)
    expected_tau = np.full_like(diag["ndotV"], tau_interior, dtype=float)
    expected_tau[is_boundary] = tau_qb
    expected_p = 0.5 * (
        np.asarray(diag["ndotV"], dtype=float)
        - (1.0 - expected_tau) * np.abs(np.asarray(diag["ndotV"], dtype=float))
    ) * (np.asarray(diag["qM"], dtype=float) - np.asarray(diag["qP"], dtype=float))

    assert np.allclose(diag["tau_face"], expected_tau, atol=1e-12, rtol=1e-12)
    assert np.allclose(diag["p"], expected_p, atol=1e-12, rtol=1e-12)


def test_split_tau_ignores_tau_qb_for_opposite_boundary() -> None:
    fx = _build_rhs_fixture()
    t_eval = 0.125

    tau_interior = 0.2
    tau_qb = 0.85
    _, diag = surface_term_from_exchange(
        q_elem=fx["q0"],
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=t_eval,
        tau=0.0,
        tau_interior=tau_interior,
        tau_qb=tau_qb,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=fx["surface_cache"],
        physical_boundary_mode="opposite_boundary",
    )

    expected_tau = np.full_like(diag["ndotV"], tau_interior, dtype=float)
    expected_p = 0.5 * (
        np.asarray(diag["ndotV"], dtype=float)
        - (1.0 - expected_tau) * np.abs(np.asarray(diag["ndotV"], dtype=float))
    ) * (np.asarray(diag["qM"], dtype=float) - np.asarray(diag["qP"], dtype=float))

    assert np.allclose(diag["tau_face"], expected_tau, atol=1e-12, rtol=1e-12)
    assert np.allclose(diag["p"], expected_p, atol=1e-12, rtol=1e-12)


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


def test_q_boundary_correction_requires_exact_boundary_source() -> None:
    fx = _build_rhs_fixture()

    with pytest.raises(ValueError, match="requires an exact boundary source"):
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
            physical_boundary_mode="opposite_boundary",
            q_boundary_correction=q_boundary_correction_zero,
            q_boundary_correction_mode="all",
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
