from __future__ import annotations

import numpy as np
import pytest

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import (
    build_projected_inverse_mass_from_rule,
    build_reference_diff_operators_from_rule,
)
from geometry.reference_triangle import reference_triangle_area
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.rhs_split_conservative_exchange import (
    _lift_surface_penalty_to_volume,
    build_surface_exchange_cache,
    rhs_split_conservative_exchange,
)
from operators.trace_policy import build_trace_policy


def q_boundary_zero(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    del y, t
    return np.zeros_like(x)


def velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    del t
    return np.ones_like(x), np.ones_like(y)


def _build_fixture() -> dict:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, 4)

    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    rng = np.random.default_rng(20260418)
    q0 = rng.normal(size=X.shape)
    u_elem, v_elem = velocity_one_one(X, Y, t=0.0)

    ndotV_precomputed = np.asarray(face_geom["nx"], dtype=float) + np.asarray(
        face_geom["ny"],
        dtype=float,
    )
    ndotV_flat_precomputed = np.asarray(
        ndotV_precomputed.reshape(-1, int(trace["nfp"])),
        dtype=float,
    )

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
        "X": X,
        "Y": Y,
        "ndotV_precomputed": ndotV_precomputed,
        "ndotV_flat_precomputed": ndotV_flat_precomputed,
    }


def _eval_rhs(
    fx: dict,
    *,
    face_order_mode: str,
    physical_boundary_mode: str,
    surface_inverse_mass_t: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    periodic = str(physical_boundary_mode).strip().lower() == "periodic_vmap"
    surface_cache = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        face_order_mode=face_order_mode,
        X_nodes=fx["X"] if periodic else None,
        Y_nodes=fx["Y"] if periodic else None,
    )

    return rhs_split_conservative_exchange(
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
        q_boundary=q_boundary_zero,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=True,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=surface_cache,
        ndotV_precomputed=fx["ndotV_precomputed"],
        ndotV_flat_precomputed=fx["ndotV_flat_precomputed"],
        surface_inverse_mass_T=surface_inverse_mass_t,
        physical_boundary_mode=physical_boundary_mode,
        face_order_mode=face_order_mode,
        X_nodes=fx["X"],
        Y_nodes=fx["Y"],
    )


def test_face_order_triangle_and_simplex_are_rhs_equivalent() -> None:
    fx = _build_fixture()

    rhs_triangle, diag_triangle = _eval_rhs(
        fx,
        face_order_mode="triangle",
        physical_boundary_mode="opposite_boundary",
    )
    rhs_simplex, diag_simplex = _eval_rhs(
        fx,
        face_order_mode="simplex",
        physical_boundary_mode="opposite_boundary",
    )

    assert np.allclose(rhs_triangle, rhs_simplex, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        np.asarray(diag_triangle["surface_rhs"], dtype=float),
        np.asarray(diag_simplex["surface_rhs"], dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )


def test_simplex_strict_requires_projected_inverse_mass() -> None:
    fx = _build_fixture()

    with pytest.raises(ValueError, match="simplex_strict"):
        _eval_rhs(
            fx,
            face_order_mode="simplex_strict",
            physical_boundary_mode="opposite_boundary",
            surface_inverse_mass_t=None,
        )


def test_simplex_strict_applies_surface_lift_scale() -> None:
    fx = _build_fixture()
    projected_inverse_mass = build_projected_inverse_mass_from_rule(fx["rule"], 4)
    surface_inverse_mass_t = np.ascontiguousarray(projected_inverse_mass.T, dtype=float)

    _, diag_simplex = _eval_rhs(
        fx,
        face_order_mode="simplex",
        physical_boundary_mode="opposite_boundary",
        surface_inverse_mass_t=surface_inverse_mass_t,
    )
    _, diag_strict = _eval_rhs(
        fx,
        face_order_mode="simplex_strict",
        physical_boundary_mode="opposite_boundary",
        surface_inverse_mass_t=surface_inverse_mass_t,
    )

    surf_simplex = np.asarray(diag_simplex["surface_rhs"], dtype=float)
    surf_strict = np.asarray(diag_strict["surface_rhs"], dtype=float)

    assert np.max(np.abs(surf_simplex)) > 1e-10
    assert float(diag_simplex["surface_lift_scale"]) == pytest.approx(1.0)
    assert float(diag_strict["surface_lift_scale"]) == pytest.approx(reference_triangle_area())
    assert not np.allclose(surf_strict, surf_simplex, atol=1e-12, rtol=1e-12)

    cache_strict = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        face_order_mode="simplex_strict",
    )
    strict_recomputed = _lift_surface_penalty_to_volume(
        p=np.asarray(diag_strict["p"], dtype=float),
        cache=cache_strict,
        use_numba=False,
        surface_inverse_mass_T=surface_inverse_mass_t,
        lift_scale=float(diag_strict["surface_lift_scale"]),
    )
    assert np.allclose(strict_recomputed, surf_strict, atol=1e-12, rtol=1e-12)


def test_periodic_vmap_regression_with_face_order_modes() -> None:
    fx = _build_fixture()

    rhs_periodic_triangle, _ = _eval_rhs(
        fx,
        face_order_mode="triangle",
        physical_boundary_mode="periodic_vmap",
    )
    rhs_periodic_simplex, _ = _eval_rhs(
        fx,
        face_order_mode="simplex",
        physical_boundary_mode="periodic_vmap",
    )

    assert np.allclose(rhs_periodic_triangle, rhs_periodic_simplex, atol=1e-12, rtol=1e-12)
