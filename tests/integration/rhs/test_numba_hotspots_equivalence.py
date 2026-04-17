from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import build_projected_inverse_mass_from_rule
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.reference_triangle import reference_triangle_area
from operators.exchange import evaluate_all_face_values
from operators.rhs_split_conservative_exact_trace import surface_term_from_exact_trace
from operators.rhs_split_conservative_exchange import (
    _build_boundary_state_from_opposite_boundary,
    _build_boundary_state_from_periodic_vmap,
    _lift_surface_penalty_to_volume,
    build_surface_exchange_cache,
)
from operators.trace_policy import build_trace_policy


def q_exact_profile(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.sin(x - t) * np.cos(y + 0.5 * t)


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

    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=2, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)
    X_nodes, Y_nodes = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    K = int(EToV.shape[0])
    Np = int(np.asarray(rule["rs"], dtype=float).shape[0])
    rng = np.random.default_rng(20260418)
    q_elem = rng.normal(size=(K, Np))

    return {
        "rule": rule,
        "trace": trace,
        "VX": VX,
        "VY": VY,
        "EToV": EToV,
        "conn": conn,
        "face_geom": face_geom,
        "X_nodes": X_nodes,
        "Y_nodes": Y_nodes,
        "q_elem": q_elem,
    }


def test_evaluate_all_face_values_embedded_numba_matches_numpy() -> None:
    fx = _build_fixture()
    q_numpy = evaluate_all_face_values(fx["q_elem"], fx["trace"], use_numba=False)
    q_numba = evaluate_all_face_values(fx["q_elem"], fx["trace"], use_numba=True)
    assert np.allclose(q_numba, q_numpy, atol=0.0, rtol=0.0)


def test_opposite_boundary_state_numba_matches_numpy() -> None:
    fx = _build_fixture()
    cache = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
    )

    q_numpy = _build_boundary_state_from_opposite_boundary(
        q_elem=fx["q_elem"],
        cache=cache,
        use_numba=False,
    )
    q_numba = _build_boundary_state_from_opposite_boundary(
        q_elem=fx["q_elem"],
        cache=cache,
        use_numba=True,
    )
    assert np.allclose(q_numba, q_numpy, atol=0.0, rtol=0.0)


def test_periodic_vmap_boundary_state_numba_matches_numpy() -> None:
    fx = _build_fixture()
    cache = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        X_nodes=fx["X_nodes"],
        Y_nodes=fx["Y_nodes"],
    )

    q_numpy = _build_boundary_state_from_periodic_vmap(
        q_elem=fx["q_elem"],
        cache=cache,
        use_numba=False,
    )
    q_numba = _build_boundary_state_from_periodic_vmap(
        q_elem=fx["q_elem"],
        cache=cache,
        use_numba=True,
    )
    assert np.allclose(q_numba, q_numpy, atol=0.0, rtol=0.0)


def test_projected_inverse_mass_lifting_numba_matches_numpy() -> None:
    fx = _build_fixture()
    cache = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
    )

    nfp = int(fx["trace"]["nfp"])
    rng = np.random.default_rng(20260419)
    p = rng.normal(size=(int(cache["K"]), 3, nfp))

    projected_inverse_mass = build_projected_inverse_mass_from_rule(fx["rule"], 4)
    surface_inverse_mass_t = np.ascontiguousarray(projected_inverse_mass.T, dtype=float)

    lifted_numpy = _lift_surface_penalty_to_volume(
        p=p,
        cache=cache,
        use_numba=False,
        surface_inverse_mass_T=surface_inverse_mass_t,
        lift_scale=1.0,
    )
    lifted_numba = _lift_surface_penalty_to_volume(
        p=p,
        cache=cache,
        use_numba=True,
        surface_inverse_mass_T=surface_inverse_mass_t,
        lift_scale=1.0,
    )
    assert np.allclose(lifted_numba, lifted_numpy, atol=1e-12, rtol=1e-12)

    strict_scale = float(reference_triangle_area())
    lifted_numpy_scaled = _lift_surface_penalty_to_volume(
        p=p,
        cache=cache,
        use_numba=False,
        surface_inverse_mass_T=surface_inverse_mass_t,
        lift_scale=strict_scale,
    )
    lifted_numba_scaled = _lift_surface_penalty_to_volume(
        p=p,
        cache=cache,
        use_numba=True,
        surface_inverse_mass_T=surface_inverse_mass_t,
        lift_scale=strict_scale,
    )
    assert np.allclose(lifted_numba_scaled, lifted_numpy_scaled, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "boundary_mode",
    ("exact_qb", "opposite_boundary", "periodic_vmap"),
)
def test_exact_trace_surface_term_numba_matches_numpy(boundary_mode: str) -> None:
    fx = _build_fixture()
    use_periodic_map = boundary_mode == "periodic_vmap"
    surface_cache = build_surface_exchange_cache(
        rule=fx["rule"],
        trace=fx["trace"],
        conn=fx["conn"],
        face_geom=fx["face_geom"],
        X_nodes=fx["X_nodes"] if use_periodic_map else None,
        Y_nodes=fx["Y_nodes"] if use_periodic_map else None,
    )

    surface_numpy, diag_numpy = surface_term_from_exact_trace(
        q_elem=fx["q_elem"],
        rule=fx["rule"],
        trace=fx["trace"],
        VX=fx["VX"],
        VY=fx["VY"],
        EToV=fx["EToV"],
        q_exact=q_exact_profile,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        tau_interior=0.2,
        tau_qb=0.75,
        face_geom=fx["face_geom"],
        physical_boundary_mode=boundary_mode,
        use_numba=False,
        conn=fx["conn"],
        surface_cache=surface_cache,
    )
    surface_numba, diag_numba = surface_term_from_exact_trace(
        q_elem=fx["q_elem"],
        rule=fx["rule"],
        trace=fx["trace"],
        VX=fx["VX"],
        VY=fx["VY"],
        EToV=fx["EToV"],
        q_exact=q_exact_profile,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        tau_interior=0.2,
        tau_qb=0.75,
        face_geom=fx["face_geom"],
        physical_boundary_mode=boundary_mode,
        use_numba=True,
        conn=fx["conn"],
        surface_cache=surface_cache,
    )

    assert np.allclose(surface_numba, surface_numpy, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        np.asarray(diag_numba["qP"], dtype=float),
        np.asarray(diag_numpy["qP"], dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        np.asarray(diag_numba["p"], dtype=float),
        np.asarray(diag_numpy["p"], dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )
