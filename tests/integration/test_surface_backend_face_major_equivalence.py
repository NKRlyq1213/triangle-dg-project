from __future__ import annotations

import numpy as np
import pytest

from data.table1_rules import load_table1_rule
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from operators.rhs_split_conservative_exchange import (
    build_surface_exchange_cache,
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


def test_face_major_surface_backend_matches_legacy() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=4, ny=3, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    K = int(EToV.shape[0])
    Np = int(len(rule["ws"]))

    rng = np.random.default_rng(20260411)
    q_elem = rng.normal(size=(K, Np))

    surface_cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
    )

    rhs_legacy, _ = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="legacy",
        surface_cache=surface_cache,
    )

    rhs_face_major, _ = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=surface_cache,
    )

    assert np.allclose(rhs_legacy, rhs_face_major, atol=1e-12, rtol=1e-12)


def test_face_major_surface_backend_numba_matches_numpy() -> None:
    pytest.importorskip("numba")

    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=4, ny=3, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    K = int(EToV.shape[0])
    Np = int(len(rule["ws"]))

    rng = np.random.default_rng(20260411)
    q_elem = rng.normal(size=(K, Np))

    surface_cache = build_surface_exchange_cache(
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
    )

    rhs_face_major_numpy, _ = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=False,
        surface_backend="face-major",
        surface_cache=surface_cache,
    )

    rhs_face_major_numba, _ = surface_term_from_exchange(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_boundary_sinx,
        velocity=velocity_one_one,
        t=0.125,
        tau=0.0,
        compute_mismatches=False,
        return_diagnostics=False,
        use_numba=True,
        surface_backend="face-major",
        surface_cache=surface_cache,
    )

    assert np.allclose(rhs_face_major_numpy, rhs_face_major_numba, atol=1e-12, rtol=1e-12)
