from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from operators.rhs_split_conservative_exact_trace import surface_term_from_exact_trace
from operators.trace_policy import build_trace_policy


def q_exact_zero(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    del y, t
    return np.zeros_like(x, dtype=float)


def velocity_one_one(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    del t
    return np.ones_like(x, dtype=float), np.ones_like(y, dtype=float)


def correction_all_ones(
    x_face: np.ndarray,
    y_face: np.ndarray,
    t: float,
    qM: np.ndarray,
    ndotV: np.ndarray,
    is_boundary: np.ndarray,
    q_boundary_exact: np.ndarray,
) -> np.ndarray:
    del x_face, y_face, t, qM, ndotV, is_boundary
    return np.ones_like(q_boundary_exact, dtype=float)


def test_exact_trace_qb_correction_applies_on_interior_faces() -> None:
    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)

    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=2, diagonal="anti")
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    K = int(EToV.shape[0])
    Np = int(np.asarray(rule["rs"], dtype=float).shape[0])
    q_elem = np.zeros((K, Np), dtype=float)

    _, diag = surface_term_from_exact_trace(
        q_elem=q_elem,
        rule=rule,
        trace=trace,
        VX=VX,
        VY=VY,
        EToV=EToV,
        q_exact=q_exact_zero,
        velocity=velocity_one_one,
        t=0.0,
        tau=0.0,
        face_geom=face_geom,
        q_boundary_correction=correction_all_ones,
        q_boundary_correction_mode="all",
    )

    qP_exact = np.asarray(diag["qP_exact"], dtype=float)
    qP = np.asarray(diag["qP"], dtype=float)
    is_boundary = np.asarray(conn["is_boundary"], dtype=bool)

    assert np.any(~is_boundary)
    assert np.allclose(qP_exact, 0.0)
    assert np.allclose(qP[is_boundary], 1.0)
    assert np.allclose(qP[~is_boundary], 1.0)
