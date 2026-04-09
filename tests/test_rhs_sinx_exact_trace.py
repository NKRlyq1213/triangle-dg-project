from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.reference_triangle import reference_triangle_area
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh
from geometry.face_metrics import affine_face_geometry_from_mesh

from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exact_trace import rhs_split_conservative_exact_trace


def build_reference_diff_operators_from_rule(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V, Vr, Vs, ws, area=reference_triangle_area()
    )


def q_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.sin(x - t)


def qt_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return -np.cos(x - t)


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def weighted_l2_error(err: np.ndarray, rule: dict, face_geom: dict) -> float:
    """
    Elementwise weighted L2 error on the physical mesh:
        sqrt( sum_k |T_k| * sum_i w_i * err_{k,i}^2 )
    """
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)

    val = 0.0
    for k in range(err.shape[0]):
        val += area[k] * np.dot(ws, err[k] ** 2)
    return float(np.sqrt(val))


def test_rhs_sinx_exact_trace_matches_qt():
    """
    Phase-1 exact-trace consistency test.

    PDE:
        q_t + div((1,1) q) = 0

    Exact field:
        q(x,y,t) = sin(x - t)

    Since
        q_t = -cos(x-t),
        q_x =  cos(x-t),
        q_y =  0,

    we have:
        q_t + q_x + q_y = 0.
    """
    order = 4
    N = 4
    t0 = 0.0

    # Table 1 rule and embedded trace
    rule = load_table1_rule(order)
    trace = build_trace_policy(rule)

    # reference differentiation operators on the rule points
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 2x2 square -> 8 triangles
    VX, VY, EToV = structured_square_tri_mesh(nx=64, ny=64, diagonal="anti")

    # physical nodal coordinates on all elements
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # exact solution sampled at volume nodes
    q_elem = q_exact_sinx(X, Y, t=t0)

    # constant velocity field V=(1,1)
    u_elem = np.ones_like(q_elem)
    v_elem = np.ones_like(q_elem)

    # geometric factors for volume term
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])

    # geometric data for face term
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    rhs, diag = rhs_split_conservative_exact_trace(
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        rule=rule,
        trace=trace,
        VX=VX,
        VY=VY,
        EToV=EToV,
        q_exact=q_exact_sinx,
        velocity=velocity_one_one,
        t=t0,
        tau=0.0,  # pure upwind
        face_geom=face_geom,
    )

    volume_rhs = diag["volume_rhs"]
    surface_rhs = diag["surface_rhs"]
    total_rhs = diag["total_rhs"]
    p = diag["p"]

    qt_exact = qt_exact_sinx(X, Y, t=t0)

    err_vol = volume_rhs - qt_exact
    err_total = total_rhs - qt_exact

    max_surface = np.max(np.abs(surface_rhs))
    max_p = np.max(np.abs(p))
    max_err_vol = np.max(np.abs(err_vol))
    max_err_total = np.max(np.abs(err_total))

    l2_err_vol = weighted_l2_error(err_vol, rule, face_geom)
    l2_err_total = weighted_l2_error(err_total, rule, face_geom)

    print("test_rhs_sinx_exact_trace_matches_qt")
    print("  max |surface_rhs|      =", max_surface)
    print("  max |p|                =", max_p)
    print("  max |volume_rhs-qt|    =", max_err_vol)
    print("  max |total_rhs-qt|     =", max_err_total)
    print("  weighted L2(volume-qt) =", l2_err_vol)
    print("  weighted L2(total-qt)  =", l2_err_total)

    # exact-trace consistency:
    # surface term should vanish, total RHS should match exact q_t
    assert np.allclose(surface_rhs, 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(p, 0.0, atol=1e-12, rtol=1e-12)

    # both volume and total should reproduce q_t
    assert np.allclose(volume_rhs, qt_exact, atol=1e-11, rtol=1e-11)
    assert np.allclose(total_rhs, qt_exact, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    test_rhs_sinx_exact_trace_matches_qt()
    print("test_rhs_sinx_exact_trace: passed")