from __future__ import annotations

import numpy as np

from data.table1_rules import load_table1_rule
from geometry.reference_triangle import reference_triangle_area
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.connectivity import build_face_connectivity

from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exchange import rhs_split_conservative_exchange


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


def q_exact_siny(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.sin(y - t)


def qt_exact_siny(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return -np.cos(y - t)


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def weighted_l2_error(err: np.ndarray, rule: dict, face_geom: dict) -> float:
    """
    Physical weighted L2 norm:
        sqrt( sum_k |T_k| * sum_i w_i * err_{k,i}^2 )
    """
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)

    val = 0.0
    for k in range(err.shape[0]):
        val += area[k] * np.dot(ws, err[k] ** 2)
    return float(np.sqrt(val))


def test_rhs_siny_exchange_matches_qt():
    """
    Exchange-based spatial RHS test.

    PDE:
        q_t + div((1,1) q) = 0

    Exact field:
        q(x,y,t) = sin(y - t)

    Since
        q_t = -cos(y-t),
        q_x =  0,
        q_y =  cos(y-t),

    we have:
        q_t + q_x + q_y = 0.
    """
    order = 4
    N = 4
    t0 = 0.0

    # Table 1 rule and embedded trace
    rule = load_table1_rule(order)
    trace = build_trace_policy(rule)

    # reference differentiation operators
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 2x2 square -> 8 triangles
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")

    # actual interior exchange connectivity
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    # physical nodal coordinates
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # exact volume state
    q_elem = q_exact_siny(X, Y, t=t0)

    # constant velocity V=(1,1)
    u_elem = np.ones_like(q_elem)
    v_elem = np.ones_like(q_elem)

    # geometry
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    rhs, diag = rhs_split_conservative_exchange(
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        rule=rule,
        trace=trace,
        conn=conn,
        face_geom=face_geom,
        q_boundary=q_exact_siny,
        velocity=velocity_one_one,
        t=t0,
        tau=0.0,  # pure upwind
    )
    volume_rhs = diag["volume_rhs"]
    surface_rhs = diag["surface_rhs"]
    total_rhs = diag["total_rhs"]
    p = diag["p"]

    print(p)
    
    mismatches = diag["interior_mismatches"]
    max_interior_mismatch = 0.0 if len(mismatches) == 0 else max(
        item["max_abs_mismatch"] for item in mismatches
    )

    qt_exact = qt_exact_siny(X, Y, t=t0)

    err_vol = volume_rhs - qt_exact
    err_total = total_rhs - qt_exact

    max_surface = np.max(np.abs(surface_rhs))
    max_p = np.max(np.abs(p))
    max_err_vol = np.max(np.abs(err_vol))
    max_err_total = np.max(np.abs(err_total))

    l2_err_vol = weighted_l2_error(err_vol, rule, face_geom)
    l2_err_total = weighted_l2_error(err_total, rule, face_geom)

    print("test_rhs_siny_exchange_matches_qt")
    print("  max interior mismatch =", max_interior_mismatch)
    print("  max |surface_rhs|      =", max_surface)
    print("  max |p|                =", max_p)
    print("  max |volume_rhs-qt|    =", max_err_vol)
    print("  max |total_rhs-qt|     =", max_err_total)
    print("  weighted L2(volume-qt) =", l2_err_vol)
    print("  weighted L2(total-qt)  =", l2_err_total)

    # interior exchange should reproduce exact traces on this smooth exact field
    assert max_interior_mismatch < 1e-12

    # exact boundary + exact interior exchange => penalty should vanish
    assert np.allclose(surface_rhs, 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(p, 0.0, atol=1e-12, rtol=1e-12)

    # non-polynomial test: do NOT require machine-precision equality to qt
    # use error thresholds instead
    assert max_err_total < 1e-4, f"max total RHS error too large: {max_err_total}"
    assert l2_err_total < 2e-5, f"L2 total RHS error too large: {l2_err_total}"


if __name__ == "__main__":
    test_rhs_siny_exchange_matches_qt()
    print("test_rhs_siny_exchange: passed")