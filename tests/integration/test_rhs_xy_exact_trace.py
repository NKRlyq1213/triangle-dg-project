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


def q_exact_xy(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    # independent of t
    return x - y


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def test_rhs_xy_exact_trace_is_zero():
    """
    Phase-1 exact-trace consistency test.

    PDE:
        q_t + div((1,1) q) = 0

    Exact field:
        q(x,y,t) = x - y

    Since q_x = 1 and q_y = -1,
        q_t = -(q_x + q_y) = 0.
    """
    order = 4
    N = 4

    # Table 1 rule and embedded trace
    rule = load_table1_rule(order)
    trace = build_trace_policy(rule)

    # reference differentiation operators on the rule points
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 2x2 square -> 8 triangles
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")

    # physical nodal coordinates on all elements
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # exact solution sampled at volume nodes
    q_elem = q_exact_xy(X, Y, t=0.0)

    # constant velocity field V=(1,1) sampled at volume nodes
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
        q_exact=q_exact_xy,
        velocity=velocity_one_one,
        t=0.0,
        tau=0.0,  # pure upwind
        face_geom=face_geom,
    )

    volume_rhs = diag["volume_rhs"]
    surface_rhs = diag["surface_rhs"]
    total_rhs = diag["total_rhs"]
    p = diag["p"]

    max_vol = np.max(np.abs(volume_rhs))
    max_surf = np.max(np.abs(surface_rhs))
    max_total = np.max(np.abs(total_rhs))
    max_p = np.max(np.abs(p))

    print("test_rhs_xy_exact_trace_is_zero")
    print("  max |volume_rhs| =", max_vol)
    print("  max |surface_rhs| =", max_surf)
    print("  max |total_rhs|   =", max_total)
    print("  max |p|           =", max_p)

    # For q = x - y and V = (1,1), exact RHS is zero.
    assert np.allclose(volume_rhs, 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(surface_rhs, 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(total_rhs, 0.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(p, 0.0, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    test_rhs_xy_exact_trace_is_zero()
    print("test_rhs_xy_exact_trace: passed")