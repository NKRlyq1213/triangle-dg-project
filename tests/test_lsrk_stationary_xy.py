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

from time_integration.lsrk54 import integrate_lsrk54


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
    return x - y


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def test_lsrk54_preserves_stationary_xy():
    """
    Stationary-solution preservation test for LSRK54.

    PDE:
        q_t + div((1,1) q) = 0

    Exact field:
        q(x,y,t) = x - y

    Since q_t = 0 exactly, the semi-discrete RHS should vanish.
    Therefore repeated LSRK steps should preserve q to roundoff.
    """
    order = 4
    N = 4
    t0 = 0.0
    tf = 1.0
    dt = 0.005

    # Table 1 rule and embedded trace
    rule = load_table1_rule(order)
    trace = build_trace_policy(rule)

    # reference differentiation operators
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # 2x2 square -> 8 triangles
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")

    # physical nodal coordinates
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # initial state
    q0 = q_exact_xy(X, Y, t=t0)

    # constant velocity
    u_elem = np.ones_like(q0)
    v_elem = np.ones_like(q0)

    # geometry
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        total_rhs, _ = rhs_split_conservative_exact_trace(
            q_elem=q,
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
            t=t,
            tau=0.0,
            face_geom=face_geom,
        )
        return total_rhs

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=t0,
        tf=tf,
        dt=dt,
    )

    q_exact_final = q_exact_xy(X, Y, t=tf_used)
    err = qf - q_exact_final

    max_err = np.max(np.abs(err))
    l2_err = np.sqrt(np.mean(err ** 2))

    print("test_lsrk54_preserves_stationary_xy")
    print("  tf_used  =", tf_used)
    print("  nsteps   =", nsteps)
    print("  max err  =", max_err)
    print("  rms err  =", l2_err)

    assert np.allclose(qf, q_exact_final, atol=1e-13, rtol=1e-13)
    assert max_err < 1e-13

if __name__ == "__main__":
    test_lsrk54_preserves_stationary_xy()
    print("test_lsrk54_preserves_stationary_xy: passed")