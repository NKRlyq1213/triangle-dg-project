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

from time_integration.lsrk54 import integrate_lsrk54
from time_integration.CFL import mesh_min_altitude, cfl_dt_from_h, vmax_from_uv


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


def empirical_rates(errors: list[float]) -> list[float]:
    rates = [np.nan]
    for i in range(1, len(errors)):
        e_prev = errors[i - 1]
        e_curr = errors[i]
        rates.append(np.log(e_prev / e_curr) / np.log(2.0))
    return rates


def solve_sinx_problem(cfl: float, tf: float):
    """
    Solve the exchange-based semi-discrete DG system with LSRK54 for

        q = sin(x-t), V=(1,1),

    using interior exchange with exact boundary data.

    Time stepping
    -------------
    The nominal step size is chosen by

        dt_nominal = CFL * hmin / (N^2 * vmax),

    and the integrator automatically takes a final short step to land exactly at tf.
    """
    order = 4
    N = 4
    t0 = 0.0

    # Table 1 rule and embedded trace
    rule = load_table1_rule(order)
    trace = build_trace_policy(rule)

    # reference differentiation operators
    Dr, Ds = build_reference_diff_operators_from_rule(rule, N)

    # Fixed mesh for temporal-order study.
    # For exchange without state projection, n=8 is stable over tf=1.0 and
    # still fine enough for temporal-rate measurement against a finer-CFL reference.
    VX, VY, EToV = structured_square_tri_mesh(nx=16, ny=16, diagonal="anti")
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    # physical nodal coordinates
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # initial state q(x,y,0) = sin(x)
    q0 = q_exact_sinx(X, Y, t=t0)

    # constant velocity
    u_elem = np.ones_like(q0)
    v_elem = np.ones_like(q0)

    # geometry
    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    # CFL-based nominal dt
    hmin = mesh_min_altitude(VX, VY, EToV)
    vmax = vmax_from_uv(u_elem, v_elem)
    dt_nominal = cfl_dt_from_h(cfl=cfl, h=hmin, N=N, vmax=vmax)

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        total_rhs, _ = rhs_split_conservative_exchange(
            q_elem=q,
            u_elem=u_elem,
            v_elem=v_elem,
            Dr=Dr,
            Ds=Ds,
            geom=geom,
            rule=rule,
            trace=trace,
            conn=conn,
            face_geom=face_geom,
            q_boundary=q_exact_sinx,
            velocity=velocity_one_one,
            t=t,
            tau=0.0,
            compute_mismatches=False,
            return_diagnostics=False,
            use_numba=False,
            surface_backend="legacy",
        )
        return total_rhs

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=t0,
        tf=tf,
        dt=dt_nominal,
    )

    return {
        "qf": qf,
        "X": X,
        "Y": Y,
        "rule": rule,
        "face_geom": face_geom,
        "tf_used": tf_used,
        "nsteps": nsteps,
        "dt_nominal": dt_nominal,
        "hmin": hmin,
        "vmax": vmax,
        "cfl": cfl,
    }


def test_lsrk54_temporal_order_sinx():
    """
    Temporal order test for LSRK54.

    IMPORTANT
    ---------
    We do NOT compare against the PDE exact solution to estimate temporal order,
    because spatial discretization error would contaminate the slope.

    Instead, we compare against a much smaller-CFL numerical reference solution,
    while keeping the spatial discretization fixed.
    """
    tf = 1.0

    # CFL sequence; halving CFL halves nominal dt
    cfl_list = [1.0 , 0.5] #, 0.25, 0.125]

    # much smaller reference CFL
    cfl_ref = cfl_list[-1] / 2.0
    
    # reference solution
    ref = solve_sinx_problem(cfl=cfl_ref, tf=tf)
    q_ref = ref["qf"]
    rule = ref["rule"]
    face_geom = ref["face_geom"]
    # exact solution at tf is q(x,y,tf) = sin(x-tf), but we use the small-CFL numerical solution as reference to avoid spatial error contamination in the convergence rates.
    
    max_errors = []
    l2_errors = []
    dt_nominals = []

    print("test_lsrk54_temporal_order_sinx")
    print(f"{'CFL':>10s} {'dt_nominal':>14s} {'nsteps':>8s} {'max_err':>16s} {'L2_err':>16s}")

    for cfl in cfl_list:
        out = solve_sinx_problem(cfl=cfl, tf=tf)
        q_num = out["qf"]
        
        err = q_num - q_ref
        max_err = float(np.max(np.abs(err)))
        l2_err = weighted_l2_error(err, rule, face_geom)

        max_errors.append(max_err)
        l2_errors.append(l2_err)
        dt_nominals.append(out["dt_nominal"])

        print(
            f"{cfl:10.4f} "
            f"{out['dt_nominal']:14.6e} "
            f"{out['nsteps']:8d} "
            f"{max_err:16.8e} "
            f"{l2_err:16.8e}"
        )

    max_rates = empirical_rates(max_errors)
    l2_rates = empirical_rates(l2_errors)

    print("\nEmpirical rates:")
    print(f"{'dt_nominal':>14s} {'rate_max':>12s} {'rate_L2':>12s}")
    for i, dt_nom in enumerate(dt_nominals):
        rmax = max_rates[i]
        rl2 = l2_rates[i]
        print(
            f"{dt_nom:14.6e} "
            f"{('-' if not np.isfinite(rmax) else f'{rmax:.6f}'):>12s} "
            f"{('-' if not np.isfinite(rl2) else f'{rl2:.6f}'):>12s}"
        )

    # Expect approximately 4th-order temporal convergence.
    assert max_rates[-1] > 3.7, f"Final max-norm rate too low: {max_rates[-1]}"
    assert l2_rates[-1] > 3.7, f"Final L2 rate too low: {l2_rates[-1]}"

    #assert max_rates[-2] > 3.5, f"Penultimate max-norm rate too low: {max_rates[-2]}"
    #assert l2_rates[-2] > 3.5, f"Penultimate L2 rate too low: {l2_rates[-2]}"


if __name__ == "__main__":
    test_lsrk54_temporal_order_sinx()
    print("test_lsrk54_temporal_order_sinx: passed")