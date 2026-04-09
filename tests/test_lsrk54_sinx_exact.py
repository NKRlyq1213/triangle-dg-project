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

        q(x,y,t) = sin(x - t),   V=(1,1)

    using:
    - actual interior exchange
    - exact physical boundary data
    - CFL-based nominal dt
    - automatic final short step to land exactly at tf
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

    # actual exchange connectivity
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    # physical nodal coordinates
    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    # initial state q(x,y,0) = sin(x)
    q0 = q_exact_sinx(X, Y, t=t0)

    # constant velocity V=(1,1) sampled at nodes
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
        )
        return total_rhs

    qf, tf_used, nsteps = integrate_lsrk54(
        rhs=rhs,
        q0=q0,
        t0=t0,
        tf=tf,
        dt=dt_nominal,
    )

    q_exact_final = q_exact_sinx(X, Y, t=tf_used)

    return {
        "qf": qf,
        "q_exact_final": q_exact_final,
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


def run_lsrk54_sinx_study(
    compare_mode: str = "exact",
    tf: float = 1.0,
    cfl_list: list[float] | None = None,
    reference_cfl_factor: float = 8.0,
):
    """
    Unified study driver.

    Parameters
    ----------
    compare_mode : str
        "exact"     -> compare q_num against exact solution
        "reference" -> compare q_num against a smaller-CFL LSRK reference solution
    tf : float
        Final time
    cfl_list : list[float] | None
        CFL sequence
    reference_cfl_factor : float
        Only used when compare_mode == "reference".
        The reference CFL is chosen as:
            cfl_ref = min(cfl_list) / reference_cfl_factor

    Returns
    -------
    dict
        Contains tables and errors.
    """
    if cfl_list is None:
        cfl_list = [1.00, 0.50, 0.25, 0.125]

    compare_mode = str(compare_mode).lower().strip()
    if compare_mode not in {"exact", "reference"}:
        raise ValueError("compare_mode must be either 'exact' or 'reference'.")

    max_errors = []
    l2_errors = []
    dt_nominals = []
    rows = []

    reference_solution = None
    cfl_ref = None

    if compare_mode == "reference":
        cfl_ref = min(cfl_list) / float(reference_cfl_factor)
        ref = solve_sinx_problem(cfl=cfl_ref, tf=tf)
        reference_solution = ref["qf"]
        rule = ref["rule"]
        face_geom = ref["face_geom"]
    else:
        ref = solve_sinx_problem(cfl=cfl_list[0], tf=tf)
        rule = ref["rule"]
        face_geom = ref["face_geom"]

    for cfl in cfl_list:
        out = solve_sinx_problem(cfl=cfl, tf=tf)

        if compare_mode == "exact":
            target = out["q_exact_final"]
        else:
            target = reference_solution

        err = out["qf"] - target
        max_err = float(np.max(np.abs(err)))
        l2_err = weighted_l2_error(err, rule, face_geom)

        max_errors.append(max_err)
        l2_errors.append(l2_err)
        dt_nominals.append(out["dt_nominal"])

        rows.append(
            {
                "cfl": cfl,
                "dt_nominal": out["dt_nominal"],
                "nsteps": out["nsteps"],
                "max_err": max_err,
                "l2_err": l2_err,
                "tf_used": out["tf_used"],
            }
        )

        # integrator should land exactly at tf
        assert abs(out["tf_used"] - tf) < 1e-14, f"Did not land on tf exactly: {out['tf_used']}"

        # basic sanity
        assert np.isfinite(max_err), "max_err is not finite"
        assert np.isfinite(l2_err), "l2_err is not finite"

    max_rates = empirical_rates(max_errors)
    l2_rates = empirical_rates(l2_errors)

    return {
        "compare_mode": compare_mode,
        "tf": tf,
        "cfl_list": cfl_list,
        "cfl_ref": cfl_ref,
        "rows": rows,
        "dt_nominals": dt_nominals,
        "max_errors": max_errors,
        "l2_errors": l2_errors,
        "max_rates": max_rates,
        "l2_rates": l2_rates,
    }


def print_study_results(result: dict) -> None:
    mode = result["compare_mode"]

    if mode == "exact":
        print("LSRK54 study against exact solution")
    else:
        print("LSRK54 study against smaller-CFL LSRK reference")
        print(f"reference CFL = {result['cfl_ref']:.8f}")

    print(f"{'CFL':>10s} {'dt_nominal':>14s} {'nsteps':>8s} {'max_err':>16s} {'L2_err':>16s}")
    for row in result["rows"]:
        print(
            f"{row['cfl']:10.4f} "
            f"{row['dt_nominal']:14.6e} "
            f"{row['nsteps']:8d} "
            f"{row['max_err']:16.8e} "
            f"{row['l2_err']:16.8e}"
        )

    print("\nEmpirical rates:")
    print(f"{'dt_nominal':>14s} {'rate_max':>12s} {'rate_L2':>12s}")
    for i, dt_nom in enumerate(result["dt_nominals"]):
        rmax = result["max_rates"][i]
        rl2 = result["l2_rates"][i]
        print(
            f"{dt_nom:14.6e} "
            f"{('-' if not np.isfinite(rmax) else f'{rmax:.6f}'):>12s} "
            f"{('-' if not np.isfinite(rl2) else f'{rl2:.6f}'):>12s}"
        )


def test_lsrk54_sinx_exact():
    """
    Exact-solution comparison:
    measures total error = spatial error + temporal error
    """
    result = run_lsrk54_sinx_study(
        compare_mode="exact",
        tf=1.0,
        cfl_list=[1.00, 0.50, 0.25, 0.125],
    )
    print_study_results(result)

    # exact-comparison only asks that refinement improves overall error
    assert result["max_errors"][-1] <= result["max_errors"][0], \
        "Refining CFL did not reduce max error overall."
    assert result["l2_errors"][-1] <= result["l2_errors"][0], \
        "Refining CFL did not reduce L2 error overall."

    # implementation-level sanity thresholds
    assert result["max_errors"][-1] < 1e-3, \
        f"Smallest-CFL max error still too large: {result['max_errors'][-1]}"
    assert result["l2_errors"][-1] < 1e-4, \
        f"Smallest-CFL L2 error still too large: {result['l2_errors'][-1]}"


def test_lsrk54_sinx_reference():
    """
    Smaller-CFL reference comparison:
    closer to a pure temporal-error study
    """
    result = run_lsrk54_sinx_study(
        compare_mode="reference",
        tf=1.0,
        cfl_list=[1.00, 0.50, 0.25, 0.125],
        reference_cfl_factor=8.0,
    )
    print_study_results(result)

    # for reference-comparison, we do expect something close to 4th-order
    assert result["max_rates"][-1] > 3.7, \
        f"Final max-norm rate too low: {result['max_rates'][-1]}"
    assert result["l2_rates"][-1] > 3.7, \
        f"Final L2 rate too low: {result['l2_rates'][-1]}"

    assert result["max_rates"][-2] > 3.5, \
        f"Penultimate max-norm rate too low: {result['max_rates'][-2]}"
    assert result["l2_rates"][-2] > 3.5, \
        f"Penultimate L2 rate too low: {result['l2_rates'][-2]}"


if __name__ == "__main__":
    print("=" * 72)
    test_lsrk54_sinx_reference()

    print("=" * 72)
    test_lsrk54_sinx_exact()
    print("=" * 72)

    print("Both studies completed.")