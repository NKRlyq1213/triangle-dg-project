from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math
import csv
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
from operators.mass import mass_matrix_from_quadrature
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exchange import rhs_split_conservative_exchange

from time_integration.lsrk54 import integrate_lsrk54
from time_integration.CFL import mesh_min_altitude, cfl_dt_from_h, vmax_from_uv


@dataclass(frozen=True)
class LSRKHConvergenceConfig:
    table_name: str = "table1"
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    # Requested fixed setup from user:
    # - q(x,y,t)=sin(x-t)
    # - CFL=1
    # - final times tf=1 and tf=10
    cfl: float = 1.0
    tf_values: tuple[float, ...] = (1.0, 10.0)

    tau: float = 0.0
    use_numba: bool | None = None
    enforce_polynomial_projection: bool = True
    verbose: bool = True


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


def build_polynomial_l2_projector_from_rule(rule: dict, N: int) -> np.ndarray:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    area = reference_triangle_area()
    M = mass_matrix_from_quadrature(V, ws, area=area)
    rhs = area * (V.T * ws[None, :])
    proj_modal = np.linalg.solve(M, rhs)
    return V @ proj_modal


def q_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(x - t)


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.ones_like(x), np.ones_like(y)


def weighted_l2_error(err: np.ndarray, rule: dict, face_geom: dict) -> float:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)

    val = 0.0
    for k in range(err.shape[0]):
        val += area[k] * np.dot(ws, err[k] ** 2)
    return float(np.sqrt(val))


def compute_convergence_rate(errors: list[float]) -> list[float]:
    rates = [math.nan]
    for i in range(1, len(errors)):
        e_prev = float(errors[i - 1])
        e_curr = float(errors[i])
        if (not np.isfinite(e_prev)) or (not np.isfinite(e_curr)):
            rates.append(math.nan)
        elif e_prev <= 0.0 or e_curr <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e_prev / e_curr, 2.0))
    return rates


def _validate_config(config: LSRKHConvergenceConfig) -> None:
    if str(config.table_name).lower().strip() != "table1":
        raise ValueError("LSRK exchange h-convergence currently supports table_name='table1' only.")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if len(config.mesh_levels) == 0:
        raise ValueError("mesh_levels must be non-empty.")
    if any(n <= 0 for n in config.mesh_levels):
        raise ValueError("All mesh levels must be positive integers.")
    if len(config.tf_values) == 0:
        raise ValueError("tf_values must be non-empty.")
    if any(tf <= 0.0 for tf in config.tf_values):
        raise ValueError("All tf values must be positive.")


def run_lsrk_h_convergence_for_tf(
    config: LSRKHConvergenceConfig,
    tf: float,
) -> list[dict]:
    _validate_config(config)

    rule = load_table1_rule(config.order)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, config.N)
    projector = None
    if config.enforce_polynomial_projection:
        projector = build_polynomial_l2_projector_from_rule(rule, config.N)

    results: list[dict] = []

    for n in config.mesh_levels:
        t0 = perf_counter()

        VX, VY, EToV = structured_square_tri_mesh(
            nx=n,
            ny=n,
            diagonal=config.diagonal,
        )

        conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

        X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

        q0 = q_exact_sinx(X, Y, t=0.0)
        if projector is not None:
            q0 = q0 @ projector.T

        u_elem, v_elem = velocity_one_one(X, Y, t=0.0)

        geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
        face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

        hmin = mesh_min_altitude(VX, VY, EToV)
        vmax = vmax_from_uv(u_elem, v_elem)
        dt_nominal = cfl_dt_from_h(cfl=config.cfl, h=hmin, N=config.N, vmax=vmax)

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
                tau=config.tau,
                compute_mismatches=False,
                return_diagnostics=False,
                use_numba=config.use_numba,
                state_projector=projector,
            )
            return total_rhs

        qf, tf_used, nsteps = integrate_lsrk54(
            rhs=rhs,
            q0=q0,
            t0=0.0,
            tf=float(tf),
            dt=dt_nominal,
        )

        tf_target = float(tf)
        reached_tf = bool(np.isclose(tf_used, tf_target, atol=1e-12, rtol=1e-12))

        q_exact_at_stop = q_exact_sinx(X, Y, t=tf_used)
        err_at_stop = qf - q_exact_at_stop
        L2_error_at_stop = weighted_l2_error(err_at_stop, rule, face_geom)
        Linf_error_at_stop = float(np.max(np.abs(err_at_stop)))

        if reached_tf:
            q_exact_final = q_exact_sinx(X, Y, t=tf_target)
            err = qf - q_exact_final
            L2_error = weighted_l2_error(err, rule, face_geom)
            Linf_error = float(np.max(np.abs(err)))
        else:
            L2_error = math.nan
            Linf_error = math.nan

        h = 1.0 / float(n)
        K = int(EToV.shape[0])
        Np = int(q0.shape[1])
        total_dof = K * Np
        elapsed = perf_counter() - t0

        row = {
            "nx": int(n),
            "ny": int(n),
            "K_tri": K,
            "h": h,
            "hmin": float(hmin),
            "Np": Np,
            "total_dof": int(total_dof),
            "tf_target": tf_target,
            "tf": float(tf_used),
            "reached_tf": reached_tf,
            "cfl": float(config.cfl),
            "dt_nominal": float(dt_nominal),
            "nsteps": int(nsteps),
            "L2_error": float(L2_error),
            "Linf_error": float(Linf_error),
            "L2_error_at_stop": float(L2_error_at_stop),
            "Linf_error_at_stop": float(Linf_error_at_stop),
            "elapsed_sec": float(elapsed),
            "projection_enabled": bool(projector is not None),
        }
        results.append(row)

        if config.verbose:
            status = "ok" if reached_tf else "stopped_early"
            l2_display = L2_error if reached_tf else L2_error_at_stop
            linf_display = Linf_error if reached_tf else Linf_error_at_stop
            print(
                f"[lsrk h-study] tf={tf:>5.1f} | n={n:>3d} | K={K:>7d} | "
                f"status={status:>12s} | "
                f"L2={l2_display:.6e} | Linf={linf_display:.6e} | "
                f"steps={nsteps:>7d} | dt={dt_nominal:.3e} | time={elapsed:.2f}s"
            )

    L2_list = [r["L2_error"] for r in results]
    Linf_list = [r["Linf_error"] for r in results]

    L2_rates = compute_convergence_rate(L2_list)
    Linf_rates = compute_convergence_rate(Linf_list)

    for i, r in enumerate(results):
        r["rate_L2"] = L2_rates[i]
        r["rate_Linf"] = Linf_rates[i]

    return results


def run_lsrk_h_convergence(config: LSRKHConvergenceConfig) -> dict[float, list[dict]]:
    _validate_config(config)

    out: dict[float, list[dict]] = {}
    for tf in config.tf_values:
        out[float(tf)] = run_lsrk_h_convergence_for_tf(config, tf=float(tf))
    return out


def print_results_table(results: list[dict], title: str | None = None) -> None:
    if title is not None:
        print(title)

    header = (
        f"{'n':>6s} {'K':>9s} {'h':>12s} "
        f"{'status':>14s} "
        f"{'dt':>12s} {'steps':>8s} "
        f"{'L2_error':>14s} {'rate':>8s} "
        f"{'Linf_error':>14s} {'rate':>8s} "
        f"{'time(s)':>10s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v: float) -> str:
        return "   -   " if not np.isfinite(v) else f"{v:8.3f}"

    for r in results:
        status = "ok" if bool(r.get("reached_tf", True)) else "stopped_early"
        print(
            f"{r['nx']:6d} {r['K_tri']:9d} {r['h']:12.4e} "
            f"{status:14s} "
            f"{r['dt_nominal']:12.4e} {r['nsteps']:8d} "
            f"{r['L2_error']:14.6e} {fmt_rate(r['rate_L2'])} "
            f"{r['Linf_error']:14.6e} {fmt_rate(r['rate_Linf'])} "
            f"{r['elapsed_sec']:10.2f}"
        )


def save_results_csv(results: list[dict], filepath: str) -> None:
    if not results:
        raise ValueError("results is empty.")

    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
