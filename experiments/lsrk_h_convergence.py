from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from time import perf_counter
import math
import csv
import numpy as np
from typing import Callable

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
from operators.rhs_split_conservative_exchange import (
    rhs_split_conservative_exchange,
    build_surface_exchange_cache,
    build_volume_split_cache,
)

from time_integration.lsrk54 import integrate_lsrk54
from time_integration.lsrk54 import RK4A, RK4B, RK4C
from time_integration.CFL import mesh_min_altitude, cfl_dt_from_h, vmax_from_uv


@dataclass(frozen=True)
class LSRKHConvergenceConfig:
    table_name: str = "table1"
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    # Requested fixed setup from user:
    # - q(x,y,t)=sin(pi*(x-t))
    # - CFL=1
    # - final times tf=2*pi and tf=20*pi
    cfl: float = 1.0
    tf_values: tuple[float, ...] = (np.pi*2, np.pi*20)

    tau: float = 0.0
    use_numba: bool | None = True
    enforce_polynomial_projection: bool = True
    projection_mode: str = "post"
    projection_frequency: str = "step"
    surface_inverse_mass_mode: str = "diagonal"
    surface_backend: str = "face-major"
    use_surface_cache: bool = True
    use_sinx_rk_stage_boundary_correction: bool = False
    q_boundary_correction: Callable | None = None
    q_boundary_correction_mode: str = "inflow"
    verbose: bool = True


class SinxRKStageBoundaryCorrection:
    """
    Notebook-style boundary state evolution synchronized with LSRK stage calls.

    This tracks [g, g_t, g_tt] using the same low-storage RK coefficients and
    returns delta_qB = g_stage - g_exact(t_stage).

    The effective step size is inferred from consecutive stage-call times,
    so the correction remains consistent on the final short step.
    """

    def __init__(self, dt: float):
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive for SinxRKStageBoundaryCorrection.")

        self._dt_default = dt
        self._initialized = False
        self._stage = 0
        self._last_t = -np.inf
        self._prev_t = 0.0
        self._prev_stage = 0
        self._have_prev_call = False

        self._g = None
        self._g_t = None
        self._g_tt = None
        self._Kg = None
        self._Kgt = None
        self._Kgtt = None

    def _reset(self, x_face: np.ndarray, t: float, expected_shape: tuple[int, int, int]) -> None:
        phase = np.pi * (x_face - float(t))

        self._g = np.sin(phase)
        self._g_t = -np.pi * np.cos(phase)
        self._g_tt = -(np.pi**2) * np.sin(phase)

        self._Kg = np.zeros(expected_shape, dtype=float)
        self._Kgt = np.zeros(expected_shape, dtype=float)
        self._Kgtt = np.zeros(expected_shape, dtype=float)

        self._stage = 0
        self._last_t = float(t)
        self._prev_t = float(t)
        self._prev_stage = 0
        self._have_prev_call = False
        self._initialized = True

    def _infer_step_dt(
        self,
        *,
        t_prev: float,
        t_curr: float,
        stage_prev: int,
        stage_curr: int,
    ) -> float:
        c_prev = float(RK4C[stage_prev])
        c_curr = float(RK4C[stage_curr])

        if stage_curr > stage_prev:
            frac = c_curr - c_prev
        else:
            frac = (1.0 - c_prev) + c_curr

        if frac <= 0.0:
            return float(self._dt_default)

        dt_step = (float(t_curr) - float(t_prev)) / frac
        if (not np.isfinite(dt_step)) or dt_step <= 0.0:
            return float(self._dt_default)
        return float(dt_step)

    def _advance_one_stage(self, x_face: np.ndarray, *, t_stage: float, stage: int, dt_step: float) -> None:
        g_ttt = (np.pi**3) * np.cos(np.pi * (x_face - float(t_stage)))

        self._Kg *= RK4A[stage]
        self._Kgt *= RK4A[stage]
        self._Kgtt *= RK4A[stage]

        self._Kg += dt_step * self._g_t
        self._Kgt += dt_step * self._g_tt
        self._Kgtt += dt_step * g_ttt

        self._g += RK4B[stage] * self._Kg
        self._g_t += RK4B[stage] * self._Kgt
        self._g_tt += RK4B[stage] * self._Kgtt

    def __call__(
        self,
        x_face: np.ndarray,
        y_face: np.ndarray,
        t: float,
        qM: np.ndarray,
        ndotV: np.ndarray,
        is_boundary: np.ndarray,
        q_boundary_exact: np.ndarray,
    ) -> np.ndarray:
        del y_face, qM, ndotV, is_boundary

        t = float(t)
        x_face = np.asarray(x_face, dtype=float)
        q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)

        if x_face.shape != q_boundary_exact.shape:
            raise ValueError("x_face and q_boundary_exact must share shape (K, 3, Nfp).")

        if (
            (not self._initialized)
            or (t < self._last_t - 1e-14)
            or (self._g is None)
            or (self._g.shape != q_boundary_exact.shape)
        ):
            self._reset(x_face=x_face, t=t, expected_shape=q_boundary_exact.shape)

        if self._have_prev_call:
            prev_stage = int(self._prev_stage)
            curr_stage = int(self._stage)

            if curr_stage != ((prev_stage + 1) % 5):
                self._reset(x_face=x_face, t=t, expected_shape=q_boundary_exact.shape)
            else:
                dt_step = self._infer_step_dt(
                    t_prev=float(self._prev_t),
                    t_curr=t,
                    stage_prev=prev_stage,
                    stage_curr=curr_stage,
                )
                self._advance_one_stage(
                    x_face=x_face,
                    t_stage=float(self._prev_t),
                    stage=prev_stage,
                    dt_step=dt_step,
                )

        delta = self._g - q_boundary_exact

        self._prev_t = t
        self._prev_stage = int(self._stage)
        self._have_prev_call = True

        self._stage = (self._stage + 1) % 5
        self._last_t = t

        return delta


def build_sinx_rk_stage_boundary_correction(dt: float) -> Callable:
    return SinxRKStageBoundaryCorrection(dt=dt)


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


def build_projected_inverse_mass_from_rule(rule: dict, N: int) -> np.ndarray:
    projector = build_polynomial_l2_projector_from_rule(rule, N)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    if projector.shape != (ws.size, ws.size):
        raise ValueError("Projected inverse-mass size must be (Np, Np).")
    if np.any(ws <= 0.0):
        raise ValueError("rule['ws'] must be strictly positive.")

    inv_ws = 1.0 / ws
    return projector * inv_ws[None, :]


def q_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(np.pi * (x - t))


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.ones_like(x), np.zeros_like(y)


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
    projection_mode = str(config.projection_mode).strip().lower()
    if projection_mode not in ("both", "pre", "post"):
        raise ValueError("projection_mode must be one of: 'both', 'pre', 'post'.")
    projection_frequency = str(config.projection_frequency).strip().lower()
    if projection_frequency not in ("rhs", "step"):
        raise ValueError("projection_frequency must be one of: 'rhs', 'step'.")
    surface_inverse_mass_mode = str(config.surface_inverse_mass_mode).strip().lower()
    if surface_inverse_mass_mode not in ("diagonal", "projected"):
        raise ValueError("surface_inverse_mass_mode must be one of: 'diagonal', 'projected'.")
    q_boundary_correction_mode = str(config.q_boundary_correction_mode).strip().lower()
    if q_boundary_correction_mode not in ("inflow", "boundary", "all"):
        raise ValueError("q_boundary_correction_mode must be one of: 'inflow', 'boundary', 'all'.")
    if config.q_boundary_correction is not None and (not callable(config.q_boundary_correction)):
        raise ValueError("q_boundary_correction must be callable or None.")
    if config.use_sinx_rk_stage_boundary_correction and config.q_boundary_correction is not None:
        raise ValueError(
            "Use either use_sinx_rk_stage_boundary_correction=True or q_boundary_correction callable, not both."
        )


def run_lsrk_h_convergence_for_tf(
    config: LSRKHConvergenceConfig,
    tf: float,
) -> list[dict]:
    _validate_config(config)

    projection_mode = str(config.projection_mode).strip().lower()
    projection_frequency = str(config.projection_frequency).strip().lower()
    surface_inverse_mass_mode = str(config.surface_inverse_mass_mode).strip().lower()

    rule = load_table1_rule(config.order)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, config.N)
    projector = None
    projector_t = None
    if config.enforce_polynomial_projection:
        projector = build_polynomial_l2_projector_from_rule(rule, config.N)
        projector_t = np.ascontiguousarray(projector.T, dtype=float)

    surface_inverse_mass_t = None
    if surface_inverse_mass_mode == "projected":
        surface_inverse_mass = build_projected_inverse_mass_from_rule(rule, config.N)
        surface_inverse_mass_t = np.ascontiguousarray(surface_inverse_mass.T, dtype=float)

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
        if projector_t is not None:
            q0_proj = np.empty_like(q0)
            np.matmul(q0, projector_t, out=q0_proj)
            q0 = q0_proj

        u_elem, v_elem = velocity_one_one(X, Y, t=0.0)

        geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
        face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)
        volume_split_cache = build_volume_split_cache(
            u_elem=u_elem,
            v_elem=v_elem,
            Dr=Dr,
            Ds=Ds,
            geom=geom,
        )
        surface_cache = None
        if config.use_surface_cache:
            surface_cache = build_surface_exchange_cache(
                rule=rule,
                trace=trace,
                conn=conn,
                face_geom=face_geom,
            )

        u_face, v_face = velocity_one_one(
            np.asarray(face_geom["x_face"], dtype=float),
            np.asarray(face_geom["y_face"], dtype=float),
            t=0.0,
        )
        ndotV_precomputed = np.ascontiguousarray(
            np.asarray(face_geom["nx"], dtype=float) * np.asarray(u_face, dtype=float)
            + np.asarray(face_geom["ny"], dtype=float) * np.asarray(v_face, dtype=float),
            dtype=float,
        )
        ndotV_flat_precomputed = np.ascontiguousarray(
            ndotV_precomputed.reshape(-1, int(trace["nfp"])),
            dtype=float,
        )

        hmin = mesh_min_altitude(VX, VY, EToV)
        vmax = vmax_from_uv(u_elem, v_elem)
        dt_nominal = cfl_dt_from_h(cfl=config.cfl, h=hmin, N=config.N, vmax=vmax)

        q_boundary_correction = config.q_boundary_correction
        q_boundary_correction_kind = "none"
        if config.use_sinx_rk_stage_boundary_correction:
            q_boundary_correction = build_sinx_rk_stage_boundary_correction(dt=dt_nominal)
            q_boundary_correction_kind = "sinx_rk_stage"
        elif q_boundary_correction is not None:
            q_boundary_correction_kind = "custom"

        projector_for_rhs_t = projector_t if (projector_t is not None and projection_frequency == "rhs") else None

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
                state_projector_T=projector_for_rhs_t,
                state_projector_mode=projection_mode,
                surface_backend=config.surface_backend,
                surface_cache=surface_cache,
                ndotV_precomputed=ndotV_precomputed,
                ndotV_flat_precomputed=ndotV_flat_precomputed,
                volume_split_cache=volume_split_cache,
                surface_inverse_mass_T=surface_inverse_mass_t,
                q_boundary_correction=q_boundary_correction,
                q_boundary_correction_mode=config.q_boundary_correction_mode,
            )
            return total_rhs

        post_step_transform = None
        if projector_t is not None and projection_frequency == "step":
            proj_buffers = (np.empty_like(q0), np.empty_like(q0))
            proj_buffer_idx = 0

            def post_step_transform(t_step: float, q_step: np.ndarray) -> np.ndarray:
                nonlocal proj_buffer_idx
                out = proj_buffers[proj_buffer_idx]
                proj_buffer_idx = 1 - proj_buffer_idx
                np.matmul(q_step, projector_t, out=out)
                return out

        qf, tf_used, nsteps = integrate_lsrk54(
            rhs=rhs,
            q0=q0,
            t0=0.0,
            tf=float(tf),
            dt=dt_nominal,
            post_step_transform=post_step_transform,
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
            "projection_mode": projection_mode,
            "projection_frequency": projection_frequency,
            "surface_inverse_mass_mode": surface_inverse_mass_mode,
            "q_boundary_correction_kind": q_boundary_correction_kind,
            "q_boundary_correction_mode": str(config.q_boundary_correction_mode).strip().lower(),
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


def run_lsrk_h_convergence_compare_qb_correction(
    config: LSRKHConvergenceConfig,
) -> dict[str, dict[float, list[dict]]]:
    """
    One-click comparison for no-correction vs notebook-style RK-stage qB correction.
    """
    if config.q_boundary_correction is not None:
        raise ValueError(
            "run_lsrk_h_convergence_compare_qb_correction requires q_boundary_correction=None."
        )

    baseline_config = replace(
        config,
        use_sinx_rk_stage_boundary_correction=False,
        q_boundary_correction=None,
    )
    rkstage_config = replace(
        config,
        use_sinx_rk_stage_boundary_correction=True,
        q_boundary_correction=None,
    )

    return {
        "baseline": run_lsrk_h_convergence(baseline_config),
        "rk_stage_correction": run_lsrk_h_convergence(rkstage_config),
    }


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
