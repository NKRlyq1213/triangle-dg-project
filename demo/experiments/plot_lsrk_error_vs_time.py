from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.table1_rules import load_table1_rule
from experiments.lsrk_h_convergence import (
    _make_q_exact,
    _make_velocity,
    _resolve_test_function_spec,
    build_polynomial_l2_projector_from_rule,
    build_projected_inverse_mass_from_rule,
    build_reference_diff_operators_from_rule,
    build_sinx_rk_stage_boundary_correction,
    weighted_l2_error,
)
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.connectivity import build_face_connectivity
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.rhs_split_conservative_exchange import (
    build_surface_exchange_cache,
    build_volume_split_cache,
    rhs_split_conservative_exchange,
)
from operators.rhs_split_conservative_exact_trace import rhs_split_conservative_exact_trace
from operators.trace_policy import build_trace_policy
from time_integration.CFL import cfl_dt_from_h, mesh_min_altitude, vmax_from_uv
from time_integration.lsrk54 import BLOWUP_BREAK_ABS, lsrk54_step


def _tf_label(tf: float) -> str:
    if float(tf).is_integer():
        return str(int(tf))
    return str(tf).replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot L2 error-vs-time for one or more LSRK transport runs (linear time, log error).",
    )
    parser.add_argument("--mesh-level", type=int, default=8, help="single mesh refinement n (nx=ny=n)")
    parser.add_argument(
        "--mesh-levels",
        type=int,
        nargs="+",
        default=None,
        help="multiple mesh refinement levels to overlay, e.g. --mesh-levels 8 16 32",
    )
    parser.add_argument("--tf", type=float, default=1.0, help="final time")
    parser.add_argument("--cfl", type=float, default=1.0, help="CFL number")
    parser.add_argument(
        "--test-function",
        choices=("sin2pi_x", "sin2pi_y", "sin2pi_xy"),
        default="sin2pi_x",
        help="transport exact-profile mode",
    )
    parser.add_argument(
        "--physical-boundary-mode",
        choices=("exact_qb", "opposite_boundary"),
        default="exact_qb",
        help="physical boundary exterior-state mode",
    )
    parser.add_argument(
        "--interior-trace-mode",
        choices=("exchange", "exact_trace"),
        default="exchange",
        help="interior-face mode: exchange uses connectivity, exact_trace uses exact exterior trace on all faces",
    )
    parser.add_argument(
        "--qb-correction",
        choices=("off", "on"),
        default="off",
        help="boundary correction mode",
    )
    parser.add_argument(
        "--q-boundary-correction-mode",
        choices=("inflow", "boundary", "all"),
        default="all",
        help="where to apply boundary correction",
    )
    parser.add_argument(
        "--surface-inverse-mass-mode",
        choices=("diagonal", "projected"),
        default="diagonal",
        help="surface lifting inverse mass mode",
    )
    parser.add_argument(
        "--state-projection",
        choices=("off", "on"),
        default="off",
        help="toggle polynomial projection on nodal state",
    )
    parser.add_argument(
        "--projection-mode",
        choices=("pre", "post", "both"),
        default="post",
        help="projection point when projection-frequency=rhs",
    )
    parser.add_argument(
        "--projection-frequency",
        choices=("rhs", "step"),
        default="step",
        help="projection frequency",
    )
    parser.add_argument(
        "--use-numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable numba kernels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output figure path (.png). default: experiments_outputs/error_vs_time_*.png",
    )
    return parser.parse_args()


def _save_csv_multi(path: Path, curves: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mesh_level", "time", "L2_error", "Linf_error", "max_abs_q"])
        for curve in curves:
            mesh_level = int(curve["mesh_level"])
            times = np.asarray(curve["times"], dtype=float)
            l2 = np.asarray(curve["l2"], dtype=float)
            linf = np.asarray(curve["linf"], dtype=float)
            qmax = np.asarray(curve["qmax"], dtype=float)
            for i in range(times.size):
                writer.writerow([mesh_level, times[i], l2[i], linf[i], qmax[i]])


def _resolve_mesh_levels(args: argparse.Namespace) -> list[int]:
    if args.mesh_levels is None or len(args.mesh_levels) == 0:
        mesh_levels = [int(args.mesh_level)]
    else:
        mesh_levels = [int(n) for n in args.mesh_levels]

    if any(n <= 0 for n in mesh_levels):
        raise ValueError("all mesh levels must be positive.")

    unique_levels: list[int] = []
    seen: set[int] = set()
    for n in mesh_levels:
        if n not in seen:
            unique_levels.append(n)
            seen.add(n)
    return unique_levels


def _run_single_mesh(
    args: argparse.Namespace,
    mesh_level: int,
    *,
    test_spec,
    q_exact,
    velocity,
    rule: dict,
    trace: dict,
    Dr: np.ndarray,
    Ds: np.ndarray,
    projector_t: np.ndarray | None,
    surface_inverse_mass_t: np.ndarray | None,
    projection_frequency: str,
    projection_mode: str,
    boundary_mode: str,
    interior_trace_mode: str,
) -> dict:
    VX, VY, EToV = structured_square_tri_mesh(
        nx=mesh_level,
        ny=mesh_level,
        diagonal="anti",
    )
    conn = None
    if interior_trace_mode == "exchange":
        conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    q0 = q_exact(X, Y, t=0.0)
    if projector_t is not None:
        q0_proj = np.empty_like(q0)
        np.matmul(q0, projector_t, out=q0_proj)
        q0 = q0_proj

    u_elem, v_elem = velocity(X, Y, t=0.0)

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
    if interior_trace_mode == "exchange":
        surface_cache = build_surface_exchange_cache(
            rule=rule,
            trace=trace,
            conn=conn,
            face_geom=face_geom,
        )

    u_face, v_face = velocity(
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
    dt_nominal = cfl_dt_from_h(cfl=float(args.cfl), h=hmin, N=4, vmax=vmax)

    q_boundary_correction = None
    q_boundary_correction_kind = "none"
    if args.qb_correction == "on":
        q_boundary_correction = build_sinx_rk_stage_boundary_correction(
            dt=dt_nominal,
            profile=test_spec,
        )
        q_boundary_correction_kind = "sinx_rk_stage"

    if interior_trace_mode == "exchange" and boundary_mode == "opposite_boundary":
        if args.qb_correction == "on":
            q_boundary_correction_kind = "disabled_for_opposite_boundary"
        q_boundary_correction = None
    elif interior_trace_mode == "exact_trace" and q_boundary_correction_kind == "sinx_rk_stage":
        q_boundary_correction_kind = "sinx_rk_stage_exact_trace"

    projector_for_rhs_t = projector_t if (projector_t is not None and projection_frequency == "rhs") else None

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        q_boundary_correction_mode = str(args.q_boundary_correction_mode).strip().lower()

        if interior_trace_mode == "exchange":
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
                q_boundary=q_exact,
                velocity=velocity,
                t=t,
                tau=0.0,
                compute_mismatches=False,
                return_diagnostics=False,
                use_numba=args.use_numba,
                state_projector_T=projector_for_rhs_t,
                state_projector_mode=projection_mode,
                surface_backend="face-major",
                surface_cache=surface_cache,
                ndotV_precomputed=ndotV_precomputed,
                ndotV_flat_precomputed=ndotV_flat_precomputed,
                volume_split_cache=volume_split_cache,
                surface_inverse_mass_T=surface_inverse_mass_t,
                physical_boundary_mode=boundary_mode,
                q_boundary_correction=q_boundary_correction,
                q_boundary_correction_mode=q_boundary_correction_mode,
            )
            return total_rhs

        q_work = np.asarray(q, dtype=float)
        if projector_for_rhs_t is not None and projection_mode in ("both", "pre"):
            q_work = q_work @ projector_for_rhs_t

        total_rhs, _ = rhs_split_conservative_exact_trace(
            q_elem=q_work,
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
            q_exact=q_exact,
            velocity=velocity,
            t=t,
            tau=0.0,
            face_geom=face_geom,
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=q_boundary_correction_mode,
            use_numba=args.use_numba,
        )

        if projector_for_rhs_t is not None and projection_mode in ("both", "post"):
            total_rhs = total_rhs @ projector_for_rhs_t
        return total_rhs

    q = np.array(q0, copy=True)
    t = 0.0
    nsteps = 0

    times = [t]
    err0 = q - q_exact(X, Y, t=t)
    l2_errors = [weighted_l2_error(err0, rule, face_geom)]
    linf_errors = [float(np.max(np.abs(err0)))]
    max_abs_q = [float(np.max(np.abs(q)))]

    tol = 1e-15 * max(1.0, abs(float(args.tf)))
    while t < float(args.tf) - tol:
        dt_step = min(dt_nominal, float(args.tf) - t)
        q = lsrk54_step(rhs, t=t, q=q, dt=dt_step)
        t += dt_step
        nsteps += 1

        if projector_t is not None and projection_frequency == "step":
            q = q @ projector_t

        q_ref = q_exact(X, Y, t=t)
        err = q - q_ref
        times.append(float(t))
        l2_errors.append(weighted_l2_error(err, rule, face_geom))
        linf_errors.append(float(np.max(np.abs(err))))
        max_abs_q.append(float(np.max(np.abs(q))))

        if max_abs_q[-1] > BLOWUP_BREAK_ABS:
            break

    reached_tf = bool(np.isclose(t, float(args.tf), atol=1e-12, rtol=1e-12))

    times_arr = np.asarray(times, dtype=float)
    l2_arr = np.asarray(l2_errors, dtype=float)
    linf_arr = np.asarray(linf_errors, dtype=float)
    qmax_arr = np.asarray(max_abs_q, dtype=float)

    eps = np.finfo(float).tiny
    l2_plot = np.maximum(l2_arr, eps)

    return {
        "mesh_level": mesh_level,
        "times": times_arr,
        "l2": l2_arr,
        "linf": linf_arr,
        "qmax": qmax_arr,
        "l2_plot": l2_plot,
        "nsteps": nsteps,
        "dt_nominal": dt_nominal,
        "reached_tf": reached_tf,
        "tf_used": float(t),
        "q_boundary_correction_kind": q_boundary_correction_kind,
    }


def main() -> None:
    args = _parse_args()

    if args.tf <= 0.0:
        raise ValueError("tf must be positive.")
    if args.cfl <= 0.0:
        raise ValueError("cfl must be positive.")

    mesh_levels = _resolve_mesh_levels(args)

    test_spec = _resolve_test_function_spec(args.test_function)
    q_exact = _make_q_exact(test_spec)
    velocity = _make_velocity(test_spec)

    rule = load_table1_rule(4)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, 4)

    projection_enabled = str(args.state_projection).strip().lower() == "on"
    projection_frequency = str(args.projection_frequency).strip().lower()
    projection_mode = str(args.projection_mode).strip().lower()
    boundary_mode = str(args.physical_boundary_mode).strip().lower()
    interior_trace_mode = str(args.interior_trace_mode).strip().lower()
    surface_inverse_mass_mode = str(args.surface_inverse_mass_mode).strip().lower()

    if interior_trace_mode == "exact_trace" and boundary_mode != "exact_qb":
        raise ValueError(
            "interior-trace-mode='exact_trace' currently supports only physical-boundary-mode='exact_qb'."
        )
    if interior_trace_mode == "exact_trace" and surface_inverse_mass_mode != "diagonal":
        raise ValueError(
            "surface-inverse-mass-mode='projected' is not supported with interior-trace-mode='exact_trace'."
        )

    projector_t = None
    if projection_enabled:
        projector = build_polynomial_l2_projector_from_rule(rule, 4)
        projector_t = np.ascontiguousarray(projector.T, dtype=float)

    surface_inverse_mass_t = None
    if surface_inverse_mass_mode == "projected":
        surface_inverse_mass = build_projected_inverse_mass_from_rule(rule, 4)
        surface_inverse_mass_t = np.ascontiguousarray(surface_inverse_mass.T, dtype=float)

    curves: list[dict] = []
    for mesh_level in mesh_levels:
        print(f"[run] starting mesh level n={mesh_level}...")
        curve = _run_single_mesh(
            args,
            mesh_level,
            test_spec=test_spec,
            q_exact=q_exact,
            velocity=velocity,
            rule=rule,
            trace=trace,
            Dr=Dr,
            Ds=Ds,
            projector_t=projector_t,
            surface_inverse_mass_t=surface_inverse_mass_t,
            projection_frequency=projection_frequency,
            projection_mode=projection_mode,
            boundary_mode=boundary_mode,
            interior_trace_mode=interior_trace_mode,
        )
        curves.append(curve)

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "experiments_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        n_tag = "-".join(str(n) for n in mesh_levels)
        stem = (
            f"lsrk_error_vs_time_tf{_tf_label(args.tf)}"
            f"_n{n_tag}"
            f"_{args.test_function}"
            f"_{boundary_mode}"
            f"_{surface_inverse_mass_mode}"
            f"_proj{str(args.state_projection).strip().lower()}"
            f"_qb{str(args.qb_correction).strip().lower()}"
        )
        if interior_trace_mode != "exchange":
            stem += f"_{interior_trace_mode}"
        fig_path = out_dir / f"{stem}.png"
    else:
        fig_path = Path(args.output)
        if not fig_path.is_absolute():
            fig_path = root / fig_path
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = fig_path.with_suffix(".csv")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for curve in curves:
        ax.plot(
            np.asarray(curve["times"], dtype=float),
            np.asarray(curve["l2_plot"], dtype=float),
            linewidth=2.0,
            label=f"L2 error (n={int(curve['mesh_level'])})",
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0e-14, top=1.0e1)
    ax.set_xlabel("time")
    ax.set_ylabel("L2 error (log scale)")
    reached_all = all(bool(curve["reached_tf"]) for curve in curves)
    mesh_label = ",".join(str(n) for n in mesh_levels)
    ax.set_title(
        "LSRK L2 error vs time"
        f" | tf={args.tf:g}, n={mesh_label}, trace={interior_trace_mode}, reached_all={reached_all}"
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    _save_csv_multi(csv_path, curves)

    for curve in curves:
        print(
            "[run]"
            f" n={int(curve['mesh_level'])}"
            f" nsteps={int(curve['nsteps'])}"
            f" dt_nominal={float(curve['dt_nominal']):.6e}"
            f" reached_tf={bool(curve['reached_tf'])}"
            f" tf_used={float(curve['tf_used']):.12f}"
        )

    print(
        f"[run] test_function={args.test_function}"
        f" physical_boundary_mode={boundary_mode}"
        f" interior_trace_mode={interior_trace_mode}"
    )
    print(f"[run] surface_inverse_mass_mode={surface_inverse_mass_mode} state_projection={args.state_projection}")
    qb_modes = ",".join(str(curve["q_boundary_correction_kind"]) for curve in curves)
    print(f"[run] qb_correction_kind={qb_modes}")
    print(f"[OK] wrote {fig_path}")
    print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()