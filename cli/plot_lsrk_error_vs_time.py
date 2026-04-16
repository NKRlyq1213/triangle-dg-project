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
    build_projected_inverse_mass_from_rule,
    build_reference_diff_operators_from_rule,
    build_rk_stage_boundary_correction,
    resolve_effective_taus,
    weighted_l2_error,
)
from experiments.output_paths import experiments_output_dir, project_root
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


def _tau_label(tau: float) -> str:
    if float(tau).is_integer():
        return str(int(tau))
    return str(tau).replace(".", "p")


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
        help="interior-face mode: exchange uses neighbor connectivity, exact_trace uses exact exterior trace on interior faces",
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--surface-inverse-mass-mode",
        choices=("diagonal", "projected"),
        default="diagonal",
        help="surface lifting inverse mass mode",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="shared surface numerical flux parameter; used for both tau-interior and tau-qb unless overridden",
    )
    parser.add_argument(
        "--tau-interior",
        type=float,
        default=None,
        help="surface numerical flux parameter for interior faces and non-exact_qb exterior traces",
    )
    parser.add_argument(
        "--tau-qb",
        type=float,
        default=None,
        help="surface numerical flux parameter for physical-boundary-mode=exact_qb faces only",
    )
    parser.add_argument(
        "--use-numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output figure path (.png). default: experiments_outputs/lsrk_error_vs_time/*.png",
    )
    return parser.parse_args()


def _save_csv_multi(path: Path, curves: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mesh_level",
                "tau_interior",
                "tau_qb",
                "time",
                "L2_error",
                "Linf_error",
                "max_abs_q",
            ]
        )
        for curve in curves:
            mesh_level = int(curve["mesh_level"])
            tau_interior = float(curve["tau_interior"])
            tau_qb = float(curve["tau_qb"])
            times = np.asarray(curve["times"], dtype=float)
            l2 = np.asarray(curve["l2"], dtype=float)
            linf = np.asarray(curve["linf"], dtype=float)
            qmax = np.asarray(curve["qmax"], dtype=float)
            for i in range(times.size):
                writer.writerow([mesh_level, tau_interior, tau_qb, times[i], l2[i], linf[i], qmax[i]])


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
    surface_inverse_mass_t: np.ndarray | None,
    boundary_mode: str,
    interior_trace_mode: str,
    tau_interior_eff: float,
    tau_qb_eff: float,
) -> dict:
    VX, VY, EToV = structured_square_tri_mesh(
        nx=mesh_level,
        ny=mesh_level,
        diagonal="anti",
    )
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    q0 = q_exact(X, Y, t=0.0)

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
    if (
        interior_trace_mode == "exchange"
        or interior_trace_mode == "exact_trace"
        or args.use_numba
        or boundary_mode == "opposite_boundary"
    ):
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
        q_boundary_correction = build_rk_stage_boundary_correction(
            dt=dt_nominal,
            profile=test_spec,
        )
        q_boundary_correction_kind = "rk_stage"

    if interior_trace_mode == "exact_trace" and q_boundary_correction_kind == "rk_stage":
        q_boundary_correction_kind = "rk_stage_exact_trace"
    elif boundary_mode == "exact_qb" and q_boundary_correction_kind == "rk_stage":
        q_boundary_correction_kind = "rk_stage_exact_qb"

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
                tau=float(args.tau),
                tau_interior=tau_interior_eff,
                tau_qb=tau_qb_eff,
                compute_mismatches=False,
                return_diagnostics=False,
                use_numba=args.use_numba,
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
            q_exact=q_exact,
            q_boundary=q_exact,
            velocity=velocity,
            t=t,
            tau=float(args.tau),
            tau_interior=tau_interior_eff,
            tau_qb=tau_qb_eff,
            face_geom=face_geom,
            physical_boundary_mode=boundary_mode,
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=q_boundary_correction_mode,
            use_numba=args.use_numba,
            conn=conn,
            surface_cache=surface_cache,
        )
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
        "tau_interior": float(tau_interior_eff),
        "tau_qb": float(tau_qb_eff),
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

    boundary_mode = str(args.physical_boundary_mode).strip().lower()
    interior_trace_mode = str(args.interior_trace_mode).strip().lower()
    surface_inverse_mass_mode = str(args.surface_inverse_mass_mode).strip().lower()
    tau_interior_eff, tau_qb_eff = resolve_effective_taus(
        tau=float(args.tau),
        tau_interior=args.tau_interior,
        tau_qb=args.tau_qb,
    )

    if interior_trace_mode == "exact_trace" and surface_inverse_mass_mode != "diagonal":
        raise ValueError(
            "surface-inverse-mass-mode='projected' is not supported with interior-trace-mode='exact_trace'."
        )
    exact_source_exists = (interior_trace_mode == "exact_trace") or (boundary_mode == "exact_qb")
    if args.qb_correction == "on" and (not exact_source_exists):
        raise ValueError(
            "qb-correction requires at least one exact source: "
            "interior-trace-mode='exact_trace' or physical-boundary-mode='exact_qb'."
        )

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
            surface_inverse_mass_t=surface_inverse_mass_t,
            boundary_mode=boundary_mode,
            interior_trace_mode=interior_trace_mode,
            tau_interior_eff=tau_interior_eff,
            tau_qb_eff=tau_qb_eff,
        )
        curves.append(curve)

    root = project_root(__file__)
    out_dir = experiments_output_dir(__file__, "lsrk_error_vs_time")

    if args.output is None:
        n_tag = "-".join(str(n) for n in mesh_levels)
        stem = (
            f"lsrk_error_vs_time_tf{_tf_label(args.tf)}"
            f"_n{n_tag}"
            f"_{args.test_function}"
            f"_{boundary_mode}"
            f"_{surface_inverse_mass_mode}"
            f"_taui{_tau_label(tau_interior_eff)}"
            f"_tauqb{_tau_label(tau_qb_eff)}"
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
        f" | tf={args.tf:g}, n={mesh_label}, tau_i={tau_interior_eff:g}, tau_qb={tau_qb_eff:g},"
        f" trace={interior_trace_mode}, reached_all={reached_all}"
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
    print(f"[run] surface_inverse_mass_mode={surface_inverse_mass_mode}")
    print(f"[run] tau={args.tau:g}")
    print(f"[run] tau_interior={tau_interior_eff:g}")
    print(f"[run] tau_qb={tau_qb_eff:g}")
    print("[run] tau_role=penalty uses tau_interior on interior/non-exact_qb faces and tau_qb on physical-boundary-mode=exact_qb faces")
    qb_modes = ",".join(str(curve["q_boundary_correction_kind"]) for curve in curves)
    print(f"[run] qb_correction_kind={qb_modes}")
    print(f"[OK] wrote {fig_path}")
    print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()
