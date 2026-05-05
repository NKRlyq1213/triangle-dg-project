from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.demo_sdg_edge_fast_advection_convergence_cfl import (
    parse_alpha_list,
    exact_state,
    choose_cfl_dt_and_steps,
    run_lsrk_fast,
    error_stats,
    add_rates,
    print_table,
)

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    build_edge_streamfunction_closed_sphere_sdg_operator,
    edge_flux_pair_summary,
)
from operators.sdg_edge_streamfunction_fast_rhs import (
    build_fast_edge_rhs_cache,
    edge_streamfunction_closed_sphere_rhs_global_corrected_fast,
)
from operators.sdg_hybrid_face_speed import apply_hybrid_face_speed_to_fast_cache
from operators.sdg_streamfunction_closed_sphere_rhs import (
    mass,
    seam_summary,
)


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
    tau: float,
    q_case: str,
    seam_tol: float,
    final_time: float,
    cfl: float,
    sigma: float,
    speed_mode: str,
    corner_tol: float,
):
    op = build_edge_streamfunction_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )

    cache = build_fast_edge_rhs_cache(op)

    apply_hybrid_face_speed_to_fast_cache(
        op,
        cache,
        speed_mode=speed_mode,
        corner_tol=corner_tol,
    )

    dt, steps, dt_nominal, hmin, vmax = choose_cfl_dt_and_steps(
        op,
        cfl=cfl,
        final_time=final_time,
    )

    q0 = exact_state(q_case, op, 0.0, sigma=sigma)
    q_exact_T = exact_state(q_case, op, final_time, sigma=sigma)

    # Warm-up / compile.
    _rhs0, info0 = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
        q0,
        op,
        fast_cache=cache,
        return_info=True,
    )

    m0 = mass(q0, op.base)
    mex = mass(q_exact_T, op.base)

    out = run_lsrk_fast(
        q0,
        op,
        cache,
        dt=dt,
        steps=steps,
    )

    qT = out["q"]
    mT = mass(qT, op.base)

    l2_err, linf_err, rel_l2_err = error_stats(qT, q_exact_T, op)

    seam = seam_summary(op.base)
    pair = edge_flux_pair_summary(q0, op)

    return {
        "nsub": int(nsub),
        "hmin": float(hmin),
        "h_proxy": float(1.0 / nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "vmax": float(vmax),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),
        "tau": float(tau),
        "cfl": float(cfl),
        "q_case": q_case,
        "sigma": float(sigma),
        "speed_mode": str(speed_mode),
        "corner_tol": float(corner_tol),

        "steps": int(steps),
        "dt": float(dt),
        "dt_nominal": float(dt_nominal),
        "final_time": float(final_time),
        "elapsed_s": float(out["elapsed_s"]),

        "L2_error": float(l2_err),
        "Linf_error": float(linf_err),
        "relative_L2_error": float(rel_l2_err),

        "mass0": float(m0),
        "mass_exact_T": float(mex),
        "massT": float(mT),
        "mass_drift": float(mT - m0),
        "exact_mass_drift": float(mex - m0),

        "max_stage_mass_rhs_abs": float(out["max_stage_mass_rhs_abs"]),
        "max_stage_rhs_L2": float(out["max_stage_rhs_l2"]),
        "max_stage_global_correction": float(out["max_stage_global_correction"]),

        "rhs0_global_correction": float(info0["global_correction_constant"]),

        "n_seam_pairs": int(seam["n_seam_pairs"]),
        "n_unmatched_boundary_faces": int(seam["n_unmatched_boundary_faces"]),
        "max_seam_match_error": float(seam["max_seam_match_error"]),
        "edge_pair_error_initial": float(pair["edge_flux_pair_max_error"]),

        "L2_rate": float("nan"),
        "Linf_rate": float("nan"),
        "relative_L2_rate": float("nan"),
    }


def write_csv(rows, path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid face-speed CFL convergence test."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)

    parser.add_argument("--final-time", type=float, default=5.0)
    parser.add_argument("--cfl", type=float, default=1.0)

    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--alphas", type=str, default="pi/4")

    parser.add_argument(
        "--q-case",
        type=str,
        default="gaussian",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "gaussian"],
    )

    parser.add_argument(
        "--speed-mode",
        type=str,
        default="corner_volavg",
        choices=["edge", "volavg", "corner_volavg"],
    )

    parser.add_argument("--corner-tol", type=float, default=0.10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_edge_fast_advection_hybrid_speed"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG hybrid face-speed advection convergence ===")
    print(f"order      = {args.order}")
    print(f"N          = {N}")
    print(f"R          = {args.R}")
    print(f"u0         = {args.u0}")
    print(f"tau        = {args.tau}")
    print(f"q_case     = {args.q_case}")
    print(f"sigma      = {args.sigma}")
    print(f"final_time = {args.final_time}")
    print(f"cfl        = {args.cfl}")
    print(f"speed_mode = {args.speed_mode}")
    print(f"corner_tol = {args.corner_tol}")
    print(f"alphas     = {alphas}")
    print()

    rows = []

    for alpha0 in alphas:
        for nsub in args.nsubs:
            rows.append(
                run_one_case(
                    nsub=nsub,
                    order=args.order,
                    N=N,
                    R=args.R,
                    u0=args.u0,
                    alpha0=alpha0,
                    tau=args.tau,
                    q_case=args.q_case,
                    seam_tol=args.seam_tol,
                    final_time=args.final_time,
                    cfl=args.cfl,
                    sigma=args.sigma,
                    speed_mode=args.speed_mode,
                    corner_tol=args.corner_tol,
                )
            )

    add_rates(rows)
    print_table(rows)

    out = Path(args.output_dir)
    csv_path = out / (
        f"hybrid_speed_{args.speed_mode}_{args.q_case}_T{args.final_time:g}_"
        f"cfl{args.cfl:g}_tol{args.corner_tol:g}_tau{args.tau:g}.csv"
    )
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
