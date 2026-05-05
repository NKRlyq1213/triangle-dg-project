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
    backward_characteristic_points,
    spherical_gaussian_from_points,
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
from operators.sdg_streamfunction_closed_sphere_rhs import (
    mass,
    seam_summary,
)


def parse_center(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--center must have format x,y,z")

    c = np.asarray(parts, dtype=float)
    n = float(np.linalg.norm(c))

    if n == 0.0:
        raise ValueError("--center vector must be nonzero.")

    return c / n


def exact_state_center(
    case: str,
    op,
    t: float,
    *,
    sigma: float,
    center: np.ndarray,
) -> np.ndarray:
    """
    Exact solution q(X,t) = q0(R(-t)X).

    Gaussian center is user-controlled.
    """
    case = case.lower().strip()
    base = op.base

    X0, Y0, Z0 = backward_characteristic_points(op, t)

    if case == "constant":
        return np.ones_like(X0)

    if case == "sphere_x":
        return X0 / base.R

    if case == "sphere_y":
        return Y0 / base.R

    if case == "sphere_z":
        return Z0 / base.R

    if case == "gaussian":
        return spherical_gaussian_from_points(
            X0,
            Y0,
            Z0,
            R=base.R,
            sigma=sigma,
            center=center,
        )

    raise ValueError("Supported q_case: constant, sphere_x, sphere_y, sphere_z, gaussian.")


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
    center: np.ndarray,
) -> dict:
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

    dt, steps, dt_nominal, hmin, vmax = choose_cfl_dt_and_steps(
        op,
        cfl=cfl,
        final_time=final_time,
    )

    q0 = exact_state_center(
        q_case,
        op,
        0.0,
        sigma=sigma,
        center=center,
    )

    q_exact_T = exact_state_center(
        q_case,
        op,
        final_time,
        sigma=sigma,
        center=center,
    )

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

        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),

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


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFL-controlled fast edge-streamfunction convergence with configurable Gaussian center."
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
    parser.add_argument("--center", type=str, default="1,0,0")
    parser.add_argument("--alphas", type=str, default="pi/4")

    parser.add_argument(
        "--q-case",
        type=str,
        default="gaussian",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "gaussian"],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_edge_fast_advection_convergence_cfl_center"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)
    center = parse_center(args.center)

    print("=== SDG edge-streamfunction CFL advection convergence with center ===")
    print(f"order      = {args.order}")
    print(f"N          = {N}")
    print(f"R          = {args.R}")
    print(f"u0         = {args.u0}")
    print(f"tau        = {args.tau}")
    print(f"q_case     = {args.q_case}")
    print(f"sigma      = {args.sigma}")
    print(f"center     = {center}")
    print(f"final_time = {args.final_time}")
    print(f"cfl        = {args.cfl}")
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
                    center=center,
                )
            )

    add_rates(rows)
    print_table(rows)

    out = Path(args.output_dir)
    center_tag = f"{center[0]:+.3f}_{center[1]:+.3f}_{center[2]:+.3f}".replace("+", "p").replace("-", "m")
    csv_path = out / (
        f"edge_fast_convergence_center_{center_tag}_{args.q_case}_"
        f"T{args.final_time:g}_cfl{args.cfl:g}_tau{args.tau:g}.csv"
    )
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
