from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    build_edge_streamfunction_closed_sphere_sdg_operator,
    edge_flux_pair_summary,
)
from operators.sdg_edge_streamfunction_fast_rhs import (
    build_fast_edge_rhs_cache,
    edge_streamfunction_closed_sphere_rhs_global_corrected_fast,
)
from operators.sdg_streamfunction_closed_sphere_rhs import (
    q_initial,
    mass,
    rhs_mass_residual,
    weighted_l2_linf,
    seam_summary,
)


RK4A = np.array(
    [
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ],
    dtype=float,
)

RK4B = np.array(
    [
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ],
    dtype=float,
)


def parse_alpha_expr(expr: str) -> float:
    s = expr.strip().lower().replace(" ", "")
    if s == "0":
        return 0.0
    if s == "pi":
        return math.pi
    if s == "-pi":
        return -math.pi
    if s.startswith("pi/"):
        return math.pi / float(s.split("/", 1)[1])
    if s.startswith("-pi/"):
        return -math.pi / float(s.split("/", 1)[1])
    return float(s)


def parse_alpha_list(text: str) -> list[float]:
    return [parse_alpha_expr(part) for part in text.split(",") if part.strip()]


def run_lsrk_fast(q0: np.ndarray, op, cache: dict, *, dt: float, steps: int):
    q = np.array(q0, dtype=float, copy=True)
    res = np.zeros_like(q)

    max_stage_mass_rhs_abs = 0.0
    max_stage_rhs_l2 = 0.0
    max_stage_rhs_linf = 0.0
    max_stage_corr = 0.0

    t0 = time.perf_counter()

    for _ in range(steps):
        for a, b in zip(RK4A, RK4B):
            rhs, info = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
                q,
                op,
                fast_cache=cache,
                return_info=True,
            )

            m_rhs = rhs_mass_residual(rhs, op.base)
            rhs_l2, rhs_linf = weighted_l2_linf(rhs, op.base)

            max_stage_mass_rhs_abs = max(max_stage_mass_rhs_abs, abs(m_rhs))
            max_stage_rhs_l2 = max(max_stage_rhs_l2, rhs_l2)
            max_stage_rhs_linf = max(max_stage_rhs_linf, rhs_linf)
            max_stage_corr = max(max_stage_corr, abs(info["global_correction_constant"]))

            res = a * res + dt * rhs
            q = q + b * res

    t1 = time.perf_counter()

    return {
        "q": q,
        "elapsed_s": float(t1 - t0),
        "max_stage_mass_rhs_abs": float(max_stage_mass_rhs_abs),
        "max_stage_rhs_l2": float(max_stage_rhs_l2),
        "max_stage_rhs_linf": float(max_stage_rhs_linf),
        "max_stage_global_correction": float(max_stage_corr),
    }


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
    steps: int,
    dt: float | None,
    final_time: float | None,
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

    q0 = q_initial(q_case, op.base)

    # Trigger / check first RHS.
    rhs0, info0 = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
        q0,
        op,
        fast_cache=cache,
        return_info=True,
    )

    rhs0_mass = rhs_mass_residual(rhs0, op.base)
    rhs0_l2, rhs0_linf = weighted_l2_linf(rhs0, op.base)

    m0 = mass(q0, op.base)

    if dt is None:
        if final_time is None:
            final_time = 0.01
        dt_eff = float(final_time) / int(steps)
    else:
        dt_eff = float(dt)
        final_time = dt_eff * int(steps)

    out = run_lsrk_fast(
        q0,
        op,
        cache,
        dt=dt_eff,
        steps=steps,
    )

    qT = out["q"]
    mT = mass(qT, op.base)
    mass_drift = mT - m0

    qT_l2, qT_linf = weighted_l2_linf(qT, op.base)

    if q_case == "constant":
        q_deviation_linf = float(np.nanmax(np.abs(qT - 1.0)))
    else:
        q_deviation_linf = float("nan")

    seam = seam_summary(op.base)
    pair = edge_flux_pair_summary(q0, op)

    dof = int(q0.size)
    rhs_calls = int(5 * steps)
    rhs_calls_per_sec = rhs_calls / out["elapsed_s"] if out["elapsed_s"] > 0.0 else float("inf")

    return {
        "nsub": int(nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),
        "tau": float(tau),
        "q_case": q_case,
        "steps": int(steps),
        "dt": float(dt_eff),
        "final_time": float(final_time),
        "dof": int(dof),
        "rhs_calls": int(rhs_calls),

        "elapsed_s": float(out["elapsed_s"]),
        "rhs_calls_per_sec": float(rhs_calls_per_sec),

        "n_seam_pairs": int(seam["n_seam_pairs"]),
        "n_unmatched_boundary_faces": int(seam["n_unmatched_boundary_faces"]),
        "max_seam_match_error": float(seam["max_seam_match_error"]),
        "edge_pair_error_initial": float(pair["edge_flux_pair_max_error"]),

        "rhs0_mass": float(rhs0_mass),
        "rhs0_L2": float(rhs0_l2),
        "rhs0_Linf": float(rhs0_linf),

        "rhs0_global_correction": float(info0["global_correction_constant"]),
        "rhs0_mass_before_correction": float(info0["global_mass_residual_before"]),
        "rhs0_mass_after_correction": float(info0["global_mass_residual_after"]),

        "mass0": float(m0),
        "massT": float(mT),
        "mass_drift": float(mass_drift),
        "relative_mass_drift": float(abs(mass_drift) / max(1.0, abs(m0))),

        "max_stage_mass_rhs_abs": float(out["max_stage_mass_rhs_abs"]),
        "max_stage_rhs_L2": float(out["max_stage_rhs_l2"]),
        "max_stage_rhs_Linf": float(out["max_stage_rhs_linf"]),
        "max_stage_global_correction": float(out["max_stage_global_correction"]),

        "qT_L2": float(qT_l2),
        "qT_Linf": float(qT_linf),
        "q_deviation_linf_if_constant": float(q_deviation_linf),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'nsub':>6s} "
        f"{'alpha/pi':>10s} "
        f"{'q_case':>15s} "
        f"{'rhsM0':>12s} "
        f"{'massDrift':>13s} "
        f"{'maxStageM':>13s} "
        f"{'corr':>12s} "
        f"{'constDev':>12s} "
        f"{'time_s':>10s} "
        f"{'rhs/s':>10s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>15s} "
            f"{r['rhs0_mass']:12.4e} "
            f"{r['mass_drift']:13.4e} "
            f"{r['max_stage_mass_rhs_abs']:13.4e} "
            f"{r['max_stage_global_correction']:12.4e} "
            f"{r['q_deviation_linf_if_constant']:12.4e} "
            f"{r['elapsed_s']:10.4f} "
            f"{r['rhs_calls_per_sec']:10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast LSRK mass check using Numba-accelerated edge-streamfunction RHS."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--final-time", type=float, default=0.01)
    parser.add_argument("--alphas", type=str, default="pi/4")

    parser.add_argument(
        "--q-case",
        type=str,
        default="gaussian",
        choices=[
            "constant",
            "sphere_x",
            "sphere_y",
            "sphere_z",
            "flat_gaussian",
            "gaussian",
            "element_jump",
            "element_checker",
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_edge_fast_lsrk_mass"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG edge-streamfunction fast LSRK mass check ===")
    print(f"order      = {args.order}")
    print(f"N          = {N}")
    print(f"R          = {args.R}")
    print(f"u0         = {args.u0}")
    print(f"tau        = {args.tau}")
    print(f"q_case     = {args.q_case}")
    print(f"steps      = {args.steps}")
    print(f"dt         = {args.dt}")
    print(f"final_time = {args.final_time}")
    print(f"alphas     = {alphas}")
    print()

    rows = []
    for nsub in args.nsubs:
        for alpha0 in alphas:
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
                    steps=args.steps,
                    dt=args.dt,
                    final_time=args.final_time,
                )
            )

    print_table(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"edge_fast_lsrk_mass_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
