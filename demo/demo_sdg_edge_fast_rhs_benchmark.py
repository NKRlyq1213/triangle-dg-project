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
)
from operators.sdg_edge_streamfunction_global_corrected_rhs import (
    edge_streamfunction_closed_sphere_rhs_global_corrected,
)
from operators.sdg_edge_streamfunction_fast_rhs import (
    build_fast_edge_rhs_cache,
    edge_streamfunction_closed_sphere_rhs_global_corrected_fast,
)
from operators.sdg_streamfunction_closed_sphere_rhs import (
    q_initial,
    rhs_mass_residual,
    weighted_l2_linf,
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


def time_call(fn, *, repeats: int) -> tuple[float, object]:
    """
    Return average seconds and last result.
    """
    last = None
    t0 = time.perf_counter()
    for _ in range(repeats):
        last = fn()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats, last


def weighted_error(a: np.ndarray, b: np.ndarray, op) -> tuple[float, float]:
    err = np.asarray(a) - np.asarray(b)
    return weighted_l2_linf(err, op.base)


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
    repeats: int,
    seam_tol: float,
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

    q = q_initial(q_case, op.base)

    # Build fast cache and trigger Numba compilation.
    cache = build_fast_edge_rhs_cache(op)

    rhs_fast0, info_fast0 = edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
        q,
        op,
        fast_cache=cache,
        return_info=True,
    )

    rhs_ref0, info_ref0 = edge_streamfunction_closed_sphere_rhs_global_corrected(
        q,
        op,
        return_info=True,
    )

    diff_l2, diff_linf = weighted_error(rhs_ref0, rhs_fast0, op)

    ref_time, rhs_ref = time_call(
        lambda: edge_streamfunction_closed_sphere_rhs_global_corrected(
            q,
            op,
            return_info=False,
        ),
        repeats=repeats,
    )

    fast_time, rhs_fast = time_call(
        lambda: edge_streamfunction_closed_sphere_rhs_global_corrected_fast(
            q,
            op,
            fast_cache=cache,
            return_info=False,
        ),
        repeats=repeats,
    )

    rhs_fast_mass = rhs_mass_residual(rhs_fast, op.base)
    rhs_ref_mass = rhs_mass_residual(rhs_ref, op.base)

    speedup = ref_time / fast_time if fast_time > 0.0 else float("inf")

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
        "repeats": int(repeats),

        "numba_available": bool(info_fast0["numba_available"]),

        "ref_time_ms": float(ref_time * 1000.0),
        "fast_time_ms": float(fast_time * 1000.0),
        "speedup": float(speedup),

        "rhs_diff_L2": float(diff_l2),
        "rhs_diff_Linf": float(diff_linf),

        "rhs_ref_mass": float(rhs_ref_mass),
        "rhs_fast_mass": float(rhs_fast_mass),
        "fast_global_correction": float(info_fast0["global_correction_constant"]),
        "ref_global_correction": float(info_ref0["global_correction_constant"]),
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
        f"{'ref_ms':>11s} "
        f"{'fast_ms':>11s} "
        f"{'speedup':>9s} "
        f"{'diff_L2':>12s} "
        f"{'diff_Linf':>12s} "
        f"{'fastMass':>12s} "
        f"{'numba':>7s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>15s} "
            f"{r['ref_time_ms']:11.4f} "
            f"{r['fast_time_ms']:11.4f} "
            f"{r['speedup']:9.3f} "
            f"{r['rhs_diff_L2']:12.4e} "
            f"{r['rhs_diff_Linf']:12.4e} "
            f"{r['rhs_fast_mass']:12.4e} "
            f"{str(r['numba_available']):>7s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Numba-accelerated edge-streamfunction RHS."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)
    parser.add_argument("--repeats", type=int, default=20)
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
        default=str(ROOT / "outputs" / "sdg_edge_fast_rhs_benchmark"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG edge-streamfunction fast RHS benchmark ===")
    print(f"order   = {args.order}")
    print(f"N       = {N}")
    print(f"R       = {args.R}")
    print(f"u0      = {args.u0}")
    print(f"tau     = {args.tau}")
    print(f"q_case  = {args.q_case}")
    print(f"repeats = {args.repeats}")
    print(f"alphas  = {alphas}")
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
                    repeats=args.repeats,
                    seam_tol=args.seam_tol,
                )
            )

    print_table(rows)

    out = Path(args.output_dir)
    csv_path = out / f"edge_fast_rhs_benchmark_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
