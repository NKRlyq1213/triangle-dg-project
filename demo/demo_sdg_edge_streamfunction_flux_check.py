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

from operators.sdg_streamfunction_closed_sphere_rhs import (
    build_closed_sphere_sdg_operator,
    q_initial,
    mass,
    rhs_mass_residual,
    weighted_l2_linf,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_edge_streamfunction_flux import (
    build_reference_face_derivative_matrices,
    sdg_surface_edge_streamfunction_strong_rhs_projected,
    edge_flux_pair_error,
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
) -> dict[str, float | int | str]:
    op = build_closed_sphere_sdg_operator(
        nsub=nsub,
        order=order,
        N=N,
        R=R,
        u0=u0,
        alpha0=alpha0,
        tau=tau,
        seam_tol=seam_tol,
    )

    q = q_initial(q_case, op)

    rhs_volume = sdg_conservative_volume_rhs(
        q,
        op.Fx_area,
        op.Fy_area,
        op.Dr,
        op.Ds,
        op.geom["rx"],
        op.geom["sx"],
        op.geom["ry"],
        op.geom["sy"],
        J_area=op.J_area,
        mask=op.cache.bad_mask,
    )

    face_Dt = build_reference_face_derivative_matrices(op.rs_nodes, op.ref_face)

    rhs_surface, p_face, a_face = sdg_surface_edge_streamfunction_strong_rhs_projected(
        q,
        op.psi,
        op.rule,
        op.rs_nodes,
        op.ref_face,
        op.conn_seam,
        N=op.N,
        J_area=op.J_area,
        tau=op.tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=op.MinvT,
        face_Dt=face_Dt,
    )

    rhs = np.where(op.cache.bad_mask, np.nan, rhs_volume + rhs_surface)

    rhsM = rhs_mass_residual(rhs, op)
    rhsL2, rhsLinf = weighted_l2_linf(rhs, op)

    surfM = rhs_mass_residual(rhs_surface, op)
    volM = rhs_mass_residual(rhs_volume, op)

    pair = edge_flux_pair_error(a_face, op.ref_face, op.conn_seam)

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

        "n_seam_pairs": int(op.seam_info.n_seam_pairs),
        "n_unmatched_boundary_faces": int(op.seam_info.n_unmatched_boundary_faces),
        "max_seam_match_error": float(op.seam_info.max_seam_match_error),

        "mass0": float(mass(q, op)),

        "I_vol": float(volM),
        "I_surf": float(surfM),
        "I_full": float(rhsM),

        "rhs_L2": float(rhsL2),
        "rhs_Linf": float(rhsLinf),

        "edgeFluxPairErr": float(pair["edge_flux_pair_max_error"]),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'nsub':>6s} "
        f"{'alpha/pi':>10s} "
        f"{'q_case':>15s} "
        f"{'I_vol':>12s} "
        f"{'I_surf':>12s} "
        f"{'I_full':>12s} "
        f"{'rhs_L2':>12s} "
        f"{'edgeErr':>12s} "
        f"{'unmatch':>7s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>15s} "
            f"{r['I_vol']:12.4e} "
            f"{r['I_surf']:12.4e} "
            f"{r['I_full']:12.4e} "
            f"{r['rhs_L2']:12.4e} "
            f"{r['edgeFluxPairErr']:12.4e} "
            f"{r['n_unmatched_boundary_faces']:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check edge-streamfunction face flux with strong correction."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)
    parser.add_argument("--alphas", type=str, default="0,pi/12,pi/6,pi/4")

    parser.add_argument(
        "--q-case",
        type=str,
        default="constant",
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
        default=str(ROOT / "outputs" / "sdg_edge_streamfunction_flux"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG edge-streamfunction flux check ===")
    print(f"order    = {args.order}")
    print(f"N        = {N}")
    print(f"R        = {args.R}")
    print(f"u0       = {args.u0}")
    print(f"tau      = {args.tau}")
    print(f"q_case   = {args.q_case}")
    print(f"alphas   = {alphas}")
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
                )
            )

    print_table(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"edge_streamfunction_flux_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
