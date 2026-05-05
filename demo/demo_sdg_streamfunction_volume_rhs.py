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

from data.table1_rules import load_table1_rule
from geometry.sphere_flat_mesh import sphere_flat_square_mesh
from geometry.sphere_flat_metrics import (
    build_sphere_flat_geometry_cache,
    geometry_diagnostics,
)
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.sdg_flattened_divergence import build_table1_reference_diff_operators
from operators.sdg_streamfunction_flux import (
    equal_area_jacobian,
    streamfunction_area_divergence,
)
from operators.sdg_conservative_rhs import (
    sdg_conservative_volume_rhs,
    sdg_conservative_volume_divergence,
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


def weighted_stats(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape.")

    if mask is None:
        good = np.isfinite(values) & np.isfinite(weights)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as values.")
        good = (~mask) & np.isfinite(values) & np.isfinite(weights)

    if not np.any(good):
        return {
            "integral": float("nan"),
            "mean": float("nan"),
            "L2": float("nan"),
            "RMS": float("nan"),
            "Linf": float("nan"),
            "measure": float("nan"),
            "n_good": 0,
        }

    vals = values[good]
    w = weights[good]

    measure = float(np.sum(w))
    integral = float(np.sum(w * vals))
    l2_sq = float(np.sum(w * vals * vals))

    return {
        "integral": integral,
        "mean": integral / measure,
        "L2": math.sqrt(max(l2_sq, 0.0)),
        "RMS": math.sqrt(max(l2_sq, 0.0) / measure),
        "Linf": float(np.max(np.abs(vals))),
        "measure": measure,
        "n_good": int(vals.size),
    }


def build_surface_weights(
    rule: dict,
    geom: dict[str, np.ndarray],
    *,
    R: float,
) -> np.ndarray:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    if J_flat.ndim != 2:
        raise ValueError("geom['J'] must have shape (K, Np).")

    K, Np = J_flat.shape
    if ws.shape != (Np,):
        raise ValueError("quadrature weights do not match Np.")

    area_flat = 2.0 * J_flat[:, 0]
    sqrtG = equal_area_jacobian(R)

    return area_flat[:, None] * ws[None, :] * sqrtG


def q_initial(case: str, cache, R: float) -> np.ndarray:
    case = case.lower().strip()

    if case == "constant":
        return np.ones_like(cache.X)

    if case == "sphere_z":
        return cache.Z / R

    if case == "sphere_x":
        return cache.X / R

    if case == "sphere_y":
        return cache.Y / R

    if case == "flat_gaussian":
        x = cache.x_flat
        y = cache.y_flat
        x0 = 0.25
        y0 = -0.15
        sigma = 0.22
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))

    raise ValueError(
        "Unknown q_case. Available: constant, sphere_x, sphere_y, sphere_z, flat_gaussian."
    )


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
    q_case: str,
) -> dict[str, float]:
    rule = load_table1_rule(order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)

    VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
        n_sub=nsub,
        R=1.0,
    )

    cache = build_sphere_flat_geometry_cache(
        rs_nodes=rs_nodes,
        VX=VX,
        VY=VY,
        EToV=EToV,
        elem_patch_id=elem_patch_id,
        R=R,
    )

    Dr, Ds = build_table1_reference_diff_operators(rule, N=N)

    geom = affine_geometric_factors_from_mesh(
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
    )

    weights = build_surface_weights(
        rule,
        geom,
        R=R,
    )

    J_area = equal_area_jacobian(R)

    Fx_area, Fy_area, psi, div_area = streamfunction_area_divergence(
        cache.X,
        cache.Y,
        cache.Z,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
        alpha0=alpha0,
        u0=u0,
        R=R,
    )

    q = q_initial(q_case, cache, R=R)

    div_Fq = sdg_conservative_volume_divergence(
        q,
        Fx_area,
        Fy_area,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
        mask=cache.bad_mask,
    )

    rhs = sdg_conservative_volume_rhs(
        q,
        Fx_area,
        Fy_area,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
        J_area=J_area,
        mask=cache.bad_mask,
    )

    div_area = np.where(cache.bad_mask, np.nan, div_area)

    div_area_stats = weighted_stats(
        div_area,
        weights,
        mask=cache.bad_mask,
    )

    div_Fq_stats = weighted_stats(
        div_Fq,
        weights,
        mask=cache.bad_mask,
    )

    rhs_stats = weighted_stats(
        rhs,
        weights,
        mask=cache.bad_mask,
    )

    gdiag = geometry_diagnostics(cache, R=R)

    return {
        "nsub": int(nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),
        "q_case": q_case,

        "n_total_nodes": int(gdiag["n_total_nodes"]),
        "n_bad_nodes": int(gdiag["n_bad_nodes"]),
        "n_good_nodes": int(rhs_stats["n_good"]),

        "J_area": float(J_area),
        "sqrtG_expected": float(gdiag["sqrtG_expected"]),
        "sqrtG_error_max_regular_from_A": float(gdiag["sqrtG_error_max_regular"]),

        "div_area_L2": float(div_area_stats["L2"]),
        "div_area_Linf": float(div_area_stats["Linf"]),
        "div_area_integral": float(div_area_stats["integral"]),

        "div_Fq_L2": float(div_Fq_stats["L2"]),
        "div_Fq_Linf": float(div_Fq_stats["Linf"]),
        "div_Fq_integral": float(div_Fq_stats["integral"]),

        "rhs_L2": float(rhs_stats["L2"]),
        "rhs_Linf": float(rhs_stats["Linf"]),
        "rhs_integral": float(rhs_stats["integral"]),
    }


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'nsub':>6s} "
        f"{'alpha/pi':>10s} "
        f"{'q_case':>13s} "
        f"{'div_area_L2':>14s} "
        f"{'rhs_L2':>14s} "
        f"{'rhs_Linf':>14s} "
        f"{'rhs_int':>14s} "
        f"{'bad':>5s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>13s} "
            f"{r['div_area_L2']:14.6e} "
            f"{r['rhs_L2']:14.6e} "
            f"{r['rhs_Linf']:14.6e} "
            f"{r['rhs_integral']:14.6e} "
            f"{r['n_bad_nodes']:5d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volume-only conservative RHS diagnostic using stream-function area flux."
    )
    parser.add_argument(
        "--nsubs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,pi/12,pi/6,pi/4",
    )
    parser.add_argument(
        "--q-case",
        type=str,
        default="constant",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "flat_gaussian"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_streamfunction_volume_rhs"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG stream-function volume RHS diagnostic ===")
    print(f"order  = {args.order}")
    print(f"N      = {N}")
    print(f"R      = {args.R}")
    print(f"u0     = {args.u0}")
    print(f"q_case = {args.q_case}")
    print(f"alphas = {alphas}")
    print()

    rows: list[dict[str, float]] = []

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
                    q_case=args.q_case,
                )
            )

    print_table(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"volume_rhs_{args.q_case}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
