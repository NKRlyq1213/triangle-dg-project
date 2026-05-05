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
from geometry.metrics import (
    affine_geometric_factors_from_mesh,
    physical_derivatives_2d,
    divergence_2d,
)
from operators.sdg_flattened_divergence import (
    build_table1_reference_diff_operators,
    sdg_flattened_cartesian_divergence,
)
from problems.sphere_advection import flattened_velocity_from_cache


def parse_alpha_expr(expr: str) -> float:
    """
    Safe parser for simple alpha expressions.

    Supported examples:
        0
        pi/12
        pi/6
        pi/4
        -pi/4
        0.7853981633974483
    """
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
    """
    Compute weighted integral, mean, L2, RMS, Linf.

    values, weights shape:
        (K, Np)

    mask:
        True means excluded.
    """
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
    """
    Build surface weights for equal-area SDG map.

    The flattened domain area element is dx dy.
    The sphere area element is:

        dS = sqrtG dx dy

    For the current SDG equal-area map, sqrtG is expected to be constant:

        sqrtG = pi R^2

    On each affine triangle:

        integral_K f dxdy ≈ area_K * sum_i w_i f_i

    where area_K = 2 * J_flat because the reference triangle area is 2.
    """
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    if J_flat.ndim != 2:
        raise ValueError("geom['J'] must have shape (K, Np).")

    K, Np = J_flat.shape
    if ws.shape != (Np,):
        raise ValueError("quadrature weights do not match Np.")

    # affine J is constant per element, but stored at every node
    area_flat = 2.0 * J_flat[:, 0]
    sqrtG = math.pi * R * R

    weights = area_flat[:, None] * ws[None, :] * sqrtG
    return weights


def streamfunction_area_flux(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict[str, np.ndarray],
    alpha0: float,
    u0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Construct area-weighted conservative flux from stream function.

    Physical solid-body rotation:

        Omega = u0 * (-sin(alpha0), 0, cos(alpha0))

    Stream function:

        psi = Omega dot X

    Area-weighted flux:

        Fx_area = - d_y psi
        Fy_area =   d_x psi

    This means:

        Fx_area = J u^x
        Fy_area = J u^y

    Therefore:

        d_x Fx_area + d_y Fy_area = 0

    analytically for divergence-free solid-body rotation.
    """
    omega = np.array(
        [
            -math.sin(alpha0) * u0,
            0.0,
            math.cos(alpha0) * u0,
        ],
        dtype=float,
    )

    psi = omega[0] * X + omega[1] * Y + omega[2] * Z

    psi_x, psi_y = physical_derivatives_2d(
        psi,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
    )

    Fx_area = -psi_y
    Fy_area = psi_x

    div_area = divergence_2d(
        Fx_area,
        Fy_area,
        Dr,
        Ds,
        geom["rx"],
        geom["sx"],
        geom["ry"],
        geom["sy"],
    )

    return Fx_area, Fy_area, div_area, psi


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
) -> dict[str, float]:
    rule = load_table1_rule(order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)

    VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
        n_sub=nsub,
        R=1.0,  # flat square remains [-1, 1]^2
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

    surface_weights = build_surface_weights(
        rule,
        geom,
        R=R,
    )

    # ------------------------------------------------------------
    # Baseline: old lambda/Ainv route
    # ------------------------------------------------------------
    q = np.ones_like(cache.X)

    u1, u2, u_sph, v_sph = flattened_velocity_from_cache(
        cache,
        u0=u0,
        alpha0=alpha0,
    )

    div_ainv = sdg_flattened_cartesian_divergence(
        q=q,
        u1=u1,
        u2=u2,
        Dr=Dr,
        Ds=Ds,
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
        mask=cache.bad_mask,
    )

    stats_ainv = weighted_stats(
        div_ainv,
        surface_weights,
        mask=cache.bad_mask,
    )

    # ------------------------------------------------------------
    # Proposed: stream-function conservative area flux
    # ------------------------------------------------------------
    Fx_area, Fy_area, div_stream, psi = streamfunction_area_flux(
        X=cache.X,
        Y=cache.Y,
        Z=cache.Z,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        alpha0=alpha0,
        u0=u0,
    )

    # Same masking convention as geometry cache
    div_stream = np.where(cache.bad_mask, np.nan, div_stream)

    stats_stream = weighted_stats(
        div_stream,
        surface_weights,
        mask=cache.bad_mask,
    )

    # ------------------------------------------------------------
    # Geometry diagnostics
    # ------------------------------------------------------------
    gdiag = geometry_diagnostics(cache, R=R)

    ratio_L2 = (
        stats_ainv["L2"] / stats_stream["L2"]
        if np.isfinite(stats_ainv["L2"])
        and np.isfinite(stats_stream["L2"])
        and stats_stream["L2"] > 0.0
        else float("nan")
    )

    ratio_Linf = (
        stats_ainv["Linf"] / stats_stream["Linf"]
        if np.isfinite(stats_ainv["Linf"])
        and np.isfinite(stats_stream["Linf"])
        and stats_stream["Linf"] > 0.0
        else float("nan")
    )

    return {
        "nsub": int(nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),

        "n_total_nodes": int(gdiag["n_total_nodes"]),
        "n_bad_nodes": int(gdiag["n_bad_nodes"]),
        "n_good_nodes": int(stats_stream["n_good"]),

        "max_radial_error": float(gdiag["max_radial_error"]),
        "sqrtG_expected": float(gdiag["sqrtG_expected"]),
        "sqrtG_error_max_regular": float(gdiag["sqrtG_error_max_regular"]),
        "A_Ainv_err_max_regular": float(gdiag["A_Ainv_err_max_regular"]),

        "integral_Ainv": float(stats_ainv["integral"]),
        "mean_Ainv": float(stats_ainv["mean"]),
        "L2_Ainv": float(stats_ainv["L2"]),
        "RMS_Ainv": float(stats_ainv["RMS"]),
        "Linf_Ainv": float(stats_ainv["Linf"]),

        "integral_stream": float(stats_stream["integral"]),
        "mean_stream": float(stats_stream["mean"]),
        "L2_stream": float(stats_stream["L2"]),
        "RMS_stream": float(stats_stream["RMS"]),
        "Linf_stream": float(stats_stream["Linf"]),

        "ratio_L2_Ainv_over_stream": float(ratio_L2),
        "ratio_Linf_Ainv_over_stream": float(ratio_Linf),
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
        f"{'L2_Ainv':>14s} "
        f"{'L2_stream':>14s} "
        f"{'ratio':>12s} "
        f"{'Linf_Ainv':>14s} "
        f"{'Linf_stream':>14s} "
        f"{'bad':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['L2_Ainv']:14.6e} "
            f"{r['L2_stream']:14.6e} "
            f"{r['ratio_L2_Ainv_over_stream']:12.4e} "
            f"{r['Linf_Ainv']:14.6e} "
            f"{r['Linf_stream']:14.6e} "
            f"{r['n_bad_nodes']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare lambda/Ainv flattened divergence with "
            "stream-function conservative-flux divergence."
        )
    )
    parser.add_argument(
        "--nsubs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Subdivisions per macro triangle.",
    )
    parser.add_argument("--order", type=int, default=4, help="Table1 order.")
    parser.add_argument("--N", type=int, default=None, help="Polynomial degree. Default: N=order.")
    parser.add_argument("--R", type=float, default=1.0, help="Sphere radius.")
    parser.add_argument("--u0", type=float, default=1.0, help="Velocity scale.")
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,pi/12,pi/6,pi/4",
        help="Comma-separated alpha expressions, e.g. '0,pi/12,pi/6,pi/4'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_streamfunction_divergence"),
        help="Output directory.",
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    rows: list[dict[str, float]] = []

    print("=== SDG stream-function divergence diagnostic ===")
    print(f"order = {args.order}")
    print(f"N     = {N}")
    print(f"R     = {args.R}")
    print(f"u0    = {args.u0}")
    print(f"alphas = {alphas}")
    print()

    for nsub in args.nsubs:
        for alpha0 in alphas:
            row = run_one_case(
                nsub=nsub,
                order=args.order,
                N=N,
                R=args.R,
                u0=args.u0,
                alpha0=alpha0,
            )
            rows.append(row)

    print_table(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "alpha_scan.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
