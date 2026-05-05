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
from operators.sdg_flattened_divergence import build_table1_reference_diff_operators


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


def weighted_scalar_stats(
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


def weighted_vector_error_stats(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    err_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
    return weighted_scalar_stats(err_norm, weights, mask=mask)


def build_surface_weights(
    rule: dict,
    geom: dict[str, np.ndarray],
    *,
    R: float,
) -> np.ndarray:
    """
    Equal-area surface weights.

    dS = sqrtG dx dy, sqrtG = pi R^2.
    """
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    if J_flat.ndim != 2:
        raise ValueError("geom['J'] must have shape (K, Np).")

    K, Np = J_flat.shape
    if ws.shape != (Np,):
        raise ValueError("quadrature weights do not match Np.")

    area_flat = 2.0 * J_flat[:, 0]
    sqrtG = math.pi * R * R

    return area_flat[:, None] * ws[None, :] * sqrtG


def solid_body_omega(alpha0: float, u0: float) -> np.ndarray:
    return np.array(
        [
            -math.sin(alpha0) * u0,
            0.0,
            math.cos(alpha0) * u0,
        ],
        dtype=float,
    )


def exact_solid_body_velocity_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    U = Omega x X.
    """
    ox, oy, oz = omega

    Ux = oy * Z - oz * Y
    Uy = oz * X - ox * Z
    Uz = ox * Y - oy * X

    return Ux, Uy, Uz


def surface_tangent_basis_and_jacobian(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute discrete X_x, X_y and J_cross = |X_x cross X_y|.

    Here x,y are flattened coordinates on [-1,1]^2.
    """
    Xx, Xy = physical_derivatives_2d(
        X, Dr, Ds, geom["rx"], geom["sx"], geom["ry"], geom["sy"]
    )
    Yx, Yy = physical_derivatives_2d(
        Y, Dr, Ds, geom["rx"], geom["sx"], geom["ry"], geom["sy"]
    )
    Zx, Zy = physical_derivatives_2d(
        Z, Dr, Ds, geom["rx"], geom["sx"], geom["ry"], geom["sy"]
    )

    Cx = Yx * Zy - Zx * Yy
    Cy = Zx * Xy - Xx * Zy
    Cz = Xx * Yy - Yx * Xy

    J_cross = np.sqrt(Cx * Cx + Cy * Cy + Cz * Cz)

    return {
        "Xx": Xx,
        "Xy": Xy,
        "Yx": Yx,
        "Yy": Yy,
        "Zx": Zx,
        "Zy": Zy,
        "J_cross": J_cross,
        "normal_x": Cx / J_cross,
        "normal_y": Cy / J_cross,
        "normal_z": Cz / J_cross,
    }


def streamfunction_flux_and_velocity_reconstruction(
    *,
    cache,
    Dr: np.ndarray,
    Ds: np.ndarray,
    geom: dict[str, np.ndarray],
    alpha0: float,
    u0: float,
    R: float,
    psi_sign: float,
) -> dict[str, np.ndarray]:
    r"""
    Construct stream-function flux and reconstruct physical velocity.

    We test both signs:

        psi = psi_sign * R * Omega dot X

    Flux convention:

        Fx_area = - psi_y
        Fy_area =   psi_x

    Then reconstruct:

        u^x = Fx_area / J
        u^y = Fy_area / J

        U_rec = u^x X_x + u^y X_y

    The expected physical velocity is:

        U_exact = Omega x X

    Theory suggests psi_sign = -1 should match U_exact
    for the flux convention above.
    """
    omega = solid_body_omega(alpha0=alpha0, u0=u0)

    psi = psi_sign * R * (
        omega[0] * cache.X
        + omega[1] * cache.Y
        + omega[2] * cache.Z
    )

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

    basis = surface_tangent_basis_and_jacobian(
        X=cache.X,
        Y=cache.Y,
        Z=cache.Z,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
    )

    J_const = math.pi * R * R
    J_cross = basis["J_cross"]

    # Reconstruction using exact equal-area constant J.
    ux_const = Fx_area / J_const
    uy_const = Fy_area / J_const

    Ux_rec_const = ux_const * basis["Xx"] + uy_const * basis["Xy"]
    Uy_rec_const = ux_const * basis["Yx"] + uy_const * basis["Yy"]
    Uz_rec_const = ux_const * basis["Zx"] + uy_const * basis["Zy"]

    # Reconstruction using discrete cross-product J.
    ux_cross = Fx_area / J_cross
    uy_cross = Fy_area / J_cross

    Ux_rec_cross = ux_cross * basis["Xx"] + uy_cross * basis["Xy"]
    Uy_rec_cross = ux_cross * basis["Yx"] + uy_cross * basis["Yy"]
    Uz_rec_cross = ux_cross * basis["Zx"] + uy_cross * basis["Zy"]

    Ux_exact, Uy_exact, Uz_exact = exact_solid_body_velocity_xyz(
        cache.X,
        cache.Y,
        cache.Z,
        omega=omega,
    )

    return {
        "psi": psi,
        "Fx_area": Fx_area,
        "Fy_area": Fy_area,
        "div_area": div_area,

        "Ux_exact": Ux_exact,
        "Uy_exact": Uy_exact,
        "Uz_exact": Uz_exact,

        "Ux_rec_const": Ux_rec_const,
        "Uy_rec_const": Uy_rec_const,
        "Uz_rec_const": Uz_rec_const,

        "Ux_rec_cross": Ux_rec_cross,
        "Uy_rec_cross": Uy_rec_cross,
        "Uz_rec_cross": Uz_rec_cross,

        "J_cross": J_cross,
    }


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
    psi_sign: float,
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

    surface_weights = build_surface_weights(
        rule,
        geom,
        R=R,
    )

    out = streamfunction_flux_and_velocity_reconstruction(
        cache=cache,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        alpha0=alpha0,
        u0=u0,
        R=R,
        psi_sign=psi_sign,
    )

    mask = cache.bad_mask

    div_stats = weighted_scalar_stats(
        out["div_area"],
        surface_weights,
        mask=mask,
    )

    # Velocity reconstruction error using J_const = pi R^2.
    e_const_x = out["Ux_rec_const"] - out["Ux_exact"]
    e_const_y = out["Uy_rec_const"] - out["Uy_exact"]
    e_const_z = out["Uz_rec_const"] - out["Uz_exact"]

    vel_const_stats = weighted_vector_error_stats(
        e_const_x,
        e_const_y,
        e_const_z,
        surface_weights,
        mask=mask,
    )

    # Velocity reconstruction error using J_cross from discrete derivatives.
    e_cross_x = out["Ux_rec_cross"] - out["Ux_exact"]
    e_cross_y = out["Uy_rec_cross"] - out["Uy_exact"]
    e_cross_z = out["Uz_rec_cross"] - out["Uz_exact"]

    vel_cross_stats = weighted_vector_error_stats(
        e_cross_x,
        e_cross_y,
        e_cross_z,
        surface_weights,
        mask=mask,
    )

    # Compare also with negative exact velocity to identify sign mistakes.
    e_cross_neg_x = out["Ux_rec_cross"] + out["Ux_exact"]
    e_cross_neg_y = out["Uy_rec_cross"] + out["Uy_exact"]
    e_cross_neg_z = out["Uz_rec_cross"] + out["Uz_exact"]

    vel_cross_neg_stats = weighted_vector_error_stats(
        e_cross_neg_x,
        e_cross_neg_y,
        e_cross_neg_z,
        surface_weights,
        mask=mask,
    )

    J_const = math.pi * R * R
    J_err = out["J_cross"] - J_const
    J_stats = weighted_scalar_stats(
        J_err,
        surface_weights,
        mask=mask,
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
        "psi_sign": float(psi_sign),

        "n_total_nodes": int(gdiag["n_total_nodes"]),
        "n_bad_nodes": int(gdiag["n_bad_nodes"]),
        "n_good_nodes": int(div_stats["n_good"]),

        "sqrtG_expected": float(gdiag["sqrtG_expected"]),
        "sqrtG_error_max_regular_from_A": float(gdiag["sqrtG_error_max_regular"]),
        "Jcross_L2_error": float(J_stats["L2"]),
        "Jcross_Linf_error": float(J_stats["Linf"]),

        "div_L2": float(div_stats["L2"]),
        "div_Linf": float(div_stats["Linf"]),
        "div_integral": float(div_stats["integral"]),

        "vel_constJ_L2_error": float(vel_const_stats["L2"]),
        "vel_constJ_Linf_error": float(vel_const_stats["Linf"]),

        "vel_crossJ_L2_error": float(vel_cross_stats["L2"]),
        "vel_crossJ_Linf_error": float(vel_cross_stats["Linf"]),

        "vel_crossJ_L2_error_against_negative_exact": float(vel_cross_neg_stats["L2"]),
        "vel_crossJ_Linf_error_against_negative_exact": float(vel_cross_neg_stats["Linf"]),
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
        f"{'sign':>7s} "
        f"{'div_L2':>12s} "
        f"{'velL2_crossJ':>15s} "
        f"{'velLinf_crossJ':>17s} "
        f"{'velL2_constJ':>15s} "
        f"{'Jcross_Linf':>14s} "
        f"{'bad':>5s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['psi_sign']:7.1f} "
            f"{r['div_L2']:12.4e} "
            f"{r['vel_crossJ_L2_error']:15.6e} "
            f"{r['vel_crossJ_Linf_error']:17.6e} "
            f"{r['vel_constJ_L2_error']:15.6e} "
            f"{r['Jcross_Linf_error']:14.6e} "
            f"{r['n_bad_nodes']:5d}"
        )


def print_best_sign_summary(rows: list[dict[str, float]]) -> None:
    print()
    print("=== Best sign summary by (nsub, alpha0), using vel_crossJ_L2_error ===")

    keys = sorted(set((r["nsub"], r["alpha0"]) for r in rows))
    header = (
        f"{'nsub':>6s} "
        f"{'alpha/pi':>10s} "
        f"{'best_sign':>10s} "
        f"{'best_vel_L2':>14s} "
        f"{'other_vel_L2':>14s} "
        f"{'ratio_other/best':>18s}"
    )
    print(header)
    print("-" * len(header))

    for nsub, alpha0 in keys:
        subset = [r for r in rows if r["nsub"] == nsub and r["alpha0"] == alpha0]
        subset = sorted(subset, key=lambda r: r["vel_crossJ_L2_error"])
        best = subset[0]
        other = subset[1] if len(subset) > 1 else None

        if other is None or best["vel_crossJ_L2_error"] <= 0:
            ratio = float("nan")
            other_l2 = float("nan")
        else:
            ratio = other["vel_crossJ_L2_error"] / best["vel_crossJ_L2_error"]
            other_l2 = other["vel_crossJ_L2_error"]

        print(
            f"{int(nsub):6d} "
            f"{alpha0 / math.pi:10.6f} "
            f"{best['psi_sign']:10.1f} "
            f"{best['vel_crossJ_L2_error']:14.6e} "
            f"{other_l2:14.6e} "
            f"{ratio:18.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether stream-function conservative flux reconstructs "
            "the intended physical solid-body velocity Omega x X."
        )
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
        "--psi-signs",
        type=float,
        nargs="+",
        default=[1.0, -1.0],
        help="Signs to test in psi = sign * R * Omega dot X.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_streamfunction_velocity_check"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    rows: list[dict[str, float]] = []

    print("=== SDG stream-function velocity reconstruction check ===")
    print(f"order     = {args.order}")
    print(f"N         = {N}")
    print(f"R         = {args.R}")
    print(f"u0        = {args.u0}")
    print(f"alphas    = {alphas}")
    print(f"psi_signs = {args.psi_signs}")
    print()

    for nsub in args.nsubs:
        for alpha0 in alphas:
            for psi_sign in args.psi_signs:
                row = run_one_case(
                    nsub=nsub,
                    order=args.order,
                    N=N,
                    R=args.R,
                    u0=args.u0,
                    alpha0=alpha0,
                    psi_sign=psi_sign,
                )
                rows.append(row)

    print_table(rows)
    print_best_sign_summary(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "velocity_check.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
