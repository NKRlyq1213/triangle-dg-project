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
from geometry.sphere_flat_metrics import build_sphere_flat_geometry_cache
from geometry.metrics import affine_geometric_factors_from_mesh
from operators.sdg_flattened_divergence import build_table1_reference_diff_operators
from operators.sdg_streamfunction_flux import (
    equal_area_jacobian,
    streamfunction_area_flux,
)
from operators.sdg_conservative_rhs import sdg_conservative_volume_rhs
from operators.sdg_surface_flux import (
    build_reference_face_data,
    build_surface_connectivity,
    surface_connectivity_summary,
)
from operators.sdg_projected_surface_flux import (
    build_projected_surface_inverse_mass_T,
    sdg_surface_penalty_rhs_diagonal_repo_lift,
    sdg_surface_penalty_rhs_projected_repo_lift,
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


def q_initial(case: str, cache, R: float) -> np.ndarray:
    case = case.lower().strip()

    if case == "constant":
        return np.ones_like(cache.X)

    if case == "sphere_x":
        return cache.X / R

    if case == "sphere_y":
        return cache.Y / R

    if case == "sphere_z":
        return cache.Z / R

    if case == "flat_gaussian":
        x = cache.x_flat
        y = cache.y_flat
        x0 = 0.25
        y0 = -0.15
        sigma = 0.22
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))

    if case == "element_jump":
        # Deliberately discontinuous DG field.
        # q_K = Z/R + eps * (-1)^K
        # This forces q^- - q^+ != 0 on interior faces.
        eps = 0.1
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return cache.Z / R + eps * sign[:, None]

    if case == "element_checker":
        # Stronger discontinuous field with no geometric smoothness assumption.
        # This isolates surface lifting / penalty behavior.
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return sign[:, None] * np.ones_like(cache.X)

    raise ValueError(
        "Unknown q_case. Available: constant, sphere_x, sphere_y, sphere_z, "
        "flat_gaussian, element_jump, element_checker."
    )


def weighted_stats(values: np.ndarray, weights: np.ndarray, *, mask=None) -> dict[str, float]:
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
            "L2": float("nan"),
            "Linf": float("nan"),
            "integral": float("nan"),
            "mean": float("nan"),
            "measure": float("nan"),
            "n_good": 0,
        }

    vals = values[good]
    w = weights[good]

    measure = float(np.sum(w))
    integral = float(np.sum(w * vals))
    l2_sq = float(np.sum(w * vals * vals))

    return {
        "L2": math.sqrt(max(l2_sq, 0.0)),
        "Linf": float(np.max(np.abs(vals))),
        "integral": integral,
        "mean": integral / measure,
        "measure": measure,
        "n_good": int(vals.size),
    }


def build_surface_weights(rule: dict, geom: dict[str, np.ndarray], *, R: float) -> np.ndarray:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    area_flat = 2.0 * J_flat[:, 0]
    sqrtG = equal_area_jacobian(R)

    return area_flat[:, None] * ws[None, :] * sqrtG


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

    weights = build_surface_weights(rule, geom, R=R)
    J_area = equal_area_jacobian(R)

    Fx_area, Fy_area, _psi = streamfunction_area_flux(
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

    ref_face = build_reference_face_data(rule)

    conn = build_surface_connectivity(
        VX=VX,
        VY=VY,
        EToV=EToV,
        rs_nodes=rs_nodes,
        ref_face=ref_face,
    )

    MinvT = build_projected_surface_inverse_mass_T(N=N, rule=rule)

    rhs_volume = sdg_conservative_volume_rhs(
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

    rhs_diag = sdg_surface_penalty_rhs_diagonal_repo_lift(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
        J_area=J_area,
        tau=tau,
        boundary_mode="same_state",
    )

    rhs_proj = sdg_surface_penalty_rhs_projected_repo_lift(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
        N=N,
        rs_nodes=rs_nodes,
        J_area=J_area,
        tau=tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=MinvT,
    )

    rhs_diag = np.where(cache.bad_mask, np.nan, rhs_diag)
    rhs_proj = np.where(cache.bad_mask, np.nan, rhs_proj)

    full_diag = np.where(cache.bad_mask, np.nan, rhs_volume + rhs_diag)
    full_proj = np.where(cache.bad_mask, np.nan, rhs_volume + rhs_proj)

    diff_proj_diag = np.where(cache.bad_mask, np.nan, rhs_proj - rhs_diag)

    st_vol = weighted_stats(rhs_volume, weights, mask=cache.bad_mask)
    st_diag = weighted_stats(rhs_diag, weights, mask=cache.bad_mask)
    st_proj = weighted_stats(rhs_proj, weights, mask=cache.bad_mask)
    st_full_diag = weighted_stats(full_diag, weights, mask=cache.bad_mask)
    st_full_proj = weighted_stats(full_proj, weights, mask=cache.bad_mask)
    st_diff = weighted_stats(diff_proj_diag, weights, mask=cache.bad_mask)

    csum = surface_connectivity_summary(conn)

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

        "n_internal_faces": int(csum["n_internal_faces"]),
        "n_boundary_faces": int(csum["n_boundary_faces"]),
        "max_face_match_error": float(csum["max_match_error"]),

        "volume_L2": float(st_vol["L2"]),

        "surface_diag_L2": float(st_diag["L2"]),
        "surface_diag_Linf": float(st_diag["Linf"]),
        "surface_diag_integral": float(st_diag["integral"]),

        "surface_proj_L2": float(st_proj["L2"]),
        "surface_proj_Linf": float(st_proj["Linf"]),
        "surface_proj_integral": float(st_proj["integral"]),

        "proj_minus_diag_L2": float(st_diff["L2"]),
        "proj_minus_diag_Linf": float(st_diff["Linf"]),
        "proj_minus_diag_integral": float(st_diff["integral"]),

        "full_diag_L2": float(st_full_diag["L2"]),
        "full_diag_Linf": float(st_full_diag["Linf"]),
        "full_diag_integral": float(st_full_diag["integral"]),

        "full_proj_L2": float(st_full_proj["L2"]),
        "full_proj_Linf": float(st_full_proj["Linf"]),
        "full_proj_integral": float(st_full_proj["integral"]),
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
        f"{'surf_diag':>13s} "
        f"{'surf_proj':>13s} "
        f"{'proj-diag':>13s} "
        f"{'full_proj':>13s} "
        f"{'full_proj_int':>14s} "
        f"{'bdry':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>13s} "
            f"{r['surface_diag_L2']:13.6e} "
            f"{r['surface_proj_L2']:13.6e} "
            f"{r['proj_minus_diag_L2']:13.6e} "
            f"{r['full_proj_L2']:13.6e} "
            f"{r['full_proj_integral']:14.6e} "
            f"{r['n_boundary_faces']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check projected surface lifting for stream-function SDG flux."
    )
    parser.add_argument("--nsubs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--alphas", type=str, default="0,pi/12,pi/6,pi/4")
    parser.add_argument(
        "--q-case",
        type=str,
        default="constant",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "flat_gaussian", "element_jump", "element_checker"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_projected_surface_lifting"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG projected surface lifting check ===")
    print(f"order  = {args.order}")
    print(f"N      = {N}")
    print(f"R      = {args.R}")
    print(f"u0     = {args.u0}")
    print(f"tau    = {args.tau}")
    print(f"q_case = {args.q_case}")
    print(f"alphas = {alphas}")
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
                )
            )

    print_table(rows)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"projected_surface_lifting_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()


