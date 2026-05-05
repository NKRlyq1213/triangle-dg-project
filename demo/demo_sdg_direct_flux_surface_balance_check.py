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
    sdg_surface_penalty_rhs_projected_repo_lift,
)
from operators.sdg_direct_flux_surface import (
    sdg_surface_direct_flux_rhs_projected,
    sdg_surface_direct_flux_rhs_diagonal,
    internal_direct_flux_pair_balance,
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
        eps = 0.1
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return cache.Z / R + eps * sign[:, None]

    if case == "element_checker":
        K = cache.X.shape[0]
        sign = np.where((np.arange(K) % 2) == 0, 1.0, -1.0)
        return sign[:, None] * np.ones_like(cache.X)

    raise ValueError(
        "Unknown q_case. Available: constant, sphere_x, sphere_y, sphere_z, "
        "flat_gaussian, element_jump, element_checker."
    )


def weighted_integral(values: np.ndarray, weights: np.ndarray, *, mask=None) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape.")

    if mask is None:
        good = np.isfinite(values) & np.isfinite(weights)
    else:
        mask = np.asarray(mask, dtype=bool)
        good = (~mask) & np.isfinite(values) & np.isfinite(weights)

    if not np.any(good):
        return float("nan")

    return float(np.sum(weights[good] * values[good]))


def weighted_l2_linf(values: np.ndarray, weights: np.ndarray, *, mask=None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if mask is None:
        good = np.isfinite(values) & np.isfinite(weights)
    else:
        mask = np.asarray(mask, dtype=bool)
        good = (~mask) & np.isfinite(values) & np.isfinite(weights)

    if not np.any(good):
        return float("nan"), float("nan")

    vals = values[good]
    w = weights[good]

    return (
        math.sqrt(max(float(np.sum(w * vals * vals)), 0.0)),
        float(np.max(np.abs(vals))),
    )


def build_surface_weights(rule: dict, geom: dict[str, np.ndarray], *, R: float) -> np.ndarray:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    J_flat = np.asarray(geom["J"], dtype=float)

    area_flat = 2.0 * J_flat[:, 0]
    sqrtG = equal_area_jacobian(R)

    return area_flat[:, None] * ws[None, :] * sqrtG


def boundary_flux_integral(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    rule: dict,
    ref_face,
    conn,
) -> float:
    we_all = np.asarray(rule["we"], dtype=float).reshape(-1)

    total = 0.0
    K = q.shape[0]

    for k in range(K):
        for f in range(3):
            if not conn.is_boundary[k, f]:
                continue

            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)
            we = we_all[ids]

            qM = q[k, ids]
            FxM = Fx_area[k, ids]
            FyM = Fy_area[k, ids]

            ndotFq = (
                conn.face_normal_x[k, f] * FxM
                + conn.face_normal_y[k, f] * FyM
            ) * qM

            total += conn.face_length[k, f] * float(np.sum(we * ndotFq))

    return float(total)


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

    MinvT = build_projected_surface_inverse_mass_T(
        N=N,
        rule=rule,
    )

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

    rhs_strong_proj = sdg_surface_penalty_rhs_projected_repo_lift(
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

    rhs_direct_proj = sdg_surface_direct_flux_rhs_projected(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
        N=N,
        J_area=J_area,
        tau=tau,
        boundary_mode="same_state",
        surface_inverse_mass_T=MinvT,
    )

    rhs_direct_diag = sdg_surface_direct_flux_rhs_diagonal(
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

    rhs_strong_proj = np.where(cache.bad_mask, np.nan, rhs_strong_proj)
    rhs_direct_proj = np.where(cache.bad_mask, np.nan, rhs_direct_proj)
    rhs_direct_diag = np.where(cache.bad_mask, np.nan, rhs_direct_diag)

    full_strong_proj = np.where(cache.bad_mask, np.nan, rhs_volume + rhs_strong_proj)
    full_direct_proj = np.where(cache.bad_mask, np.nan, rhs_volume + rhs_direct_proj)
    full_direct_diag = np.where(cache.bad_mask, np.nan, rhs_volume + rhs_direct_diag)

    I_vol = weighted_integral(rhs_volume, weights, mask=cache.bad_mask)
    I_strongP = weighted_integral(rhs_strong_proj, weights, mask=cache.bad_mask)
    I_directP = weighted_integral(rhs_direct_proj, weights, mask=cache.bad_mask)
    I_directD = weighted_integral(rhs_direct_diag, weights, mask=cache.bad_mask)

    I_full_strongP = weighted_integral(full_strong_proj, weights, mask=cache.bad_mask)
    I_full_directP = weighted_integral(full_direct_proj, weights, mask=cache.bad_mask)
    I_full_directD = weighted_integral(full_direct_diag, weights, mask=cache.bad_mask)

    full_directP_L2, full_directP_Linf = weighted_l2_linf(
        full_direct_proj,
        weights,
        mask=cache.bad_mask,
    )

    B = boundary_flux_integral(
        q,
        Fx_area,
        Fy_area,
        rule,
        ref_face,
        conn,
    )

    pair = internal_direct_flux_pair_balance(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        tau=tau,
    )

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

        "I_vol": float(I_vol),
        "I_strongP_surf": float(I_strongP),
        "I_directP_surf": float(I_directP),
        "I_directD_surf": float(I_directD),

        "I_full_strongP": float(I_full_strongP),
        "I_full_directP": float(I_full_directP),
        "I_full_directD": float(I_full_directD),

        "B_bdry": float(B),
        "directP_plus_boundary": float(I_full_directP + B),
        "strongP_plus_boundary": float(I_full_strongP + B),

        "full_directP_L2": float(full_directP_L2),
        "full_directP_Linf": float(full_directP_Linf),

        "fhat_pair_max_error": float(pair["max_pair_error"]),
        "fhat_pair_l2_unweighted": float(pair["pair_l2_unweighted"]),
        "fhat_n_pairs": int(pair["n_pairs"]),
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
        f"{'q_case':>15s} "
        f"{'I_vol':>12s} "
        f"{'I_strongP':>12s} "
        f"{'I_directP':>12s} "
        f"{'B_bdry':>12s} "
        f"{'direct+B':>12s} "
        f"{'pairErr':>12s} "
        f"{'bdry':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>15s} "
            f"{r['I_vol']:12.4e} "
            f"{r['I_full_strongP']:12.4e} "
            f"{r['I_full_directP']:12.4e} "
            f"{r['B_bdry']:12.4e} "
            f"{r['directP_plus_boundary']:12.4e} "
            f"{r['fhat_pair_max_error']:12.4e} "
            f"{r['n_boundary_faces']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct numerical flux surface balance diagnostic."
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
        default="element_checker",
        choices=[
            "constant",
            "sphere_x",
            "sphere_y",
            "sphere_z",
            "flat_gaussian",
            "element_jump",
            "element_checker",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_direct_flux_surface_balance"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG direct numerical flux surface balance check ===")
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
    csv_path = output_dir / f"direct_flux_surface_balance_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
