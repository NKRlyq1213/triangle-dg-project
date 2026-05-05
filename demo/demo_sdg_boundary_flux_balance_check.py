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
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as values.")
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

    l2 = math.sqrt(max(float(np.sum(w * vals * vals)), 0.0))
    linf = float(np.max(np.abs(vals)))

    return l2, linf


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
    r"""
    Compute flat open-boundary conservative flux integral:

        B = sum_{boundary faces} int_e (n_x Fx_area + n_y Fy_area) q ds

    Since the PDE is:

        d_t(J q) + div(F_area q) = 0

    the global mass derivative on an open flat boundary should satisfy:

        I_full ≈ -B

    when interior numerical fluxes are conservative.
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

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

    volume_int = weighted_integral(rhs_volume, weights, mask=cache.bad_mask)
    surface_diag_int = weighted_integral(rhs_diag, weights, mask=cache.bad_mask)
    surface_proj_int = weighted_integral(rhs_proj, weights, mask=cache.bad_mask)
    full_diag_int = weighted_integral(full_diag, weights, mask=cache.bad_mask)
    full_proj_int = weighted_integral(full_proj, weights, mask=cache.bad_mask)

    full_proj_l2, full_proj_linf = weighted_l2_linf(
        full_proj,
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

    balance_error = full_proj_int + B

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

        "volume_integral": float(volume_int),
        "surface_diag_integral": float(surface_diag_int),
        "surface_proj_integral": float(surface_proj_int),
        "full_diag_integral": float(full_diag_int),
        "full_proj_integral": float(full_proj_int),

        "boundary_flux_integral": float(B),
        "balance_error_full_plus_boundary": float(balance_error),

        "full_proj_L2": float(full_proj_l2),
        "full_proj_Linf": float(full_proj_linf),
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
        f"{'I_vol':>13s} "
        f"{'I_surfP':>13s} "
        f"{'I_fullP':>13s} "
        f"{'B_bdry':>13s} "
        f"{'I+B':>13s} "
        f"{'bdry':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>15s} "
            f"{r['volume_integral']:13.6e} "
            f"{r['surface_proj_integral']:13.6e} "
            f"{r['full_proj_integral']:13.6e} "
            f"{r['boundary_flux_integral']:13.6e} "
            f"{r['balance_error_full_plus_boundary']:13.6e} "
            f"{r['n_boundary_faces']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Boundary flux balance check for stream-function projected surface lifting."
        )
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
        default=str(ROOT / "outputs" / "sdg_boundary_flux_balance"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG boundary flux balance check ===")
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
    csv_path = output_dir / f"boundary_flux_balance_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
