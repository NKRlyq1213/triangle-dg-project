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
from operators.sdg_surface_flux import (
    build_reference_face_data,
    build_surface_connectivity,
    sdg_surface_penalty_rhs,
    surface_connectivity_summary,
)
from operators.rhs_split_conservative_exchange import numerical_flux_penalty


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

    raise ValueError(
        "Unknown q_case. Available: constant, sphere_x, sphere_y, sphere_z, flat_gaussian."
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


def surface_rhs_using_repo_penalty(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    ref_face,
    conn,
    *,
    J_area: float,
    tau: float,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
) -> np.ndarray:
    """
    Same lifting/scaling as sdg_surface_penalty_rhs, but the penalty p
    is computed by the repo's existing numerical_flux_penalty function.
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if not (q.shape == Fx_area.shape == Fy_area.shape):
        raise ValueError("q, Fx_area, Fy_area must have the same shape.")

    mode = boundary_mode.lower().strip()
    if mode not in ("same_state", "constant"):
        raise ValueError("boundary_mode must be 'same_state' or 'constant'.")

    K, _Np = q.shape
    rhs_surface = np.zeros_like(q, dtype=float)

    for k in range(K):
        for f in range(3):
            ids = ref_face.face_node_ids[f]
            wratio = ref_face.face_wratio[f]

            qM = q[k, ids]
            FxM = Fx_area[k, ids]
            FyM = Fy_area[k, ids]

            ndotF = (
                conn.face_normal_x[k, f] * FxM
                + conn.face_normal_y[k, f] * FyM
            )

            if conn.is_boundary[k, f]:
                if mode == "same_state":
                    qP = qM
                else:
                    qP = np.full_like(qM, float(q_boundary_value))
            else:
                kp = conn.neighbor_elem[k, f]
                idsP = conn.neighbor_node_ids[k, f, :]
                qP = q[kp, idsP]

            p = numerical_flux_penalty(
                ndotF,
                qM,
                qP,
                tau=tau,
            )

            scale = conn.face_length[k, f] / conn.area_flat[k] / float(J_area)
            rhs_surface[k, ids] += scale * wratio * p

    return rhs_surface


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

    rhs_ours = sdg_surface_penalty_rhs(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        J_area=J_area,
        tau=tau,
        boundary_mode="same_state",
    )

    rhs_repo = surface_rhs_using_repo_penalty(
        q,
        Fx_area,
        Fy_area,
        ref_face,
        conn,
        J_area=J_area,
        tau=tau,
        boundary_mode="same_state",
    )

    diff = rhs_ours - rhs_repo

    rhs_ours = np.where(cache.bad_mask, np.nan, rhs_ours)
    rhs_repo = np.where(cache.bad_mask, np.nan, rhs_repo)
    diff = np.where(cache.bad_mask, np.nan, diff)

    st_ours = weighted_stats(rhs_ours, weights, mask=cache.bad_mask)
    st_repo = weighted_stats(rhs_repo, weights, mask=cache.bad_mask)
    st_diff = weighted_stats(diff, weights, mask=cache.bad_mask)

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

        "ours_L2": float(st_ours["L2"]),
        "repo_L2": float(st_repo["L2"]),
        "diff_L2": float(st_diff["L2"]),
        "diff_Linf": float(st_diff["Linf"]),
        "diff_integral": float(st_diff["integral"]),
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
        f"{'tau':>7s} "
        f"{'ours_L2':>13s} "
        f"{'repo_L2':>13s} "
        f"{'diff_L2':>13s} "
        f"{'diff_Linf':>13s} "
        f"{'bdry':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['q_case']:>13s} "
            f"{r['tau']:7.3f} "
            f"{r['ours_L2']:13.6e} "
            f"{r['repo_L2']:13.6e} "
            f"{r['diff_L2']:13.6e} "
            f"{r['diff_Linf']:13.6e} "
            f"{r['n_boundary_faces']:6d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generated SDG surface penalty with existing repo numerical_flux_penalty."
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
        default="sphere_z",
        choices=["constant", "sphere_x", "sphere_y", "sphere_z", "flat_gaussian"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_repo_penalty_alignment"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG repo penalty alignment check ===")
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
    csv_path = output_dir / f"repo_penalty_alignment_{args.q_case}_tau{args.tau:g}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
