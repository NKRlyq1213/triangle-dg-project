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

from operators.sdg_edge_streamfunction_closed_sphere_rhs import (
    build_edge_streamfunction_closed_sphere_sdg_operator,
)
from operators.sdg_edge_streamfunction_flux import (
    edge_streamfunction_normal_flux,
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


def face_node_ids_array(ref_face) -> np.ndarray:
    return np.asarray(
        [np.asarray(ref_face.face_node_ids[f], dtype=int) for f in range(3)],
        dtype=int,
    )


def volume_normal_flux(op) -> np.ndarray:
    """
    Compute a_vol = n_x Fx_area + n_y Fy_area on local face nodes.

    Shape:
        (K, 3, Nfp)
    """
    base = op.base
    conn = base.conn_seam
    ref_face = base.ref_face

    K = base.Fx_area.shape[0]
    nfp = ref_face.nfp

    a = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)

            a[k, f, :] = (
                conn.face_normal_x[k, f] * base.Fx_area[k, ids]
                + conn.face_normal_y[k, f] * base.Fy_area[k, ids]
            )

    return a


def face_weights_all(op) -> np.ndarray:
    """
    Face quadrature physical weights:

        length[k,f] * we_local[f,i]
    """
    base = op.base
    ref_face = base.ref_face
    conn = base.conn_seam

    we_all = np.asarray(base.rule["we"], dtype=float).reshape(-1)

    K = conn.face_length.shape[0]
    nfp = ref_face.nfp

    w = np.zeros((K, 3, nfp), dtype=float)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)
            w[k, f, :] = conn.face_length[k, f] * we_all[ids]

    return w


def face_category_masks(op) -> dict[str, np.ndarray]:
    """
    Categorize local faces:
    - all: all local faces
    - flat_interior: faces that were already interior before seam closure
    - seam: faces that were boundary in flat connectivity but became paired
    """
    base = op.base

    all_mask = np.ones_like(base.conn_seam.face_length, dtype=bool)
    seam_mask = np.asarray(base.conn_flat.is_boundary, dtype=bool)
    flat_mask = ~seam_mask

    return {
        "all": all_mask,
        "flat_interior": flat_mask,
        "seam": seam_mask,
    }


def weighted_stats_face(values: np.ndarray, weights: np.ndarray, face_mask: np.ndarray) -> dict[str, float]:
    """
    Weighted face L2/Linf/integral over selected faces.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    face_mask = np.asarray(face_mask, dtype=bool)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape.")
    if face_mask.shape != values.shape[:2]:
        raise ValueError("face_mask must have shape (K, 3).")

    mask = face_mask[:, :, None] & np.isfinite(values) & np.isfinite(weights)

    if not np.any(mask):
        return {
            "L2": float("nan"),
            "Linf": float("nan"),
            "integral": float("nan"),
            "measure": float("nan"),
        }

    vals = values[mask]
    w = weights[mask]

    measure = float(np.sum(w))
    integral = float(np.sum(w * vals))
    l2 = math.sqrt(max(float(np.sum(w * vals * vals)), 0.0))
    linf = float(np.max(np.abs(vals)))

    return {
        "L2": l2,
        "Linf": linf,
        "integral": integral,
        "measure": measure,
    }


def pair_error_for_flux(a_face: np.ndarray, op) -> dict[str, float]:
    return edge_flux_pair_error(
        a_face,
        op.base.ref_face,
        op.base.conn_seam,
    )


def run_one_case(
    *,
    nsub: int,
    order: int,
    N: int,
    R: float,
    u0: float,
    alpha0: float,
    tau: float,
    seam_tol: float,
) -> dict[str, float | int]:
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

    a_edge = edge_streamfunction_normal_flux(
        op.base.psi,
        op.base.ref_face,
        op.base.conn_seam,
        op.face_Dt,
    )

    a_vol = volume_normal_flux(op)

    diff = a_edge - a_vol

    w = face_weights_all(op)
    masks = face_category_masks(op)

    st_all = weighted_stats_face(diff, w, masks["all"])
    st_flat = weighted_stats_face(diff, w, masks["flat_interior"])
    st_seam = weighted_stats_face(diff, w, masks["seam"])

    st_edge_norm = weighted_stats_face(a_edge, w, masks["all"])

    pair_edge = pair_error_for_flux(a_edge, op)
    pair_vol = pair_error_for_flux(a_vol, op)

    rel_L2 = st_all["L2"] / max(st_edge_norm["L2"], 1.0e-300)

    return {
        "nsub": int(nsub),
        "h_proxy": float(1.0 / nsub),
        "order": int(order),
        "N": int(N),
        "R": float(R),
        "u0": float(u0),
        "alpha0": float(alpha0),
        "alpha0_over_pi": float(alpha0 / math.pi),
        "tau": float(tau),

        "n_seam_pairs": int(op.base.seam_info.n_seam_pairs),
        "n_unmatched_boundary_faces": int(op.base.seam_info.n_unmatched_boundary_faces),
        "max_seam_match_error": float(op.base.seam_info.max_seam_match_error),

        "diff_all_L2": float(st_all["L2"]),
        "diff_all_Linf": float(st_all["Linf"]),
        "diff_all_integral": float(st_all["integral"]),
        "diff_all_rel_L2": float(rel_L2),

        "diff_flat_L2": float(st_flat["L2"]),
        "diff_flat_Linf": float(st_flat["Linf"]),

        "diff_seam_L2": float(st_seam["L2"]),
        "diff_seam_Linf": float(st_seam["Linf"]),

        "a_edge_L2": float(st_edge_norm["L2"]),

        "edge_pair_error": float(pair_edge["edge_flux_pair_max_error"]),
        "vol_pair_error": float(pair_vol["edge_flux_pair_max_error"]),

        "diff_all_L2_rate": float("nan"),
        "diff_all_Linf_rate": float("nan"),
        "diff_seam_L2_rate": float("nan"),
    }


def add_rates(rows: list[dict]) -> None:
    groups = {}
    for r in rows:
        key = (r["alpha0"], r["order"], r["N"])
        groups.setdefault(key, []).append(r)

    for group in groups.values():
        group.sort(key=lambda x: x["nsub"])

        prev = None
        for r in group:
            if prev is not None:
                ratio = float(r["nsub"]) / float(prev["nsub"])

                if r["diff_all_L2"] > 0.0 and prev["diff_all_L2"] > 0.0:
                    r["diff_all_L2_rate"] = math.log(prev["diff_all_L2"] / r["diff_all_L2"]) / math.log(ratio)

                if r["diff_all_Linf"] > 0.0 and prev["diff_all_Linf"] > 0.0:
                    r["diff_all_Linf_rate"] = math.log(prev["diff_all_Linf"] / r["diff_all_Linf"]) / math.log(ratio)

                if r["diff_seam_L2"] > 0.0 and prev["diff_seam_L2"] > 0.0:
                    r["diff_seam_L2_rate"] = math.log(prev["diff_seam_L2"] / r["diff_seam_L2"]) / math.log(ratio)

            prev = r


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
        f"{'diff_L2':>12s} "
        f"{'rate':>7s} "
        f"{'diff_Linf':>12s} "
        f"{'rate':>7s} "
        f"{'seam_L2':>12s} "
        f"{'rate':>7s} "
        f"{'edgePair':>12s} "
        f"{'volPair':>12s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['alpha0_over_pi']:10.6f} "
            f"{r['diff_all_L2']:12.4e} "
            f"{r['diff_all_L2_rate']:7.3f} "
            f"{r['diff_all_Linf']:12.4e} "
            f"{r['diff_all_Linf_rate']:7.3f} "
            f"{r['diff_seam_L2']:12.4e} "
            f"{r['diff_seam_L2_rate']:7.3f} "
            f"{r['edge_pair_error']:12.4e} "
            f"{r['vol_pair_error']:12.4e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face flux consistency check: a_edge=-d_s psi vs a_vol=n dot F."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)
    parser.add_argument("--alphas", type=str, default="pi/4")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_face_flux_consistency"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG face flux consistency check ===")
    print(f"order  = {args.order}")
    print(f"N      = {N}")
    print(f"R      = {args.R}")
    print(f"u0     = {args.u0}")
    print(f"tau    = {args.tau}")
    print(f"alphas = {alphas}")
    print()

    rows = []
    for alpha0 in alphas:
        for nsub in args.nsubs:
            rows.append(
                run_one_case(
                    nsub=nsub,
                    order=args.order,
                    N=N,
                    R=args.R,
                    u0=args.u0,
                    alpha0=alpha0,
                    tau=args.tau,
                    seam_tol=args.seam_tol,
                )
            )

    add_rates(rows)
    print_table(rows)

    out = Path(args.output_dir)
    csv_path = out / f"face_flux_consistency_order{args.order}_N{N}.csv"
    write_csv(rows, csv_path)

    print()
    print("=== Output ===")
    print(csv_path)


if __name__ == "__main__":
    main()
