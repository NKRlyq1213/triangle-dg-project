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
    reference_face_parameters,
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


def volume_normal_flux(op) -> np.ndarray:
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


def endpoint_node_mask(op, *, tol: float) -> np.ndarray:
    """
    Return mask of shape (3, Nfp) selecting face endpoints.

    Endpoint means local edge coordinate t is 0 or 1.
    """
    ts = reference_face_parameters(op.base.rs_nodes, op.base.ref_face)
    nfp = op.base.ref_face.nfp

    mask = np.zeros((3, nfp), dtype=bool)

    for f in range(3):
        t = np.asarray(ts[f], dtype=float)
        mask[f, :] = (np.abs(t) <= tol) | (np.abs(t - 1.0) <= tol)

    return mask


def face_category_masks(op) -> dict[str, np.ndarray]:
    seam_mask = np.asarray(op.base.conn_flat.is_boundary, dtype=bool)
    flat_mask = ~seam_mask
    all_mask = np.ones_like(seam_mask, dtype=bool)

    return {
        "all": all_mask,
        "flat": flat_mask,
        "seam": seam_mask,
    }


def weighted_stats(values: np.ndarray, weights: np.ndarray, face_mask: np.ndarray, node_mask: np.ndarray):
    """
    values, weights: (K, 3, Nfp)
    face_mask:       (K, 3)
    node_mask:       (3, Nfp)
    """
    mask = (
        face_mask[:, :, None]
        & node_mask[None, :, :]
        & np.isfinite(values)
        & np.isfinite(weights)
    )

    if not np.any(mask):
        return {
            "L2": float("nan"),
            "Linf": float("nan"),
            "integral": float("nan"),
            "measure": 0.0,
            "count": 0,
        }

    vals = values[mask]
    w = weights[mask]

    return {
        "L2": math.sqrt(max(float(np.sum(w * vals * vals)), 0.0)),
        "Linf": float(np.max(np.abs(vals))),
        "integral": float(np.sum(w * vals)),
        "measure": float(np.sum(w)),
        "count": int(vals.size),
    }


def collect_top_errors(
    *,
    op,
    diff: np.ndarray,
    a_edge: np.ndarray,
    a_vol: np.ndarray,
    max_records: int,
):
    base = op.base
    ref_face = base.ref_face
    conn = base.conn_seam

    t_faces = reference_face_parameters(base.rs_nodes, ref_face)

    records = []

    K = diff.shape[0]
    nfp = ref_face.nfp

    seam_face_mask = np.asarray(base.conn_flat.is_boundary, dtype=bool)

    for k in range(K):
        for f in range(3):
            ids = np.asarray(ref_face.face_node_ids[f], dtype=int)

            kp = int(conn.neighbor_elem[k, f])
            fp = int(conn.neighbor_face[k, f])

            for i in range(nfp):
                nid = int(ids[i])
                val = float(diff[k, f, i])

                records.append(
                    {
                        "abs_diff": abs(val),
                        "diff": val,
                        "a_edge": float(a_edge[k, f, i]),
                        "a_vol": float(a_vol[k, f, i]),
                        "elem": int(k),
                        "face": int(f),
                        "local_face_node": int(i),
                        "volume_node_id": int(nid),
                        "t_face": float(t_faces[f][i]),
                        "is_endpoint": bool(abs(t_faces[f][i]) <= 1.0e-12 or abs(t_faces[f][i] - 1.0) <= 1.0e-12),
                        "is_seam_face": bool(seam_face_mask[k, f]),
                        "neighbor_elem": kp,
                        "neighbor_face": fp,
                        "X": float(base.cache.X[k, nid]),
                        "Y": float(base.cache.Y[k, nid]),
                        "Z": float(base.cache.Z[k, nid]),
                        "patch_id": int(base.elem_patch_id[k]) if hasattr(base, "elem_patch_id") else -1,
                    }
                )

    records.sort(key=lambda r: r["abs_diff"], reverse=True)
    return records[:max_records]


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
    endpoint_tol: float,
):
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
    weights = face_weights_all(op)

    ep_mask = endpoint_node_mask(op, tol=endpoint_tol)
    int_mask = ~ep_mask

    face_masks = face_category_masks(op)

    st_ep_all = weighted_stats(diff, weights, face_masks["all"], ep_mask)
    st_int_all = weighted_stats(diff, weights, face_masks["all"], int_mask)

    st_ep_seam = weighted_stats(diff, weights, face_masks["seam"], ep_mask)
    st_int_seam = weighted_stats(diff, weights, face_masks["seam"], int_mask)

    st_ep_flat = weighted_stats(diff, weights, face_masks["flat"], ep_mask)
    st_int_flat = weighted_stats(diff, weights, face_masks["flat"], int_mask)

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

        "endpoint_all_L2": float(st_ep_all["L2"]),
        "endpoint_all_Linf": float(st_ep_all["Linf"]),
        "endpoint_all_measure": float(st_ep_all["measure"]),

        "interior_all_L2": float(st_int_all["L2"]),
        "interior_all_Linf": float(st_int_all["Linf"]),
        "interior_all_measure": float(st_int_all["measure"]),

        "endpoint_seam_L2": float(st_ep_seam["L2"]),
        "endpoint_seam_Linf": float(st_ep_seam["Linf"]),
        "endpoint_seam_measure": float(st_ep_seam["measure"]),

        "interior_seam_L2": float(st_int_seam["L2"]),
        "interior_seam_Linf": float(st_int_seam["Linf"]),
        "interior_seam_measure": float(st_int_seam["measure"]),

        "endpoint_flat_L2": float(st_ep_flat["L2"]),
        "endpoint_flat_Linf": float(st_ep_flat["Linf"]),

        "interior_flat_L2": float(st_int_flat["L2"]),
        "interior_flat_Linf": float(st_int_flat["Linf"]),

        "endpoint_all_L2_rate": float("nan"),
        "interior_all_L2_rate": float("nan"),
        "endpoint_seam_L2_rate": float("nan"),
        "interior_seam_L2_rate": float("nan"),
    }, collect_top_errors(
        op=op,
        diff=diff,
        a_edge=a_edge,
        a_vol=a_vol,
        max_records=20,
    )


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

                for name in [
                    "endpoint_all_L2",
                    "interior_all_L2",
                    "endpoint_seam_L2",
                    "interior_seam_L2",
                ]:
                    rate_name = name + "_rate"
                    if r[name] > 0.0 and prev[name] > 0.0:
                        r[rate_name] = math.log(prev[name] / r[name]) / math.log(ratio)

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
        f"{'ep_L2':>12s} "
        f"{'rate':>7s} "
        f"{'int_L2':>12s} "
        f"{'rate':>7s} "
        f"{'epSeam_L2':>12s} "
        f"{'rate':>7s} "
        f"{'intSeam_L2':>12s} "
        f"{'rate':>7s} "
        f"{'ep_Linf':>12s} "
        f"{'int_Linf':>12s}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['nsub']:6d} "
            f"{r['endpoint_all_L2']:12.4e} "
            f"{r['endpoint_all_L2_rate']:7.3f} "
            f"{r['interior_all_L2']:12.4e} "
            f"{r['interior_all_L2_rate']:7.3f} "
            f"{r['endpoint_seam_L2']:12.4e} "
            f"{r['endpoint_seam_L2_rate']:7.3f} "
            f"{r['interior_seam_L2']:12.4e} "
            f"{r['interior_seam_L2_rate']:7.3f} "
            f"{r['endpoint_all_Linf']:12.4e} "
            f"{r['interior_all_Linf']:12.4e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Endpoint diagnostic for a_edge - a_vol face flux mismatch."
    )

    parser.add_argument("--nsubs", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seam-tol", type=float, default=1.0e-10)
    parser.add_argument("--endpoint-tol", type=float, default=1.0e-12)
    parser.add_argument("--alphas", type=str, default="pi/4")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "sdg_face_flux_endpoint_diagnostic"),
    )

    args = parser.parse_args()

    N = args.order if args.N is None else args.N
    alphas = parse_alpha_list(args.alphas)

    print("=== SDG face flux endpoint diagnostic ===")
    print(f"order  = {args.order}")
    print(f"N      = {N}")
    print(f"R      = {args.R}")
    print(f"u0     = {args.u0}")
    print(f"tau    = {args.tau}")
    print(f"alphas = {alphas}")
    print()

    rows = []
    top_rows = []

    for alpha0 in alphas:
        for nsub in args.nsubs:
            row, top = run_one_case(
                nsub=nsub,
                order=args.order,
                N=N,
                R=args.R,
                u0=args.u0,
                alpha0=alpha0,
                tau=args.tau,
                seam_tol=args.seam_tol,
                endpoint_tol=args.endpoint_tol,
            )
            rows.append(row)

            # Keep top records only for the finest run of each alpha by default.
            if nsub == max(args.nsubs):
                for t in top:
                    out = dict(row)
                    out.update(t)
                    top_rows.append(out)

    add_rates(rows)
    print_table(rows)

    out = Path(args.output_dir)
    csv_path = out / f"face_flux_endpoint_summary_order{args.order}_N{N}.csv"
    top_path = out / f"face_flux_endpoint_top_errors_order{args.order}_N{N}.csv"

    write_csv(rows, csv_path)
    write_csv(top_rows, top_path)

    print()
    print("=== Output ===")
    print(csv_path)
    print(top_path)


if __name__ == "__main__":
    main()
