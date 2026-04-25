from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.table1_rules import load_table1_rule
from geometry.sphere_flat_mesh import sphere_flat_square_mesh
from geometry.sphere_flat_metrics import build_sphere_flat_geometry_cache, geometry_diagnostics
from operators.sdg_flattened_divergence import (
    build_table1_reference_diff_operators,
    sdg_flattened_cartesian_divergence,
)
from problems.sphere_advection import flattened_velocity_from_cache


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate SDG flattened divergence using constant-state test."
    )
    p.add_argument("--order", type=int, default=4, help="Table1 order.")
    p.add_argument("--N", type=int, default=None, help="Polynomial degree. Default: N=order.")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=np.pi / 4.0)
    p.add_argument(
        "--n-sub-list",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated n_sub values.",
    )
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def triangle_areas(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> np.ndarray:
    VX = np.asarray(VX, dtype=float)
    VY = np.asarray(VY, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    x1 = VX[EToV[:, 0]]
    y1 = VY[EToV[:, 0]]
    x2 = VX[EToV[:, 1]]
    y2 = VY[EToV[:, 1]]
    x3 = VX[EToV[:, 2]]
    y3 = VY[EToV[:, 2]]

    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    return area


def masked_weighted_norms(
    values: np.ndarray,
    weights: np.ndarray,
    areas: np.ndarray,
    bad_mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute area-weighted L2 and Linf.

    Integral approximation:
        ∫_K f^2 dxdy ≈ area_K * Σ_i w_i f_i^2

    Table1 weights are assumed normalized for the triangle rule.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    areas = np.asarray(areas, dtype=float).reshape(-1)
    bad_mask = np.asarray(bad_mask, dtype=bool)

    if values.ndim != 2:
        raise ValueError("values must have shape (K,Np).")
    K, Np = values.shape

    if weights.shape != (Np,):
        raise ValueError("weights must have shape (Np,).")
    if areas.shape != (K,):
        raise ValueError("areas must have shape (K,).")
    if bad_mask.shape != values.shape:
        raise ValueError("bad_mask must have same shape as values.")

    regular = (~bad_mask) & np.isfinite(values)

    l2_sq = 0.0
    linf = 0.0
    n_regular = 0
    n_bad = int(np.sum(~regular))

    for k in range(K):
        g = regular[k]
        if not np.any(g):
            continue

        vk = values[k, g]
        wk = weights[g]

        l2_sq += areas[k] * float(np.dot(wk, vk * vk))
        linf = max(linf, float(np.max(np.abs(vk))))
        n_regular += int(np.sum(g))

    return {
        "L2": math.sqrt(l2_sq),
        "Linf": linf,
        "n_regular_nodes": n_regular,
        "n_bad_or_nan_nodes": n_bad,
    }


def convergence_rates(errors: list[float], nsubs: list[int]) -> list[float]:
    rates = [math.nan]
    for i in range(1, len(errors)):
        e0 = errors[i - 1]
        e1 = errors[i]
        h0 = 1.0 / nsubs[i - 1]
        h1 = 1.0 / nsubs[i]

        if e0 <= 0.0 or e1 <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e0 / e1) / math.log(h0 / h1))

    return rates


def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(rows: list[dict], out: Path):
    nsubs = np.array([r["n_sub"] for r in rows], dtype=float)
    h = 1.0 / nsubs
    L2 = np.array([r["L2_div_constant"] for r in rows], dtype=float)
    Linf = np.array([r["Linf_div_constant"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, L2, marker="o", label="L2")
    ax.loglog(h, Linf, marker="s", label="Linf")
    ax.invert_xaxis()
    ax.set_xlabel("h ~ 1/n_sub")
    ax.set_ylabel("error for div_flat(u), q=1")
    ax.set_title("SDG flattened divergence constant-state validation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    N = args.order if args.N is None else args.N
    nsubs = [int(s.strip()) for s in args.n_sub_list.split(",") if s.strip()]

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sdg_divergence_validation" / (
            f"order{args.order}_N{N}_R{args.R:g}_alpha{args.alpha0:.6g}"
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rule = load_table1_rule(args.order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)
    weights = np.asarray(rule["ws"], dtype=float).reshape(-1)

    Dr, Ds = build_table1_reference_diff_operators(rule, N=N)

    rows: list[dict] = []

    for n_sub in nsubs:
        VX, VY, EToV, elem_patch_id = sphere_flat_square_mesh(
            n_sub=n_sub,
            R=1.0,
        )

        cache = build_sphere_flat_geometry_cache(
            rs_nodes=rs_nodes,
            VX=VX,
            VY=VY,
            EToV=EToV,
            elem_patch_id=elem_patch_id,
            R=args.R,
        )

        u1, u2, _, _ = flattened_velocity_from_cache(
            cache,
            u0=1.0,
            alpha0=args.alpha0,
        )

        q = np.ones_like(cache.x_flat)

        div = sdg_flattened_cartesian_divergence(
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

        areas = triangle_areas(VX, VY, EToV)
        norms = masked_weighted_norms(
            values=div,
            weights=weights,
            areas=areas,
            bad_mask=cache.bad_mask,
        )

        geom_diag = geometry_diagnostics(cache, R=args.R)

        row = {
            "n_sub": int(n_sub),
            "K": int(EToV.shape[0]),
            "Np": int(rs_nodes.shape[0]),
            "total_dof": int(EToV.shape[0] * rs_nodes.shape[0]),
            "h": float(1.0 / n_sub),
            "L2_div_constant": norms["L2"],
            "Linf_div_constant": norms["Linf"],
            "n_regular_nodes": norms["n_regular_nodes"],
            "n_bad_or_nan_nodes": norms["n_bad_or_nan_nodes"],
            "detA_error_max_regular": geom_diag["detA_error_max_regular"],
            "sqrtG_error_max_regular": geom_diag["sqrtG_error_max_regular"],
            "A_Ainv_err_max_regular": geom_diag["A_Ainv_err_max_regular"],
            "Ainv_numpy_err_max_regular": geom_diag["Ainv_numpy_err_max_regular"],
        }
        rows.append(row)

        print(
            f"[constant div validation] n_sub={n_sub:4d} | "
            f"K={row['K']:6d} | DOF={row['total_dof']:8d} | "
            f"L2={row['L2_div_constant']:.6e} | "
            f"Linf={row['Linf_div_constant']:.6e} | "
            f"bad={row['n_bad_or_nan_nodes']}"
        )

    L2_rates = convergence_rates([r["L2_div_constant"] for r in rows], nsubs)
    Linf_rates = convergence_rates([r["Linf_div_constant"] for r in rows], nsubs)

    for r, r2, ri in zip(rows, L2_rates, Linf_rates):
        r["rate_L2"] = r2
        r["rate_Linf"] = ri

    csv_path = outdir / "constant_divergence_validation.csv"
    fig_path = outdir / "constant_divergence_convergence.png"

    save_csv(rows, csv_path)
    plot_convergence(rows, fig_path)

    print("\n=== Results table ===")
    header = f"{'n_sub':>8s} {'K':>8s} {'L2':>14s} {'rate':>8s} {'Linf':>14s} {'rate':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt_rate(v):
            return "   -   " if not np.isfinite(v) else f"{v:8.3f}"

        print(
            f"{r['n_sub']:8d} {r['K']:8d} "
            f"{r['L2_div_constant']:14.6e} {fmt_rate(r['rate_L2'])} "
            f"{r['Linf_div_constant']:14.6e} {fmt_rate(r['rate_Linf'])}"
        )

    print("\n=== Output ===")
    print(csv_path)
    print(fig_path)


if __name__ == "__main__":
    main()
